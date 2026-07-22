# 03 · action chunk 不是 action queue

[上一篇：Policy 接口](02_policy_inference.md) · [返回分支入口](../README.md) · [下一篇：Temporal Ensembling](04_temporal_ensembling.md)

这两个概念总在同一段代码里出现，却属于不同层级：

| 概念 | 属于哪里 | 一句话定义 |
| --- | --- | --- |
| action chunk | 模型输出 | ACT 一次预测的未来动作序列 |
| action queue | Policy 推理缓存 | 普通推理模式中等待逐帧执行的动作列表 |

queue 更像待办清单，不是第二个神经网络。📬

## action chunk：模型一次做出的局部计划

底层 `ACT.forward()` 输出：

```text
actions: [B, C, A]
```

- `B`：batch size；真机单实例通常为 1，但实现保留 batch 维。
- `C`：`chunk_size`，模型一次预测多少个环境时间步。
- `A`：每个动作的维度。

`C` 不是推理迭代次数。Decoder 使用 `C` 个动作 query，一次前向并行产生这些位置的隐藏表示，然后 action head 映射成动作。

### chunk 为什么能增强局部一致性

如果每帧只预测一个动作，每个动作主要对当前 observation 独立负责。ACT 同时预测一段动作，Decoder 中不同 query 还会通过 self-attention 交流，因此“向胶水靠近—微调—合拢夹爪”可以在同一段序列表示里共同建模。

这不等于输出必然平滑：

- 示范本身不一致，模型也可能学到冲突；
- 新旧 chunk 的边界仍可能有跳变；
- 控制频率和硬件通信也会制造停顿；
- action 后处理或机器人驱动还可能改变实际轨迹。

### chunk 太短与太长的代价

| 选择 | 好处 | 代价 |
| --- | --- | --- |
| 短 chunk | 预测目标更近，模型不必覆盖太长未来 | 序列约束变弱，若每次都执行完则推理更频繁 |
| 长 chunk | 一次表达更长的局部动作结构 | 网络要预测更远未来；若执行太多会依赖旧 observation，开环误差累积 |

## action queue：Policy 保存的待执行动作

普通模式下，`reset()` 创建：

```python
self._action_queue = deque([], maxlen=self.config.n_action_steps)
```

queue 保存的是按时间拆开的 `[B,A]` tensor。它不包含 Transformer 隐藏状态，也不重新计算注意力。只要 queue 还有动作，`select_action()` 就直接 `popleft()`；因此这一步便宜得多。

queue 为空后，Policy 才用当前 observation 再运行 `predict_action_chunk()`。这个“观察—规划一段—执行若干步—再观察”的节奏，构成普通 ACT 在部署端的 receding-horizon 行为。

## `chunk_size` 与 `n_action_steps` 的关系

当前 `ACTConfig` 明确要求：

```text
1 ≤ n_action_steps ≤ chunk_size
```

模型总是预测 `chunk_size` 个动作；普通 queue 只保留前 `n_action_steps`。因此：

```text
chunk_size = 100
n_action_steps = 20

预测：a0 ... a99
入队：a0 ... a19
丢弃：a20 ... a99
```

丢弃后半段看起来有点“浪费”，但换来的是更早用新 observation 重规划。模型训练时仍学习完整 100 步结构，部署时不必把旧计划从头吃到尾。

## SO-101 抓胶水的假设示例

> ⚠️ 下表中的 `chunk_size=100` 与各个 `n_action_steps` 是**假设示例**，不是我已确认的 checkpoint 配置。

假设控制频率固定，模型每次预测 100 个动作：

| `n_action_steps` | 重新观察并推理 | 推理次数 | 闭环程度 | 动作连续性倾向 | 纠偏速度 |
| --- | --- | --- | --- | --- | --- |
| 1 | 每个控制步 | 最高 | 最强 | 容易受相邻 chunk 差异影响 | 最快 |
| 10 | 每 10 步 | 较高 | 较强 | 同一 chunk 内较一致 | 较快 |
| 50 | 每 50 步 | 较低 | 较弱 | 长段内连续，边界少 | 较慢 |
| 100 | 整个 chunk 执行完 | 最低 | 最弱 | 最少重新规划，但最依赖旧观测 | 最慢 |

例如机械臂在第 12 步接近胶水时发生轻微偏差：

- `n_action_steps=10` 已经进行第二次推理，可能在较新图像上修正；
- `n_action_steps=100` 仍在执行最初看到目标时生成的计划，直到整段结束才重新观察。

但 `n_action_steps=1` 也不是无条件最好。每步都推理会增加 GPU 负担；如果推理赶不上控制周期，机器人可能等待。闭环更频繁与实时性之间需要一起测。

## 为什么 queue 清空才重新运行 Transformer

普通模式是在主动复用已经算好的计划，而不是“忘记看相机”：

```mermaid
flowchart LR
    O1["观测 t0"] --> P1["预测 C 步"] --> Q1["执行 N 步"] --> O2["观测 tN"]
    O2 --> P2["重新预测 C 步"]
```

`N=n_action_steps`。在 queue 消费期间，外部控制循环可能仍在采集 observation，但 `select_action()` 不会用它重新跑模型，直到 queue 空。

这也是为什么 action queue 影响闭环程度：模型多久愿意重新听取环境意见，由 `n_action_steps` 决定。

## `reset()` 为什么不能省

episode reset 往往意味着机器人、目标或计时器回到新起点。Policy 的内部状态也必须同步：

- 普通模式：清空 `_action_queue`；
- Temporal Ensembling：清空在线融合张量与计数。

若普通 queue 跨 episode 保留，下一轮第一个 observation 到来时 queue 不是空的，Policy 会直接返回上一轮剩余动作，甚至不会调用模型。这不是“模型泛化失败”，而是缓存生命周期错误。

## 容易混淆的四句话

- action chunk 是神经网络输出，queue 是 Python `deque`。
- `predict_action_chunk()` 生成完整 chunk，不负责执行。
- `select_action()` 每次只返回一个动作，即使它刚刚预测了 100 个。
- Temporal Ensembling 开启时没有普通 action queue；它每步预测并在线融合。

## 证据表

| 我的理解 | LeRobot 源码证据 | ACT 论文/官方代码 | SO-101 实验对应 |
| --- | --- | --- | --- |
| `chunk_size` 与 `n_action_steps` 是两个参数 | [`ACTConfig`](https://github.com/huggingface/lerobot/blob/1427d35ef58ab46651dc7ef78bde81642090c861/src/lerobot/policies/act/configuration_act.py#L80-L177) | 原始实现用 `num_queries` 预测 chunk，普通评估用 `query_frequency` 消费 | 我的 checkpoint 精确数值待读取 `config.json` |
| queue 只在为空时重新推理 | [`select_action()`](https://github.com/huggingface/lerobot/blob/1427d35ef58ab46651dc7ef78bde81642090c861/src/lerobot/policies/act/modeling_act.py#L86-L122) | 原始 [`eval_bc()`](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/imitate_episodes.py#L210-L269) 按 query frequency 更新 `all_actions` | 真机动作连续是定性观察，尚未做 `n_action_steps` 对照 |
| episode reset 必须清缓存 | [`ACTPolicy.reset()`](https://github.com/huggingface/lerobot/blob/1427d35ef58ab46651dc7ef78bde81642090c861/src/lerobot/policies/act/modeling_act.py#L78-L85) | 原始聚合张量在每个 rollout 内重新创建 | 部署框架是否在每轮正确调用 reset 仍可通过日志/入口继续核对 |
