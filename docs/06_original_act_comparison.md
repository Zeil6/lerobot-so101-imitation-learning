# 06 · LeRobot ACT 与原始 ACT 官方实现对照

[上一篇：Transformer 层](05_transformer_layers.md) · [返回分支入口](../README.md) · [下一篇：源码地图](07_source_map.md)

“都叫 ACT”只说明算法家族相同，不代表数据文件、Policy API、checkpoint 或部署循环可以直接互换。本篇固定比较：

- LeRobot：[`1427d35ef58ab46651dc7ef78bde81642090c861`](https://github.com/huggingface/lerobot/tree/1427d35ef58ab46651dc7ef78bde81642090c861)
- ACT 官方代码：[`742c753c0d4a5d87076c8f69e5628c79a8cc5488`](https://github.com/tonyzhaozh/act/tree/742c753c0d4a5d87076c8f69e5628c79a8cc5488)
- ACT 论文：[arXiv:2304.13705](https://arxiv.org/abs/2304.13705)

## 核心机制哪些保持一致

两边最重要的继承关系仍然清楚：

- 当前 observation 条件下预测一个 action chunk；
- 训练时用真实动作和 state 经过 CVAE encoder 得到 latent 分布；
- 用重参数化采样 latent，推理时 latent 取零；
- ResNet 图像特征、state/latent token 进入 Transformer；
- Decoder query 输出整段动作；
- loss 由 masked L1 与 `kl_weight × KL` 组成；
- 部署可选择分块执行或 temporal aggregation/ensembling。

这些是算法骨架。真正让我绕一下的是：LeRobot 改变了“谁负责准备数据、谁负责归一化、谁保存 queue、谁把动作发出去”。

## 逐项比较

| 维度 | ACT 原始官方实现 | LeRobot ACT | 对我的理解有什么影响 |
| --- | --- | --- | --- |
| 任务与硬件 | 面向 ALOHA 双臂操作与模拟任务 | 面向统一机器人生态，SO-101 由 feature schema/robot 接口适配 | 我复现的是 ACT 算法在 SO-101 上的 LeRobot 实现，不是原 ALOHA 硬件原样复刻 |
| 数据接口 | HDF5 episode，读取 `qpos`、`qvel`、`action`、相机图像 | `LeRobotDataset`、metadata、`PolicyFeature`、`delta_timestamps` | action chunk 的监督含义相同，取样与元数据接口不同 |
| 单臂/双臂适配 | `state_dim=14`、多处 Linear 输入 14 被硬编码 | state/action 维度来自 `input_features`/`output_features` | SO-101 不需要把 6 维硬塞进 14 维原始模型 |
| 摄像头组织 | 输入 `[B,K,C,H,W]`，按 `camera_names` 循环；视觉分支共享 `backbones[0]`，特征沿宽度拼接 | 多个 image feature key 整理成 list，共用一个 torchvision backbone，特征 token 加入 Encoder sequence | 相机 key、数量和 shape 必须与训练一致；组织方式不是 checkpoint 兼容保证 |
| 配置方式 | argparse + task config dict，例如 `num_queries`、`temporal_agg` | `ACTConfig` dataclass，例如 `chunk_size`、`n_action_steps`、`temporal_ensemble_coeff` | 参数名不能机械一一替换，要看执行语义 |
| Policy 封装 | `policy.py::ACTPolicy` 包装 `DETRVAE`，optimizer 也由构建函数带回 | `PreTrainedPolicy` 下的 `ACTPolicy` 包装 `ACT`，统一 `forward/select_action/predict_action_chunk/reset` | LeRobot 把训练 loss 与部署缓存放进统一 API |
| 训练入口 | `imitate_episodes.py` 负责加载数据、训练/验证、保存 best/last | 可走通用训练 CLI；教程另有 `examples/tutorial/act/act_training_example.py` | 教程是一条最小链路，不等于我的 CLI 训练脚本原样执行它 |
| normalization | Dataset 中归一化 qpos/action；Policy 内对图像用 ImageNet normalize；评估循环外 post-process action | pre/post processor 按 feature schema 与 stats 做 normalization/unnormalization | 当前 `predict_action_chunk()` 输出仍在模型空间，机器人尺度由 postprocessor 恢复 |
| action queue | 没有 `deque` 类；评估循环用 `all_actions` 与 `t % query_frequency` 选择当前动作 | `ACTPolicy` 内维护 `_action_queue`，空时填充，逐次 `popleft()` | action queue 是 LeRobot 工程封装，不是论文中新增加的网络层 |
| Temporal Ensembling | `imitate_episodes.py` 外部保存 `all_time_actions`，筛当前列后指数加权 | `ACTTemporalEnsembler` 在线维护未来平均，`select_action()` 内调用 | 数学目标接近，存储位置和在线实现不同 |
| decoder 层数 | config 设置 `dec_layers=7`；Transformer 计算并 stack 各层输出，但 `DETRVAE` 对返回值取 `[0]`，action head 实际使用第 1 层结果 | 默认 `n_decoder_layers=1`，注释明确为匹配原始实际行为 | “原始 ACT 使用 7 层有效 Decoder”需要修正；7 层被构造/计算不等于 action head 使用最后一层 |
| L1 的 padding 缩放 | `masked_l1.mean()`：padding 置零后仍以完整 tensor 元素数作分母 | 有效元素求和后除以有效时间位置数 × action 维度 | 两边都是 masked L1，但 episode 尾部 padding 比例会让 loss 数值缩放略有不同 |
| checkpoint 格式 | `torch.save(state_dict)` 的 `.ckpt`，归一化统计另存 `dataset_stats.pkl` | `model.safetensors` + `config.json`，pre/post processor 资产另存 | 两边权重不能因算法同名直接互载 |
| 部署接口 | `eval_bc()` 直接创建 ALOHA/sim env，做预处理、查询、temporal aggregation、`env.step()` | Policy 与 processor 通过 LeRobot robot/record/eval 管线调用 | `select_action()` 只返回动作；发送串口/机器人仍由更外层运行时负责 |

## backbone：是否复用了 torchvision

是。当前 LeRobot 直接通过：

```python
getattr(torchvision.models, config.vision_backbone)(...)
```

创建 ResNet，默认 `resnet18` 与 ImageNet 预训练权重，再用 `IntermediateLayerGetter` 取 `layer4` feature map。它不是从零手写一套 CNN。

原始 ACT 的 DETR backbone 也[建立在 torchvision ResNet 上](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/detr/models/backbone.py#L1-L108)，但封装在 `detr/models/backbone.py`，再通过 DETR 的位置编码/Joiner 输出。两边都使用 ResNet 家族，不代表中间模块命名和 checkpoint key 相同。

## decoder “7 层变 1 层”的真实原因

原始 `imitate_episodes.py` 设置：

```python
dec_layers = 7
```

原始 `TransformerDecoder.forward()` 确实循环 7 层，并在 `return_intermediate=True` 时返回各层 stack。问题发生在下游：

```python
hs = self.transformer(...)[0]
a_hat = self.action_head(hs)
```

`self.transformer(...)[0]` 选择的是 decoder stack 的第 0 层，不是最后一层。后面的层虽然被算了，却没有进入 action head。LeRobot 因此把默认 decoder 层数设成 1 来匹配真正影响输出的路径。

> 🔍 这类问题提醒我：配置里写了几层，只能证明对象被构造；要知道输出真正用了哪一层，还要继续追返回 tensor 的索引。

## Temporal Ensembling 放在哪里

原始实现把 temporal aggregation 放在评估循环：

```text
policy(qpos, image) → all_time_actions[t, t:t+C]
→ 取 all_time_actions[:, current_t]
→ 过滤未填项
→ exp(-k × arange) 加权
```

LeRobot 把它移进 Policy：

```text
ACTPolicy.select_action
→ predict_action_chunk
→ ACTTemporalEnsembler.update
→ return current action
```

LeRobot 的在线实现只保存尚未消费的未来平均，而原始实现按整个 rollout 建二维时间表。前者更适合统一部署接口，也避免显式保存完整 episode 历史。

## 归一化由哪个层级负责

原始 ACT 的职责分散在三个位置：

- `EpisodicDataset` 归一化 qpos/action；
- `policy.py` 对图像做 ImageNet normalize；
- `eval_bc()` 在环境 step 前反归一化 action。

当前 LeRobot 通过 `make_act_pre_post_processors()` 复用默认 processor pipeline，按 `normalization_mapping` 和 dataset stats 统一处理。这是工程封装差异，不是论文算法公式的改变。

## 哪些地方仍不能过度下结论

- 没有逐权重验证两套模型在相同输入上的数值等价。
- 没有证明 LeRobot 的在线 ensembler 与原始离线时间表在所有 batch、episode 边界上完全一致，只确认了对重叠当前动作的指数加权目标与索引方向。
- 我的训练 checkpoint 对应的 LeRobot SHA 仍未知，不能把 2026-07-22 的 processor/checkpoint 格式当成当时的精确实现。
- 没有做 `n_decoder_layers=1/7` 的 SO-101 消融，也没有必要虚构这一结果。

## 证据表

| 我的理解 | LeRobot 源码证据 | ACT 论文/官方代码 | SO-101 实验对应 |
| --- | --- | --- | --- |
| LeRobot 用 feature schema 替代 14 维硬编码 | [`ACT.__init__`](https://github.com/huggingface/lerobot/blob/1427d35ef58ab46651dc7ef78bde81642090c861/src/lerobot/policies/act/modeling_act.py#L315-L365) 从 feature shape 建 Linear | 原始 [`detr_vae.py`](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/detr/models/detr_vae.py#L35-L62) 多处固定 14 | 适配 SO-101 是 LeRobot 通用接口带来的工程差异 |
| 默认 1 层 Decoder 匹配原始实际输出 | [`ACTConfig`](https://github.com/huggingface/lerobot/blob/1427d35ef58ab46651dc7ef78bde81642090c861/src/lerobot/policies/act/configuration_act.py#L120-L142) 注释 | 原始 [`TransformerDecoder`](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/detr/models/transformer.py#L98-L132) 返回 stack，`DETRVAE` 随后取 `[0]` | 本次源码核对，不是 SO-101 消融结论 |
| 两边 Temporal Ensembling 所在层级不同 | LeRobot [`select_action/update`](https://github.com/huggingface/lerobot/blob/1427d35ef58ab46651dc7ef78bde81642090c861/src/lerobot/policies/act/modeling_act.py#L86-L258) | 原始 [`imitate_episodes.py`](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/imitate_episodes.py#L220-L266) | 我的 checkpoint 是否启用仍需从 config 核对 |
| checkpoint 不直接互通 | LeRobot [`PreTrainedPolicy.save_pretrained`](https://github.com/huggingface/lerobot/blob/1427d35ef58ab46651dc7ef78bde81642090c861/src/lerobot/policies/pretrained.py#L103-L153) | 原始训练用 `torch.save(state_dict)` 并单存 stats | 我使用的是 LeRobot checkpoint 加载链路，没有声称转换过原始权重 |
