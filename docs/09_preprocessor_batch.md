# 09 · `preprocessor(batch)`：Dataset 和 ACT 之间的“安检通道”

[上一篇：数据闭环](08_data_closed_loop.md) · [返回分支入口](../README.md)

我沿着 `LeRobotDataset.__getitem__()` 往后追时，最开始把 `preprocessor(batch)` 想成“给图像做一下 resize，再顺手 normalization”。真正打开 `processor_act.py` 后才发现，它更像 Dataset 和 ACT 之间的安检与分拣线：改字段名、检查是否缺 batch 维、把 Tensor 送到正确设备，再按 feature 类型统一数值尺度。

Dataset 像仓库管理员，负责把货找出来；`preprocessor` 负责贴标签、装箱、过安检和统一计量单位。箱子还是那箱子，只是终于符合 ACT 这位“挑剔收货员”的数据契约。📦

## 先固定我实际追的源码版本

本篇以聊天中上传的 `processor_act.py` 为主要分析对象。该文件显式组装了四个 input steps 和两个 output steps。上传文件没有 Git metadata，因此我不能擅自说它“就是某个 commit 的 blob”。

进一步对照远程源码后，能确认两层版本关系：

| 来源 | 固定版本 | `processor_act.py` 的写法 | 能否视为旧 checkpoint 版本 |
| --- | --- | --- | --- |
| 聊天上传源码 | commit 未知 | ACT 文件内显式列出 4 + 2 个 Step | 不能；只作为本次学习的主要代码对象 |
| LeRobot | [`3f2179f3b69708b6ad009b2e7685dd9d05269ee1`](https://github.com/huggingface/lerobot/tree/3f2179f3b69708b6ad009b2e7685dd9d05269ee1) | 与上传内容的函数结构、Step 顺序和关键参数一致 | 只能说结构核对一致，不能证明上传文件 blob 相同 |
| 本分支既有基线 | [`1427d35ef58ab46651dc7ef78bde81642090c861`](https://github.com/huggingface/lerobot/tree/1427d35ef58ab46651dc7ef78bde81642090c861) | ACT 文件调用 `make_default_pre_post_processors()`，通用 factory 再组装同样顺序 | 是本次远程核对基线，不是旧 checkpoint 证明 |

也就是说，后一个版本把脚手架搬进了通用 helper，不是把处理顺序偷偷洗牌。文档下面讲 Step 时，优先对应上传源码和 `3f2179f3…` 的显式实现，同时用 `1427d35e…` 检查重构后的行为契约。

## 真实调用链：它不是 Dataset 的成员函数

教程训练链在固定基线中是：

```text
LeRobotDataset.__getitem__()
→ DataLoader batch
→ policies.factory.make_pre_post_processors(...)
→ policies.act.processor_act.make_act_pre_post_processors(...)
→ preprocessor(batch)
→ ACTPolicy.forward(batch)
```

当前 `1427d35e` 中，`make_pre_post_processors()` 会根据 `config.type == "act"` 动态找到 `make_act_pre_post_processors()`；ACT factory 再调用 `make_default_pre_post_processors()`。聊天上传版本则在 ACT factory 内直接列出 Step。

这条链说明：

- `preprocessor` 不是 `LeRobotDataset` 的一部分；
- Dataset 负责每条 sample 的读取与 Dataset 级 transform；
- DataLoader 负责把多条 sample collate 成 batch；
- processor 负责把 batch 整理成 Policy 约定的输入；
- `policy.forward(batch)` 默认接收的不是完全原始的 Dataset 输出。

训练入口的关键顺序很短：

```python
for batch in dataloader:
    batch = preprocessor(batch)
    loss, _ = policy.forward(batch)
```

越短的两行越容易被一眼略过，但这里正好是 Dataset 世界和模型世界的分界线。

## `processor_act.py` 自己做了什么

上传版本中的 `make_act_pre_post_processors(config, dataset_stats)` 主要负责**组装**，并没有把每种处理的数学和设备逻辑重新写一遍。

输入：

- `config: ACTConfig`：提供 `device`、input/output features 与 `normalization_mapping`；
- `dataset_stats`：各 feature 的统计量，可为 `None`。

返回：

- 一个接收 batch dict 的 `PolicyProcessorPipeline`；
- 一个接收 Policy action 的 `PolicyProcessorPipeline`。

实际工作由多个可组合 Step 完成：

```text
preprocessor
RenameObservationsProcessorStep(rename_map={})
→ AddBatchDimensionProcessorStep()
→ DeviceProcessorStep(device=config.device)
→ NormalizerProcessorStep(...)
```

```text
postprocessor
UnnormalizerProcessorStep(...)
→ DeviceProcessorStep(device="cpu")
```

`PolicyProcessorPipeline.__call__()` 会先把外部输入转换成统一的 `EnvTransition`，再按 `steps` 列表顺序调用每个 Step，最后转换回 batch 或 Policy action。Step 被拆开后，训练、推理、保存和加载可以复用同一套数据契约，也可以单独检查中间状态；Policy 本身不必同时承担字段适配、硬件迁移和数值还原。

## Step 1：`RenameObservationsProcessorStep`

它解决的是“数据集/环境字段名”和“Policy 期待字段名”不一致。例如某个环境叫 `camera_left`，Policy 配置可能期待 `observation.images.left`，这时可以用 `rename_map` 做映射。

本次上传的 ACT factory 明确传入：

```python
RenameObservationsProcessorStep(rename_map={})
```

空字典意味着：

- observation key 会原样保留；
- 这一步实际没有发生字段重命名；
- 它仍作为通用 pipeline 的稳定接口保留，方便其他环境或后续配置插入映射。

不能因为类名里有 `Rename`，就写成“ACT 把相机字段重命名了”。源码里的空映射已经把这件事否掉。

## Step 2：`AddBatchDimensionProcessorStep`

神经网络通常以 batch 为单位处理数据，所以 batch dimension 是 Tensor 最前面的“样本数量”维。

```text
单条 state: [D_s]       → batch size 1: [1, D_s]
单张 image: [C,H,W]     → batch size 1: [1,C,H,W]
单个 action: [D_a]      → batch size 1: [1,D_a]
```

但这个 Step **不是每次无条件 `unsqueeze(0)`**。固定源码中的判断是按维数进行：

- action 只有在 `dim() == 1` 时补 batch 维；
- `observation.state`/environment state 只有在 1D 时补；
- 单张图像只有在 3D `[C,H,W]` 时补；
- 单个字符串 `task` 会包装成 list；
- 已经带 batch 维的 Tensor 原样返回。

训练时 DataLoader 通常已经把多条 sample 组成 `[B,...]`，所以这一步更多是在确认数据契约，而不是再变成 `[1,B,...]`。真机推理常从单条 observation 开始，这时它才会实际补出 `B=1`。

## Step 3：`DeviceProcessorStep`

CPU Tensor 存在系统内存中；CUDA Tensor 存在 GPU 显存中。Policy 参数在哪个设备上，参与计算的输入 Tensor 通常也要到同一设备。

上传版本使用：

```python
DeviceProcessorStep(device=config.device)
```

在本次配置方式下没有传 `float_dtype`，因此它主要做设备迁移：

- CPU → `cuda`/`mps`/`cpu`；
- 不改变 Tensor 的 shape；
- 不进行 normalization；
- 默认不主动改 dtype；
- 数值语义不变，只是存放和计算的位置改变。

源码本身还支持显式 `float_dtype` 和 MPS 的兼容处理，所以“Device Step 永远不可能改 dtype”也说得太满。更准确的表述是：**本次 ACT factory 没有要求 dtype cast**。

postprocessor 末尾使用 `DeviceProcessorStep(device="cpu")`，因为后续 robot runtime 通常在 CPU 侧继续处理动作。GPU 上出现一个 action Tensor，只能说明模型算完了；它还没有越过 USB 线自己跑去找舵机。

## Step 4：`NormalizerProcessorStep`

上传源码的构造参数是：

```python
NormalizerProcessorStep(
    features={**config.input_features, **config.output_features},
    norm_map=config.normalization_mapping,
    stats=dataset_stats,
    device=config.device,
)
```

这四项分别回答：

- `features`：哪些字段存在、每个字段属于 VISUAL、STATE 还是 ACTION；
- `norm_map`：每一类 feature 采用哪一种 normalization mode；
- `stats`：实际使用的 mean/std、min/max 或其他统计量；
- `device`：normalization stats 放在哪个设备上参与运算。

### input features 与 output features

- input features 是 Policy 的条件输入，例如相机图像和 `observation.state`；
- output features 是 Policy 要预测的字段，ACT 中必须包含 action；
- 训练 batch 同时包含 observation 与真实 action，所以两边都可能被 normalization；
- 推理只有当前 observation，没有真实 action。`NormalizerProcessorStep` 检查 action 是否存在；不存在就只处理 observation，不会报出一条“未来正确动作”。

### `dataset_stats` 从哪里来

教程先创建：

```python
dataset_metadata = LeRobotDatasetMetadata(dataset_id)
```

随后把 `dataset_metadata.stats` 传给 `make_pre_post_processors()`。这些统计量来自数据集元数据中的 stats，而不是每个 batch 临时重新计算。训练保存 processor assets 后，部署应加载与 checkpoint 匹配的统计量。

### 不是所有字段天生都用 mean/std

`NormalizerProcessorStep` 根据 feature type 查 `normalization_mapping`，可支持 `MEAN_STD`、`MIN_MAX`、quantile 类模式或 `IDENTITY`；如果 mapping 为 identity，或该 key 没有 stats，数据可以保持原样。

本分支固定的 ACT `1427d35e` 默认配置确实是：

| FeatureType | 默认 mode |
| --- | --- |
| `VISUAL` | `MEAN_STD` |
| `STATE` | `MEAN_STD` |
| `ACTION` | `MEAN_STD` |

这是**该固定 commit 的 ACT 默认值**，不是“LeRobot 所有 Policy、所有 checkpoint、所有字段永远统一用 mean/std”。旧 checkpoint 可能保存了不同 mapping，仍要读取其 `config.json` 和 processor 配置。

### normalization 为什么有用

图像像素、关节状态和 action 的原始量纲与范围可能差很多。normalization 把它们变换到模型更容易共同优化的数值尺度，避免某个数值范围很大的字段仅因为单位不同就支配梯度。

它不等于：

- 图像 resize 或 crop；
- 数据增强；
- ResNet 的 feature extraction；
- Transformer 内部的 LayerNorm。

这些操作可能在 Dataset transform、模型 backbone 或其他 Step 中发生，但不能因为都叫“预处理”就塞进 `preprocessor(batch)`。

## postprocessor：把模型动作带回控制链

`policy.select_action()` 的输出仍处于模型动作空间。postprocessor 做两件事：

```text
normalized/model-space action
→ UnnormalizerProcessorStep
→ DeviceProcessorStep("cpu")
→ CPU 上的机器人尺度 action
```

`UnnormalizerProcessorStep` 只使用 `config.output_features`，按与训练匹配的 mapping 和 stats 对 action 做逆变换。例如训练若使用 mean/std：

```text
normalize:   x_model = (x_robot - mean) / (std + eps)
unnormalize: x_robot = x_model × std + mean
```

normalization 和 unnormalization 是一对。只保存模型权重、不保留正确 stats，就像把地图留下却把比例尺扔了。

不过，postprocessor 输出依然只是控制链中的数据。它不会调用 `predict_action_chunk()`，不会管理 action queue，也不会直接写舵机。真实顺序仍然是：

```text
predict_action_chunk()
→ select_action()
→ postprocessor()
→ robot runtime
→ 机器人通信/执行
```

## 训练和推理复用同一套契约，但输入不同

| 阶段 | processor 输入 | 是否包含真实 action | normalization 对象 | Policy 输出 | postprocessor |
| --- | --- | ---: | --- | --- | --- |
| 训练 | DataLoader batch | 是 | 存在且配置覆盖时处理 observation + action | `(loss, loss_dict)` | 不参与这次 loss 计算，但与 Policy 一起保存 |
| 推理 | 当前 observation | 否 | 只处理实际存在的 observation | normalized/model-space 单步 action | unnormalize，再移回 CPU |

训练时 `ACTPolicy.forward()` 要用真实 action chunk 构造监督目标和 CVAE latent；推理时 `select_action()` 没有未来真值。两种阶段共用 processor 约定，不代表 batch 内容或 Policy 返回值相同。

## Shape 在各 Step 前后怎样变化

符号定义：

- `B`：batch size；
- `T`：action chunk 的时间长度；
- `C`：图像通道数；
- `H,W`：图像尺寸；
- `D_s`：robot state 维度；
- `D_a`：action 维度。

| 数据/Step | 进入 Step 前 | 进入 Step 后 | 说明 |
| --- | --- | --- | --- |
| DataLoader 的 image | `[B,C,H,W]` | `[B,C,H,W]` | 训练时通常已有 batch 维 |
| 单条推理 image / AddBatch | `[C,H,W]` | `[1,C,H,W]` | 只有 3D 图像才补维 |
| DataLoader 的 state | `[B,D_s]` | `[B,D_s]` | 不会再无条件 unsqueeze |
| 单条推理 state / AddBatch | `[D_s]` | `[1,D_s]` | 只有 1D state 才补维 |
| 训练 action chunk / AddBatch | `[B,T,D_a]` | `[B,T,D_a]` | DataLoader 已组成 batch |
| Device | 任意上述 shape | shape 不变 | 主要改变 tensor.device |
| Normalizer | 任意上述 shape | shape 不变 | 改变数值尺度，不改语义维 |
| `select_action()` 输出 | — | `[B,D_a]` | 已从 chunk/queue/ensemble 选择单步 |
| Unnormalizer | `[B,D_a]` | `[B,D_a]` | 恢复 action 尺度 |
| postprocessor Device | `[B,D_a]` on model device | `[B,D_a]` on CPU | 仍未直接发送给舵机 |

我没有把 SO-101 的 `D_s`、`D_a` 写死。这个分支尚未保存旧数据集的完整 feature metadata，猜一个“常见 6 维”反而会把示例写成实验事实。

> **假设示例：** 若单条推理 state 恰好有 6 个数，则 AddBatch 后从 `[6]` 变成 `[1,6]`。这里只演示 batch 维，不代表旧 checkpoint 的 state 一定为 6 维。

## 我这次真正修正的理解

> 🧠 **我最初的理解**
> `preprocessor(batch)` 大概就是给图像做 resize 或 normalization。
>
> 🔍 **继续追代码后发现**
> 它是一条由多个 Processor Step 组成的统一入口，负责字段命名、batch 维、设备迁移和按 feature 类型归一化。图像裁剪、增强和模型内部视觉处理没有源码证据时，不能全部塞进这个函数。

另外几处也一起理顺了：

- Dataset 负责取数据，processor 负责让数据符合 Policy 契约；
- DataLoader batch 与单条 inference observation 的 batch 维来源不同；
- `RenameObservationsProcessorStep({})` 在本次配置中没有真的改名；
- Device Step 只把模型输入送到计算设备，不会把 GPU action 发送给机器人；
- normalization 与 unnormalization 是成对的两个方向；
- `predict_action_chunk()`、`select_action()`、postprocessor 和机器人发送指令是四个不同节点；
- processor 与 Policy 在教程中分别 `save_pretrained()`，加载时也应按实际 checkpoint 的 processor assets 恢复，不能只拿当前最新版 factory 重建后就假设完全兼容。

## 源码与版本边界

- 源码检查日期：**2026-07-23**。
- 主要分析对象：聊天上传的 `processor_act.py`，commit 未知。
- 显式 Step 对照版本：LeRobot [`3f2179f3b69708b6ad009b2e7685dd9d05269ee1`](https://github.com/huggingface/lerobot/tree/3f2179f3b69708b6ad009b2e7685dd9d05269ee1)。
- 本分支远程核对基线：LeRobot [`1427d35ef58ab46651dc7ef78bde81642090c861`](https://github.com/huggingface/lerobot/tree/1427d35ef58ab46651dc7ef78bde81642090c861)。
- 上传源码与 `3f2179f3…` 在函数结构、Step 顺序和关键构造参数上相符；因为上传文件缺少 Git metadata，本篇不声称二者 blob SHA 相同。
- `1427d35e…` 的 `processor_act.py` 已调用通用 helper；同 commit 的 `processor/factory.py` 明确保留相同的 4 + 2 顺序。
- 旧 ACT checkpoint 的精确 LeRobot SHA、normalization mapping、feature 维度和 processor serialization 版本仍需结合旧环境与 checkpoint assets 核对。

## 证据表

| 我的理解 | LeRobot 源码证据 | SO-101 实验对应 | 真实性状态 |
| --- | --- | --- | --- |
| ACT 文件主要组装 pipeline，不包办全部处理逻辑 | 上传 `processor_act.py`；[`3f2179f3…/processor_act.py`](https://github.com/huggingface/lerobot/blob/3f2179f3b69708b6ad009b2e7685dd9d05269ee1/src/lerobot/policies/act/processor_act.py) | 我从 Dataset 继续追到 processor factory | 上传源码已读；上传 commit 未知 |
| 当前重构仍保留 `Rename → Batch → Device → Normalize` | [`1427d35e…/processor_act.py`](https://github.com/huggingface/lerobot/blob/1427d35ef58ab46651dc7ef78bde81642090c861/src/lerobot/policies/act/processor_act.py) 调用 [`processor/factory.py`](https://github.com/huggingface/lerobot/blob/1427d35ef58ab46651dc7ef78bde81642090c861/src/lerobot/processor/factory.py) | 不把新 helper 倒推为旧 checkpoint 一定使用 | 两个固定版本已分别核对 |
| pipeline 按列表顺序逐个调用 Step | [`PolicyProcessorPipeline`](https://github.com/huggingface/lerobot/blob/3f2179f3b69708b6ad009b2e7685dd9d05269ee1/src/lerobot/processor/pipeline.py) | 训练调用 `batch = preprocessor(batch)` | 源码已确认 |
| AddBatch 只在缺少 batch 维时补维 | [`batch_processor.py`](https://github.com/huggingface/lerobot/blob/3f2179f3b69708b6ad009b2e7685dd9d05269ee1/src/lerobot/processor/batch_processor.py) 的 dim 判断 | DataLoader batch 与单条真机 observation 要区分 | 源码已确认；SO-101 精确维度待核对 |
| Device Step 不等于机器人通信 | [`device_processor.py`](https://github.com/huggingface/lerobot/blob/3f2179f3b69708b6ad009b2e7685dd9d05269ee1/src/lerobot/processor/device_processor.py) 只处理 Tensor device/dtype | 推理输出还需 postprocessor 与 robot runtime | 源码已确认 |
| normalization 按 feature type 与 mapping 决定 | [`normalize_processor.py`](https://github.com/huggingface/lerobot/blob/3f2179f3b69708b6ad009b2e7685dd9d05269ee1/src/lerobot/processor/normalize_processor.py)；[`ACTConfig`](https://github.com/huggingface/lerobot/blob/1427d35ef58ab46651dc7ef78bde81642090c861/src/lerobot/policies/act/configuration_act.py) | 训练/部署必须匹配 stats 与 config | 固定默认值已确认；旧 checkpoint 值待核对 |
| 推理 action 要 unnormalize 并回 CPU | 上传 `processor_act.py` 的 output steps；[当前 default factory](https://github.com/huggingface/lerobot/blob/1427d35ef58ab46651dc7ef78bde81642090c861/src/lerobot/processor/factory.py) | postprocessor 后仍由更外层发送机器人 | 源码已确认；未写成舵机驱动 |
| Policy 与 processor 分别保存 | [`act_training_example.py`](https://github.com/huggingface/lerobot/blob/1427d35ef58ab46651dc7ef78bde81642090c861/examples/tutorial/act/act_training_example.py#L93-L101) | 旧 checkpoint 需要继续检查 processor assets | 教程链已确认；实验文件待核对 |
