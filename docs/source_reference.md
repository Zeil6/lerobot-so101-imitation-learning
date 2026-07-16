# 源码、论文与实验依据索引

[返回本分支入口](../README.md) · [ACT 独立笔记](act.md) · [Diffusion Policy 独立笔记](diffusion_policy.md)

## 核对范围

检查日期：**2026-07-16**。

| 来源 | 固定版本 | 用途 |
| --- | --- | --- |
| Hugging Face LeRobot | [`3f2179f3b69708b6ad009b2e7685dd9d05269ee1`](https://github.com/huggingface/lerobot/tree/3f2179f3b69708b6ad009b2e7685dd9d05269ee1) | 核对当前配置类、Policy、模型、processor 与动作队列 |
| ACT 原始官方代码 | [`742c753c0d4a5d87076c8f69e5628c79a8cc5488`](https://github.com/tonyzhaozh/act/tree/742c753c0d4a5d87076c8f69e5628c79a8cc5488) | 核对 DETRVAE、损失、数据接口与 temporal aggregation |
| Diffusion Policy 原始官方代码 | [`5ba07ac6661db573af695b419a7947ecb704690f`](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f) | 核对 U-Net policy、scheduler、receding horizon、EMA 与裁剪 |
| ACT 论文 | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) | 核对 action chunking、CVAE 与 temporal ensemble 的原始动机 |
| Diffusion Policy 论文 | [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137) | 核对条件动作扩散、receding-horizon control 与多模态动机 |
| Diffusion Policy 官方项目页 | [diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu/) | 论文、代码与官方实验入口 |

固定 SHA 的意义是让链接在上游继续更新后仍可复查。它们是本次整理的检查基线，不足以证明我的训练环境当时正好使用同一 SHA；训练版本仍需从本地环境记录或 checkpoint metadata 追溯。

## LeRobot ACT 路径与关键符号

| 路径 | 关键符号 | 本次核对内容 |
| --- | --- | --- |
| [`configuration_act.py`](https://github.com/huggingface/lerobot/blob/3f2179f3b69708b6ad009b2e7685dd9d05269ee1/src/lerobot/policies/act/configuration_act.py) | `ACTConfig` | `chunk_size`、`n_action_steps`、`use_vae`、`kl_weight`、`temporal_ensemble_coeff`；Temporal Ensembling 默认关闭及配置约束 |
| [`modeling_act.py`](https://github.com/huggingface/lerobot/blob/3f2179f3b69708b6ad009b2e7685dd9d05269ee1/src/lerobot/policies/act/modeling_act.py) | `ACTPolicy`, `ACT`, `ACTTemporalEnsembler` | `forward()` 的 L1/KL、`select_action()` 的队列/聚合、VAE train/inference 分支、Transformer 数据流 |
| [`processor_act.py`](https://github.com/huggingface/lerobot/blob/3f2179f3b69708b6ad009b2e7685dd9d05269ee1/src/lerobot/policies/act/processor_act.py) | `make_act_pre_post_processors` | 输入与 feature 归一化、输出动作反归一化 |

原始 ACT 对照：

| 路径 | 核对内容 |
| --- | --- |
| [`policy.py`](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/policy.py) | L1 + KL 损失、推理不传真实 action |
| [`detr/models/detr_vae.py`](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/detr/models/detr_vae.py) | DETRVAE、latent encoder、queries、action head |
| [`imitate_episodes.py`](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/imitate_episodes.py) | query frequency、`all_time_actions` 和指数权重 temporal aggregation |
| [`utils.py`](https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/utils.py) | HDF5 episode 与 qpos/action mean-std |

## LeRobot Diffusion Policy 路径与关键符号

| 路径 | 关键符号 | 本次核对内容 |
| --- | --- | --- |
| [`configuration_diffusion.py`](https://github.com/huggingface/lerobot/blob/3f2179f3b69708b6ad009b2e7685dd9d05269ee1/src/lerobot/policies/diffusion/configuration_diffusion.py) | `DiffusionConfig` | horizons、scheduler、训练/推理步数、prediction target、resize/crop 与输入约束 |
| [`modeling_diffusion.py`](https://github.com/huggingface/lerobot/blob/3f2179f3b69708b6ad009b2e7685dd9d05269ee1/src/lerobot/policies/diffusion/modeling_diffusion.py) | `DiffusionPolicy`, `DiffusionModel`, `DiffusionRgbEncoder`, `DiffusionConditionalUnet1d` | observation/action queues、加噪训练、迭代推理、执行窗口、视觉编码与 train/eval crop |
| [`processor_diffusion.py`](https://github.com/huggingface/lerobot/blob/3f2179f3b69708b6ad009b2e7685dd9d05269ee1/src/lerobot/policies/diffusion/processor_diffusion.py) | `make_diffusion_pre_post_processors` | 输入/feature 归一化和动作反归一化 |

原始 Diffusion Policy 对照：

| 路径 | 核对内容 |
| --- | --- |
| [`diffusion_unet_hybrid_image_policy.py`](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py) | robomimic encoder、ConditionalUnet1D、加噪 loss、多轮采样、`start=To-1` 的执行窗口 |
| [`train_diffusion_unet_hybrid_workspace.yaml`](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/train_diffusion_unet_hybrid_workspace.yaml) | DDPM 100 train/inference steps、16/2/8 horizon 示例、EMA 与裁剪配置 |
| [`train_diffusion_unet_ddim_hybrid_workspace.yaml`](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/train_diffusion_unet_ddim_hybrid_workspace.yaml) | DDIM scheduler 与较少 inference steps 示例 |
| [`train_diffusion_unet_hybrid_workspace.py`](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py) | EMA model copy、训练更新与评估选择 |

## 已确认的实现差异

1. **通用接口而非原硬件脚本原样搬运。** LeRobot 用 `PolicyFeature`、LeRobotDataset、processor 和 `PreTrainedPolicy` 统一 ACT/DP；原始仓库分别使用 HDF5/task runner/Hydra 等接口。
2. **ACT decoder 层数处理。** 当前 LeRobot 的配置注释说明原始 ACT 虽声明多层 decoder，但调用问题导致只使用第一层；LeRobot 默认一层以匹配这一行为。
3. **动作消费封装。** LeRobot 把普通 action queue、ACT Temporal Ensembling 或 DP observation/action deque 放进 Policy；原始仓库更依赖外部评估/runner 循环。
4. **归一化位置。** 原始 ACT/DP 各自管理统计；LeRobot 通过 processor 按 policy feature schema 统一预处理和反归一化。
5. **Diffusion 视觉编码细节。** 原始 U-Net hybrid policy 借用 robomimic observation encoder；当前 LeRobot 直接组织 torchvision ResNet、SpatialSoftmax 和按相机共享/独立 encoder。
6. **Diffusion EMA。** 原始官方 workspace 明确维护 EMA policy；本次检查的 LeRobot Diffusion policy/config 没有同类 EMA model copy。这个结论只对上述固定 commit 的检查范围负责。
7. **默认 horizon 已变化。** 原始官方 U-Net 示例为 16/2/8；当前 LeRobot 默认 64/2/32。实验 checkpoint 记录仍以 16/8 为准。

## 实验事实、源码事实和推断如何分层

| 类型 | 可以写什么 | 本分支示例 |
| --- | --- | --- |
| 真实实验 | 仓库日志、配置和视频能支持的过程或现象 | ACT/DP 10ep 同数据；DP 50ep 为重新采集；DDIM 16 步后抽动周期减小 |
| 本次源码核对 | 固定 commit 中实际存在的类、字段和调用 | ACT Temporal Ensembling 默认关闭；DP 队列为空才生成新动作块 |
| 原论文/官方代码 | 作者提出或官方实现的机制 | ACT action chunk + CVAE；Diffusion Policy 条件动作扩散与 receding horizon |
| 当前推断 | 由多个证据支持但尚未单独实验的解释 | DP 推理延迟是抽动的重要原因之一；块边界也可能贡献 |

## 已主动保留的版本疑问

- 实验时 LeRobot 的精确 tag/SHA 尚未从 checkpoint metadata 确认。
- 实验版本只支持 `crop_shape`，当前上游支持 `resize_shape`/`crop_ratio`；文档没有用当前字段倒推旧实验。
- 尚未确认 ACT checkpoint 的 `temporal_ensemble_coeff`、`chunk_size` 和 `n_action_steps` 保存值。
- 尚未进行 ACT Temporal Ensembling 开关、DDIM 10/16 步或不同 `n_action_steps` 的严格消融。
- 没有定量成功率、训练时长、loss 曲线或统一 checkpoint 对比，因此未写入相应结论。
