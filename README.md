# Diffusion Policy 复刻：训练之外的实时部署问题

本分支记录我在同一套 SO-101 数据上训练和部署 Diffusion Policy 的过程。基本训练和真机部署已经完成；初始部署出现周期性停顿与抽动，改用 DDIM 16 步并关闭显示、启用 AMP 后，抽动周期明显减小。这个现象支持“推理延迟是主要原因之一”，但不意味着动作块边界、数据质量和通信因素已经被完全排除。

[返回 `main` 项目导航](https://github.com/Zeil6/lerobot-so101-imitation-learning/tree/main)

## 数据复用边界

- ACT 与 Diffusion Policy 可以使用同一个 LeRobot 数据集。
- 数据集保存的是图像、机器人状态、动作和任务信息，并不从属于某个策略。
- ACT 权重不能转换成 Diffusion Policy 权重；两者网络结构和训练目标不同。
- 我先用较少 episode 验证流程，后来把数据扩充到约 50 组。
- 约 50 组数据没有自动消除抽动，因此我把注意力从“只有数据量不足”转向部署推理链路。

## 文档导航

- [`docs/training-deployment.md`](docs/training-deployment.md)：数据复用、训练、checkpoint 与部署流程。
- [`docs/inference-analysis.md`](docs/inference-analysis.md)：周期性停顿、动作队列和 DDPM/DDIM 分析。
- [`docs/image-and-algorithm.md`](docs/image-and-algorithm.md)：图像裁剪、Diffusion Policy 原理与反思。

## 阶段性结论

- 同一原始数据可用于 ACT 和 Diffusion Policy，但需要分别训练权重。
- Diffusion Policy 的真机表现对推理步数、动作队列长度和视觉预处理更敏感。
- 从 100 步 DDPM 形态调整为 DDIM 16 步后，停顿/抽动周期明显缩短，说明推理时间是核心变量之一。
- 增加 episode 不能解决计算延迟；数据问题和实时性问题必须分开验证。

## 尚未完成

- 对 DDIM 10 步和 16 步做同条件重复测试并记录实际推理时间。
- 重新确定双摄像头裁剪区域，训练输入一致的新 checkpoint。
- 系统比较不同 `n_action_steps` 和多个 checkpoint。
- 验证动作块融合、动作平滑和更高频闭环是否必要。

