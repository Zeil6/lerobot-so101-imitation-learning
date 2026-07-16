# Diffusion Policy 训练与部署流程

[返回本分支 README](../README.md) · [返回 `main`](https://github.com/Zeil6/lerobot-so101-imitation-learning/tree/main)

## 1. 先复用数据，不复用权重

我使用 ACT 阶段采集的 SO-101 LeRobot 数据集：两路图像 `handeye`、`fixed`，机器人状态和动作都保持不变。策略类型改为 diffusion 后重新训练。这样做能让两个策略面对相同的示范内容，也避免把采集差异误当成算法差异。

最开始我只用较少 episode 验证“数据能读、训练能启动、checkpoint 能加载”的链路，之后才把数据扩充到约 50 组。这里的“约 50 组”来自阶段性记录，不补写每组时长或总帧数。

## 2. 检查本地版本与依赖

```bash
conda activate lerobot
python -m pip check
lerobot-train --help
lerobot-record --help
```

Diffusion Policy 所需依赖应以当前 LeRobot 项目的 extras/依赖声明为准。仓库不固定一个未经日志确认的额外安装命令，避免把别的版本说明伪装成我的实际记录。

## 3. 训练

命令骨架如下，batch size 必须填入经过当前 GPU 验证的值：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lerobot-train \
  --dataset.repo_id=<HF_USER>/so101_test \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_so101_test \
  --job_name=diffusion_so101_test \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --batch_size=<BATCH_SIZE_VERIFIED_ON_THIS_GPU>
```

我会保留多个 checkpoint，而不是只留下 `last`。训练 loss 可以用来发现发散或异常，但不能替代真机评价：较低 loss 并不直接告诉我推理是否及时、动作块是否衔接或抓取是否稳定。

## 4. 部署前检查

1. checkpoint 的 policy 类型确实为 diffusion；
2. `handeye`、`fixed` 两路相机名称、数量和 shape 与训练一致；
3. 训练与部署使用同一裁剪规则；
4. 自动控制时不配置 Leader teleop；
5. 关闭不必要的实时显示，避免把可视化开销混入控制延迟；
6. 在安全姿态先测一次推理耗时和动作队列更新。

部署命令的 checkpoint 参数名随 LeRobot 版本变化，因此先用本地 `lerobot-record --help` 确认。仓库只固定已经确认过的关键部署配置：

```text
noise_scheduler_type=DDIM
num_inference_steps=16
use_amp=true
display_data=false
```

## 5. 真机观察

我最初重点观察“能不能完成抓取”，后来增加了更细的事件记录：

- 动作队列开始和结束时间；
- 新动作块生成前后的停顿；
- 停顿是否近似周期性；
- 抽动发生在块边界还是块内部；
- 单次推理时间与控制周期的比例；
- checkpoint、初始位置和相机帧是否一致。

这些记录可以帮助区分数据导致的错误动作、推理导致的停顿和通信导致的丢包。

