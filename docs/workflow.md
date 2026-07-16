# ACT 实际操作流程

[返回本分支 README](../README.md) · [返回 `main`](https://github.com/Zeil6/lerobot-so101-imitation-learning/tree/main)

以下顺序按照我实际遇到问题的先后整理。尖括号内容是本机相关占位符，不能直接照抄为别人的串口或账号。

## 1. 创建环境

我第一次使用 Python 3.10 安装时，LeRobot 的版本约束直接拒绝了安装。之后把环境边界先固定到 Python 3.12：

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
python --version
```

删除失败环境时，我使用：

```bash
conda deactivate
conda env remove -n lerobot
```

## 2. 安装 LeRobot、Feetech 与 FFmpeg

在 LeRobot 源码根目录执行：

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[feetech]"
conda install -c conda-forge ffmpeg=7.1.1
```

`-e .` 只把当前源码以 editable 模式安装；`-e ".[feetech]"` 还会安装项目声明的 Feetech 额外依赖。editable 的意义是源码修改可以直接反映到环境中，不需要每改一次都重新打包安装。

安装后先检查命令入口，而不是沿用其他版本的教程：

```bash
lerobot-record --help
lerobot-train --help
ffmpeg -version
```

## 3. 连接机器人和双摄像头

本实验使用：

- SO-101 Leader：人工示范输入；
- SO-101 Follower：复现 Leader 动作并在部署时执行策略；
- `handeye`：手眼视角；
- `fixed`：固定视角。

先确认设备，而不是直接假设 `/dev/ttyACM0` 和相机索引永远不变：

```bash
ls -l /dev/ttyACM*
v4l2-ctl --list-devices
```

## 4. 录制数据集

下面保留我使用过的参数结构，但用占位符移除了机器相关端口。我的旧命令曾写成 `python -m lerobot.record`，本地版本实际应使用 `lerobot-record`。

```bash
lerobot-record \
  --robot.disable_torque_on_disconnect=true \
  --robot.type=so101_follower \
  --robot.port=<FOLLOWER_PORT> \
  --robot.id=<FOLLOWER_ID> \
  --robot.cameras="{'handeye': {'type': 'opencv', 'index_or_path': 0, 'width': 1280, 'height': 720, 'fps': 30}, 'fixed': {'type': 'opencv', 'index_or_path': 2, 'width': 1280, 'height': 720, 'fps': 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=<LEADER_PORT> \
  --teleop.id=<LEADER_ID> \
  --display_data=true \
  --dataset.repo_id=<HF_USER>/so101_test \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=20 \
  --dataset.single_task="Grab the glue"
```

实际录制时我会逐轮检查：两路画面是否更新、Follower 是否跟随、episode 是否达到有效长度、重置阶段是否误触退出键，以及保存结束前是否出现编码或上传错误。

## 5. 检查 episode

第三轮录制曾因按键触发提前退出，后续保存空 episode 失败。因此“机械臂已经停下”不等于“episode 已成功落盘”。我会把终端中的 `Episode`、`Reset`、相机断开和 `save_episode` 放在一条时间线上看，并检查本地数据目录中的 episode 数量和元数据是否一致。

若暂时不上传 Hub，应按当前 LeRobot 版本选择本地保存相关参数；若需要上传，先独立验证认证状态，避免把 401 和本地数据损坏混为一类。

## 6. 训练 ACT

基础训练命令：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lerobot-train \
  --dataset.repo_id=<HF_USER>/so101_test \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_test \
  --job_name=act_so101_test \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --batch_size=<BATCH_SIZE_VERIFIED_ON_THIS_GPU>
```

仓库不写死最终 batch size，因为目前保留下来的对话日志只证明我在 OOM 后降低了 batch size，没有足够证据确认最终数值。复现时应从较小值开始，记录每次修改和峰值显存。

## 7. 加载 checkpoint 并部署

部署命令的具体参数名依 LeRobot 版本而异，先执行：

```bash
lerobot-record --help
```

部署配置必须满足三个约束：

1. 载入 ACT checkpoint；
2. 机器人仍为 SO-101 Follower；
3. 相机字典的名称和数量与训练数据一致。

自动策略部署时不再配置 Leader teleop。Leader 是采集示范时的输入设备，如果部署时仍把 teleop 和 policy 同时接入，就会让控制源含糊，甚至发生互相覆盖。

## 8. 验证顺序

1. 断开负载或把机械臂放在安全区域，确认通信和急停方式。
2. 只验证模型能加载、观测键匹配、两路相机正常。
3. 低风险执行短时间策略，观察动作方向和幅度。
4. 再进入完整抓取测试，记录 checkpoint、初始位姿和失败类型。

