# LeRobot × SO-101 排错记录

这个分支不做“报错字符串大全”，而是按问题所在层级整理我在环境安装、数据采集、训练和真机部署中的判断过程。相同的最后一行报错可能由不同原因触发，因此每条记录都尽量保留时间顺序和日志证据。

[返回 `main` 项目导航](https://github.com/Zeil6/lerobot-so101-imitation-learning/tree/main)

## 使用方式

```text
现象
→ 第一判断
→ 日志证据
→ 排查步骤
→ 真正原因
→ 修复方法
→ 验证方式
→ 可以迁移到其他项目的经验
```

先用下面的分类定位问题，再进入 [`docs/cases.md`](docs/cases.md) 查看记录。

| 类别 | 典型问题 | 首先确认 |
| --- | --- | --- |
| 环境与依赖 | Python 版本、editable install、Feetech extra、FFmpeg、下载中断、resolver | 解释器、pip、包版本和错误发生阶段 |
| 命令与配置 | CLI 入口变化、YAML/字典、引号、参数名 | 本地 `--help` 和解析器停在哪一步 |
| 数据采集 | 空 episode、方向键、相机断开、编码、Hub 401 | 事件时间线、本地落盘和远端上传 |
| GPU 与训练 | OOM、batch size、碎片、AMP、双相机、恢复训练 | 峰值显存、其他进程、输入规模、checkpoint 元数据 |
| 真机通信 | status packet、舵机 5 扭矩、串口、电源、ID、波特率 | 物理链路与总线配置，避免盲目重写舵机参数 |

## 快速诊断命令

```bash
python --version
which python
python -m pip --version
python -m pip check
lerobot-record --help
lerobot-train --help
nvidia-smi
ls -l /dev/ttyACM*
v4l2-ctl --list-devices
```

命令只负责收集证据，不等于自动修复。尤其是舵机通信错误，不应在未确认电源、线材、端口占用、ID 和波特率前重新运行配置命令；配置写入可能让原本局部的通信问题扩大。

## 阶段性结论

- 最后一条 traceback 经常只是清理或保存阶段的结果，第一个异常事件更接近根因。
- 安装失败、配置解析失败、设备连接失败、训练失败和远端认证失败应该分层处理。
- 能重新运行不等于已修复；验证必须回到原触发条件，并检查预期产物。

## 尚未完成

- 补充原始日志的日期、LeRobot commit/版本和最终修复所用的精确包版本。
- 对 Feetech 通信问题增加经过万用表/替换线材验证的硬件记录。
- 将无敏感信息的典型日志片段归档到 `logs/sanitized/`。

