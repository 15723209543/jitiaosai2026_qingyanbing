# 青眼病题

本仓库是 2026 年青光眼眼底图像二分类比赛项目。本地使用官方训练集训练、独立官方验证集按 AUROC 选模，再对官方无标签测试集生成预测。初赛平台上传的文件名为 `submission.csv`；规则原文见根目录两份 PDF。

## 当前目录

- `glaucoma_first/`、`glaucoma_first_my/`：原始数据，只读。
- `submission/`：当前正式模型、推理代码与 `submission.csv`。
- `submission2/`：margin AUC 候选代码与 `submission.csv`。
- `submission3/`：统一双模型联合训练，`--mode holdout` 对应 80/20，`--mode independent` 对应独立 80/80，`--percent` 可调整；含候选 `submission.csv`。
- `项目文档/当前状态/`：最新说明、训练结果、配方及 AI 交接。
- `项目文档/历史研究记录/`、`项目文档/训练记录/`：历史文档、运行证据及源码。
- `项目文档/归档_本地/`：本机 ZIP 备份，Git 忽略；旧 submission2/3 ZIP 另见 [GitHub Release](https://github.com/15723209543/jitiaosai2026_qingyanbing/releases/tag/archive-V1.18.2-submission2-3)。

## 模型状态

正式模型官方 valid AUROC 为 **0.9859689935**。`submission2` 候选为 **0.9856580776**，`submission3` 候选为 **0.9859218009**，均未超过正式模型，故两目录没有晋升的 `model/best_model.pth`。两份同名 CSV 位于不同目录，均是候选，不能当作模型已提升的证据。官方测试集仅用于生成 CSV，不用于选模。详细核验见 [训练结果](项目文档/当前状态/训练结果.md)。

## 运行

在本机 GPU 环境中先确认没有重复训练进程；中断后从原 run 继续：

```powershell
python submission3/code/train.py --mode holdout --percent 80 --epochs 1
python submission3/code/train.py --mode independent --percent 80 --epochs 1
python submission2/code/train.py --epochs 1
```

推理入口分别为各目录 `code/predict.py`，计时入口为 `cost.py`。候选权重只在本机忽略的 `runs/`，可通过 `--checkpoint` 显式指定；默认推理只接受已晋升的 `model/best_model.pth`。完整用法与目录清理记录见 [使用说明](项目文档/当前状态/使用说明.md)。
