# 2026-10-30 已完成但未晋升的候选权重

两份权重来自已完成 GPU 训练，元数据 `training_complete=True`、`selection.name=ensemble`，并经复制前后 SHA256 核对。它们的官方独立 valid AUROC 均低于正式 `submission/model/best_model.pth`，因此**不是** `model/best_model.pth`，也不替换正式模型。两份 PTH 由 Git LFS 管理，克隆后需运行 `git lfs pull`。

| 权重 | 字节 | 官方 valid AUROC | SHA256 | 对应本地 CSV |
|---|---:|---:|---|---|
| `submission2/candidate_epoch1.pth` | 200821471 | 0.9856580776142385 | `788408b86f6eb95e61286a83fe2a268699a300e6ae53d3968f4817f0c7fdfd5b` | `submission2/submission.csv` |
| `submission3/candidate_epoch1.pth` | 200823263 | 0.9859218009434817 | `c4929e3f52426be0ed5b27f7c72e6c531c6f113d5e0dea47637d8a50f4d53ab0` | `submission3/submission.csv` |

重现当前预测时显式指定对应 `--checkpoint`，例如：

```powershell
python submission2/code/predict.py submission2/submission.csv --checkpoint 项目文档/训练记录/20261030_候选权重/submission2/candidate_epoch1.pth
python submission3/code/predict.py submission3/submission.csv --checkpoint 项目文档/训练记录/20261030_候选权重/submission3/candidate_epoch1.pth
```

官网分数以用户提供的提交对应关系记录于 `项目文档/当前状态/训练结果.md`；平台分数不用于模型晋升。
