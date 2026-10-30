# 历史模型归档：旧 submission2 与旧 submission3

本 Release 对应标签 `archive-V1.18.2-submission2-3`（旧提交 `3dc3dbe`），保存的是 **2026-10-30 整理前的两个旧目录**。它们与当前 `main` 中同名的 `submission2/`（margin AUC）和 `submission3/`（统一 80/20、独立 80/80 联合训练）是不同实验，请勿混用代码、权重或成绩。

## submission2_20261030.zip：引入额外数据训练

旧 submission2 在官方 train 之外引入本地额外图像进行训练；旧训练脚本 `train_round` 依次训练官方数据和额外数据，再用官方 valid 评估。ZIP 含旧代码、`model/best_model.pth` 与 `submission.csv`。这属于历史实验归档；额外医学数据不符合目前项目约定的训练限制，不能据此把旧权重当作当前合规正式模型。

## submission3_20261030.zip：train+valid 混合全集训练

旧 submission3 把官方 train 与 valid 合并为全集进行训练，并在同一批图上评估。该集内 AUROC **不是独立验证 AUROC**。ZIP 含旧代码、`model/best_model.pth`、暂停时权重和 `submission.csv`。这也是历史实验，与当前独立保留官方 valid 的 submission3 不同。

| 资产 | 大小（字节） | SHA256 |
|---|---:|---|
| `submission2_20261030.zip` | 186346224 | `90d0ea859943ab89000dc95725331a3aae748a1cdda1e6bc45152be2ec721ae7` |
| `submission3_20261030.zip` | 372622040 | `13497dbf801db3819bc99799fdca7a69bd11623629a60182de9188eb4b861251` |

ZIP 作为 Release 资产保存，普通 Git push/pull 不传输。当前模型和 2026-10-30 晚上两次新提交的结果以仓库 `项目文档/当前状态/训练结果.md` 为准。
