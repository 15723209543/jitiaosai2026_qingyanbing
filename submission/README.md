# v1.28.1：2026-11-06 官网成绩 98.05

用户提供的官网截图显示：2026-11-06 18:14:29，提交 ID 34323，成绩 **98.05**。截图仅显示文件名 `submission.csv`，不含上传文件 SHA256，因此不能独立证明平台文件与本地字节完全相同；本地正式 CSV SHA256 为 `46f5202f6ebac88299f25b4d81f751aa3d28211d954b417ce3ddacd96389072d`。v1.27.1 的用户反馈成绩为 97.57（提交 ID 32895），更早 v1.26.1 为 97.97。

正式方案是 Small 第 6 轮成对 AUC 训练 EMA + 原 Tiny，固定各 0.5 概率融合。完整官方 valid 5,849 张 AUROC **0.9883804426480413**；模型权重分别为 `ab36e4f40f5c630db94d93fff35ece25c522f303194906b1b8a72ea0ee533555` 与 `dd4253f1ddc6a103478a43b32b21a06d9ff087567be8b95f4936e3930141a647`，合计 309,878,666 字节。本机同实现单张 47.845 ms；指定比赛硬件仍待验。CSV 已核验 5,855 行、官方 test 顺序、三列及 0.5 标签阈值。官网分数只作结果记录，不用于模型选择。

# submission：Small320 + Tiny320 固定等权融合

**当前版本：v1.28.1。** 官网 98.05 为用户截图所示；平台未提供上传文件 SHA256。两个骨干仅使用非医学 ImageNet-1K 预训练，官方 train 训练，官方 valid 只用于 AUROC 选模，test 只用于输出；无 TTA、外部医学数据或测试集调参。

## 正式结果

- Small 第 6 轮固定 70% 官方 valid 选模子集 AUROC 0.9855777928539122；Tiny 对照第 7 轮完整官方 valid AUROC 0.9864871867475788。
- Small+Tiny 固定概率各 0.5，完整官方 valid AUROC **0.9883804426480413**，证据为 `model/ensemble_valid.json`，含两权重 SHA256。此前 v1.26.1 同口径为 0.9875753924850881。
- 正式权重文件 `model/best_model.pth` 与 `model/tiny_best_model.pth` 合计 309,878,666 字节；本机同实现 47.845 ms/张，见 `model/ensemble_cost.json`。指定设备仍需验收。
- 正式 `submission.csv` 为 5,855 行，官方 test 顺序与三列格式已核验；SHA256 `46f5202f6ebac88299f25b4d81f751aa3d28211d954b417ce3ddacd96389072d`。

## 使用

首次拉取执行 `git lfs pull`。在仓库根目录以官方只读数据作为 `glaucoma_first/` 或 `GLAUCOMA_DATA_ROOT`：

```powershell
D:/anaconda3.0/envs/deepStudy/python.exe -u submission/code/predict.py --mode test --small-checkpoint submission/model/best_model.pth --tiny-checkpoint submission/model/tiny_best_model.pth --selection-record submission/model/ensemble_valid.json --output submission/submission.csv --workers 1
D:/anaconda3.0/envs/deepStudy/python.exe -u submission/cost.py <pictures100> <pictures5> --output submission/model/ensemble_cost.json
```

预测、验证和计时共用 `code/predict.py` 的推理实现。训练配方、70% 选模与 30% 一次审计的边界见 `RECIPE.md` 和项目文档。原 v1.27.1 及 v1.26.1 记录保留在训练记录中。
