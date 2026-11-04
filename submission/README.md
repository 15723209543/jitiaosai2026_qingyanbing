# submission：Small320 + Tiny320 固定等权融合

**当前版本：v1.25.2。** 本方案是用户提供的 S23 报告思路的近似复现，作者的 98.55 不是本机成绩。两个骨干分别从非医学 ImageNet-1K 预训练，在官方 train 中训练；官方 valid 5,849 张仅用于 AUROC 选模，test 5,855 张仅用于输出。没有 TTA、外部医学数据或测试集调参。

## 正式结果

- Small 第5轮官方 valid AUROC：0.9863724439657326；Tiny 对照第7轮：0.9864871867475788。
- 预先固定概率各 0.5 的 Small+Tiny 融合，完整官方 valid AUROC：**0.9875753924850881**；比原正式 Small 高 0.0012029485193555。记录：`model/ensemble_valid.json`，含两权重 SHA256。
- SFA 特征噪声 0.05 独立实验最佳第5轮 0.987248，低于固定融合，未晋升。
- 正式模型文件 `model/best_model.pth`（Small）与 `model/tiny_best_model.pth`（Tiny）合计 309,878,282 字节，低于 500 MB。本机 RTX 5060 Laptop 使用同一推理实现测得 84.677 ms/张；指定设备仍需验收，记录 `model/ensemble_cost.json`。
- 正式 CSV 为 `submission.csv`，5855 行，官方 test 顺序与三列格式须按本次部署核验；今日官网分数未知，不能承诺平台提升。

## 使用

首次拉取先执行 `git lfs pull`。在仓库根目录，`glaucoma_first/` 或 `GLAUCOMA_DATA_ROOT` 指向官方只读数据：

```powershell
D:/anaconda3.0/envs/deepStudy/python.exe -u submission/code/predict.py --output submission/submission.csv --workers 1
D:/anaconda3.0/envs/deepStudy/python.exe -u submission/cost.py runs/submission_new/timing_20261102/100 runs/submission_new/timing_20261102/5 --output submission/model/ensemble_cost.json
```

`predict.py` 和 `cost.py` 均调用 `code/ensemble_tiny_small.py` 的同一个 `score()` 实现。预测会校验官方 valid 记录、两个权重的 SHA256 及 AUROC 门槛。原 Small 单模型入口保留为 `code/predict_small.py` 与 `cost_small.py`。完整训练配方与实验差异见 `RECIPE.md`，今日原始证据见 `../项目文档/训练记录/20261104_Tiny320/README.md`。旧四路线已存本地 ZIP 和仓库 Release。