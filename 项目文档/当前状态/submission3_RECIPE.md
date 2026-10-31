# submission3 联合训练配方

- 统一源码：`submission3/code/joint_train.py`；统一入口 `submission3/code/train.py`，由 `--mode holdout|independent` 和 `--percent` 控制，两种模式的运行均写到 `submission3/runs`。
- `--mode holdout --percent 80`：官方 train 内分层抽 80% 训练，余 20% 作诊断；`--mode independent --percent 80`：两份独立 80% 样本，允许重叠。百分比可通过 `DEFAULT_PERCENT` 或命令行设置。官方 valid 始终保留为唯一晋升指标。
- 初始双骨干参数来自正式 `submission/model/best_model.pth`（其原始预训练仅为非医学 ImageNet）。两个骨干的原训练见过官方 train，因此这两种内部抽样的诊断分数均不作为独立泛化依据。
- ConvNeXt-Tiny 与 EfficientNetV2-S 共用同一输入批次，融合概率为两支 sigmoid 均值；一个 focal + 0.1 pairwise AUC 损失反传，单个 AdamW 同时更新两支分类头。骨干冻结，384 等比补边，FP32 单张无 TTA 推理。
- 只有本轮官方 valid AUROC 严格高于本轮初始基线时复制到 `model/best_model.pth`。未晋升候选仅保存在 `runs`。后续训练采用 BF16 自动混合精度，配方 ID：`official_train_joint_heads_rgb384_focal_pairauc_bf16_v3`。已完成的首轮旧配方 `...pairauc_v2` 使用 FP16，保留原始候选但不跨配方续训。
## 2026-10-31 多轮低学习率实验（已训练）

旧 independent 80/80、LR `1e-5` run 的官方 valid AUROC 第 1 轮 0.9859218009，第 2 轮 0.9858542508。下一次独立 run 从正式 `submission/model/best_model.pth` 初始化，沿用 independent 80/80 抽样、focal + 0.1 pairwise AUC、BF16、384 预处理以及未参与训练的官方 valid 选模，将学习率降至 `5e-6`。每轮检查独立 valid AUROC；只有严格超过正式起点 0.9859689935 才晋升 `model/best_model.pth`。新 run 不复用旧 run 的优化器进度。实际 run `submission3/runs/20261031_140418_260003` 完成2轮；官方独立 valid AUROC 第1轮 0.9859541880190029、第2轮 0.9859162488733925，选择第1轮为本路线本轮最佳。两轮均低于正式基线 0.9859689935392411，未晋升 `model/best_model.pth`。
