# 新 submission2 配方

- 旧版 submission2 已于 2026-10-30 压缩归档为本地 ZIP，并上传至 GitHub Release `archive-V1.18.2-submission2-3`。本目录随后用于新候选。
- 从正式 `submission/model/best_model.pth` 双骨干初始化，冻结骨干，用一份融合输出训练两个分类头。默认使用官方 train 全量；官方 valid 只用于 AUROC 选模。
- 目标函数为 focal + 0.1 成对 margin AUC surrogate（margin 0.5）。参考 [ICCV 2021 DAM 论文](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html) 的 margin 思想；[CCF 人工智能会议目录](https://www.ccf.org.cn/Academic_Evaluation/AI/) 将 ICCV 列为 A 类。本实现**不等同于**论文完整 min-max 目标和 PESG 优化。
- 只有官方 valid AUROC 严格高于初始基线才更新默认 `model/best_model.pth`；候选留在忽略的 `runs`。FP16 训练在第约 2753 批发生非有限损失，已标记作废；本配方改用 BF16 且禁用梯度缩放。配方 ID：`official_train_joint_heads_rgb384_focal_marginauc_bf16_v2`。
## 2026-10-31 多轮低学习率实验（已训练）

在 2026-10-30 已完成的 full/marginauc、LR `5e-5` run 中，官方 valid AUROC 第 1 轮为 0.9856580776，第 2 轮降至 0.9855026197。因此下一次独立 run 仍从正式 `submission/model/best_model.pth` 初始化，保持 full 官方 train、focal + margin pairwise AUC、BF16、384 预处理及官方 valid 独立选模，但将学习率降至 `1e-5`，分轮训练并按官方 valid AUROC 保留最佳候选。旧 run 与新 run 不能跨参数 `--resume`。实际 run `submission2/runs/20261031_143405_202421` 完成2轮；官方独立 valid AUROC 第1轮 0.9859329050836605、第2轮 0.985854250757395，选择第1轮为本路线本轮最佳。两轮均低于正式基线 0.9859689935392411，未晋升 `model/best_model.pth`。
