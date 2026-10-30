# 新 submission2 配方

- 旧版 submission2 已于 2026-10-30 压缩归档为本地 ZIP，并上传至 GitHub Release `archive-V1.18.2-submission2-3`。本目录随后用于新候选。
- 从正式 `submission/model/best_model.pth` 双骨干初始化，冻结骨干，用一份融合输出训练两个分类头。默认使用官方 train 全量；官方 valid 只用于 AUROC 选模。
- 目标函数为 focal + 0.1 成对 margin AUC surrogate（margin 0.5）。参考 [ICCV 2021 DAM 论文](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html) 的 margin 思想；[CCF 人工智能会议目录](https://www.ccf.org.cn/Academic_Evaluation/AI/) 将 ICCV 列为 A 类。本实现**不等同于**论文完整 min-max 目标和 PESG 优化。
- 只有官方 valid AUROC 严格高于初始基线才更新默认 `model/best_model.pth`；候选留在忽略的 `runs`。FP16 训练在第约 2753 批发生非有限损失，已标记作废；本配方改用 BF16 且禁用梯度缩放。配方 ID：`official_train_joint_heads_rgb384_focal_marginauc_bf16_v2`。
