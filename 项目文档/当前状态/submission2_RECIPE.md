# 新 submission2 配方

- 旧版 submission2 已于 2026-10-30 压缩归档为本地 ZIP，并上传至 GitHub Release `archive-V1.18.2-submission2-3`。本目录随后用于新候选。
- 从正式 `submission/model/best_model.pth` 双骨干初始化，冻结骨干，用一份融合输出训练两个分类头。默认使用官方 train 全量；官方 valid 只用于 AUROC 选模。
- 目标函数为 focal + 0.1 成对 margin AUC surrogate（margin 0.5）。参考 [ICCV 2021 DAM 论文](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html) 的 margin 思想；[CCF 人工智能会议目录](https://www.ccf.org.cn/Academic_Evaluation/AI/) 将 ICCV 列为 A 类。本实现**不等同于**论文完整 min-max 目标和 PESG 优化。
- 只有官方 valid AUROC 严格高于初始基线才更新默认 `model/best_model.pth`；候选留在忽略的 `runs`。FP16 训练在第约 2753 批发生非有限损失，已标记作废；本配方改用 BF16 且禁用梯度缩放。配方 ID：`official_train_joint_heads_rgb384_focal_marginauc_bf16_v2`。
## 2026-10-31 多轮低学习率实验（已训练）

在 2026-10-30 已完成的 full/marginauc、LR `5e-5` run 中，官方 valid AUROC 第 1 轮为 0.9856580776，第 2 轮降至 0.9855026197。因此下一次独立 run 仍从正式 `submission/model/best_model.pth` 初始化，保持 full 官方 train、focal + margin pairwise AUC、BF16、384 预处理及官方 valid 独立选模，但将学习率降至 `1e-5`，分轮训练并按官方 valid AUROC 保留最佳候选。旧 run 与新 run 不能跨参数 `--resume`。实际 run `submission2/runs/20261031_143405_202421` 完成2轮；官方独立 valid AUROC 第1轮 0.9859329050836605、第2轮 0.985854250757395，选择第1轮为本路线本轮最佳。两轮均低于正式基线 0.9859689935392411，未晋升 `model/best_model.pth`。

## 2026-10-31 续试验（已完成）

用户要求继续至20:00前选最佳；从正式 submission/model/best_model.pth 新开 run，仍用 full 官方 train、marginauc、BF16、官方 valid 独立选模，学习率降至 5e-6。不得跨学习率 resume；验证完成后仅用更优候选替换已交付 CSV。仅官方 valid 严格高于正式基线 0.9859689935392411 才晋升模型。


实际run `20261031_172549_567435` 第1轮官方valid AUROC 0.9859523373289731，高于此前本路线0.9859329050836605，因此用此候选重新预测CSV；仍低于正式基线，未晋升模型。

## 2026-10-31 晚间同配方续训（已完成）

从 run `20261031_172549_567435` 的完整第1轮进度以相同 full/marginauc、BF16、LR5e-6 参数 `--resume` 至 `--epochs 2`；仅独立官方 valid AUROC 高于该run第1轮0.9859523373289731时才更新当前submission2候选。正式晋升仍需超过0.9859689935392411。


实际第2轮官方独立valid AUROC 0.9859329050836605，低于第1轮0.9859523373289731；不选第2轮，不替换已交付CSV和正式模型。

## 2026-11-01 上午短轮（已完成）

为在 12:00 前交付一份可提交 CSV，从正式 baseline 新开 full/marginauc/BF16 一轮，学习率 3e-6。官方 train 训练、官方 valid AUROC 选模，绝不并入 valid；新 run 不跨参数 resume。仅当此轮严格高于当前本路线 0.9859523373289731 时替换 submission2 CSV；严格超过正式 0.9859689935392411 才能晋升正式权重。若未改善，保留现有已核验 CSV。


实际结果：新 run 20261101_094241_178122 第1轮官方独立 valid AUROC 0.9859551133640179，严格高于本路线旧候选 0.9859523373289731，但增幅仅 0.000002776 且低于正式基线 0.9859689935392411。已据此更新本路线 CSV；正式 best_model.pth 未晋升。
