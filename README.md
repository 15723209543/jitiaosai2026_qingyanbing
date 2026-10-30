## 2026-10-30 当前工作

旧 `submission2/3` 已作为两个完整 ZIP 归档到 [GitHub Release](https://github.com/15723209543/jitiaosai2026_qingyanbing/releases/tag/archive-V1.18.2-submission2-3)，本机 ZIP 在 `D:/青眼病题/历史归档_本地/`，不参与日常 Git push/pull。旧 `submission3/` 已从当前工作树移除；当前 `submission2/` 是新 AUC margin 候选代码。

`submission4/code/joint_train.py` 统一 4/5：`--mode holdout` 表示 80/20，`--mode independent` 表示独立 80/80，`--percent` 可改。两个骨干在同一批次融合预测、共同训练两个分类头；8GB 显存下骨干冻结。官方 train 用于训练，官方 valid 仅用于 AUROC 晋升。新 `submission2` 基于 [ICCV 2021 AUC margin 思路](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html)，不是论文完整 PESG 实现。两份不同的候选 CSV 已生成，均未超过正式模型的官方 valid AUROC，故正式模型仍为 `submission/`。具体成绩、CSV 校验和见 `docs/训练结果.md`。

<!-- EXTRA_20261028_START -->
## 2026-10-28 本地增强数据训练

本轮增强训练已完成并停止，两分支候选均未刷新最佳。保留原正式模型及CSV，最佳验证AUROC仍为0.985968994；完整网络独立重载复测已通过。

- convnext_tiny.fb_in1k 第1轮，增强后单支AUROC 0.970791485，与保留分支固定均值AUROC 0.979703482，是否刷新最佳：False。
- efficientnetv2_rw_s.ra2_in1k 第1轮，增强后单支AUROC 0.899279897，与保留分支固定均值AUROC 0.959342191，是否刷新最佳：False。

正式模型及CSV未替换，仍保留原最佳AUROC 0.985968994。

用户明确要求利用新增强数据集训练，且不得向GitHub发布数据。数据位于仓库外D:/青眼病/青光眼检测数据集；本轮不复制、暂存、上传任何真实图片、YOLO标签或数据ZIP。Git忽略规则包含数据目录及ZIP。发布内容仅为源码、汇总指标、模型及匹配预测，不含逐图清单。

原始3209张（train2881、valid197、test131）仅作额外训练，标签0=Glaucoma映射阳性1，1=No Glaucoma映射阴性0。通过图片可解码检查、标签/框坐标检查和SHA精确重复检查后，剔除118张重复图，保留3091张（阳1273、阴1818）。与官方valid及test精确相同的额外图片不进入训练。没有患者ID，不能验证患者级或增强变体的跨集合重合。

从正式SHA288f8e0af64623fdb3a20bdc17fe3157b79cefa7c91e5d59d459d5b6d76f4813暖启动，配方增加_extra_v1。每分支本次最多1轮：完整官方train → 本地增强集 → 仅官方valid一次选模；两训练阶段共用同一模型和优化器。初次验证超越起点就停止；未改善则不替换正式模型/CSV。只改训练目标来源，不改变384整图预处理或FP32推理，无TTA。训练AMP FP16、梯度裁剪2.0；损失focal+0.1成对AUC。原始数据只读。

执行入口：`python submission/code/train.py --extra-data "D:/青眼病/青光眼检测数据集" --max-epochs 1`。本轮运行目录runs/continuation_20261028_223149，初次日志D:/青眼病/work/extra_training_20261028.log，关机后恢复日志为同目录extra_training_20261028_resume.log。中断用同一--extra-data和--max-epochs参数加--resume runs/continuation_20261028_223149。优化器及完整运行缓存仅本机保留。

本轮汇总证据和源码在training/20261028_aug/。当前正式文件仍以submission/和上方正式成果记录为准；训练完成不等于新权重已晋升。本机速度旧限制仍存在，不宣称官网成绩或官方速度检查通过。

<!-- EXTRA_20261028_END -->

# 青光眼项目

# 当前成果（2026-10-28）

正式模型、对应CSV及代码现在均在本仓库 `submission/`，不再只存在于被Git忽略的runs目录。

- `submission/model/best_model.pth`：ConvNeXt-Tiny原权重 + 本轮EfficientNetV2第1轮参数，固定等权组合。
- `submission/submission.csv`：该模型的5855行预测，官方名称/顺序通过，实际交付predict/cost复测不一致0张。
- 官方独立验证AUROC：0.984618915 → 0.985968994，增量0.001350078。这是验证指标，非官网成绩。
- 本机端到端耗时0.127828秒/张，**尚未达到100毫秒要求**，不得宣称已通过速度限制；官方指定硬件复测未完成。
- 模型SHA256：`288f8e0af64623fdb3a20bdc17fe3157b79cefa7c91e5d59d459d5b6d76f4813`；CSV SHA256：`b8800414ac2c39173cb51551b73a7f6fe47faff7a6ba63449194dd1e31579765`。

基于[远端v1.15.6](https://github.com/15723209543/jitiaosai2026_qingyanbing/commit/a64b008f67f9eb3030ffb8e251329ca92e3a4022)，源模型SHA256 `2b5a15db0246ee0858e06708105d6c9a36f5b1f028439b2925cb3cabd6ad865e`。仅官方train51396张训练、valid5849张选模；无外部医学数据、valid并入训练或TTA。384整图等比补边，ImageNet归一化。AMP训练、FP32推理。ConvNeXt实际训练3轮均未采用；EfficientNetV2第1轮虽单支AUROC下降，与原ConvNeXt组合提升，按预定组合AUROC标准采用。首次提升后已停止。

训练源码、不可变上游代码、基准权重和记录已纳入版本控制：`training/20261028/`。实际历史脚本为run_training_original.py，配置/逐轮结果/重载核验分别为config.json、history.json、TRAINING_COMPLETE.json和verification.json。优化器与大图缓存继续留在本机runs/work，不推送。

预测：`python submission/code/predict.py`。计时：`python submission/cost.py pictures_100 pictures_5`。两者共用FP32、batch1、单CPU线程、禁用TF32的推理函数。训练：`python submission/code/train.py` 会以当前正式权重创建独立run，最多每支3轮、首次验证提升停止；`--resume runs/<名称>`从已有轮次边界恢复。本次正式成果不会被训练中的临时文件覆盖，完成后需核验再晋升。

`submission2/`、`submission3/`是远端保留的另两条实验，不代表本轮成果；尤其submission3的train+valid集内结果不能称独立验证。本次未启动新训练、未上传官网。旧正式目录完整备份在本机runs/20261028_github_latest/legacy_submission_before_promotion；用户原提交0a099ea保留在合并历史及backup/before-sync-20261028-0a099ea分支。

Git LFS管理PTH与safetensors。首次克隆后运行`git lfs pull`，避免把134字节指针当模型。大型原图、缓存、优化器、ZIP不提交。
