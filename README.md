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
