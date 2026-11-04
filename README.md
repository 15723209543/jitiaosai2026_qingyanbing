# 青眼病题

当前正式方案是 [`submission/`](submission/README.md) 中的 ConvNeXt-Small320 与 ConvNeXt-Tiny320 固定等权融合，版本 **v1.26.1**。2026-11-04 本机官方独立 valid AUROC **0.9875753924850881**，高于此前 Small 单模型 0.9863724439657326。用户反馈官网成绩 **97.97**（未提供提交 ID 或上传文件哈希）；作者报告的 98.55 不属于本机成绩。正式 CSV 为 [`submission/submission.csv`](submission/submission.csv)，最终字节哈希及核验记录见[今日实验](项目文档/训练记录/20261104_Tiny320/README.md)。

模型只用非医学 ImageNet-1K 预训练；官方 valid 不参与训练，test 不参与选模，无 TTA。两模型权重合计309,878,282字节，本机同实现计时84.677ms/张，比赛指定设备仍需验收。首次拉取需 `git lfs pull`。命令与权重来源见[使用说明](项目文档/当前状态/使用说明.md)和[`submission/RECIPE.md`](submission/RECIPE.md)。旧四路线 ZIP 在本机归档及 GitHub Release；原始数据和 junction 目标只读。