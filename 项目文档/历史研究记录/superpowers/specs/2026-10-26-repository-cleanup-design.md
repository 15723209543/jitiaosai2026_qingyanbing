# 仓库清理设计（2026-10-26）

目标：将不再用于当前训练/预测的历史实验移出本地工作树和 GitHub 当前分支；删除前制作可恢复的本地 ZIP。用户已选择保守清理范围。

保留：`submission/code`、`submission/model/best_model.pth`、`submission/submission.csv`、`submission/cost.py`、核验表格、`submission2`、根目录 `docs`、`项目文档`、比赛 PDF、原始数据 `glaucoma_first`、旧有两份本地 ZIP。不得训练或改变正式权重及 CSV。当前正式 PTH 在本机为 LFS 指针，不能声称该 ZIP 包含真实正式权重。

清理并纳入新 ZIP：九个根目录 `continuation*` 实验文件夹；`submission/docs`；`submission/pretrain_model` 中的旧轮权重及其历史说明（整体归档后以简明 README 替换）；`submission/results`；过时的 `submission/file_manifest.json`。这批文件不被当前正式预测、计时或新候选训练导入。旧 `submission/code` 的辅助脚本暂保留，因为现有 `train.py` 仍导入其中部分文件。

备选方案：只清理九个实验目录会保留提交目录的旧轮大权重与历史快照；连根目录文档一起清理会违背既有“说明文档不要删除”的要求。用户选定上述中间范围。备份 ZIP 放在本地忽略目录，不推送；远端保留 Git 历史可恢复。先验证 ZIP 的条目数量、大小及关键权重字节，再按解析后的明确路径删除，最后更新文档、测试并推送。
