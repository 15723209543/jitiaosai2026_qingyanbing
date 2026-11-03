**当前代码版本：v1.25.1（明日待训）。** 正式 Small320 模型对应 2026-11-02 官网 97.65（ID 29668）；2026-11-03 低学习率候选官网 96.95（ID 30606）。Tiny320 和特征增强仅完成代码，尚无训练成绩。

# 青眼病题

这是 2026 年青光眼眼底图像二分类比赛项目。当前根目录唯一提交方案是 [`submission/`](submission/README.md)：非医学 ImageNet-1K ConvNeXt-Small、320 输入的 S23 g07 **近似复现**。旧 `submission`、`submission2`、`submission3`、`submission4` 已本地打包为四份 ZIP，保存在 Git 忽略的 `项目文档/归档_本地/20261102_旧submission/`；原目录额外保留在该处 `原目录/`，不会随普通 Git 拉取。

当前最佳本机官方独立 valid AUROC **0.9863724439657326**（第5轮），提交文件 [`submission/submission.csv`](submission/submission.csv) 已核验 5855 行。用户 2026-11-02 官网反馈 **97.65**（ID 29668），见[提交记录](项目文档/提交记录/20261102_submission官网97.65.md)。报告作者的 98.55 与本机不是同一训练轨迹。后续续训至第11轮早停、低学习率从头训练10轮均未超过正式 AUROC；明日将训练 Tiny320 对照和可选特征增强，见[复现方案](项目文档/当前状态/20261103_S23_Tiny320复现方案.md)。

原始 `glaucoma_first/`、`glaucoma_first_my/` 及其 junction 目标只读；只使用规则允许的非医学预训练，官方 valid 不并入训练，测试集不用于调参，预测不使用 TTA。比赛规则见根目录两份 PDF。训练与预测命令、权重来源、速度测量见 [`submission/README.md`](submission/README.md) 和[使用说明](项目文档/当前状态/使用说明.md)。跨电脑先执行 `git lfs pull`。`runs/` 为本地忽略的训练断点，普通拉取不会获得完整断点。Tiny 训练将从合规 ImageNet-1K 权重新建独立 run；启动前先确认 GPU 与胸片任务状态。
