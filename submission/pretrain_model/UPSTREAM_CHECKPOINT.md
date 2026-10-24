# 本次续训起点

本轮不使用外部通用预训练或医学预训练权重。续训起点是用户指定项目以赛方训练数据得到的比赛模型检查点。

- 仓库：https://github.com/15723209543/jitiaosai2026_qingyanbing
- 提交：`1e9bf1e26573b130be76e63a52182d8a3a1da053`（main，2026-10-23，v1.12.1）
- 文件路径：`submission/model/best_model.pth`，通过 Git LFS 下载。
- 文件大小：73,217,565 bytes。
- Git LFS 与本地原始文件 SHA-256：`def7e5c5be79903cd863421156f0557786d03bc567722ffd083360c18418d1a6`。
- 原始检查点本机备份：`work/upstream/best_model_v1.12.1.pth`（仓库根目录下）。
- 上游训练数据声明：仅使用赛方 `glaucoma_first/train`，验证集仅用于选模，不下载外部预训练权重。见该提交的README及submission说明。
- 上游模型：ResNet-18 + DenseNet-121，当前256整图补边配方。DenseNet分支需在256输入上重验证。

复现时先恢复该上游文件，再运行本轮 `code/train.py`。最终 `model/best_model.pth` 是本轮续训选出的检查点，不能再用于复现“同一续训起点”；复现应使用上列SHA匹配的原始文件。

ImageNet候选权重仅作过来源调研与下载，未参与本轮初始化、训练或预测，已移至仓库 `work/research_pretrain`，不纳入材料包。
