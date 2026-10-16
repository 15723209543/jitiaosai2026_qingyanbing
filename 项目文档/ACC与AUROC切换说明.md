# ACC与AUROC切换说明（v1.4.1）

日期：2026-10-16。当前默认启用ACC方案，AUROC方案按用户原样保留为三引号注释。

## 1. 怎么切换

打开submission/code/train.py，找到每轮训练打印下方的“方案1”和“方案2”。

- 使用ACC：保留方案1前后的三引号，方案2不加三引号。当前已经是这种状态。
- 使用AUROC：删除方案1前后的两个三引号；在方案2的metrics = evaluate之前和“方案2结束”之前各加一行三引号，包住整个方案2。
- 三引号沿用现有缩进。只启用一段，不能同时启用，也不能全部注释；运行过程中不要切换。
- 不需要去工具里改参数。两段分别包含验证、最佳模型保存、学习率调整、记录与早停。

## 2. 两种标准

| 项目 | 方案1：AUROC | 方案2：ACC（当前） |
|---|---|---|
| 比较数据 | 完整验证集 | 完整验证集 |
| 保存条件 | AUROC更高；完全相同则AUPRC更高 | accuracy严格更高 |
| 相等处理 | 两项都相等不保存 | ACC相等不保存，不比较其他指标 |
| 学习率与早停 | 本次候选自身AUROC | 本次候选自身accuracy |
| selection_metric | validation_auroc | validation_accuracy |
| tie_break_metric | validation_auprc | null，无平局指标 |

首轮建立本次候选的指标基准，不用旧冠军卡住候选训练。之后连续2轮未超过自身最高值则学习率减半；连续10轮未提高则早停，最多12轮。batch、学习率、网络结构不变。

注意ACC是验证集打印的大写Acc，不是训练过程的小写acc。比赛仍按AUROC为主、AUPRC决胜；使用ACC是可选实验方式，不保证平台分数提高。

## 3. 文件与终端

train.py只训练和验证；只有运行predict.py才写submission/results/submission.csv。两种方案共用submission/model/best_model.pth，不另建模型文件；以后训练胜出时会替换同一文件，切换前如需保留不同实验请自行备份。

结束时统一打印“保留模型验证Acc、AUROC、AUPRC、阳性召回率”，这些是按所选方案留下的同一个模型的指标，不代表每个指标分别达到最高。

utils.save_training_record新增可选参数selection_metric，由正在运行的方案传入，避免ACC保存后记录仍误写AUROC。最初仅验证旧模型的epoch=0还未进入选模分支，selection_metric和tie_break_metric暂记null；首个训练轮后写入实际方案。best_epoch=0仍不会重写正式模型配置。

## 4. 检查与提交范围

使用jitiaosai进行10项轻量测试：ACC升/降/相等，ACC元数据，ACC自身进步时学习率不误降，停滞降学习率及早停，AUROC反向变化与AUPRC平局决胜，AUROC元数据，epoch=0记录。均在合成指标和临时目录执行实际选模/调度/保存代码，没有训练或整批预测。

既有模型、配置和CSV不修改；model.py、predict.py不修改。原PDF在本次开始时已从本地删除，本次不恢复、不提交该删除，仅同步Markdown说明。数据集、.idea和缓存不提交。

提交版本由用户指定为v1.4.1，格式：v1.4.1 feat(train):增加ACC注释切换。下一次默认v1.4.2，前两位仅由用户修改。
