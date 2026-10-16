# AUROC唯一选模说明（v1.4.3）

更新日期：2026-10-16。本文件名为兼容旧链接保留；**旧ACC/AUROC三引号切换方法已停用**。

## 当前规则

仅在完整验证集的新AUROC严格大于已有正式模型AUROC时替换权重；第一次没有旧模型时保存首轮结果。比较完整精度，不比较终端四舍五入数值。

AUROC相等时，无论ACC、AUPRC、AP、召回率如何变化，都不替换。这是用户本次指定的本地规则；官网仍会在AUROC相等时比较AUPRC，二者需区分。

调度器和早停只观察本次候选自身AUROC。首轮建立候选基准，随后连续2轮未提高时学习率减半，连续5轮未提高时早停，最多12轮。候选不必每轮都超过正式冠军。

## 保存位置

- 正式预测只读取submission/model/best_model.pth。
- 每次候选最佳另存项目文档/训练记录/时间目录/best_model.pth。
- 新权重中selection_metric为validation_auroc，tie_break_metric为null，并保存validation和epoch。
- run_config.json顶层记录正式冠军，candidate_best单独记录候选路径、轮数、指标和结构；path_base=project_root，避免同名pth混淆。候选pth自身也带指标与轮数。
- 若未超过正式冠军，原模型及同目录config.json不变，可能仍显示历史ACC选模，这是未被替换的旧模型真实记录，不应手工改成AUROC。

train.py只训练+验证，不生成答案；只有运行predict.py才写submission.csv。本次源码升级不训练、不重写任何既有模型或结果。
