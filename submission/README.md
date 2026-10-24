# 第二轮续训后的青光眼模型

验证AUROC 0.962409，高于第一轮0.957992。

最终模型model/best_model.pth，预测submission.csv。先验证选模后锁定，256整图补边，0.5归一化，FP32推理，阈值0.5，无TTA。

预测：python code/predict.py。计时：python cost.py pictures_100 pictures_5。训练依赖本机保留的上一轮优化器状态，详见docs/使用说明.md与AI_HANDOFF.md。
