# 2026 青光眼眼底图像二分类

当前活动目录 `submission/` 是已确认官网得分最高的非五模型版本：ResNet-18 + DenseNet-121 双模型，历史提交 92.37 分（ID 14086）。两网络独立训练、阳性概率等权平均；原来两份最佳权重已装入一份 `submission/model/best_model.pth`，预测行为不变。对应 `submission/results/submission.csv` 已恢复。详情见 [submission 说明](submission/README.md) 和 [提交记录](项目文档/提交记录/第6次提交-92.37分.md)。

使用 `deepStudy` 解释器，在根目录运行 `submission/code/predict.py` 可重新生成 CSV；运行 `submission/code/train.py` 默认跳过已有参数的重训，并用官方验证集复核。数据放在 `glaucoma_first/`，不纳入 Git。训练记录在 `项目文档/训练记录/`。

五模型源码、权重和 CSV 保留在本地 `五模型归档.zip`（Git 忽略），可从 [v1.9.2 Release 附件](https://github.com/15723209543/jitiaosai2026_qingyanbing/releases/download/v1.9.2/five-model-archive.zip)恢复；旧 `submission`、`submission2`、`submission3` 的归档仍为本地 `三个不用的模型.zip`。用户未提交的项目文档改动没有纳入本次恢复。

本项目仅使用主办方提供的训练数据，不使用外部训练数据或预训练权重。单张推理是否满足 100 ms 需在目标设备实测，历史官网分数不保证重新提交时保持不变。
