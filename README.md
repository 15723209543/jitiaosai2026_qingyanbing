# 2026 青光眼眼底图像二分类

当前活动目录 `submission/` 使用ResNet-18 + DenseNet-121 双模型结构；其历史64×64版本曾提交92.37分（ID 14086）。现在已改为256×256整图等比例缩放补边，当前PTH在新配置下尚未完成双模型训练验证，不能把历史92.37分或历史CSV当作当前模型成绩。详情见 [submission说明](submission/README.md)、[新赛规适配说明](项目文档/修改说明/v1.12.1新参赛规则适配.md)和[历史提交记录](项目文档/提交记录/第6次提交-92.37分.md)。

使用 `deepStudy` 解释器，运行 `submission/code/train.py` 会从现有权重继续、每个模型全量训练5轮；当前输入为整图等比例缩小并补边至256×256，不裁剪、不拉伸。训练先重新验证新输入下的基线，只有验证AUROC提高才更新权重。训练不写CSV；两个分支均验证完成后，单独运行`submission/code/predict.py`才会在`submission/`根部生成新CSV。历史`submission/results/submission.csv`不覆盖。数据放在`glaucoma_first/`，不纳入Git。训练记录在`项目文档/训练记录/`。

五模型源码、权重和 CSV 保留在本地 `五模型归档.zip`（Git 忽略），可从 [v1.9.2 Release 附件](https://github.com/15723209543/jitiaosai2026_qingyanbing/releases/download/v1.9.2/five-model-archive.zip)恢复；旧 `submission`、`submission2`、`submission3` 的归档仍为本地 `三个不用的模型.zip`。用户未提交的项目文档改动没有纳入本次恢复。

本项目当前仅使用主办方提供的训练数据，不使用外部训练数据或预训练权重。根目录新增《比赛参赛规则_青眼病.pdf》要求最终另交模型核验材料；已准备`submission/cost.py`和核验清单，但**未打包、未重新训练或生成当前权重的最终CSV**。新规则的端到端单张平均推理上限为100秒，仍须按指定设备实测。历史测试集预测加权的分析CSV不再用于调参或选模。
