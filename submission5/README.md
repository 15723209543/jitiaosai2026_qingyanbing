# submission5 模式入口

`submission5/code/train.py` 只转发到 `submission4/code/joint_train.py`，并默认设置 `--mode independent`。这表示从官方 train 两次独立分层抽取相同比例（默认 80%）供训练和内部诊断，可能重叠；正式选模仍只看未参与训练的官方 valid AUROC。

新的权重、运行记录和预测 CSV 都在 `submission4/`，`submission5/code/predict.py` 与 `submission5/cost.py` 也仅转发到 4 的同一推理与计时实现，不再把两种模式分别训练成两份互相竞争的包。旧 `submission5/runs/` 原训练进度保留在本机，和新联合配方不兼容，不可拿来 `--resume`。
