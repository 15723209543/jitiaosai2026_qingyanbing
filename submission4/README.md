# submission4：五模型三票方案

`code/model.py` 定义旧五层 CNN、ResNet18、DenseNet121、EfficientNet-B0、ConvNeXt-Tiny；`code/train.py` 只训练和验证；`code/predict.py` 单独运行时才写提交 CSV；`code/utils.py` 保存数据、指标及单文件检查点工具。仅使用主办方 `glaucoma_first` 数据，所有 torchvision 模型均设 `weights=None`。

在项目根目录用 `deepStudy` 解释器直接运行 `submission4/code/train.py`。首次运行会尝试读取 `submission/model/best_model.pth` 中的旧五层 CNN，以及 `submission3/model/best_model.pth` 中的三模型参数；若没有旧参数，相应模型从随机初始化开始。每个模型训练轮数在 `code/utils.py` 的 `EPOCHS` 中分别设置，`0` 表示不训练该分支，但仍参与基线验证与五票推理。训练前先验证五个分支和五模型组合；之后每轮依次训练各分支并验证，整轮结束再验证五模型组合。单模最佳和组合最佳根据各自验证 AUROC 更新，统一保存在 `model/best_model.pth`；训练记录保存在项目根目录的 `项目文档/训练记录/submission4/`，训练过程不会写提交文件。

训练结束后单独运行 `submission4/code/predict.py`，输出 `results/submission.csv`。五个阳性概率的中位数作为 `Probability`；它大于等于 0.5 时，恰好代表至少三模型投阳性票，写为 `Pred_Label=1`。不要用无标签测试集衡量准确率。

单文件包含两套快照（各模型历史最佳、联合历史最佳），本机空模型保存实测约 433 MB；程序在训练完成时检查是否超过 500 MB。是否满足单张图像 100 ms 推理限制仍需在实际提交设备上实测，不能仅凭参数量保证。建议先以小轮数试运行并检查终端验证 AUROC、AUPRC，再进行完整训练。
