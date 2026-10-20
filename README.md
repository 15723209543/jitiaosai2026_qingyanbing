# 2026 青光眼眼底图像二分类

当前源码版本：v1.8.2（待用户续训验证）。活动方案只有 `submission/`：五个模型分别训练、分别按官方验证集 AUROC 保存独立最佳；每轮还用五模型联合预测的 AUROC 保存整组最佳。两个快照都装在同一份 `submission/model/best_model.pth` 中。现有 `.pth` 与 CSV 未被本次代码修改覆盖，新参数是否提分尚未实测。

五个模型为旧五层 CNN、ResNet-18、DenseNet-121、EfficientNet-B0、ConvNeXt-Tiny。对每张图片的五个阳性概率降序排列后，依次乘以 `submission/code/utils.py` 中的 `RANK_WEIGHTS`，加总得到最终阳性概率；当前占比以该文件和运行开始时的终端打印为准，最终概率不低于 0.5 时输出阳性。训练阶段的整组验证、最终验证和单独预测调用同一加权函数；各分支的梯度更新仍使用自己的交叉熵损失。

## 文件位置

| 路径 | 用途 |
| --- | --- |
| `submission/code/model.py` | 五个模型定义和数据集入口 |
| `submission/code/train.py` | 单独训练与验证，不生成提交 CSV |
| `submission/code/predict.py` | 读取最佳整组快照并生成提交 CSV |
| `submission/code/utils.py` | 数据、指标、参数保存、排序占比、CSV 工具；调整轮数与占比在此文件 |
| `submission/code/tools/test_five_models.py` | 不使用比赛图片的快速行为测试；`code/` 下不再有 `tests` 文件夹 |
| `submission/model/best_model.pth` | 一份五模型参数文件；请勿手动删除或覆盖 |
| `submission/results/submission.csv` | 单独运行预测脚本后生成，提交前核对 |
| `glaucoma_first/` | 官方训练、验证、测试数据，仅保留本地，不进入 Git |
| `项目文档/训练记录/` | 逐轮训练记录 |
| `三个不用的模型.zip` | 旧 `submission`、`submission2`、`submission3` 的本地备份；被 Git 忽略，不进入普通 clone/push |

## 运行

在项目根目录，用 `deepStudy` 解释器运行：

```powershell
& "C:\Users\LENOVO\.conda\envs\deepStudy\python.exe" submission/code/train.py
& "C:\Users\LENOVO\.conda\envs\deepStudy\python.exe" submission/code/predict.py
```

训练入口先打印 CPU/GPU、五个排序占比和每个模型的独立训练参数，再读取完整官方训练集和有标签验证集；第一轮前先测基线。每个模型的累计目标轮数、学习率、正则、阳性损失占比、Focal Loss 系数及梯度截断均在 `submission/code/utils.py` 的 `MODEL_TRAINING` 中单独修改；`epochs=0` 表示不训练该模型，但仍参加验证。现有检查点已训练30轮，本版 CNN 保持30轮不再训练，另外四个目标轮数分别为38、40、42、42。训练中显示 loss、真实阳性、报对阳性和误报；验证显示 AUROC、AUPRC、AP、真实阳性、报对阳性、误报、漏报及阳性召回率，不再把 Acc 打印到终端。各模型独立最佳和联合最佳仍只按官方验证 AUROC 严格提高更新。

单独运行预测入口时，先按同样格式验证有标签验证集，再读取无标签测试集并写 CSV。CSV 列为 `Image_name,Probability,Pred_Label`，保持官方测试清单顺序。无标签测试集不能计算召回率或 AUROC。

四模型续训配方与本次修改的依据详见[参数调整说明](项目文档/修改说明/v1.8.2四模型独立续训.md)。本次只改代码和轻量检查，完整训练由用户执行；新配置不保证官网分数上升。

更改 `RANK_WEIGHTS` 后，旧 `.pth` 的整组最佳快照未必仍是新占比下的最佳；重新运行训练入口会先在官方验证集上复核并按新占比继续选模。不要仅凭改权重后的 CSV 推断验证指标或官网分数。

本项目不使用外部训练数据或预训练参数。模型文件需低于 500 MB；单张图像推理 100 ms 需在目标机器实测。大模型文件由 Git LFS 管理，克隆后请运行 `git lfs pull`。旧版本 ZIP 已作为 [v1.7.6 Release 附件](https://github.com/15723209543/jitiaosai2026_qingyanbing/releases/tag/v1.7.6)保存，不随普通 `git clone` 下载。
