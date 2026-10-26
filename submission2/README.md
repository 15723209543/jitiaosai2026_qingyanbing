# submission2：待训练的预训练候选

此目录不替代当前 `submission/`。当前最新的 5,855 行提交 CSV 仍在 `../submission/submission.csv`。远端已有用户训练到一半时上传的候选 PTH；本机为避免下载模型，只保留 Git LFS 指针，不能直接加载。候选尚未完成训练，也未生成可提交的候选 CSV。

## 赛规目录与文件

| 路径 | 用途 |
| --- | --- |
| `code/model.py` | ConvNeXt-Tiny 与 EfficientNetV2-S 定义；下载并核验 ImageNet-1K 原始权重。 |
| `code/train.py` | GPU 全量训练、官方验证集 AUROC 选模；单个 PTH。 |
| `code/predict.py` | 固定模型验证与测试推理；仅运行此文件才写根目录 `submission.csv`。 |
| `code/utils.py` | 整图缩放补边、数据读取、指标、提交格式。 |
| `code/requirements.txt` | 依赖；PyTorch 须按训练机 CUDA 版本先安装。 |
| `model/best_model.pth` | 训练成功后才生成；保存最佳分支参数和固定推理方案。 |
| `pretrain_model/` | 训练时下载的非医学原始权重及来源信息。 |
| `licenses/` | 训练时保存的 Apache-2.0 全文。 |
| `cost.py` | 与预测共用推理函数；5 张预热、100 张单张计时三轮。 |
| `submission.csv` | 训练完成后单独运行 `predict.py` 才生成。 |

项目根目录 `glaucoma_first/train.csv`、`valid.csv`、`test.csv` 与对应图片文件夹维持原位。代码只用官方训练集更新参数；官方验证集只用于选择模型，测试集只用于最后推理。整图等比例缩至 384×384 并补边，不裁剪；训练期轻度翻转、旋转和颜色扰动，验证/测试期固定。两模型的学习率、轮次、早停独立，使用阳性加权 focal 损失和全骨干微调。推理方案只在验证集比较两单模型及固定等权融合，不用测试结果调参；输出阈值固定 0.5，无 TTA。

## 在 GPU 训练机操作

1. 安装适配 CUDA 的 PyTorch，然后安装 `code/requirements.txt`。建议先用一张图确认 `timm`、`safetensors` 和 GPU 可用。
2. 从项目根目录运行 `python submission2/code/train.py`。需要访问公开模型卡与 Apache 官方许可页面。`train.py` 不写测试 CSV。训练与验证均在首批、每 25 批及末批打印**当前批次**的真实阳性、报阳、报对、误报、漏报；整轮验证结束再打印 AUROC 与召回等汇总。最佳权重存入一个 `model/best_model.pth`。当前脚本**不支持断点续训**；已有候选权重时会拒绝覆盖。中断前先备份权重，不能把中途 PTH 当作最终可预测模型。
3. 核对官方验证 AUROC、阳性召回、误报及模型大小；若未优于当前正式验证 AUROC **0.964011**，不要替换当前 `submission/`。**0.99 是待检验目标，不保证达到。**
4. 仅当决定提交此候选时，运行 `python submission2/code/predict.py` 生成候选根目录的 `submission.csv`，检查 5,855 行、列顺序和官方 test.csv 顺序。运行 `python submission2/cost.py <100张计时图目录> <5张预热图目录>`；须在官方指定设备复测。
5. 最终材料还需按规程补齐核验表格、模型卡/论文、原始权重 SHA-256、许可证全文与数据声明；**不要直接把此未训练完成的目录打包提交。**

官方 `valid` 必须独立于训练样本：把其中图片加入训练会使选模 AUROC 失去原有的独立验证含义，也违反本项目的数据约定。若要研究新划分，只能在 `train` 内部另外划分，且不得拿这个新分数与原官方验证 AUROC 直接比较。

若 `model.safetensors` 与当前 `timm` 版本结构不兼容，严格加载会报错并停止，不能静默随机初始化。若最终模型或必要推理文件超过 500MB，也必须改方案后重训。
