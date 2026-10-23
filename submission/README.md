# submission：ResNet-18与DenseNet-121双模型（历史92.37分方案）

> 新参赛规则适配状态：本目录当前**不是可直接提交的最终材料包**。`model/best_model.pth` 在256×256配置下仅完成ResNet-18第1轮；DenseNet-121尚未完成该配置的验证。新版`predict.py`会拒绝从未验证完整的模型生成最终CSV。历史`results/submission.csv`与当前权重不对应，继续保留供追溯。

当前活动方案恢复自 Git 提交 `6abba8e` 的 `03/`。ResNet-18 与 DenseNet-121 分别训练和验证，预测时对两者的阳性概率取算术平均，以 0.5 为阳性阈值。2026-10-18 官网记录 ID 14086 对应 92.37 分、当时 64×64 输入的本地验证 AUROC 为 0.943116；这两项都是历史结果，不能作为改为 256×256 后的成绩。

## 文件

| 路径 | 用途 |
| --- | --- |
| `code/model.py` | 两种网络结构，随机初始化，不下载外部权重 |
| `code/train.py` | 从已有权重继续训练，每个网络各跑完整训练集 5 轮，按各自验证 AUROC 更新参数，最后验证双模型；不写提交 CSV |
| `code/predict.py` | 读取双模型参数，等权平均阳性概率，将新提交CSV写入本目录根部；不修改历史`results/` |
| `code/utils.py` | 将整张图等比例缩到 256×256 范围内并补黑边、不裁剪或拉伸；指标、学习率、单文件权重及 CSV 工具 |
| `model/best_model.pth` | 一份文件，内含两种网络的最佳参数；训练会更新此文件，原版可从 Git 历史恢复 |
| `results/submission.csv` | 历史 92.37 分对应的 5,855 行提交文件；不由新版predict.py覆盖 |
| `cost.py` | 按新规程5张预热、100张计时、batch=1、3轮测端到端耗时，并与当前模型生成的根部`submission.csv`比较 |
| `模型核验材料与数据申明清单.xlsx` | A1—A6、数据使用与材料状态；未完成项须在最终提交前补齐 |
| `pretrain_model/`、`licenses/` | 当前网络均为`weights=None`，没有外部预训练权重；目录说明见各自README |

网络采用自适应池化，原 64×64 权重可作为 256×256 输入的初始化；训练开始前会在新预处理下重测官方验证集，旧指标不可直接比较。缩放到 256 仍会损失部分像素信息，但比 64 保留更多细节；黑边保持原始长宽比和全部视野。新规程写明单张平均端到端耗时上限为**100秒**，而非100毫秒；仍需按`cost.py`在指定硬件上实测，不能用此前仅前向耗时代替。

## 运行

在项目根目录使用具备 `requirements.txt` 依赖的 Python 环境：

```powershell
& 'C:\Users\LENOVO\.conda\envs\deepStudy\python.exe' submission/code/train.py
& 'C:\Users\LENOVO\.conda\envs\deepStudy\python.exe' submission/code/predict.py
```

运行 `train.py` 会直接开始全量五轮训练，不再跳过已有权重；本机 `deepStudy` 是 CPU 版 PyTorch，耗时可能很长。更换为 GPU 环境后，程序会自动使用 CUDA。官方数据目录为根目录 `glaucoma_first/`，原始数据不进入 Git。训练记录保存在 `项目文档/训练记录/`。

两个模型都完成当前配置的验证后，才可单独生成新CSV：

```powershell
python submission/code/predict.py <测试图片目录> submission/submission.csv
python submission/cost.py <pictures_100> <pictures_5>
```

若有官方`test.csv`需保持其行顺序，可在预测命令末尾追加`--test-csv <路径>`。`cost.py`要求与本次模型对应的根部`submission.csv`存在，输出FP32计时、100张逐图3轮均值及CSV不一致数量；最终以主办方指定设备复测为准。

新规程允许合规的非医学领域开源预训练权重，也允许训练阶段的数据增强和类别不平衡处理。当前代码已有水平翻转、轻度颜色增强与类别加权交叉熵；仍保持从零训练，不会在未核实许可证与来源时自动下载预训练权重。[ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)与[DenseNet-121](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html)有ImageNet权重选项，但是否提高本赛题AUROC须另做合规实验。若启用预训练，需同时提供权重公开链接、许可证全文、上游训练数据说明和原始权重SHA-256，并重新训练验证。

第10、11次及98.10分参考衍生的历史分析CSV使用了测试集预测做加权/规则组合；按照新规程第5条，**不得再利用这些文件调阈值、选模型或拟合后处理参数**。本目录训练与推理代码不读取这些分析CSV。最终材料中的模型、CSV、计时和核验清单须相互对应；此次按你的要求先准备材料，**不创建ZIP、不代替你向官网上传**。
