# 2026 青光眼眼底图像二分类

代码版本 **v1.6.2**。`submission2/` 是新方案：ResNet-18 与 DenseNet-121 在**特征层连接**，经过同一个融合分类头，联合训练并保存为**一份** `model/best_model.pth`。这与历史版本中「分别训练，再平均两个阳性概率」不同。原 `submission/` 是独立保留的旧单模型基线，不受本次修改影响。

## 目录

| 路径 | 用途 |
| --- | --- |
| `submission2/code/model.py` | 定义双骨干特征融合网络；直接运行可看网络结构和参数量 |
| `submission2/code/train.py` | 在官方训练集上联合训练；每轮用官方验证集计算 AUROC，仅在提高时更新权重；**不写提交 CSV** |
| `submission2/code/predict.py` | 读取单个融合权重，预测无标签测试集，生成提交 CSV |
| `submission2/code/utils.py` | 数据读取、图像变换、指标、权重读写和 CSV 工具 |
| `submission2/model/best_model.pth` | 唯一一份融合模型参数，含两骨干和分类头；当前需继续训练 |
| `submission2/results/submission.csv` | `predict.py` 生成的待提交文件 |
| `submission/` | 旧单模型基线；不作为本版训练初始化或提交目录 |
| `glaucoma_first/` | 官方训练、验证、测试原图与 CSV；只留本地，不提交 GitHub |
| `项目文档/训练记录/` | 各次训练的日志和逐轮指标，不混入 `results/` |

旧版 01–08 双模型筛选过程见[历史实验结果](项目文档/八组双模型训练结果_2026-10-18.md)。03 的官网历史提交得分是 **92.37**；它不是本次联合融合网络的官网成绩。新方案的运行实测见[本版记录](项目文档/修改说明/v1.6.1联合融合与后续方案.md)。

`submission2/code/` 与原 `submission/code/` 一样，只保留 `model.py`、`train.py`、`predict.py`、`utils.py` 和 `requirements.txt`；数据与权重分别在同级的 `glaucoma_first/`、`submission2/model/`，训练记录在 `项目文档/`。`utils.py` 就是原项目约定的工具文件，不另增测试源码目录。

## 运行

本机解释器是 `C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe`，依赖见 `submission2/code/requirements.txt`；原 `deepStudy` 环境不受影响。

```powershell
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission2/code/train.py
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission2/code/predict.py
```

本版 `best_model.pth` 是融合模型的**续训起点**：当前仅完成 3 轮（第 1–2 轮预热、第 3 轮联合更新），最佳权重来自第 2 轮，验证 AUROC 0.942946；第 4 轮被用户要求中断。你点击 `train.py` 后会读取这一份权重，从第 1 轮开始继续**同时更新两个骨干和融合头**，仅在完整验证集 AUROC 更高时覆盖这一份文件；不会写测试 CSV。它不是把两份旧模型文件简单打包。若主动移除这份权重，代码仍能随机初始化训练，但失去当前起点，效果可能下降。

单独运行 `predict.py` 才生成 `submission2/results/submission.csv`。格式是 `Image_name,Probability,Pred_Label`，与官方 `test.csv` 的顺序一致；`Probability` 是青光眼阳性概率。无标签测试集没有可计算的真实准确率。提交前确认本版 CSV 的行数、列名、概率范围和模型大小；官网分数只能在提交后得知。

本项目不使用外部数据或预训练权重。模型文件上限 500 MB；「小于 500 MB」是约束，不代表要尽量填满。单张推理 100 ms 的要求须在目标环境实测，不应仅按参数量推断。也不能保证本版超过历史官网 92.37。

为使单个模型文件可上传至 GitHub，`submission2/model/*.pth` 使用 Git LFS；克隆仓库时请安装 Git LFS 并运行 `git lfs pull`。原始数据集始终不进入 Git。
