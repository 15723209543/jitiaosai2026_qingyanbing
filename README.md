# 2026 青光眼眼底图像二分类

代码版本 **v1.6.10**。`submission3/` 是正在验证的三模型实验：ResNet-18、DenseNet-121、EfficientNet-B0 独立训练和选模，在预测时对三个阳性概率取中位数，三份参数封装在**一份** `model/best_model.pth`。每个模型每轮遍历全部 51,396 张训练图片，而非抽样 4,096 次；训练批量 64 时为 804 批。首次训练前并行解码并缓存缩小后的训练/验证图片，后续轮次复用缓存。仅在三种模型都完成训练后，才单独运行 `submission3/code/predict.py` 写 CSV。`submission2/` 是已保留的双骨干特征融合方案；原 `submission/` 是旧单模型基线。

## 目录

| 路径 | 用途 |
| --- | --- |
| `submission3/code/` | 三模型实验的训练、模型、预测、工具及测试源码；用 `deepStudy` 解释器运行 |
| `submission3/model/best_model.pth` | 三份模型参数合为一份；`complete=False` 时不可预测 |
| `submission3/results/submission.csv` | 只在单独执行 `submission3/code/predict.py` 后生成 |
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

`submission3` 使用你指定的 `deepStudy` 解释器。`train.py` 启动时会打印实际设备；CUDA 可用时训练及验证优先用 GPU，否则用 CPU。当前本机 `deepStudy` 安装的是 PyTorch CPU 版，故仍会显示 CPU；代码本身不能代替 CUDA 版 PyTorch 或 NVIDIA 显卡。已有单文件参数时，直接运行训练入口续训，不要再添加 `--bootstrap-dir`；全量训练会比旧抽样方案耗时明显更长。训练全部结束后才单独运行预测入口：

```powershell
& "C:\Users\LENOVO\.conda\envs\deepStudy\python.exe" submission3/code/train.py
& "C:\Users\LENOVO\.conda\envs\deepStudy\python.exe" submission3/code/predict.py
```

`submission3` 的逐轮规则、保存错误处理和验证说明见[三模型方案](项目文档/修改说明/submission3三模型独立训练方案.md)。下方是保留的 `submission2` 使用说明。

本机解释器是 `C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe`，依赖见 `submission2/code/requirements.txt`；原 `deepStudy` 环境不受影响。

```powershell
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission2/code/train.py
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission2/code/predict.py
```

点击 `train.py` 时，如已有 `best_model.pth`，会先用**有标签的官方验证集**打印当前模型指标及已保存的最佳 AUROC，再从这份权重继续同时更新两个骨干和融合头；只有后续完整验证集 AUROC 更高才覆盖权重。它不会用无标签 `test` 计算准确率，也不会写测试 CSV。若主动移除这份权重，代码仍可随机初始化训练，但失去已训练起点。权重当前实际轮次与指标以最近的 `项目文档/训练记录/` 和权重内元数据为准。

单独运行 `predict.py` 才生成 `submission2/results/submission.csv`。格式是 `Image_name,Probability,Pred_Label`，与官方 `test.csv` 的顺序一致；`Probability` 是青光眼阳性概率。无标签测试集没有可计算的真实准确率。提交前确认本版 CSV 的行数、列名、概率范围和模型大小；官网分数只能在提交后得知。

本项目不使用外部数据或预训练权重。模型文件上限 500 MB；「小于 500 MB」是约束，不代表要尽量填满。单张推理 100 ms 的要求须在目标环境实测，不应仅按参数量推断。也不能保证本版超过历史官网 92.37。

为使单个模型文件可上传至 GitHub，`submission2/model/*.pth` 使用 Git LFS；克隆仓库时请安装 Git LFS 并运行 `git lfs pull`。原始数据集始终不进入 Git。
