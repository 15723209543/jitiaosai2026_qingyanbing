# 2026 青光眼眼底图像二分类

当前源码版本：v1.8.8（待用户续训验证）。活动方案只有 `submission/`：五个模型分别训练、分别按官方验证集 AUROC 保存独立最佳；每轮还用五模型联合预测的 AUROC 保存整组最佳。两个快照都装在同一份 `submission/model/best_model.pth` 中。现有 `.pth` 与 CSV 未被本次代码修改覆盖，新参数是否提分尚未实测。

五个模型为旧五层 CNN、ResNet-18、DenseNet-121、EfficientNet-B0、ConvNeXt-Tiny。后四个完整网络由 `torchvision` 提供，`model.py` 只构造它们并替换二分类头；原来近 200 行的课堂示例还包含训练和测试，当前已分别放进 `train.py`、`predict.py`。对每张图片的五个阳性概率降序排列后，乘以训练确定的五个占比，加总得到最终阳性概率；`utils.py` 的 `RANK_WEIGHTS` 只是初始值。每轮五个分支各自训练后，再用五模型预测完整训练集来调整排序占比，随后用官方有标签验证集验收；最佳占比和模型参数一起写入 `.pth`。`predict.py` 从 `.pth` 读取最佳占比，仍以 0.5 判阳性。各模型独立训练，CNN 用加权交叉熵，其余模型用各自参数的 Focal Loss。

## 文件位置

| 路径 | 用途 |
| --- | --- |
| `submission/code/model.py` | 五个模型定义和数据集入口 |
| `submission/code/train.py` | 单独训练与验证，不生成提交 CSV |
| `submission/code/predict.py` | 读取最佳整组快照并生成提交 CSV |
| `submission/code/utils.py` | 数据、指标、参数保存、排序占比、CSV 工具；调整轮数与占比在此文件 |
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

训练入口先打印 CPU/GPU、五个初始排序占比和每个模型的独立训练参数，再读取完整官方训练集和有标签验证集；第 0 轮前先测基线。每次运行的轮次从 0 计数，即使读取旧 `.pth` 权重，也不继承旧日志里的轮号。`MODEL_TRAINING` 中的 `epochs` 是本次运行的轮数，当前五模型均为 30；`epochs=0` 表示不训练但仍验证。学习率、正则、阳性损失占比、Focal Loss 系数及梯度截断也在 `utils.py` 独立修改。训练进度保持第1批、每25批和最后一批输出：loss 是本轮截至当前的平均损失，真实阳性、报对阳性、误报、漏报只统计本次输出区间；每轮结束打印整轮总计，`train_fp`、`train_fn` 等记录在 `项目文档/训练记录/.../training_history.csv`。每轮五模型重新预测整个训练集来搜索排序占比，最多移动 0.05；候选占比只有在官方验证 AUROC 提高时才连同整组最佳模型一起保存，同训练 AUROC 时优先减少误报与漏报之和。轮末打印候选及最终保留的占比，记录在 `run_config.json`。这一步增加每轮五模型遍历训练集的时间。训练期间的验证输出 AUROC、AUPRC、AP，不再打印 Acc。动态占比不能保证官网成绩提升，也不能替代独立测试。

单独运行预测入口时，先按同样格式验证有标签验证集，再读取无标签测试集并写 CSV。CSV 列为 `Image_name,Probability,Pred_Label`，保持官方测试清单顺序。无标签测试集不能计算召回率或 AUROC。

四模型续训配方与本次修改的依据详见[参数调整说明](项目文档/修改说明/v1.8.2四模型独立续训.md)。本次只改代码和轻量检查，完整训练由用户执行；新配置不保证官网分数上升。

更改初始 `RANK_WEIGHTS` 后，旧 `.pth` 的整组最佳快照未必仍是新占比下的最佳；重新运行训练入口会先用已保存的最佳占比复核旧整组，然后用新初始值验证当前五模型，再逐轮调整。单独运行 `predict.py` 始终使用 `.pth` 内保存的最佳占比；只改 `utils.py` 而不训练，不会改变预测结果。官方 `test` 无标签，无法用来计算 AUROC 或判断占比好坏，最终仅用于生成提交 CSV。不要仅凭验证指标推断官网分数。

本项目不使用外部训练数据或预训练参数。模型文件需低于 500 MB；单张图像推理 100 ms 需在目标机器实测。大模型文件由 Git LFS 管理，克隆后请运行 `git lfs pull`。旧版本 ZIP 已作为 [v1.7.6 Release 附件](https://github.com/15723209543/jitiaosai2026_qingyanbing/releases/tag/v1.7.6)保存，不随普通 `git clone` 下载。
