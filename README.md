# 2026 青光眼眼底图像二分类

当前代码版本：v1.7.6。活动方案只有 `submission/`：五个模型分别训练、分别按官方验证集 AUROC 保存独立最佳；每轮还用五模型联合预测的 AUROC 保存整组最佳。两个快照都装在同一份 `submission/model/best_model.pth` 中。

五个模型为旧五层 CNN、ResNet-18、DenseNet-121、EfficientNet-B0、ConvNeXt-Tiny。对每张图片的五个阳性概率降序排列后，依次乘以 `submission/code/utils.py` 中的 `RANK_WEIGHTS`，加总得到最终阳性概率；目前占比为 `(0.3, 0.3, 0.2, 0.1, 0.1)`，最终概率不低于 0.5 时输出阳性。训练阶段的整组验证、最终验证和单独预测调用同一加权函数；各分支的梯度更新仍使用自己的交叉熵损失。

## 文件位置

| 路径 | 用途 |
| --- | --- |
| `submission/code/model.py` | 五个模型定义和数据集入口 |
| `submission/code/train.py` | 单独训练与验证，不生成提交 CSV |
| `submission/code/predict.py` | 读取最佳整组快照并生成提交 CSV |
| `submission/code/utils.py` | 数据、指标、参数保存、排序占比、CSV 工具；调整轮数与占比在此文件 |
| `submission/code/tests/` | 不使用比赛图片的快速行为测试 |
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

训练入口首先打印 CPU/GPU 与五个排序占比，然后读取完整官方训练集和有标签验证集；第一轮前先测基线。`EPOCHS` 为每个模型独立的目标总轮数，设为 0 表示不训练该模型，但仍参加联合验证。只有单独运行预测入口才写 CSV，CSV 列为 `Image_name,Probability,Pred_Label`，保持官方测试清单顺序。无标签测试集不能计算准确率。

更改 `RANK_WEIGHTS` 后，旧 `.pth` 的整组最佳快照未必仍是新占比下的最佳；重新运行训练入口会先在官方验证集上复核并按新占比继续选模。不要仅凭改权重后的 CSV 推断验证指标或官网分数。

本项目不使用外部训练数据或预训练参数。模型文件需低于 500 MB；单张图像推理 100 ms 需在目标机器实测。大模型文件由 Git LFS 管理，克隆后请运行 `git lfs pull`。旧版本备份 ZIP 若未作为 GitHub Release 附件上传，则仅存在当前本地工作区，不会随 `git clone` 下载。
