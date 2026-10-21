# 2026 青光眼眼底图像二分类

活动方案为 `submission/` 中的双视野共享骨干 CNN：同一张眼底图生成整图与候选视盘局部，两路特征在同一网络中融合，训练后保存一份 `model/best_model.pth`。这是原 `submission2/` 实验更名后的版本，目前尚未正式训练，不能直接运行 `predict.py` 提交。

原五模型 `submission/` 的代码、权重和 CSV 已打包为本地 `五模型归档.zip`；其中权重已更新为远端 v1.9.1 提交的最新版本并完成 SHA-256 校验，归档校验通过后才移出活动目录。ZIP 约 401 MB，被 Git 忽略，不随普通 clone 下载；可从 [v1.9.2 Release 附件](https://github.com/15723209543/jitiaosai2026_qingyanbing/releases/download/v1.9.2/five-model-archive.zip)下载，线上文件名为 `five-model-archive.zip`。旧权重也可从历史 Git LFS 提交恢复。此前的 `三个不用的模型.zip` 仍保持原样。

## 文件位置

| 路径 | 用途 |
| --- | --- |
| `submission/code/model.py` | 双视野共享骨干 CNN |
| `submission/code/train.py` | 用官方 train 训练、官方 valid 选模；不生成提交 CSV |
| `submission/code/predict.py` | 读取训练好的参数并生成提交 CSV |
| `submission/code/utils.py` | 数据预处理、指标和 CSV 工具；调整轮数与批量在此文件 |
| `submission/model/best_model.pth` | 训练后生成的一份模型参数；当前尚不存在 |
| `submission/results/submission.csv` | 单独运行预测脚本后生成；当前尚不存在 |
| `glaucoma_first/` | 官方训练、验证、测试数据，仅保留本地，不进入 Git |
| `项目文档/训练记录/` | 逐轮训练记录 |
| `三个不用的模型.zip` | 旧 `submission`、`submission2`、`submission3` 的本地备份；被 Git 忽略，不进入普通 clone/push |
| `五模型归档.zip` | 刚归档的五模型源码、权重与 CSV；被 Git 忽略，不进入普通 clone/push |

## 运行

在项目根目录，用 `deepStudy` 解释器运行：

```powershell
& "C:\Users\LENOVO\.conda\envs\deepStudy\python.exe" submission/code/train.py
& "C:\Users\LENOVO\.conda\envs\deepStudy\python.exe" submission/code/predict.py
```

训练入口先打印 CPU/GPU，在第一轮前用有标签官方验证集建立基线；每轮遍历完整训练集并在完整验证集计算 AUROC，仅当 AUROC 严格提高才更新一份 `.pth`。训练进度在第 1 批、每 25 批和末批打印；训练记录在 `项目文档/训练记录/`。训练完成后再单独运行 `predict.py`，它先验证有标签官方验证集，随后预测无标签 test 并写 `Image_name,Probability,Pred_Label` 三列。具体模型结构及风险见 [submission 说明](submission/README.md)。

本项目不使用外部训练数据或预训练参数。模型文件需低于 500 MB；单张图像推理 100 ms 需在目标机器实测。当前双视野方案尚无正式 AUROC 与时延结果，不能保证达标。历史大模型文件由 Git LFS 管理，克隆历史版本后请运行 `git lfs pull`。更早的 ZIP 已作为 [v1.7.6 Release 附件](https://github.com/15723209543/jitiaosai2026_qingyanbing/releases/tag/v1.7.6)保存。
