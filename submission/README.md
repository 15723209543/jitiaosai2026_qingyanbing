# submission：历史最高分双模型（92.37 分）

当前活动方案恢复自 Git 提交 `6abba8e` 的 `03/`。ResNet-18 与 DenseNet-121 分别训练和验证，预测时对两者的阳性概率取算术平均，以 0.5 为阳性阈值。2026-10-18 官网记录 ID 14086 对应 92.37 分；这是项目文档中已核实的非五模型最高提交分数。本地官方验证集双模型 AUROC 为 0.943116。成绩是历史结果，并不承诺再次提交仍得同分。

## 文件

| 路径 | 用途 |
| --- | --- |
| `code/model.py` | 两种网络结构，随机初始化，不下载外部权重 |
| `code/train.py` | 各网络独立训练，按各自验证 AUROC 更新参数；已有参数默认跳过重训，最后验证双模型 |
| `code/predict.py` | 读取双模型参数，等权平均阳性概率，生成提交 CSV |
| `code/utils.py` | 官方数据读图、指标、学习率、单文件权重及 CSV 工具 |
| `model/best_model.pth` | 一份文件，内含两种网络的原始最佳参数；不是五模型 |
| `results/submission.csv` | 历史 92.37 分对应的 5,855 行提交文件 |

`best_model.pth` 只是将历史两份 `.pth` 原样封装进同一文件，没有联合训练，也没有改变预测公式。它约 73.2 MB，低于 500 MB；单张推理 100 ms 限制仍需在目标设备实测。

## 运行

在项目根目录使用具备 `requirements.txt` 依赖的 Python 环境：

```powershell
& 'C:\Users\LENOVO\.conda\envs\deepStudy\python.exe' submission/code/train.py
& 'C:\Users\LENOVO\.conda\envs\deepStudy\python.exe' submission/code/predict.py
```

若只需重新生成提交文件，直接运行 `predict.py`；`train.py` 目前会识别已有的两种最佳参数并跳过重新训练，然后用官方 `valid` 验证。若确需重训，先备份当前 `best_model.pth`，再修改 `train.py` 的 `RETRAIN_EXISTING`。官方数据目录为根目录 `glaucoma_first/`，原始数据不进入 Git。训练记录保存在 `项目文档/训练记录/`，历史分数证据见 `项目文档/提交记录/第6次提交-92.37分.md`。
