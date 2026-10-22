# submission：历史最高分双模型（92.37 分）

当前活动方案恢复自 Git 提交 `6abba8e` 的 `03/`。ResNet-18 与 DenseNet-121 分别训练和验证，预测时对两者的阳性概率取算术平均，以 0.5 为阳性阈值。2026-10-18 官网记录 ID 14086 对应 92.37 分、当时 64×64 输入的本地验证 AUROC 为 0.943116；这两项都是历史结果，不能作为改为 256×256 后的成绩。

## 文件

| 路径 | 用途 |
| --- | --- |
| `code/model.py` | 两种网络结构，随机初始化，不下载外部权重 |
| `code/train.py` | 从已有权重继续训练，每个网络各跑完整训练集 5 轮，按各自验证 AUROC 更新参数，最后验证双模型；不写提交 CSV |
| `code/predict.py` | 读取双模型参数，等权平均阳性概率，生成提交 CSV |
| `code/utils.py` | 将整张图等比例缩到 256×256 范围内并补黑边、不裁剪或拉伸；指标、学习率、单文件权重及 CSV 工具 |
| `model/best_model.pth` | 一份文件，内含两种网络的最佳参数；训练会更新此文件，原版可从 Git 历史恢复 |
| `results/submission.csv` | 历史 92.37 分对应的 5,855 行提交文件；只有单独运行 predict.py 才会重写 |

网络采用自适应池化，原 64×64 权重可作为 256×256 输入的初始化；训练开始前会在新预处理下重测官方验证集，旧指标不可直接比较。缩放到 256 仍会损失部分像素信息，但比 64 保留更多细节；黑边保持原始长宽比和全部视野。单张推理 100 ms 限制仍需在目标设备实测，本地 CPU 双模型仅前向测试约 218 ms，未达到该限制。

## 运行

在项目根目录使用具备 `requirements.txt` 依赖的 Python 环境：

```powershell
& 'C:\Users\LENOVO\.conda\envs\deepStudy\python.exe' submission/code/train.py
& 'C:\Users\LENOVO\.conda\envs\deepStudy\python.exe' submission/code/predict.py
```

运行 `train.py` 会直接开始全量五轮训练，不再跳过已有权重；本机 `deepStudy` 是 CPU 版 PyTorch，耗时可能很长。更换为 GPU 环境后，程序会自动使用 CUDA。若只需生成新提交文件，在训练完成后单独运行 `predict.py`。官方数据目录为根目录 `glaucoma_first/`，原始数据不进入 Git。训练记录保存在 `项目文档/训练记录/`，历史分数证据见 `项目文档/提交记录/第6次提交-92.37分.md`。
