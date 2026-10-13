# 2026 青光眼眼底图像二分类

本项目依据《2026 年第八届全国高校计算机能力挑战赛赛项规程》赛项一更新。
代码仍以学习资料中的 day07 课堂 CNN 为模板；主要变化是训练和预测都在本地完成，
初赛只上传最终生成的 `submission.csv`，不再把代码交给服务器运行。

## 1. 2026 官方要求

- 任务：判断眼底图像为青光眼阳性的概率；标签 `1` 为阳性，`0` 为阴性。
- 数据：训练集 51396 张、验证集 5849 张、测试集 5855 张；图片为 JPG/PNG，尺寸不固定。
- 评价：主要指标 AUROC，辅助指标 AUPRC；每日最多提交 2 次。
- 初赛：只提交测试集预测结果 `submission.csv`。
- 复赛：提交 `submission.csv` 和模型以供验证。
- CSV：UTF-8 编码，字段顺序必须是 `Image_name,Probability,Pred_Label`。
- `Probability`：类别 1（青光眼阳性）的概率，而不是最大类别置信度。
- `Pred_Label`：默认以 `0.5` 为阈值。
- 推荐环境：Python 3.12、PyTorch 2.10.0、torchvision 0.25.0。
- 约束：模型压缩前不超过 500MB，单张端到端延迟不超过 100ms；不得使用外部标注数据。

## 2. 目录

```text
项目根目录/
├─ submission/
│  ├─ code/                         # 唯一 Python 源码目录
│  │  ├─ model.py
│  │  ├─ train.py
│  │  ├─ predict.py
│  │  ├─ utils.py
│  │  └─ requirements.txt
│  ├─ model/
│  │  ├─ best_model.pth             # 运行 train.py 后生成
│  │  └─ config.json
│  └─ results/
│     └─ submission.csv             # 运行 predict.py 后生成；初赛只上传这个文件
├─ glaucoma_first/
│  ├─ train.csv                     # Image_name/Age/True_Label，共 51396 行
│  ├─ valid.csv                     # Image_name/Age/True_Label，共 5849 行
│  ├─ test.csv                      # Image_name/Age，无标签，共 5855 行
│  ├─ train/
│  ├─ valid/
│  └─ test/
├─ 项目文档/
└─ README.md
```

数据集不进入提交文件。`submission/code/` 继续只保留四个 Python 文件和
`requirements.txt`。

## 3. 数据接口

真实数据位于 `glaucoma_first/`。`train.csv`、`valid.csv` 的列为 `Image_name`、`Age`、
`True_Label`，无标签的 `test.csv` 只有 `Image_name`、`Age`。训练/验证/测试规模分别为
51396/5849/5855，图片分别位于 `train/`、`valid/`、`test/`。
其中 `Age` 为辅助信息；课堂 CNN 当前仍按图像二分类流程训练。

```csv
Image_name,Age,True_Label
000001.jpg,62,0
000002.png,71,1
```

项目已完成 1 epoch 的首轮 CPU 全量基线；后续应依据下文实测指标安排改进实验。

## 4. 本地训练

使用项目专用解释器运行：

```powershell
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission/code/train.py
```

`train.py` 会：

1. 读取训练集和官方验证集；
2. 根据 0/1 样本数自动计算交叉熵类别权重，减轻阳性样本极少的问题；
3. 保持课堂 CNN 的 DataLoader、前向传播、反向传播和 Adam 更新流程；
4. 每轮按课堂格式打印 `epoch/loss/acc/time`；
5. 保存参数到 `submission/model/best_model.pth`；
6. 调用 `predict.py` 的 `evaluate()` 打印验证集 `Acc`。

当前文件名是 `best_model.pth`，但代码保存的是最后一轮参数，并没有自动选择最佳轮。
AUROC/AUPRC 最终由比赛平台根据提交概率计算，训练准确率不能代替排行榜指标。

## 5. 本地预测并生成提交文件

确认测试图片已位于 `glaucoma_first/test/` 且模型参数已经生成，然后直接运行：

```powershell
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission/code/predict.py
```

无需命令行参数。程序会自动生成：

```text
submission/results/submission.csv
```

格式为：

```csv
Image_name,Probability,Pred_Label
test_00000.jpg,0.920000,1
test_00001.jpg,0.120000,0
```

预测时对两类 logits 执行 Softmax，只保存索引 1 的概率；概率大于或等于 `0.5` 时
`Pred_Label=1`。CSV 每张测试图片一行，图片名不含本机绝对路径。

## 6. 最终提交

初赛只上传这一份文件：

```text
submission/results/submission.csv
```

不要上传 `submission` 文件夹、源码、测试集或模型。上传前检查：

1. 文件名严格为 `submission.csv`；
2. UTF-8 编码；
3. 表头严格为 `Image_name,Probability,Pred_Label`；
4. 数据行数为 5855，且与测试图片一一对应；
5. `Probability` 全部位于 `[0,1]`；
6. `Pred_Label` 只能是 0 或 1；
7. 没有空值、重复文件名或本机绝对路径。

复赛时再按主办方通知同时提交该 CSV 和 `best_model.pth`。

## 7. 模型与环境

模型输入为 `3×64×64`，使用三组卷积、ReLU 和最大池化，随后为
`2048→2048→2` 全连接层及 Dropout 0.5。参数约 1373 万，float32 权重约 54.9MB，
低于规程的 500MB 上限。

`requirements.txt` 已按规程固定：

```text
torch==2.10.0
torchvision==0.25.0
```

本机 `jitiaosai` 当前是 Python 3.12.13、PyTorch 2.13.0+cpu、torchvision
0.28.0+cpu；这套环境用于接口验证，未自动降级。正式大规模训练或复赛环境复现时，
应按规程推荐版本重新安装。原有 `deepStudy` 解释器没有被删除、覆盖或改名。

## 8. 首轮实测结果与当前边界

首轮全量训练已完成：训练集 51396 张、验证集 5849 张，1 epoch；训练 loss `0.49328`，
训练循环打印 acc `0.97`，耗时 `1624.53s`。验证结果为 accuracy `0.967345`、
AUROC `0.628867`、AP/AUPRC `0.048873`。验证概率范围为 `0.064293`～`0.308502`。

验证集以阈值 0.5 判定时 TP=0、TN=5658、FP=0、FN=191，sensitivity=0、
specificity=1。由此可见，约 97% 的准确率主要受类别不平衡影响，不能说明阳性识别有效。
仅作阈值诊断参考：F1 最优阈值 `0.257520`（F1 `0.094390`、sensitivity
`0.277487`、specificity `0.844645`）；Youden 阈值 `0.190134`
（sensitivity `0.790576`、specificity `0.446271`）。这些验证集阈值不应直接视为最终
竞赛阈值或排行榜成绩。

提交 CSV 为 5855 个数据行，三列及测试文件名/顺序匹配正确；但按默认 0.5 阈值，
`Pred_Label` 全为 0。连续 `Probability` 仍可用于 AUROC 排序评价。模型文件大小为
54935871 bytes。当前模型和 CSV 只是首轮流程基线，不建议视为最终竞赛方案；应优先改善
阳性检出能力，并以验证 AUROC/AP 等指标评估后续实验。
