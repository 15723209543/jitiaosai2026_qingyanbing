# submission4 / submission5：可配置比例的 train+valid 抽样实验

## 来源与目标

用户已澄清：所说的 **97.78** 指当前 `submission/` 文件夹中的模型；此官网成绩由用户提供，仓库成绩台账尚无对应记录。当前 `submission/code/model.py`、`utils.py` 及 `training/20261028/` 记录了 ConvNeXt-Tiny + EfficientNetV2-S、384 像素整图等比补边、均衡批次、focal 加成对 AUC 排序损失等配方。本次复制当前 `submission/` 的模型结构、预处理、损失和学习率，分别建立 `submission4/` 和 `submission5/`；两套新代码彼此只在抽样方式及对应指标标记上不同。相对当前 `submission/` 的续训脚本，新实验从原始 ImageNet 权重开始并使用新的固定划分，因此不能把旧权重或旧成绩算作本轮结果。历史提交 `e801fe5` 中的旧 `submission4` 是 RGB64 五模型方案，与本实验无关；仓库另有 `v1.15.1` 的历史官网 97.82 记录。上述旧成绩均不属于本轮新实验。

用户本次指定将官方 `glaucoma_first/train` 与 `valid` 合成抽样池，训练与评估均从这个池中取样，并可自行修改百分比。两套代码各自默认 `80`，也支持命令行 `--percent 80`。每个目录固定一种抽样方式，无需 `--mode` 参数。一次运行只抽一次，整次训练中的每轮都使用同一划分和固定随机种子；新建一次运行可换种子再抽。这样每轮的 AUROC 可以在同一批验证图片上比较。

两个新训练器均采用底层配置中的每支最多 30 轮、连续 10 轮未提升早停。当前 `submission/` 的 2026-10-28 续训是从已有医学任务权重启动、每支最多 3 轮且首次整体提升即停止；新实验不能沿用该权重，否则 `submission4` 的随机留出样本早已被旧模型见过。因此它们并非旧训练过程的逐轮复现，官网成绩也不能预先推断。

## 两个独立实验

| 目录 | 百分比为 80 时 | 指标解释 |
| --- | --- | --- |
| `submission4/` | 从合并池分层随机取约 80% 训练；剩余约 20% 验证 | 同一次运行中训练和验证图片互斥，可用于该次运行选模。它不再是原始官方 `valid` 的结果。 |
| `submission5/` | 从合并池独立抽约 80% 训练，再独立抽约 80% 评估 | 两组允许重叠，期望约有全池 64% 同时出现在两组；AUROC 含已训练样本，只能称为**重叠评估**。 |

抽样按标签分层，使两组都有阳性和阴性；实际张数、类别数、重叠数和划分指纹以运行日志为准。`80` 是可修改的默认值，比例需在 0 与 100 之间。当前官方池记录约 57,245 张，80/20 的理论张数约为 45,796/11,449，分层取整后可能相差一两张。

`submission4` 的样本互斥仅指本次抽出的图片。若从 `submission/`、`submission3/` 或其他已经见过官方图片的权重暖启动，随机留出的图片可能早已参与过训练或选模，不能再称为独立留出验证。两个实验分别从有来源与许可证的**非医学 ImageNet-1K 原始预训练权重**初始化，训练进度、最终模型和 CSV 互不共用；不能使用外部医学数据或医学权重。没有患者标识时，也无法证明患者级互斥。

论文依据：[CCF 人工智能 A 类会议目录](https://www.ccf.org.cn/Academic_Evaluation/AI/zgjsjxhtjgjxshy/al/)收录 ICCV；[ICCV 2021 的 Deep AUC Maximization 论文](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html)在医学图像实验中使用分层的 8:2 无泄漏验证，并强调同一患者图片不能跨训练与验证。这里据此在 `submission4` 提供固定、互斥的留出验证；本题目前没有患者 ID，因此代码只能保证图片记录互斥，不能达到论文的患者级划分。当前 `submission/` 已有成对 AUC 排序损失，但它不是该论文 AUC margin 损失的原样实现；本实验不同时改变损失，以便看清抽样方式的影响。

## 在 GPU 训练机执行

1. 先确认没有其他同项目训练进程；保持 `glaucoma_first/` 原图、CSV、压缩包及 junction 目标只读。安装与驱动匹配的 CUDA 版 PyTorch，并安装对应目录 `code/requirements.txt` 中的其他依赖。当前 `model.py` 会通过 Hugging Face API 查询 ImageNet 原始权重的仓库修订版并下载到各自 `pretrain_model/`，因此首次启动需能访问 Hugging Face；仓库中现有 `submission/pretrain_model/` 不会自动被新实验读取。确认权重、许可证、来源和 SHA-256 可用。
2. 从仓库根目录分别运行两个实验；为避免抢占显存，先完成一个再启动另一个：

   ```powershell
   python submission4/code/train.py --percent 80 --seed 2026
   python submission5/code/train.py --percent 80 --seed 2026
   ```

3. 要改变训练比例，分别把命令中的 `80` 改成所需数字；也可改各自 `train.py` 顶部的默认比例。为比较两个实验，固定其他训练参数和随机种子。中断后只从**本目录的同一运行目录**恢复，沿用该运行保存的比例和种子：

   ```powershell
   python submission4/code/train.py --resume submission4/runs/实际运行目录名
   python submission5/code/train.py --resume submission5/runs/实际运行目录名
   ```

   `run_config.json` 保存的划分指纹覆盖图片名、标签、顺序和抽样索引，不覆盖原图文件字节。若把运行目录搬到另一台电脑续训，先按原始数据审计记录核对两台机器上的图片内容一致；不要仅凭划分指纹推断原图完全相同。

4. 各自训练结束后核对日志中的实际抽样、AUROC、最终选模、模型来源及 `training_complete`，再分别生成 CSV 和计时：

   ```powershell
   python submission4/code/predict.py
   python submission4/cost.py pictures_100 pictures_5
   python submission5/code/predict.py
   python submission5/cost.py pictures_100 pictures_5
   ```

   `submission4/model/best_model.pth` 与 `submission4/submission.csv` 属于留出验证实验；`submission5/model/best_model.pth` 与 `submission5/submission.csv` 属于重叠评估实验。每个目录的**首次完成 run** 会把最终权重复制到默认 `model/best_model.pth`；后续新 run 的模型保存在该 run 的 `final_model.pth`，不会覆盖默认权重。若要预测或计时后续 run，给两个脚本传同一个 `--checkpoint`，并给 `predict.py` 指定单独的输出 CSV 路径，例如：

   ```powershell
   python submission4/code/predict.py submission4/runs/实际运行目录名/submission.csv --checkpoint submission4/runs/实际运行目录名/final_model.pth
   python submission4/cost.py pictures_100 pictures_5 --checkpoint submission4/runs/实际运行目录名/final_model.pth
   ```

   `submission5` 相应替换目录。预测脚本仅读取官方无标签 `test`，CSV 列为 `Image_name,Probability,Pred_Label`，顺序与官方 `test.csv` 一致。各目录内的预测与计时共用同一 FP32 推理函数，不使用 TTA；测试集预测和官网分数不得用于挑轮次、阈值或调权重。

## 当前状态与下一步

本说明写作时，`submission4` 与 `submission5` 的抽样代码已完成，但尚未执行真实 GPU 训练；两个实验均没有完成权重、预测 CSV、本轮验证成绩或官网成绩。代码已通过合成数据单测与语法检查；训练机还需依次执行两个新运行，按日志确认 `submission4` 划分互斥、`submission5` 交叠数量及训练进程状态。各自训练完成并通过 CSV 格式和推理计时检查后，才能决定是否交官网；版本号中间位保持 `18`，只有用户明确指示官网上传次数改变时才修改。
