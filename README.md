# 2026 青光眼眼底图像二分类

当前代码版本：`v1.2.0`。以 day07 课堂 CNN 四步为主体，在本地训练、验证、预测，初赛只上传 `submission.csv`。

用户反馈旧版本平台成绩为 **79 分**，不代表准确率 79%。本版不承诺提升平台分数。10月15日收尾检查发现工作区已有3轮训练记录及更新后的模型，已保留；助手不继续训练。该记录中最佳验证 AUROC 为0.896657，仍需由用户运行预测并提交确认平台表现。

## 1. 固定协作约定

- 每次任务开始先 `git fetch origin`，核对工作区、本地提交与远端。不同步时先保留本地训练产物再同步；有冲突先确认，不用强制重置覆盖本地文件。
- 修改和轻量检查完成后提交并推送。提交格式：`版本号 类型(影响范围):行为`；本版为 `v1.2.0 feat(train):优化正则化与训练记录`。
- `glaucoma_first/` 全部原始数据、`.idea/`、缓存和临时检查文件不提交；其余模型、答案、记录和文档提交到私密仓库。
- 助手只改代码和做轻量验证，不启动正式训练或整批预测；由用户运行。
- 仅 `submission/code/` 存放四个 Python 源文件，保持课堂编号注释和函数来源说明。

仓库：[jitiaosai2026_qingyanbing](https://github.com/15723209543/jitiaosai2026_qingyanbing)。

## 2. 本版修改

保持 `32→64→128` 卷积通道、`4608→2048→2048→2` 全连接结构和 64×64 输入，不增加神经元。

| 项目 | 新训练默认设置 |
|---|---|
| 中间激活 | LeakyReLU，负半轴斜率 0.01 |
| 卷积失活 | 第三组卷积/池化后 Dropout2d(0.1) |
| 全连接失活 | 保留 Dropout(0.5) |
| 优化器 | 保留课堂 Adam，学习率 5e-5，L2 权重衰减 1e-4 |
| 学习率调整 | 验证 AUROC 连续 2 轮无显著提升时减半，最低 1e-6 |
| 训练上限 | 12 轮；连续 5 轮未改善最佳模型则提前停止 |
| 最佳模型 | 优先比较验证 AUROC，相等时比较梯形 AUPRC |
| 训练记录 | 项目文档/训练记录/时间目录，每次运行独立保存 |

旧版已含全连接 Dropout；本次新增卷积通道失活。验证和预测均用 `eval()` 关闭随机失活。
旧参数仍按旧 ReLU 配置预测，新参数将权重和配置一起保存，避免错误套用激活函数。

## 3. 目录与数据

```text
项目根目录/
├─ submission/
│  ├─ code/
│  │  ├─ model.py                 # todo1数据接口、todo2模型；main只打印结构
│  │  ├─ train.py                 # todo3训练、每轮验证；不生成提交答案
│  │  ├─ predict.py               # todo4验证/预测；main生成提交答案
│  │  ├─ utils.py                 # 必要的数据、加载、记录和CSV工具
│  │  └─ requirements.txt
│  ├─ model/
│  │  ├─ best_model.pth           # 当前最佳权重；新版同时包含模型配置
│  │  └─ config.json              # 当前最佳模型说明
│  └─ results/
│     └─ submission.csv          # 仅单独运行predict.py时更新
├─ glaucoma_first/               # 不提交GitHub，也不上传比赛
│  ├─ train.csv / train/          # 51396张，有True_Label
│  ├─ valid.csv / valid/          # 5849张，有True_Label
│  └─ test.csv / test/            # 5855张，无真实标签
├─ 项目文档/
│  ├─ 训练记录/
│  │  ├─ 20261014_v1.1.1_79分基线/ # 修改前的模型、配置、答案和训练历史
│  │  └─ 时间目录/                # 用户以后每次运行train.py时新建
│  │     ├─ training_history.csv  # 每轮loss/acc/验证指标/学习率
│  │     └─ run_config.json       # 本次参数、完成轮数、最佳模型信息
│  └─ 各项说明文档
└─ README.md
```

训练和验证 CSV 包含 `Image_name,Age,True_Label`，测试 CSV 包含 `Image_name,Age`。
标签 1 为阳性，0 为阴性。当前 CNN 只用图片，不使用 Age。CSV 的 .jpg 与磁盘 .JPG 忽略大小写匹配。

## 4. 你接下来怎么运行

使用 `jitiaosai` 解释器，先运行训练（也可以在编辑器中运行 train.py）：

```powershell
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission/code/train.py
```

程序先按旧模型自己的配置验证旧权重，再把权重加载到精调模型。每轮显示进度、loss/acc、学习率和验证指标；只有验证改善才覆盖最佳参数。没有改善则保持旧模型及其配置。
训练结束打印记录目录，不读取无标签 test，不更新答案 CSV。`RESUME=True` 只恢复权重，Adam 和调度器重新初始化，不是完整断点恢复。

需要提交时，再单独运行预测：

```powershell
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission/code/predict.py
```

只上传 `submission/results/submission.csv`。训练完成但未运行预测时，旧 CSV 仍是旧答案。

## 5. 参数在哪里调整

utils.py：`EPOCHS=12`、`LEARNING_RATE=5e-5`、`WEIGHT_DECAY=1e-4`、`EARLY_STOPPING_PATIENCE=5`、`LR_PATIENCE=1`、`MIN_LEARNING_RATE=1e-6`、`BATCH_SIZE=64`、`CPU_THREADS=8`、`RESUME=True`、`LOG_EVERY=25`。

model.py 的 ImageModel.__init__：`negative_slope=0.01`、`conv_dropout=0.1`、`dropout=0.5`。
改默认值影响下一次训练；预测始终以已保存权重对应的配置为准。不要只改 IMAGE_SIZE，全连接输入维度也要匹配。后续实验尽量一次只改一项。

## 6. CSV 与规则

```csv
Image_name,Probability,Pred_Label
test_00000.jpg,0.920000,1
test_00001.jpg,0.120000,0
```

- UTF-8，三列顺序固定，5855行，文件名/顺序与 test.csv 一致。
- Probability 是类别1概率，不是最大类别置信度；保留6位小数后按0.5判标签。
- test无真实标签，不能计算本地真实测试准确率。主要比较验证 AUROC，兼看梯形 AUPRC、AP 和召回率。
- 初赛只上传 CSV；复赛按主办方通知提交模型和结果。模型小于500MB，单张延迟要求需在官方环境实测。
- 根据用户提供的提交页：每天成功提交1次；失败允许重试，当天最多2次，最终以网站当日规则为准。

## 7. 环境与说明

规程推荐 Python 3.12、PyTorch 2.10.0、torchvision 0.25.0。依赖列表继续保留推荐版本，本次不升级或降级环境，不修改 deepStudy。

- [完整代码、函数及算法说明](项目文档/项目代码与算法说明.md)
- [PDF说明](项目文档/青光眼分类项目代码与算法说明.pdf)
- [终端输出解释](项目文档/终端输出与评价指标说明.md)
- [后续调整与规划](项目文档/后续调整与规划.md)
- [环境检查报告](项目文档/jitiaosai环境检查报告.md)
- [本版轻量检查记录](项目文档/v1.2.0代码检查记录.md)

修改前的本地最佳验证 AUROC 为0.893341、梯形 AUPRC 为0.394133、AP 为0.396894。这些是旧模型的结果，不是新方案的实测成绩。
