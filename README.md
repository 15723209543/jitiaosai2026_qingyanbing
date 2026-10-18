# 2026 青光眼眼底图像二分类

当前代码版本：**v1.5.4**。原`submission/`仍是v1.5.2单模型基线；根目录`01/`～`08/`为八套独立双模型实验。现已获准由我训练和验证；只有选定最终方案后才生成对应提交答案。旧92分模型仍位于`submission/model/best_model.pth`。

## 八组双模型实验（新）

每个编号目录都只有`code/`、`model/`、`results/`三个子目录；四份Python源码仅在它自己的`code/`内。八套共用本地`glaucoma_first/`官方数据，不复制原图，也不触碰旧`submission/`的权重与CSV。组合清单、运行顺序及风险见[八组实验说明](项目文档/修改说明/v1.5.3八组双模型实验.md)。

CPU实测后，八套均改为64×64输入、每轮均衡抽样8192张、最多20轮；具体取舍见[训练策略记录](项目文档/修改说明/v1.5.4训练策略.md)。

建议依次运行`01`、`02`、`03`、`04`，可训练五种单模型；运行`05`～`08`时，代码会核验相同架构/训练方案的参数并复制进本方案`model/`，避免重复训练。若你改变训练参数，必须同步修改`utils.py`中的`RECIPE`标记以免误复用旧权重。每套`train.py`都只训练或复用两种单模型，并打印双模型验证AUROC；**不会写提交CSV**。只有对应`predict.py`运行后，才会生成该套`results/submission.csv`。

```powershell
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" 01/code/train.py
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" 01/code/predict.py
```

把`01`替换为其他编号即可。每套可能生成自己的`submission.csv`供比较，但**不要把八份全部提交**；根据独立验证AUROC、AUPRC、模型总大小及单张端到端延迟筛选后，只手动提交选中的一份。若双模型不如最佳单模型或超出赛项限制，应保留单模型。本机轻量检查不保证官网成绩或最终设备的100ms合规性。

## 文件放置

| 位置 | 内容 |
|---|---|
| `glaucoma_first/train/`、`train.csv` | 主办方训练图与标签，留在本地 |
| `glaucoma_first/valid/`、`valid.csv` | 主办方独立验证图与标签，留在本地 |
| `glaucoma_first/test/`、`test.csv` | 主办方无标签测试集，留在本地 |
| `submission/code/model.py` | todo1数据接口、todo2模型；直接运行只打印结构 |
| `submission/code/train.py` | todo3训练、验证和保存模型；不生成提交答案 |
| `submission/code/predict.py` | todo4验证/预测；直接运行才生成提交答案 |
| `submission/code/utils.py` | 图片、指标、权重兼容、记录与CSV工具 |
| `submission/model/` | 正式最佳模型和配置 |
| `submission/results/submission.csv` | 最终提交的CSV |
| `项目文档/训练记录/时间目录/` | 每次训练的历史、当次最佳和训练前模型备份 |

源码仅在上述四个Python文件；历史记录和说明文档不包含额外可运行源码。原始图片不上传GitHub。

## 如何运行

解释器使用已配置的`jitiaosai`；原有`deepStudy`不受影响。复现依赖见`submission/code/requirements.txt`，赛项参考Python 3.12、PyTorch 2.10.0、torchvision 0.25.0。

```powershell
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission/code/model.py
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission/code/train.py
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission/code/predict.py
```

先运行训练，再根据训练结果决定是否运行预测。预测会写入`submission/results/submission.csv`，字段为`Image_name,Probability,Pred_Label`；`Probability`是阳性类别1的概率。

## 本版训练规则

- 新模型ResNet-18、输入128×128、两类输出，`weights=None`，不使用外部预训练权重。
- 批量32；均衡有放回采样；温和水平翻转、±15°旋转、亮度/对比度变化只用于训练。
- AdamW；随机初始化时前两轮学习率`3e-5→3e-4`，之后余弦下降；同结构续训峰值`1e-5`。最多30轮，至少10轮，候选自身验证AUROC连续8轮不提高则早停。
- **正式模型只在独立验证AUROC严格超过原模型时替换**。训练前旧模型与JSON另备份到本次训练记录目录；候选最佳即使未超过旧冠军也单独保留。
- 本轮更新后如仍未超过旧冠军，预测继续使用旧模型的**64像素**输入；新模型被保存后预测自动使用**128像素**。输入尺寸及预处理标记随权重保存，不凭全局常量猜测。
- 现有92分模型的CNN参数无法迁移到ResNet。代码更新不会自动产生新权重或新成绩。

先完成128像素实验，记录验证AUROC和单张端到端耗时，再另做160像素版本；不同时改变多个变量。目标是争取验证AUROC≥0.98与官网更高分，**没有事先保证**。官网限制模型≤500MB、单张端到端≤100ms；本机抽样测速不代表官方最终测量。

## 说明文档

- [代码与算法说明](项目文档/项目代码与算法说明.md)
- [本版修改及检查](项目文档/修改说明/v1.5.2方案A实施与检查.md)
- [后续规划](项目文档/修改说明/后续调整与规划.md)
- [终端与评价指标](项目文档/评价指标说明/终端输出与评价指标说明.md)
