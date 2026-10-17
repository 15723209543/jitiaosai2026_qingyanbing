# 2026 青光眼眼底图像二分类

当前代码版本：**v1.5.2**。方案A使用ResNet-18和128×128眼底图像，从随机参数开始训练。正式训练、生成提交答案由你运行；旧92分模型在训练前仍位于`submission/model/best_model.pth`。

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
