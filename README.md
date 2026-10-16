# 2026 青光眼眼底图像二分类

当前版本：**v1.4.3**（2026-10-16）。沿用课堂CNN四步；方案B1精简分类头，**验证AUROC是唯一选模标准**。助手只改代码和做轻量检查，正式训练、预测由用户运行。

## 1. 本版决定

| 项目 | 本版设置与理由 |
|---|---|
| 模型层数/神经元 | 五层卷积32→64→128→256→512不变；全连接4608→2048→2048→2改为4608→512→2，减少过大分类头 |
| 参数量 | 3929410个，FP32参数约15.72MB；旧模型15208258个，不承诺速度同比提升 |
| 学习率 | 已训练卷积1e-5，新分类头1e-4；无迁移时全网1e-4；同结构续训仍采用这两组学习率 |
| 图片数量 | 每批64；每轮按51396张的数量均衡有放回抽样，不删减数据；验证完整5849张 |
| 失活/激活 | 卷积Dropout2d(0.1)、分类头Dropout(0.5)、LeakyReLU(0.01)保留 |
| 梯度截断 | 全部梯度L2范数上限5，非有限梯度报错；这是异常保护，不保证提分 |
| 选模/调度 | AUROC严格提高才保存，相等不保存；ACC/AUPRC/AP只展示；自身AUROC连续2轮不提高降学习率、5轮不提高早停，最多12轮 |
| 输入 | 仍为64×64；96×96属于下一步单独实验，本版不修改 |

官网仍以AUROC为主、相等时AUPRC决胜；**本地按用户要求只看AUROC**，不采用平局决胜。旧ACC注释切换段已移除，不要照旧说明解除注释。

## 2. 固定职责与运行

- submission/code/model.py：todo1数据接口、todo2模型；main只打印结构。
- submission/code/train.py：todo3训练、验证、保存权重/训练记录；不写答案。
- submission/code/predict.py：todo4验证/预测；单独运行main才生成提交CSV。
- submission/code/utils.py：必要的数据、迁移/兼容、指标、保存和CSV工具。
- 仅上述四个Python源码文件，不新增训练框架、依赖或解释器。

先用jitiaosai运行训练，完成后需要答案再运行预测：

```powershell
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission/code/train.py
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission/code/predict.py
```

deepStudy解释器保持不变。依赖文件仍以规程Python3.12、PyTorch2.10.0、torchvision0.25.0为复现目标，本次不安装库。

## 3. 数据与产物

- 数据在glaucoma_first/train.csv、train/、valid.csv、valid/、test.csv、test/，不移动、不上传原图。
- 正式冠军：submission/model/best_model.pth，说明为同目录config.json。
- 本次候选最佳：项目文档/训练记录/时间目录/best_model.pth，连同training_history.csv和run_config.json保存。即使未超过正式冠军也会保留自己的最佳参数。
- 训练记录里的pth包含其自身epoch、validation和model_config；run_config.json顶层描述正式冠军，candidate_best单独描述本次候选的文件、轮数、指标和结构，路径均相对项目根目录。
- 提交答案：submission/results/submission.csv；UTF-8三列Image_name、Probability、Pred_Label。Probability为阳性概率，保留6位小数后按0.5判标签。
- 无标签测试集5855张，不能计算真实测试准确率。results只放提交答案。

## 4. 旧权重与续训

工具按pth结构恢复旧三层、窄五层、宽五层，旧参数不会被新分类头误读。宽五层迁移到精简五层时继承五层卷积，分类头重新初始化；旧三层/窄五层只迁移匹配的卷积。同一精简结构续训恢复全部权重，但Adam和调度器重新初始化，不是完整断点恢复。

运行train时先重新验证磁盘现有模型，再训练候选；即使RESUME=False也先保护现有冠军。候选只有严格超过其验证AUROC才替换正式文件。源码升级不会自动把现有模型变成新结构。

**本次不回滚或覆盖已有pth/config/CSV。默认迁移当前保存的ACC模型，而非自动取回昨日权重。** 昨日AUROC模型仍在Git提交97618d6；若要从它开始，先备份当前文件，再恢复对应模型/配置，不能仅改JSON伪装为旧模型。

## 5. 调整位置与提交约定

训练参数在utils.py；模型固定结构、激活和失活在model.py。改图片尺寸必须同步数据处理和权重配置，不能拿旧64尺寸权重直接假称96输入。

先核对GitHub，本次起点13c65a9(v1.4.2)与远端一致。原始数据、.idea、缓存不提交；前两位版本号由用户维护，助手只递增末位。本次：v1.4.3 feat(model):精简分类头恢复AUROC。

## 6. 说明与检查

- [文件、函数、流程与算法](项目文档/项目代码与算法说明.md)
- [后续调整与规划](项目文档/后续调整与规划.md)
- [本版七项取舍、检索及检查](项目文档/v1.4.3修改与检查说明.md)
- [终端字段解释](项目文档/终端输出与评价指标说明.md)
- [选模规则与旧切换方法停用说明](项目文档/ACC与AUROC切换说明.md)

未进行正式训练或整批预测，不保证官网分数提高；旧测速也不能直接代表新模型或评测机的延迟。
