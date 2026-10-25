# AI 交接文档

## 持续训练授权与第五轮训练中

用户要求继续直到额度剩余不足10%，按5小时与周额度中较低剩余值执行；不重置或购买额度。起查剩余49%/92%。第五轮入口continuation_rank/code/train.py，run=runs/20261025_rank_head，日志D:/青眼病/work/rank_training.log。方案docs/第五轮排序训练方案.md，额度策略docs/quota_policy.json。当前正式仍第四轮0.962730，不可提前声称提高。先查进程，再--resume。

## 以下为已完成历史与通用约定


## 第四轮分类头训练已完成（2026-10-25）

本轮AUROC 0.962409 → 0.962730；是否晋升：True。当前正式AUROC 0.962730。
实际新增分类头轮数（非整网训练）：{'resnet18': 30, 'densenet121': 6}。权威证据runs/20261025_frozen_head/TRAINING_COMPLETE.json、POST_SELECTION_COMPLETE.json、final_verification.json及历史CSV；后处理的最终候选哈希以POST_SELECTION_COMPLETE.json为准。当前正式权重SHA256 5ed7b0ff021c65920bd15651df4004be1381047060e78819553ff1e084627f9d，正式CSV SHA256 8a4b82ba3da212a3f5504903262ca0675212994020865ef543d914c594353112。

今天代码continuation_head/code，配方v4；来源为9e3b64198ba150ff6b65929dc38d274ccd2102a037976c6834d2e21632630ddf，新建分类头AdamW与随机状态，冻结卷积与BN，FP32训练，明确不是匹配优化器的精确恢复。只在本轮断点恢复时用--resume。再次训练需新run、不可变基准及正确来源配置；不要直接重跑旧finalize脚本。

本轮分类头轮数以PTH的additional_head_epochs_completed和报告classification_head_epochs为准。epochs_trained_each是源检查点遗留的整网历史，不应把分类头轮数当作额外整网训练。*_best_head_optimizer.pth匹配分支最佳头，最终组合可能采用更早轮次；再次续训前必须逐张量核对分类头参数，不能自动套用最后轮优化器。特征缓存和优化器只在本机run目录，不装入最终材料；特征来源和哈希在feature_cache_manifest.json。

当前submission保留可直接预测的模型与CSV，旧成果在baseline_snapshot与输出第三轮_0.962409。验证选模、无测试调参/TTA，未提交官网。本轮训练与后处理已结束，再开始新训练前仍需先查进程。

## 以下为前三轮历史和通用约定


## 第三轮已完成（2026-10-25）

本轮AUROC 0.962409 → 0.962409；是否晋升：False。当前正式AUROC 0.962409。
实际新增轮数：{'resnet18': 5, 'densenet121': 4}。权威证据runs/20261025_continue_lr1e5/TRAINING_COMPLETE.json、POST_SELECTION_COMPLETE.json、final_verification.json及历史CSV；后处理的最终候选哈希以POST_SELECTION_COMPLETE.json为准。当前正式权重SHA256 9e3b64198ba150ff6b65929dc38d274ccd2102a037976c6834d2e21632630ddf，正式CSV SHA256 e6ed53536e270a261dbb08c2e367a88076bb219a1dc08ccc74babe97af1ff489。

今天代码continuation_today/code，配方v3；来源为9e3b64198ba150ff6b65929dc38d274ccd2102a037976c6834d2e21632630ddf，新建AdamW/scaler/RNG，明确不是匹配优化器的精确恢复。只在本轮断点恢复时用--resume。再次训练需新run、不可变基准及正确来源配置；不要直接重跑旧finalize脚本。

当前submission保留可直接预测的模型与CSV，旧成果在baseline_snapshot与输出上一轮_0.962409。验证选模、无测试调参/TTA，未提交官网。本轮训练与后处理已结束，再开始新训练前仍需先查进程。

## 以下为第二轮历史和通用约定


## 第二轮已完成（2026-10-24）

验证AUROC从0.957992提高到0.962409，采用fixed_mean_ensemble。新增轮数：{'resnet18': 5, 'densenet121': 8}。模型已通过预测和本机计时检查后提升为submission，旧材料有备份。

当前PTH SHA-256：`9e3b64198ba150ff6b65929dc38d274ccd2102a037976c6834d2e21632630ddf`。当前CSV SHA-256：`e6ed53536e270a261dbb08c2e367a88076bb219a1dc08ccc74babe97af1ff489`。

本轮权威记录是runs/20261024_continue_lr3e5/TRAINING_COMPLETE.json及PROMOTED.json，当前docs/训练结果.md为第二轮。下文第一轮数字属于历史记录。continuation和submission使用RECIPE v2，旧基准备份保持v1。

再次推进训练需新建run和不可变基准，按新的起点SHA及匹配优化器来源调整配置；现有--resume只恢复本轮已规划轮数，不会自动扩展8轮上限。先核查进程与完成标志。

本轮最终快照可能来自较早轮次，与*_last.pth的参数不同。下一次若从最终最佳权重开始，先逐张量核对优化器对应关系；没有匹配的最佳轮优化器时，应从最佳权重新建优化器并如实记录，不能把最后一轮的优化器误称为最佳轮的精确恢复。


## 第二轮续训状态

用户已要求“继续训练以获得更大的分数”。本轮运行入口为 `continuation/code/train.py`，新run为 `runs/20261024_continue_lr3e5`，日志 `D:\青眼病\work\continuation_training.log`。先查看进程和完成标志，不重复启动。方案见 `docs/第二轮续训方案.md`。候选权重在continuation，已经完成核验并提升为活动submission。第一轮成果保留在基准备份和输出目录上一轮_0.957992中。

新run从第一轮最佳PTH恢复匹配的optimizer/scaler/RNG，LR 3e-5余弦到1e-6，每分支最多额外8轮，至少4轮、patience=3。除分支最佳外独立保存全局最优推理快照，按验证AUROC在两个单模型与固定等权组合间比较。候选RECIPE更新为continuation_cosine_v2。首次检查点来源SHA与下面第一轮最终PTH一致。

## 第一轮完成状态（历史，2026-10-24）

训练、最终预测和本机核验均已完成。两个分支各训练5轮，每轮51,396张。ResNet-18验证AUROC 0.954715，DenseNet-121为0.952740，最终按验证结果采用等权双模型，AUROC 0.957992。上游同预处理双模型重测基线为0.754550。

最终5855行CSV已核对。FP32本机5张预热、100张×3轮端到端平均0.055303秒/张，与CSV不一致0张。尚未进行主办方指定硬件/样本复测、填写队名或官网上传，不得把本地验证结果称作官网分数。

- 最终PTH SHA-256：`134650d23ce1f2c48bdf2e9abf612faff61b3ccf6cd0b39530cba8c1963b8eaf`。
- 最终CSV SHA-256：`2eac133281e36ea711e5e7c08e7a9c9e36ffa2c6bd8980579283aadb11e04ff7`。
- 原始续训起点已复制到 `submission/pretrain_model/upstream_competition_initialization.pth`，最终推理不加载它。
- 最终文件清单及哈希在 `submission/file_manifest.json`。原始训练阶段代码哈希在各阶段config中，最终推理代码哈希以该清单为准。

## 任务与根目录

用户要求完成 `D:\青眼病` 的计挑赛青光眼项目，检索 GitHub 最新模型与资料、实际完成训练，并留下人类与 AI 文档。

工作仓库：`D:\青眼病\project`。数据真实目录：`D:\青眼病\glaucoma_first\glaucoma_first`。仓库 `glaucoma_first` 是指向该目录的 junction，因此与 submission 在同一根目录。不要复制、删除或改写原始数据和33GB压缩包。

## 先读

1. `docs/使用说明.md`：操作入口与成果。
2. `docs/训练结果.md`：最终实测结果和未完成的外部事项；生成前不能宣称训练完成。
3. `docs/模型与资料调研_20261024.md`：模型筛选和来源。
4. `runs/20261024_upstream_dual/config.json`、`validation_metrics.json`、`TRAINING_COMPLETE.json`：本轮机器可读证据。只有完成标志存在并核对日志后才能认定训练完成。
5. 根目录比赛规则PDF与 `项目文档/修改说明/v1.12.1新参赛规则适配.md`。旧文档中的“此次不训练/不打包”属于历史会话，当前用户已明确要求完成训练。

## 第一轮实现（历史）

- Python：`D:\青眼病\.venv\Scripts\python.exe`，通过 system-site-packages 复用本机 CUDA torch，venv 中补充 torchvision。
- torch 2.11.0+cu128、torchvision 0.26.0+cu128，RTX 5070 Laptop GPU 8GB。
- 两分支 ResNet-18 / DenseNet-121，从GitHub最新main的比赛训练权重继续，分别完整训练5轮。该PTH通过Git LFS取得，SHA-256 def7e5c5be79903cd863421156f0557786d03bc567722ffd083360c18418d1a6。上游备份 `work/upstream/best_model_v1.12.1.pth`。
- 用户补充强调“在人家GitHub项目最新最优模型的肩膀上跑”。因此不从随机参数或ImageNet重新开基线，不随意更换模型结构。曾下载但未用于训练的ImageNet资料移至 `work/research_pretrain`，不混入最终材料。
- train/valid 沿用上游整图256等比例缩小与黑边补齐，均值/标准差均0.5。训练额外水平翻转、亮度与对比度轻扰动。
- 自然类别比例、交叉熵阳性权重 sqrt(阴性数/阳性数)。AdamW，AMP训练，FP32验证/预测。ResNet使用channels_last，DenseNet使用contiguous NCHW，分别沿用适合本机的计算布局，不改变输入数值或网络结构。
- ResNet完成5轮并完整保存后，为DenseNet切换计算布局，通过 --resume 恢复。过程记录见 runs/20261024_upstream_dual/execution_phases.json 和 config_phase1.json。phase2应跳过已完成的ResNet，从上游DenseNet分支开始5轮，未冻结网络层。
- valid按各分支AUROC选择checkpoint，不加入训练。完整训练后由select_inference.py在两个单模型和固定等权双模型中按验证AUROC选优，再锁定测试推理。阈值0.5，无TTA。实际选择见PTH的inference_selection及runs中的inference_selection.json。
- `prepare.py` 仅缓存 train/valid 的确定性预处理，同时对原图解码与SHA-256审计。与valid完全重复的train图会排除，报告位于 `work/cache256/audit.json`。无病人ID，不能证明病人层面的无泄漏。
- `train.py` 先重测每个上游分支的256验证基线，只有AUROC提高才更新最佳权重；`--resume` 从本轮 trusted local last checkpoint 恢复参数、优化器、AMP和随机状态。cuDNN benchmark开启，未承诺跨硬件逐位确定复现。
- 最佳模型单文件：`submission/model/best_model.pth`。中断恢复文件 `runs/.../*_last.pth` 不用于最终推理。
- `predict.py` 与 `cost.py` 共用模型加载与 score_batch，最终预测batch=1，避免计时batch变化产生概率差异。
- 最终CSV只在训练完成和参数固定后产生。历史 `submission/results/submission.csv` 保留，不是本轮输出。

## 第一轮运行方式（历史参考）

在仓库根目录运行：

```powershell
& 'D:\青眼病\.venv\Scripts\python.exe' -u submission/code/train.py --resume
& 'D:\青眼病\.venv\Scripts\python.exe' submission/code/select_inference.py
& 'D:\青眼病\.venv\Scripts\python.exe' -u submission/code/predict.py
```

先核查运行中的进程与 `D:\青眼病\work\` 日志，不重复启动训练。训练未结束不能调用最终预测或虚填完成状态。续训会重查缓存源图哈希。

推理只读PTH，使用eval、inference_mode和FP32，不下载预训练权重。本次不使用外部预训练权重，使用的是该项目此前以赛方训练数据得到的比赛检查点。

如果修改模型结构、RECIPE或归一化，旧权重不可继续按同一配置推理；必须新建run并重新训练，保留原成果。不要使用历史测试预测、平台成绩、测试集统计来调模型、阈值或融合。

## 第一轮交付标准（历史）

最终需要：每分支5轮真实记录、当前PTH、固定模型验证结果、5855行CSV、5预热+100图片×3轮的本机计时与CSV一致性、许可证与哈希、核验清单、人类与AI文档。必须说明本机抽样计时不等于主办方指定硬件复测。官网提交、队名、官方计时样本仍由用户提供/操作。

不要将原始数据、缓存、last优化器文件、虚拟环境、历史CSV或无关资料装进最终材料包。不得擅自向GitHub push或替用户向官网提交。
