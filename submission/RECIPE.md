# 冻结配方与差异

配方 ID：`small320_erasing_ema_aux_consistency_official_valid_v1`。依据用户提供的 2026-11-01 报告 S23 g07；原作者没有随报告提供完整源码、权重、原始分割索引，所以本实现为**注明差异的近似复现**，不是原训练轨迹恢复。

- 单模型 `convnext_small.fb_in1k`，仅非医学 ImageNet-1K 预训练，320×320 整图等比补边；不裁视盘、不用测试 TTA。
- 官方 train 内分层随机取 41,116 张；随机种子 20261024。官方 valid 全部 5,849 张始终只用于 AUROC 选模；其余官方 train 仅留作内部数据，不用于模型选取。
- AdamW 学习率 1e-4，weight decay 1e-4，40轮余弦调度、最多25轮、最少10轮且连续6轮无 AUROC 提升早停。BF16 优先，否则 FP16；默认 micro-batch 2×累积32，显存允许可改为报告的8×8；梯度范数截断2。
- 训练增强：水平翻转0.5，旋转±15°，ColorJitter(0.22,0.22,0.15)，随机擦除0.1、面积0.01–0.05。验证与测试不增强。
- 主 BCE + 0.2 辅助 BCE + 至第5轮线性升至0.05的一致性 MSE。阳性 BCE 权重取 `min(6, sqrt(阴性/阳性))`；报告只写“截断平方根”，未给截断值，本实现固定6，属明确近似。
- EMA 衰减0.996006，选 EMA 的官方 valid AUROC 严格提升时保存单份权重；AP、召回、误报只报告，不用于选模。预测固定阈值0.5。
- 原报告按内部 selection 的 `sqrt(AUROC×梯形PR)` 选第7轮；此处为遵守项目规则改用独立官方 valid AUROC，实际最佳轮次不预设，不能套用原报告指标或官网98.55。
- 原报告的空间注意力与辅助头确切结构、归一化与随机数状态未知；本目录用可运行的通道注意力与辅助头实现。若取得作者原源码/权重，应重新建立配方版本并重新验证。

## 2026-11-02 本机时间盒复现（执行前记录）

用户要求北京时间20:30前交付，8GB GPU与胸片题共享。先按报告的 micro-batch 8×累积8、workers2、CPU线程4，从非医学ImageNet-1K原始权重新开run；本次最多7轮（报告高分选择第7轮），保留代码内40轮余弦调度，官方valid每完整轮选AUROC。因原报告训练上限25轮且按内部joint选模，本次是受时限约束的近似复现，不得宣称精确复刻98.55。显存探针batch8三步峰值分配约3922MiB，稳定单步约0.163秒，完整数据加载耗时仍待实测。任何参数更改记录后新建run，原run只用同参数resume。


## 2026-11-02 submission_new 本机近似复现结果

依据远端 `submission_new` 与用户提供的高分报告，在本机 RTX 5060 Laptop 8GB 使用非医学 ImageNet-1K `convnext_small.fb_in1k` 初始化。官方 train 51,396 张中分层抽取 41,116 张训练；官方 valid 5,849 张只用于 AUROC 选模；test 5,855 张只用于最终预测。实际命令：`D:/anaconda3.0/envs/deepStudy/python.exe -u submission_new/code/train.py --epochs 7 --batch-size 8 --accumulation 8 --workers 2`。运行目录 `runs/submission_new/20261102_133834`，原始日志 `runs/submission_new/launch_20261102_full.out.log`。7 轮已完整结束，官方 valid AUROC 依次为 0.957677、0.977851、0.982447、0.983859、**0.9863724439657326**、0.984559、0.984456；第 5 轮最佳，模型 `submission_new/model/best_model.pth` SHA256 `33ce0cccaf8bdaa9d132a75c6fd25d19dc2293ae9a5955b03d2b1f2647ff7a93`，`training_complete=True`、`metric_scope=official_valid`。该 AUROC 高于既有正式验证基线 0.9859689935392411，但官网分数未知，不能等同报告作者的 98.55。

已用该最佳权重显式预测 `submission_new/submission.csv`，2026-11-02 15:58:32 北京时间生成，SHA256 `39412da3b861de2e1781bed85e16de91367ba40793de8deeec7056aeaeee7f30`。独立核验 5,855 行、官方 test 顺序、三列表头 `Image_name,Probability,Pred_Label`、概率范围与 0.5 标签，错误 0。无 TTA、外部医学数据、医学预训练或测试调参。该实现与报告作者仍有结构、划分、轮次和选模差异，属可运行近似复现。比赛平台分数及指定设备每张 100ms 计时尚未实测。共享 GPU 已于 16:01 确认空闲并交还胸片任务；本日不再重复启动训练。预训练文件 SHA256 `cb43de67007c925cddcddea01655191aeb805dfd9ec76eb1455e8b8de398e8a6`。

### 2026-11-02 19:36 本机速度补测

胸片任务明确释放 GPU 后，在同一台 RTX 5060 Laptop 8GB 上执行 `D:/anaconda3.0/envs/deepStudy/python.exe -u submission_new/cost.py runs/submission_new/timing_20261102/100 runs/submission_new/timing_20261102/5`。计时图片为官方 valid 的只读副本，5 张预热、100 张×3 轮；单张从读图、预处理至模型输出含 CUDA 同步，与预测共用 `score_batch`。实测平均 **0.057311 秒/张（57.311 ms/张）**，低于 100 ms 参考门槛。此结果仅代表本机与该图像集，不冒称比赛指定设备正式验收。19:36 确认无 Python 进程、GPU 显存 26 MiB，并明确归还胸片任务。此前文档“速度尚未实测”已由本段更新。

## 2026-11-02 晚间目录归档与续训修复

用户反馈本候选官网 **97.65**（ID 29668，20:57:24；[截图记录](../项目文档/提交记录/20261102_submission官网97.65.md)），低于报告作者 98.55。报告 g07 实际训练13轮、joint选第7轮；本机第5轮官方 valid AUROC 最高，第6/7轮下降，不能据此认定轮次不足。高分报告原文已放入 `项目文档/历史研究记录/`。本目录由 `submission_new/` 重命名为 `submission/`；四个旧方案另存本地 ZIP。运行历史断点仍位于 `runs/submission_new/20261102_133834`，不移动，以保留完整优化器/EMA状态。代码仅修复续训目标轮数及早停状态恢复，不改变模型、数据、预处理、学习率或推理配方；此前训练的权重仍有效。明日同配方执行 `python submission/code/train.py --epochs 25 --batch-size 8 --accumulation 8 --workers 2 --resume runs/submission_new/20261102_133834`，只按官方 valid AUROC 严格提升更新最佳。当前断点第7轮，最佳第5轮，连续未提升2轮。旧 CSV 与权重在今晚不覆盖；详细执行门槛见 `项目文档/当前状态/20261103_submission复现续训方案.md`。

2026-11-02 版本补充：用户已指定本轮完成训练及官网 97.65 的版本为 **v1.24.1**；明日续训结果下一提交为 **v1.24.2**，中间位 24 保持不变。

## 2026-11-03 条件重训方案（执行前记录）

用户要求：若同配方断点续训触发早停且正式模型没有更新，则从第 1 轮开启新 run。本实验配方 ID `small320_lr3e5_ema_aux_consistency_official_valid_v2`，脚本 `submission/code/train_restart.py`。与 v1 的唯一训练配方变化是 AdamW 初始学习率及 40 轮余弦调度峰值从 `1e-4` 降至 `3e-5`；最低学习率仍为 `1e-6`。数据划分种子、41,116 张训练样本、独立官方 valid 5,849 张、模型结构、ImageNet-1K 预训练、320 像素、增强、损失、EMA、batch8×累积8、workers2 均保持 v1。理由：v1 第 5 轮后官方 valid AUROC 连续下降，降低更新幅度以检验是否能延缓退化，不保证提升。新实验从非医学 ImageNet-1K 原始权重初始化，优化器/EMA从零开始，绝不跨配方 resume；本轮时间盒目标 10 轮，若超过 17:15 不再启动新轮，18:30 前完成 CSV。

待旧 run 确认早停且胸片并行显存允许，再执行：

```powershell
D:/anaconda3.0/envs/deepStudy/python.exe -u submission/code/train_restart.py --epochs 10 --batch-size 8 --accumulation 8 --workers 2
```

新 run 的最佳候选只写到其 `runs/submission_new/<新run>/best.pth`，脚本不会覆盖正式 `submission/model/best_model.pth` 或 `submission/submission.csv`。仅在官方独立 valid AUROC 严格超过 v1 最佳 `0.9863724439657326` 时，才人工晋升并重新预测；test 和平台分数不参与选模。若无提升，保留已验证的旧 CSV，并如实标注今天没有更高分候选。完整训练日志与实际轮次待执行后填写。

### 2026-11-03 实际执行结果

v1旧run第8–11轮官方valid AUROC为0.983186、0.984842、0.982625、0.982553，连续6轮无提升在第11轮早停。v2低学习率新run runs/submission_new/20261103_135745 从ImageNet-1K原始权重完整训练10轮；第1–10轮AUROC为0.935235、0.962165、0.972783、0.977784、0.978090、0.979014、0.979959、0.981102、0.979172、0.975056。最佳第8轮0.981102低于正式v1第5轮0.9863724439657326，因此未晋升正式best_model.pth，也不覆盖正式submission.csv。候选权重已归档 项目文档/训练记录/20261103_低学习率重训/candidate_best_epoch8.pth，测试预测仅作为实验候选，不能用测试/官网反馈选模。

## S23 g03 Tiny320 随机擦除独立复现（2026-11-03，执行前冻结）

详细证据、已知差异、选模门槛与命令见 项目文档/当前状态/20261103_S23_Tiny320复现方案.md。配方ID s23_g03_tiny320_erasing_official_valid_v3；独立代码 model_tiny.py/train_tiny.py/predict_tiny.py/cost_tiny.py。只替换骨干为非医学ImageNet-1K ConvNeXt-Tiny；其余沿S23 320 erasing设置，官方train内41116张训练、官方valid5849张只用于AUROC选模。报告原精确源码/分割索引缺失，仍为明确差异的可执行复现。正式Small模型与CSV只在独立valid AUROC严格提升且体积/速度过关后更换。模型的真实训练结果待运行后记录。
## S23 Tiny 特征增强对照（v1.25.1 代码，尚未训练）

Tiny 原配方 `s23_g03_tiny320_erasing_official_valid_v3`，`--feature-noise-std 0`；论文启发消融 `s23_g03_tiny320_erasing_sfa_official_valid_v1`，`--feature-noise-std 0.05`。在池化特征进入分类头前，仅训练态加入独立高斯噪声，幅度为该样本特征标准差的5%；验证、预测、计时 `eval()` 无噪声。参考 ICCV 2021 Li 等人的特征增强，但未实现类别条件协方差估计，不能称为完整复现。两配方独立 run，参数和官方 train/valid 指纹写入断点。今晚仅完成代码与方案，明日训练后填写真实 AUROC。详见 `项目文档/当前状态/20261103_S23_Tiny320复现方案.md`。
## Tiny 与正式 Small 的固定等权候选（执行前记录，2026-11-04）

用户提供报告第48–49行记录过 Tiny+Small 双模型等权融合；其作者分数不代表本机效果。本机新增 `submission/code/ensemble_tiny_small.py`，只预设两个模型概率各0.5，绝不根据 test/官网分数搜权重。先用 Tiny 已通过官方 valid 的最佳权重与正式 Small 第5轮权重，对完整官方 valid 5849 张评估并保存 JSON；只有融合 AUROC 严格超过正式 Small `0.9863724439657326`，且总模型文件小于500MB、同实现单张端到端计时不超过100ms，才考虑生成融合 test CSV。脚本 test 模式强制读取官方 valid 选择记录并核对两个权重 SHA256，避免误用。融合不需要额外训练或外部数据；Tiny 训练继续在独立 run 进行。若融合验证不升，则继续看预先冻结的特征噪声实验或其他可解释改动，不把低分候选晋升正式模型。
### 2026-11-04 对照路线提前结束与独立SFA时间盒

Tiny对照run `runs/submission_tiny/20261104_095229`从ImageNet-1K初始化，第1–10轮官方valid AUROC为0.956466、0.978117、0.981997、0.984912、0.986096、0.986198、**0.9864871867475788**、0.986446、0.982382、0.982655。第7轮best.pth的`training_complete=True`、`metric_scope=official_valid`；第10轮last.pth可读且`patience=3`。在连续3轮未超过第7轮且最近两轮显著退化后，13:00前安全停止仅我方对照PID1584，保留完整断点，绝不把第10轮误称为最优。

为赶上18:30交付，第二个已冻结配方 `s23_g03_tiny320_erasing_sfa_official_valid_v1` 初始目标改为**10轮**，而非原计划13轮；优化器、40轮学习率曲线、数据划分、增强、骨干、batch8×累积8、特征噪声0.05均未更改。若10轮后仍改善且时间允许，可在同配方run中`--resume --epochs 13`。新run必须从原始非医学ImageNet-1K初始化，不能从对照断点跨配方resume；16:30后不启动新训练。共享GPU前先与胸片任务核对显存。最终模型只用独立官方valid AUROC选择；Tiny与Small固定等权融合也须独立valid验证且满足体积/延迟门槛。
### 2026-11-04 共享GPU下的SFA执行参数

胸片后台训练PID13132当前占GPU约2055MiB；Tiny对照batch8运行时约5070MiB，两者同时运行余量不足。为在用户要求的剩余显存内完成独立SFA实验，本次将micro-batch从8改为4、梯度累积从8改为16，**有效batch仍为64**；workers从2改为1、CPU线程限制为2。特征噪声0.05、骨干、数据划分、ImageNet-1K初始化、学习率与40轮曲线保持原配方。batch形态已写入checkpoint config，不与batch8 run混用；这是可复现的GPU资源变体，不冒称与报告逐字节一致。新run初始目标10轮，达到截止时间可在完整last断点后提前停，选模仍只用独立官方valid AUROC。若显存不足，只停止我方新run、记录错误并继续使用已通过验证的Tiny第7轮候选；绝不停止胸片进程。

### 2026-11-04 实际选模
Tiny对照第7轮官方valid AUROC0.9864871867475788；SFA噪声0.05 run 第5轮0.987248、第6轮0.987096，仍低于预先固定0.5/0.5的Small+Tiny融合0.9875753924850881。完整官方valid5849行选择记录和权重SHA位于model/ensemble_valid.json；正式预测与计时均调用code/ensemble_tiny_small.py。融合两PTH合计309878282字节，本机同实现84.677ms/张。作者精确源码、切分和权重缺失，仍为近似复现；平台分数未知。

## 2026-11-04 通宵备选：Small 全官方 train 低学习率续调（执行前冻结）

实验脚本`code/train_small_full_finetune.py`，配方ID`small320_full_official_train_low_lr_finetune_v1`，与正式Small模型隔离保存到`runs/submission_small_full_finetune/<时间>/`，不自动覆盖`submission/model/`。起点是正式Small第5轮已训练EMA权重（本目录`model/best_model.pth`，SHA256 `33ce0cccaf8bdaa9d132a75c6fd25d19dc2293ae9a5955b03d2b1f2647ff7a93`），不是新的外部医学权重。官方train的51,396张全部训练；官方valid5,849张仍完全留出，仅AUROC选模；test只在最终选择后预测。与原Small配方不同：训练量从41,116增至51,396、AdamW和EMA从所选权重重置、micro-batch4×累积16（有效64）、学习率固定6轮余弦1e-5到1e-6、双视图一致性系数固定0.05、seed20261105。其他骨干/320像素/随机擦除/辅助BCE0.2/EMA0.996006/weight decay1e-4一致。最长6轮，连续3个完整轮次不提升则早停；只有官方valid AUROC严格超过原Small起点才保存新候选，最终是否替换正式融合仍须与其0.9875753924850881比较，体积/延迟过关。该实验参考用户报告关于“数据量与优化步数不可混淆”的结论，是新的可测假设，不宣称精确重现98.55。变配方绝不从旧`last.pth`跨run resume；同run中断后只以`--resume`恢复完整断点。每轮日志与真实AUROC待运行后填写。

## 2026-11-04 固定等权 Tiny 模型权重平均（执行前冻结与实际结果）

依据 [ICML 2022 Model soups](https://proceedings.mlr.press/v162/wortsman22a.html) 的权重平均思路，本机在看test前固定：Tiny无噪声对照第7轮EMA与Tiny训练态特征噪声0.05第5轮EMA的同名浮点参数各0.5，整数buffer沿用对照；两模型均从非医学ImageNet-1K起点训练、结构和320推理相同，噪声只在训练态。脚本`runs/submission_soup/try_soup.py`，来源SHA和完整官方valid5849张评估在`runs/submission_soup/equal_soup_valid.json`。实际单Tiny soup AUROC0.9868980399341895；与正式Small第5轮概率再各0.5融合的官方valid AUROC **0.9883693385078627**，高于v1.26.1正式等权融合0.9875753924850881。候选权重`runs/submission_soup/equal_soup_candidate.pth`；SFA来源第5轮PTH已按SHA备份`runs/submission_soup/sfa_source_epoch5.pth`，防止续训覆盖。该结果仅为验证候选，尚未生成test CSV或晋升正式模型；若最终采用，正式预测/计时须与其相同结构和权重、门槛核对后重新测量。权重平均不是新的医学预训练，也不使用官方valid训练模型或test调参。

## 2026-11-04 通宵新增：Tiny320 温和 RandAugment 独立实验（执行前冻结）

参考 [NeurIPS 2020 RandAugment](https://proceedings.neurips.cc/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html) 的简化共享强度思路，新增`code/train_tiny_randaugment.py`，配方ID`s23_g03_tiny320_randaugment_n1m4_official_valid_v1`。仅训练态在原水平翻转±15°旋转、颜色扰动、低概率随机擦除之前加入**1个RandAugment操作、强度4/30**；valid/test单视图和预处理不变，无TTA。采用与Tiny对照相同的非医学ImageNet-1K起点、官方train内41116张、独立官方valid5849张、seed20261024、双视图BCE+辅助0.2+一致性0.05/前5轮、EMA0.996006、AdamW1e-4与固定40轮余弦；共享GPU用micro-batch4×累积16、workers1、OMP/MKL线程2（有效batch64）。新run在`runs/submission_tiny_randaugment/`，与SFA或对照断点隔离，目标上限13轮；至少10轮且patience6自动早停，但用户要求连续3完整轮不提高时可在last.pth完整后提前停。预先选择官方valid AUROC为唯一晋升标准，必须超过当前最优融合0.9886265844220017且满足体积/速度才替换。该方法可能损伤眼底色彩/结构，属于消融假设，不能预称提升或报告作者的98.55；不使用任何外部医学数据、权重或test调参。训练结果与源码指纹待完成后据实记录。

## 2026-11-04 通宵备选：Small 同轨迹固定权重平均（执行前冻结）

参考 [ICML 2022 Model soups](https://proceedings.mlr.press/v162/wortsman22a.html)，在看到官方valid之前固定：正式Small第5轮EMA与独立全官方train低LR续调第3轮EMA的同名参数各0.5，整数buffer沿用正式Small；先评估单Small权重平均，再与已固定的SFA Tiny第5轮概率各0.5融合，仅用完整官方valid5849张比较当前最高0.9886265844220017。虽然低LR续调第1–3轮单模型AUROC均低于起点，参数平均可能平滑轨迹，也可能退化；不能预设成功。两源均为非医学ImageNet起点经官方train训练，valid不入训练；test只在最终候选选择后预测。实验输出隔离在被Git忽略的`runs/submission_soup/`，正式权重/CSV不覆盖。若双模型继续受500MB/100ms限制，须同实现测量后才能晋升。

## 2026-11-05 03:19 RandAugment 停滞后独立低学习率实验（执行前冻结）

RandAugment N1/M4 run `runs/submission_tiny_randaugment/20261104_205005` 第7轮官方独立 valid AUROC 0.9889134413766172，随后第8/9/10轮依次 0.988797/0.988726/0.985943；训练损失仍从0.16636降到0.11824，出现泛化退化。已核实第10轮 `last.pth` 完整、早停计数3，只停止我方 PID19360，原第7轮最佳权重保留（SHA256 `6686f5e5ae0042c8e4d7334206307d12a7111ce7c9b7d6d1f0d53d93b8890e30`）。

参考 [ICML 2022 Model soups](https://proceedings.mlr.press/v162/wortsman22a.html) 对同起点微调权重可平均的研究，拟从第7轮 EMA 最佳权重开启**独立新 run**，重置优化器和轮次，以 LR 1e-5、权重衰减 5e-4、8轮余弦降到1e-6，继续官方 train 内 41116 张、RandAugment N1/M4，batch4×累积16，其他损失和验证预处理与旧配方一致。新run结果只在独立官方 valid 5849 张评估；best只在AUROC严格提高时保存，且不自动覆盖正式模型/CSV。若有效，再固定等权试同架构权重平均，须独立valid证明提升、体积和同实现计时合规。该实验针对训练loss继续下降而valid退化的现象，不能保证达到0.995；测试集不用于选择。新配方与旧断点绝不跨配方resume。结果待完成后填写。

## 2026-11-05 06:18 低学习率再停滞与固定等权跨阶段Tiny权重平均

独立低学习率run `runs/submission_tiny_randaugment_low_lr/20261105_032034` 从Rand第7轮EMA起点0.988913441377开始。第1轮官方valid AUROC 0.9889643353524361略升，第2/3/4轮依次0.988268/0.987911/0.986970；训练loss 0.10030→0.05106而valid退化。第4轮 `last.pth` 已核验 epoch4、best_auroc0.9889643353524361、patience3，训练自动早停。原最佳和正式CSV保持不动。

参考 [ICML 2022 Model soups](https://proceedings.mlr.press/v162/wortsman22a.html)，在进一步看验证结果之前固定：同架构RandAugment Tiny原run第7轮EMA与低LR新run第1轮EMA的同名浮点参数各0.5；整数buffer沿用原run。先测单Tiny soup，再与正式Small固定0.5/0.5概率融合，使用完整官方valid5849张选模，不看test。实验独立输出`runs/submission_soup/`，不得覆盖正式权重；只有AUROC、模型体积和同实现单张计时门槛均通过才可晋升。此权重平均不产生新医学预训练。结果待运行。

## 2026-11-05 06:20 SAM 跨配方独立训练（执行前冻结）

原RandAugment Tiny第7轮官方valid0.988913；从其最佳权重独立低LR训练，第1轮0.988964，但第2–4轮降到0.988268/0.987911/0.986970，训练loss持续下降。已核验第4轮完整last及patience3，程序自动早停。参考 [ICLR 2021 Sharpness-Aware Minimization](https://openreview.net/forum?id=6Tm1mposlrM) 和 [作者官方实现](https://github.com/google-research/sam)，在看到本实验验证结果前固定下一实验：从**RandAugment第7轮EMA**作为起点，独立新run、重置AdamW优化器，使用SAM rho0.05（每个有效batch的同一批样本两次前后向）、LR1e-5按6轮余弦降至1e-6、WD1e-4；RandAugment N1/M4、320像素、batch4×累积16、官方train内41116训练及官方valid5849全量AUROC选模保持不变。SAM计算约两倍，显存需与胸片共存；若OOM只停我方进程。连续3个完整轮次无提升早停。实验候选仅保存在新run，不覆盖正式模型或CSV；测试集不参与调参。该文献给出改善泛化的一般方法，不保证本题0.995。验证结果和计时待填写。

## 2026-11-05 09:50 SAM停滞后的成对AUC损失独立实验（执行前冻结）

SAM从Rand第7轮最佳EMA启动，官方valid第1/2/3轮0.988729/0.987696/0.987934，均未超起点0.988913441377；第3轮last已完整，patience3，程序自动早停，正式权重不变。当前最佳**单Tiny**是同架构Rand第7轮与低LR第1轮固定等权soup，官方valid AUROC0.988965260697451，权重`runs/submission_soup/rand_low_lr_equal_soup_candidate.pth` SHA256 `fff98be5212da2ee2b20aa3aced1a12e7ba4b9a36d8f907208da646926ba25ce`。

参考 [ICCV 2021 Deep AUC Maximization](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html) 及其先以交叉熵训练再以AUC目标微调的思路，下一独立run**从上述最高单Tiny权重**启动，重置优化器，非从零开始。继续官方train内41116张，WeightedRandomSampler以约25%阳性比例采样每轮41116次，避免microbatch4大多没有阳性的情况；Loss=原双视图BCE+辅助0.2+一致性0.05 + 0.1×成对平方hinge AUC margin0.5（仅当当前batch同时含阳性、阴性计算）。RandAugment N1/M4、320像素、batch4×累积16、workers1、BF16；AdamW LR5e-6按4轮余弦至1e-6、WD1e-4，最多4轮，连续3轮独立官方valid AUROC不提高早停。与论文的精确DAM优化器不同，这是本机可控的近似消融，不预称论文复现或结果提升。正式模型/CSV不覆盖；原始数据和junction只读，测试集不选模。实验结果待运行。
配方补充（执行前）：WeightedRandomSampler已把阳性抽样比例设为25%，因此BCE的pos_weight固定为1，避免对阳性重复加权；其他原损失项不变。

## 2026-11-05 10:10 报告高分方法组合的单Tiny模型平均（执行前冻结）

用户报告S23开发组显示Small320+erasing g07平台98.55、全量Tiny+Small双模型98.43、全量Tiny+Small+Swin三模型98.35；这些平台反馈的本地分区与当前官方valid5849不可直接比，历史高分权重并未作为当前可用checkpoint核实。当前本机正式Small+Tiny已测同实现84.677ms、总权重约310MB；再增加第三次骨干前向可能突破100ms，故先试**同架构Tiny权重平均**，不增加推理模型数量和结构。

参考 [ICML 2022 Model soups](https://proceedings.mlr.press/v162/wortsman22a.html)，在看新验证结果前固定：当前最高Tiny Rand+低LR等权soup（官方valid0.988965260697451）权重0.75，与训练态SFA噪声0.05第5轮EMA权重0.25做逐参数平均；整型buffer沿用前者。先评估单Tiny，再固定Small与Tiny各0.5概率评估，官方valid全5849张只用于选模。若无提高则丢弃；不使用平台分数或test预测选择。该方案融合报告里高分的Tiny增强路线，仍为单Tiny推理，须最终同实现计时。实验脚本和候选置于被Git忽略的runs/submission_soup，正式权重/CSV暂不覆盖。结果待运行。
2026-11-05 12:18阶段结果：成对AUC新run第1/2/3/4轮官方valid AUROC依次0.9894010982/0.9896361358/0.9897295957/0.989682（AP仅记录）。最佳第3轮权重已单独归档candidate_epoch3.pth，第4轮完整last保存，只有1轮未提升。保持配方不变，允许同run --resume把目标轮数从4延长至7；每轮AUROC严格升高即保存best及不可变候选，连续3轮不提升提前止训，最晚17:00停训准备交付。

## 2026-11-05 13:50 成对AUC停滞后的温和重启（执行前冻结）

上一同配方run `runs/submission_tiny_pair_auc/20261105_095121` 第3轮官方valid AUROC 0.9897295956797492；第4/5/6轮分别约0.989682/0.989656/0.989608，训练loss仍低而验证连续三轮退化，完整第6轮last.pth记录patience=3并早停。依据 [ICCV 2021 Deep AUC Maximization](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html) 对AUC指标易受少数排序错误影响的分析，以及本机连续退化现象，提出更温和的独立验证：从第3轮最佳EMA权重 `runs/submission_tiny_pair_auc/candidate_epoch3.pth`（SHA256 `0feb9619de1c5316bb2d0947bd5ea122fe9957f77d36a7bef174468f2da516c8`）初始化**新run**，优化器重置；阳性重采样目标由25%降至15%，成对平方hinge系数由0.1降至0.05，学习率恒定2e-6、最多3轮。保留Tiny320、RandAugment N1/M4、batch4×累积16、官方train内41116张训练、官方valid5849张仅以AUROC选模，test完全不参与调整。每轮AUROC严格提升才更新候选；若未超0.9897295956797492，保持上一最佳。此为本机温和消融，不声称精确复现论文提出的完整优化器。
## 2026-11-05 16:18 全量官方train温和AUC微调（执行前冻结）

前一个温和AUC run `runs/submission_tiny_pair_auc_mild/20261105_135132` 从当前最高Tiny第3轮EMA起训练，第1/2/3轮官方valid AUROC约0.989632/0.989522/0.989427，连续3轮均低于起点0.9897295956797492，epoch3完整last.pth已核验patience3并正常早停。参考[ICCV 2021 Large-Scale Robust Deep AUC Maximization](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html)对大规模AUC训练的实证思路，现做**独立本机消融**：从已冻结的原成对AUC第3轮最佳EMA `runs/submission_tiny_pair_auc/candidate_epoch3.pth`（SHA256 `0feb9619de1c5316bb2d0947bd5ea122fe9957f77d36a7bef174468f2da516c8`）出发，不从零；将官方train的51,396张全部用于训练而非之前内部41,116张；官方valid5,849张完全留出。恢复原成对AUC配方的25%阳性采样及pairwise平方hinge系数0.1，优化器重新初始化、固定1e-6学习率、至多2轮、batch4×累积16/workers1。RandAugment N1/M4、Tiny320模型、EMA等保持不变。新配方独立run，不跨旧run恢复；每完整轮只以官方valid AUROC严格高于0.9897295956797492才保存新候选。此实验不是论文DAM优化器的精确复现；test仅最终预测，不能选模。19:00训练硬截止，之后选择已完成的最高且合规候选。
## 2026-11-05 18:08 固定等权Tiny快照平均（执行前冻结）

全量官方train独立实验第1轮完整官方valid AUROC约0.989619，第2轮约0.989592，均未超过原成对AUC第3轮最佳0.9897295956797492。依据[ICML 2022 Model soups](https://proceedings.mlr.press/v162/wortsman22a.html)和[SWA](https://proceedings.mlr.press/r16/izmailov18a.html)的已训练权重平均思路，在评估前固定原成对AUC第3轮最佳EMA权重与全量官方train第2轮EMA权重各0.5进行同架构逐参数平均，整数buffer取原第3轮。官方valid5849全量仅用于AUROC选模，绝不读取test预测选择；若不严格超过0.9897295956797492则保留原第3轮。平均后的单Tiny仍须核对体积≤500MB与同实现单张≤100ms。本实验不增加推理模型数，不声称论文精确复现。## 2026-11-05 v1.26.2 选模与交付

正式路线现为**单Tiny320**，从已训练权重继续的成对AUC实验第3轮在完整官方valid 5,849张的AUROC为 **0.9897295956797492**，高于上一正式Small+Tiny融合的0.9875753924850881。最佳权重SHA256为`0feb9619de1c5316bb2d0947bd5ea122fe9957f77d36a7bef174468f2da516c8`，已复制至`submission/model/tiny_best_model.pth`；训练证据为`runs/submission_tiny_pair_auc/20261105_095121/best.pth`和不可变`runs/submission_tiny_pair_auc/candidate_epoch3.pth`。第4/5/6轮未提升后早停。温和成对AUC新run第1/2/3轮约0.989632/0.989522/0.989427；全量官方train新run第1/2轮约0.989619/0.989592；固定等权Tiny快照平均0.9896916565341388，均未超过第3轮，故不晋升。仅AUROC选模，AP和test/官网成绩不参与。正式`submission/code/predict.py`与`submission/cost.py`均调用单Tiny的同一推理实现；单权重111,644,807字节，RTX 5060 Laptop本机100张×3轮平均43.440ms/张，指定比赛硬件仍待验。CSV最终生成、独立核验和远端状态见`项目文档/训练记录/20261105_最终交付/README.md`；官网分数未知，不承诺达到98.5。历史Small模型和融合记录保留供回退。

## 2026-11-05 元数据勘误

成对AUC原run保存的候选PTH `config.initial_lr=1e-5`、`config.weight_decay=5e-4` 是启动脚本中的**展示字段错误**；真实训练执行的学习率由同脚本`learning_rate()`定义为5e-6预热后4轮余弦至1e-6，真实`AdamW`权重衰减为1e-4，优化器状态亦按真实值保存。该勘误不修改已冻结的最佳PTH及其SHA；代码现把新run元数据改为实际值，并在同配方resume时兼容上述旧元数据记录。历史配方与权重选择仍以真实日志和代码为准。