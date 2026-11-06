# v1.28.1：2026-11-06 官网成绩 98.05

用户提供的官网截图显示：2026-11-06 18:14:29，提交 ID 34323，成绩 **98.05**。截图仅显示文件名 `submission.csv`，不含上传文件 SHA256，因此不能独立证明平台文件与本地字节完全相同；本地正式 CSV SHA256 为 `46f5202f6ebac88299f25b4d81f751aa3d28211d954b417ce3ddacd96389072d`。v1.27.1 的用户反馈成绩为 97.57（提交 ID 32895），更早 v1.26.1 为 97.97。

正式方案是 Small 第 6 轮成对 AUC 训练 EMA + 原 Tiny，固定各 0.5 概率融合。完整官方 valid 5,849 张 AUROC **0.9883804426480413**；模型权重分别为 `ab36e4f40f5c630db94d93fff35ece25c522f303194906b1b8a72ea0ee533555` 与 `dd4253f1ddc6a103478a43b32b21a06d9ff087567be8b95f4936e3930141a647`，合计 309,878,666 字节。本机同实现单张 47.845 ms；指定比赛硬件仍待验。CSV 已核验 5,855 行、官方 test 顺序、三列及 0.5 标签阈值。官网分数只作结果记录，不用于模型选择。

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

## 2026-11-05 晚至11月6日：Small320 每轮随机子集独立续调实验

官网97.97的v1.26.1正式双模型权重、推理代码与CSV已经按SHA恢复；该成绩为用户反馈，历史验证AUROC 0.9875753924850881。2026-11-05单Tiny本地验证0.9897295956797492但用户反馈官网97.57，因此保持本节实验与正式文件隔离，不使用官网或test调参。

本实验参考用户的《青光眼_真实实验与提交报告_20261101.md》中的Small320、随机擦除、双视图/EMA路径，和[ICCV 2021 Deep AUC Maximization](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html)、[UAI 2018 SWA](https://proceedings.mlr.press/r16/izmailov18a.html)关于高质量初始权重与权重平均/泛化的研究。论文是设计依据，并非声称严格复现。考虑此前Tiny成对AUC本地AUROC上升而官网下降，本轮首先采用官网最高97.97对应Small EMA权重作为暖启动，不改变正式提交。训练态增强和BCE+辅助头+一致性沿用现有Small配方；优化器重置，AdamW LR3e-6到5e-7/固定30轮余弦、WD3e-4、EMA0.996006。每个epoch从官方train 51,396张按seed 20261106+epoch不放回抽24,000张，完整官方valid 5,849张仅以AUROC选模，不进入训练；不使用test、TTA或外部医学数据/权重。microbatch4×累积16、workers1、CPU线程2，目标最多30轮、4个完整轮次不提升则停止并分析。每轮子集指纹、训练loss、官方valid AUROC和best/last原子保存。代码`submission/code/train_small_random_subset.py`，独立run`runs/submission_small_random_subset/`；中断只按同配方`--resume`恢复。候选不自动覆盖正式模型/CSV。正式预测与计时共用现有推理实现；晋升前还要完整官方valid评估与Tiny固定0.5融合、体积≤500MB及同实现单张≤100ms核验。目标0.995及官网98.5均为用户期望，结果待实测。

## 2026-11-05 晚：按用户要求分割官方 valid 的独立实验

用户希望避免反复在完整官方 valid 5,849张上选轮次而导致选择偏差。先保留当前独立run `runs/submission_small_random_subset/20261105_200003` 的完整轮次断点与最佳权重，再只停止我方PID57388。新代码 `submission/code/train_small_select_holdout.py` 是独立配方：从官网97.97的Small权重 SHA256 `33ce0cccaf8bdaa9d132a75c6fd25d19dc2293ae9a5955b03d2b1f2647ff7a93` 重新暖启动，不从97.57 Tiny或零初始化。官方valid按seed20261106分层固定70%为轮次选择子集、30%留作最终一次审计；正式训练时只评估选择子集，审计子集不用于调参或早停。每轮官方train仍以seed+epoch随机不放回抽24000张，其他优化与EMA沿用上一随机子集配方。两run绝不跨配方resume，也不同时运行。此前所有实验已使用过完整官方valid，因此30%只是本轮以后减少重复查看的阶段性审计，不能声称全新盲测；最终报告同时列出选择子集、审计子集、完整官方valid AUROC，不用审计分数再反复挑模型。正式97.97版CSV与权重保持不变直至候选经完整验证与推理约束核对。

## 2026-11-06 Small 轨迹平均与长窗口EMA独立配方（执行前冻结）

当前70/30实验第1–3轮的训练损失和模型轨迹与前一完整valid实验完全相同：只换验证口径，没有改变训练抽样和优化。这一配方不能作为新的提分方法重复消耗GPU。参考用户报告的Small320高分开发路线及其“同轨迹权重平均/长窗口EMA”待测建议、[ICML2022 Model soups](https://proceedings.mlr.press/v162/wortsman22a.html)的参数平均研究，本机尝试独立Small训练配方。论文只是方法依据，不声称论文保证本题提升。

新代码`submission/code/train_small_swa_cycle.py`从当前70%选模第2轮最佳Small权重的不可变副本`runs/submission_small_swa_cycle/source_best.pth`初始化；该权重本身继承官网97.97版Small，不从零或97.57版Tiny起。新run与旧run完全分离，优化器重置。官方train内每轮seed20261107+epoch无放回抽20,000张，训练态双视图/随机擦除沿用Small；EMA衰减由0.996006改为0.999，学习率在1e-6至3e-6之间每2轮余弦循环，AdamW WD3e-4，batch4×累积16/workers1、最多20轮、连续4轮选模AUROC不提升停。官方valid沿用固定seed20261106分层70%选模/30%最终一次审计，选模子集及分割指纹保存；不读test调参。每轮EMA快照单独原子保存，供预先固定同轨迹权重平均候选；平均后仍须先在70%选模子集评估，30%只在最终审计一次，不可反复用审计集调配比。无医学外部数据/权重或TTA。新候选不自动覆盖正式97.97版模型/CSV，体积与同实现单张时延最终验证。

### 2026-11-06 12:00 固定四快照Small参数平均候选

长窗口EMA+周期LR独立run第1轮选模子集AUROC0.983802578019，第2–5轮逐轮下降至0.983559475351，连续4轮无提升自动早停；第5轮last与每轮EMA快照均完整保存。不继续重复该配方。按[ICML2022 Model soups](https://proceedings.mlr.press/v162/wortsman22a.html)及用户报告对同轨迹权重平均的建议，在读取平均后性能前预先固定取该run第1–4轮EMA同名浮点权重各0.25，非浮点buffer取第1轮，形成一个Small单模型候选。仅在固定70%选模子集评估并与第1轮最佳0.983802578019比较；30%留出部分暂不评估。参数平均不增加推理模型数，也不保证提升。若不优，再依据训练loss降低但选模AUROC下降的现象改训练正则化与采样，另开独立run，不覆盖官网97.97正式CSV和权重。

### 2026-11-06 12:00 停滞后的训练态随机深度候选（执行前冻结）

周期LR+长EMA第1轮选择子集AUROC0.983802578019，此后第2–5轮连续4轮下降且训练loss由0.130675降至0.116537，自动早停；第5轮last与第1–5轮快照保留。参考[ECCV2016 Stochastic Depth原论文](https://arxiv.org/abs/1603.09382)的训练态残差路径随机丢弃思路，尝试ConvNeXt Small的训练态drop_path_rate=0.1、推理态禁用。此为本题独立实验，不宣称论文保证效果。新run从上一最高选择子集第1轮Small EMA（继承官网97.97 Small）不可变权重`runs/submission_small_drop_path/source_best.pth`初始化并重置优化器；仅官方train随机子集每轮20000张、seed20261108+epoch；官方valid仍为固定70%选模/30%最终一次审计。LR2e-6余弦至5e-7、WD3e-4、EMA0.996006、batch4×累积16/workers1、最多15轮、4轮无改善早停。drop path无新可学习参数，推理用原Small结构加载同一state_dict。最佳PTH只存在独立run，不覆盖正式97.97版。模型体积、时延及完整官方valid/融合AUROC必须在晋升前实测。

### 2026-11-06 约14:00 随机深度停滞与固定融合检验

训练态drop_path_rate0.1独立run第1–4轮选择子集AUROC依次0.983435097241、0.983063847430、0.982809437660、0.982504145937，全部低于起点0.983802578019，完整第4轮last.pth落盘并自动早停，不续训该配方。为判断Small单模型小幅变化在原官网97.97双模型结构中是否真正有效，在读取融合结果前固定比较：原Small+原Tiny与当前70%最佳Small（长EMA周期LR第1轮）+相同原Tiny，各自概率固定50/50；两条仅在同一固定70%选模子集计算AUROC，30%继续不使用。若新融合不超过原融合，不用该候选覆盖正式97.97 CSV。评估脚本为本地`eval_small_tiny_selection.py`，输出`runs/submission_small_swa_cycle/20261106_102323/fixed_ensemble_selection.json`。用户目标0.99不是允许缩小验证集反复挑选权重的理由。

### 2026-11-06 下午：完整 valid 对照后独立 Small 成对 AUC 实验（启动前记录）

固定 Small+原 Tiny 各 0.5 的完整官方 valid 5849 张评估：官网 97.97 版权重 AUROC 0.9875753924850881；长 EMA 周期 LR 第 1 轮 Small 最佳权重与同一 Tiny 融合 AUROC 0.9877067914772022，确有小幅提高但仍低于目标 0.99。70% 固定选择子集上分别为 0.9849031358359717 和 0.9851518920548771。两者是不同口径，不把子集分数用于解释同一子集内的退化。30% 审计部分暂不单独评估。

参考 ICCV 2021 Large-Scale Robust Deep AUC Maximization 的 AUC 导向训练思想，建立独立 Small 候选 `submission/code/train_small_pair_auc_subset.py`。从上述长 EMA 第 1 轮最佳 Small EMA 权重暖启动，不从零开始、不跨配方 resume。每轮从官方 train 51,396 张按 `20261109+epoch` 无放回抽 12,000 张作为基础池，再在该池内用有放回加权抽样形成 12,000 次训练输入，目标阳性采样比例 15%；重复抽样只发生在官方 train 内。loss 为 BCE 主头 + 0.2 BCE 辅助头 + 0.05 双视图一致性 + 0.05 成对平方 hinge AUC 项（margin 0.5）；该实现是论文启发的简化配方，不声称严格复现论文。正样本重采样后 BCE 的 pos_weight=1；AdamW 起点 LR 2e-6、12 轮余弦至 5e-7、WD3e-4，EMA 沿用 Small 原实现。batch4×累积16/workers1、最多12轮，连续4完整轮次无选择子集 AUROC 提升早停。官方 valid 固定 seed20261106 的70%只用于轮次选择，30%待最终单次审计；历史曾看过完整 valid，不称新盲测。每轮保存原子 last/best，候选不自动覆盖官网 97.97 版提交。最终须在完整官方 valid 同口径对比固定 Small+Tiny 融合、核验模型总量≤500MB和同实现单张≤100ms。test仅最终生成CSV，不按test或官网反馈调参；无外部医学数据/权重或 TTA。用户要求每15分钟低额度监测，18:00后不新增实验，19:00前完成选择和如实交付。0.99/0.995与官网98.5为目标，不保证达到。

### 2026-11-06 约16:45 Small 成对 AUC 停滞后的温和独立配方（启动前记录）

前一独立run `runs/submission_small_pair_auc_subset/20261106_141143` 从长EMA Small权重暖启动，固定70%选择集起点AUROC0.983802578019，第1–6轮连续提高至第6轮0.985577792854，第7–10轮连续4轮下降，完整第10轮last.pth记录patience4正常早停。第6轮最佳权重已冻结`candidate_epoch6.pth`，SHA256 AB36E4F40F5C630DB94D93FFF35ECE25C522F303194906B1B8A72EA0EE533555。其与原Tiny固定概率各0.5在完整官方valid5849张AUROC **0.9883804426480413**，高于官网97.97版0.9875753924850881；当前仅候选，正式文件未替换。第7–10轮选择集AUROC下降但训练loss未一致上涨，继续原配方意义有限。

参考 [ICCV2021 Large-Scale Robust Deep AUC Maximization](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Large-Scale_Robust_Deep_AUC_Maximization_A_New_Surrogate_Loss_and_ICCV_2021_paper.html) 的AUC导向训练，以及当前本机回落现象，独立新run从上述第6轮最佳Small EMA权重暖启动，重置优化器，不跨配方resume。每轮官方train seed20261110+epoch无放回抽10000张基础池，池内有放回加权抽样10000次，目标阳性比例由15%降至10%；成对平方hinge AUC loss系数由0.05降至0.02，margin仍0.5；BCE主头、0.2辅助头、0.05双视图一致性不变且pos_weight=1。LR1e-6至3e-7/6轮余弦，AdamW WD3e-4，batch4×累积16/workers1，连续3轮选择集AUROC不提高即早停。该配方只是论文启发的本题消融，不宣称严格复现。固定seed20261106的70%官方valid仅选模、30%只最终一次审计；历史曾看过完整valid，不能称全新盲测。每次新best原子保存，不自动覆盖正式97.97文件。18:00后不新增实验，19:00前对最终候选做完整valid同口径评估、部署体积和时延、5855行CSV顺序/三列/阈值/哈希核验；按用户最新要求以**今天训练得到的完整官方valid AUROC最高且合规版本**最终生成CSV，即使官网未知。无test或官网调参、外部医学数据权重和TTA。


## 2026-11-06 今日正式提交（本地，未获官网评分）

按用户最新要求，正式 `submission/` 已改为今天训练的 Small 第6轮 EMA + 原 Tiny，推理概率固定各0.5；**尚未上传官网，不能把本地结果当作平台分数**。从官网97.97版 Small 暖启动，逐轮官方 train 随机12,000张基础池并在池内按15%阳性比例加权抽样，加入0.05成对平方hinge AUC项；固定70%官方valid用于轮次选择。第1–6轮选择集AUROC从0.984091升至最佳 **0.9855777928539122**，第7–10轮连续4轮下降后自动早停，完整last已核验。第6轮冻结Small权重SHA256 `ab36e4f40f5c630db94d93fff35ece25c522f303194906b1b8a72ea0ee533555`。随后从该最佳权重做10%阳性抽样、0.02成对AUC项的温和新run，第1–3轮选择集AUROC依次0.985442/0.985289/0.985107，自动早停，未晋升。

第6轮Small与原Tiny在**完整官方valid 5,849张**固定等权AUROC **0.9883804426480413**，比旧官网97.97版同口径 **0.9875753924850881** 高0.0008050501629532。最终保留的30%分区仅作一次阶段性审计：新旧AUROC分别 **0.9931808319385035** / **0.9936457752154236**，候选在该分区反而低0.0004649432769201；该分区在历史实验中曾被看到，不能称真正盲测。这个分歧提示官网泛化收益不确定，须如实告知。

正式双权重总量 **309,878,666字节**，同一`ensemble_tiny_small.py`推理路径在本机100张×3轮平均 **47.845 ms/张**，指定比赛硬件仍需验收。正式`submission/submission.csv`按该权重和完整valid选择记录生成，独立验证**5,855行、官方test顺序、Image_name/Probability/Pred_Label、概率范围和0.5阈值**；CSV SHA256 `46f5202f6ebac88299f25b4d81f751aa3d28211d954b417ce3ddacd96389072d`。Small/Tiny权重哈希分别为`ab36e4f40f5c630db94d93fff35ece25c522f303194906b1b8a72ea0ee533555`与`dd4253f1ddc6a103478a43b32b21a06d9ff087567be8b95f4936e3930141a647`。旧97.97版CSV/Small权重与推理代码备份在`runs/final_20261106/previous_*`；完整评估、计时和审计证据也在该目录。没有Git推送，用户可自行提交CSV；官网分数待实际上传后记录。
