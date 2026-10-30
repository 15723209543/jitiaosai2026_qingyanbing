## 2026-10-30 15:20 起：归档、统一训练与今晚交付进行中

- 用户要求旧 `submission2/3` 归档、新 4/5 以一个参数控制并联合训练，以及独立的新版 `submission2` 候选。旧目录已完整压缩到训练机 `D:/青眼病题/历史归档_本地/submission2_20261030.zip` 与 `submission3_20261030.zip`；[Release `archive-V1.18.2-submission2-3`](https://github.com/15723209543/jitiaosai2026_qingyanbing/releases/tag/archive-V1.18.2-submission2-3) 的两个 ZIP 资产均已核验为 `uploaded`。旧 Git 历史尚在。
- `submission4/code/joint_train.py` 为统一实现；`submission4/code/train.py` 与 `submission5/code/train.py` 仅为入口。`--mode holdout|independent` 控制 80/20 或独立 80/80，`--percent` 可改。训练同一批次上两骨干平均概率的共同损失，单个 AdamW 优化两个分类头；8GB 显存约束下冻结两个已训练骨干。初始化使用正式 `submission/model/best_model.pth`。其原训练已覆盖官方 train，因此内部抽样验证不独立，**只有未参与训练的官方 valid AUROC 用于晋升**。
- 新 `submission2` 复制同一双骨干融合训练框架，改用 focal + margin pairwise AUC surrogate。它受 ICCV 2021 DAM 论文启发，**不是原论文 PESG/min-max 的精确实现**。默认使用官方 train 全量与官方 valid 选模。图片建议中的 EfficientNet-B3、DenseNet384 仅作为后续架构候选，本轮为保证可运行和遵守权重来源，沿用已验证的非医学 ImageNet 初始化骨干与现有正式训练参数。
- 旧 4/5 的独立训练进程已仅对本项目停止，保存的 `runs` 进度未删除；胸片项目进程未动。统一 4/5 首轮在训练机使用 `--mode holdout --percent 30 --epochs 1`（时间限制），运行日志 `D:/青眼病题/work/joint4_20261030.out.log`。是否完成、是否提高 AUROC、CSV 是否生成，必须查日志和真实文件，不可按计划推定。
- 统一 4/5 首轮官方 valid：起点 0.9859689935，训练后 0.9859440092，未提高；候选只在 `runs/20261030_152201_379075/`，默认模型未晋升。新版 submission2 的两个 FP16 run 在约第 2753 批出现非有限损失，均以 `INVALID_RUN.txt` 标记作废。BF16、关闭梯度缩放的新配方 `runs/20261030_160656_195989/` 完成全量 3316 批，官方 valid AUROC **0.9856580776**，低于起点 **0.9859689935**，未晋升。
- 后台顺序任务于北京时间 **17:26** 完成两份候选 CSV：`submission2/candidate_marginauc.csv`（SHA256 `5faee981ce5c0e7f68e2f023fd34d4bb46aaef45e13dec0a949ba2f0465b9044`）和 `submission4/candidate_joint.csv`（SHA256 `91654a68e149cde2c3fdb82a58465a74f373dd08e6611f28c915c5e762da6156`）。两份均已独立复核为 5855 行、官方顺序、概率与标签合法，且内容不同。统一 4/5 独立 80/80 run `runs/20261030_164052_879617/` 的两轮官方 valid AUROC 分别为 **0.9859218009** 和 **0.9858542508**，仍低于起点，最佳候选是第 1 轮，**未晋升**。两份 CSV 仅供比较，不得称为已提高的正式模型；对应 PTH 暂在本机忽略的 runs 中。
- 版本号中间位固定 18；下一次 Git 提交为 `V1.18.3`。官方 test 仅做最终预测，不用于选模。

<!-- FRACTIONAL_SUBMISSIONS_20261030_START -->
## 2026-10-30 submission4 / submission5 新抽样实验（待训练）

用户澄清“97.78”指当前 `submission/` 文件夹模型；这是用户报告的官网分数，现有成绩台账没有该次上传的版本或 CSV 哈希对应关系。历史旧 `submission4` 是不同的 RGB64 五模型，现新建的 `submission4/` 和 `submission5/` 均复制**当前 submission 双骨干及 balanced focal+pairAUC 配方**，没有把历史模型或成绩冒充新实验。

- `submission4`：官方 train+valid 合并池按标签分层固定抽 80% 训练，余 20% 验证，图片记录互斥。`submission5`：从相同合并池独立抽两份 80% 分别训练/评估，允许重叠，日志会记录真实重叠数；其 AUROC 是集内指标。两个目录的 `code/train.py` 顶部都可改默认 `80`，命令行 `--percent` 也可覆盖；一次 run 内划分固定，`--resume` 校验指纹并从上一完整轮继续。
- 两个实验默认仅由合规的非医学 ImageNet-1K 原始权重初始化。已有 `submission/` 和 `submission3/` 权重都见过官方数据，不能用于随机 80/20 留出实验的独立验证。官方 test 只供完成模型预测，不能用历史测试预测或平台分数选模；预测和计时共用 `score_batch`、FP32、无 TTA。
- 新训练器沿用当前双骨干、384 预处理、balanced focal+0.1 pairAUC、学习率，并设置每支上限 30 轮/耐心 10；当前 `submission/` 的 2026-10-28 实际训练是旧任务权重续训且最多 3 轮、首次提升即停。新实验为避免留出样本先前泄漏，改用 ImageNet 初始权重，故不是旧训练日志的逐轮重放。
- **尚无新实验训练结果**：本工作树没有 `glaucoma_first/`，当前 `submission/model/best_model.pth` 是 134 字节 Git LFS 指针，且可调用 Python 环境没有 PyTorch；未启动 GPU 训练，`submission4/5` 均无完成 PTH、预测 CSV、AUROC、计时或官网分数。训练机应先确认没有重复训练进程，检查真实数据和 ImageNet 原始权重，再按 `docs/submission4_5_抽样方案.md` 运行。原始数据/junction 目标只读。
- 新实验当前的 ImageNet 初始化需访问 Hugging Face API/下载；它不会自动读取 `submission/pretrain_model/` 的 LFS 文件。`--resume` 的划分指纹覆盖图片名和标签，不覆盖原图字节；跨机器恢复前须核对原图内容一致。详见操作方案。
- 版本号保持官网上传次数位 `18`；除非用户明确更改，后续提交只增加末位。同步状态见 `docs/使用说明.md`、`docs/训练结果.md`、`docs/平台成绩台账.md`。

<!-- FRACTIONAL_SUBMISSIONS_20261030_END -->

<!-- INSAMPLE_20261029_START -->
## 2026-10-29 submission3（train+valid 合并）集内训练

用户要求把 `submission3` 跑一轮并提交看平台分数（"把 valid 并入训练"这条路到底行不行）。已完成并**已提交官网：平台 97.48**。产物：`submission3/model/best_model.pth`（200,826,145 字节，`recipe=..._train_valid_insample_v1`，`training_complete=True`，`selection=ensemble`）与 `submission3/submission.csv`（5,855 行，阳性 154 张，SHA256 `de4869c593d28660b52a370a4e0ca8ae0a496fcfcb73472ac0bb3b6fccff6cd9`，格式与官方顺序核验通过）。

- **三版对照（同测试集、同指标，分数可比）**：v1.15.1 = **97.82**（仅 train，30/38 轮 + valid 早停）；v1.16.4 = 97.22（仅 train，valid 0.98597）；v1.17.1 = **97.48**（train+valid 合并，两支各 1 轮）。详见 `docs/平台成绩台账.md`。
- **判读**：相对 v1.16.4 涨 0.26，但该对比不干净（数据与训练量同时变）；相对 v1.15.1 仍**低 0.34**，且 v1.15.1 用的是一模一样的架构与学习率，差别只在 30/38 轮 + valid 早停。→ **合并 valid 救不回放弃 30 轮的损失**。
- **关键量化**：官方 valid 仅 191 个阳性，其 AUROC 的 1 个标准误约 0.006（95% 区间 ±0.012）。v1.16.4 号称的 +0.00135 只有 0.2 个标准误，**低于噪声底**，当初按严格提升晋升的判断本身不成立。今后 valid 上 <0.005 的变化一律不算提升。
- 集内（**含数据泄漏，不是验证指标**）：ConvNeXt 基线 0.997001 → 0.998374；EffNet 基线 0.996459 → 0.999449；等权组合 0.999855 → 选 ensemble。**不得当作泛化提升，也不得与官方验证 AUROC 横向比较。**
- **BUG-1（任何机器都会踩，修复必须保留）**：`submission3/code/utils.py` 的 `BASE_RECIPE` 原写 `..._fullframe_v1`，而正式权重内是 `..._balanced_pairauc_v4_official_only_20261028`（10-28 晋升时改过），导致暖启动第一秒抛"配方不一致"、**完全无法从正式权重续训**。已改为实际字符串，训练数学未动。
- **本机适配（可整体还原，不影响训练数学）**：`NUM_WORKERS` 4→6、`persistent_workers` True→False（本机提交内存 25.7 GB 上限所限；原设置会让 worker 加载 torch DLL 时 `WinError 1455` 崩掉、主进程静默卡死）。`epochs=1`、`patience=3`、学习率与队长原值一致，未提高。还原清单见 `docs/集内训练_20261029.md` 第 5 节。
- **下一步建议**：要么恢复 30/38 轮 + valid 早停（97.82 的来源），要么做 5 折交叉验证（用满数据又保住干净尺子，约 3 小时，需分几个晚上）。**不要**在集内指标上继续加轮。最终交哪份 CSV（97.82 那份 vs 本轮）需队长按平台"取最高分/取最后一次"的规则定。
- 记录：`docs/集内训练_20261029.md`（实验记录）、`docs/平台成绩台账.md`（历次提交与判读规则）。本机硬件/环境/坑点与逐条代码改动明细在 `LOCAL_NOTES/`（已 gitignore，不随仓库交付）。日志：`runs/train_s3_20261029_run2.log`、`runs/predict_s3_20261029.log`、核验脚本 `runs/verify_csv.py`；历史最好 CSV 备份 `runs/fallback_97.82/submission_97.82.csv`。
- 未完成：每支仅 1 轮（笔记本资源限制，30/38 轮上限未尝试）；`cost.py` 缺 `pictures_100`/`pictures_5`，100 ms 速度限制仍未验证。
- 清理备忘：仓库根目录 4 个 `hs_err_pid*.log` 是 10-28 误提交的 IDE/JVM 崩溃日志，无价值；`runs/wheels/` 留有 2.6 GB torch 轮子（已 ignore）。十一月中任务结束后统一清理，届时不要动原始数据 junction 的目标。

<!-- INSAMPLE_20261029_END -->

<!-- EXTRA_20261028_START -->
## 2026-10-28 本地增强数据训练

本轮增强训练已完成并停止，两分支候选均未刷新最佳。保留原正式模型及CSV，最佳验证AUROC仍为0.985968994；完整网络独立重载复测已通过。

- convnext_tiny.fb_in1k 第1轮，增强后单支AUROC 0.970791485，与保留分支固定均值AUROC 0.979703482，是否刷新最佳：False。
- efficientnetv2_rw_s.ra2_in1k 第1轮，增强后单支AUROC 0.899279897，与保留分支固定均值AUROC 0.959342191，是否刷新最佳：False。

正式模型及CSV未替换，仍保留原最佳AUROC 0.985968994。

用户明确要求利用新增强数据集训练，且不得向GitHub发布数据。数据位于仓库外D:/青眼病/青光眼检测数据集；本轮不复制、暂存、上传任何真实图片、YOLO标签或数据ZIP。Git忽略规则包含数据目录及ZIP。发布内容仅为源码、汇总指标、模型及匹配预测，不含逐图清单。

原始3209张（train2881、valid197、test131）仅作额外训练，标签0=Glaucoma映射阳性1，1=No Glaucoma映射阴性0。通过图片可解码检查、标签/框坐标检查和SHA精确重复检查后，剔除118张重复图，保留3091张（阳1273、阴1818）。与官方valid及test精确相同的额外图片不进入训练。没有患者ID，不能验证患者级或增强变体的跨集合重合。

从正式SHA288f8e0af64623fdb3a20bdc17fe3157b79cefa7c91e5d59d459d5b6d76f4813暖启动，配方增加_extra_v1。每分支本次最多1轮：完整官方train → 本地增强集 → 仅官方valid一次选模；两训练阶段共用同一模型和优化器。初次验证超越起点就停止；未改善则不替换正式模型/CSV。只改训练目标来源，不改变384整图预处理或FP32推理，无TTA。训练AMP FP16、梯度裁剪2.0；损失focal+0.1成对AUC。原始数据只读。

执行入口：`python submission/code/train.py --extra-data "D:/青眼病/青光眼检测数据集" --max-epochs 1`。本轮运行目录runs/continuation_20261028_223149，初次日志D:/青眼病/work/extra_training_20261028.log，关机后恢复日志为同目录extra_training_20261028_resume.log。中断用同一--extra-data和--max-epochs参数加--resume runs/continuation_20261028_223149。优化器及完整运行缓存仅本机保留。

本轮汇总证据和源码在training/20261028_aug/。当前正式文件仍以submission/和上方正式成果记录为准；训练完成不等于新权重已晋升。本机速度旧限制仍存在，不宣称官网成绩或官方速度检查通过。

<!-- EXTRA_20261028_END -->

# 当前成果（2026-10-28）

正式模型、对应CSV及代码现在均在本仓库 `submission/`，不再只存在于被Git忽略的runs目录。

- `submission/model/best_model.pth`：ConvNeXt-Tiny原权重 + 本轮EfficientNetV2第1轮参数，固定等权组合。
- `submission/submission.csv`：该模型的5855行预测，官方名称/顺序通过，实际交付predict/cost复测不一致0张。
- 官方独立验证AUROC：0.984618915 → 0.985968994，增量0.001350078。这是验证指标，非官网成绩。
- 本机端到端耗时0.127828秒/张，**尚未达到100毫秒要求**，不得宣称已通过速度限制；官方指定硬件复测未完成。
- 模型SHA256：`288f8e0af64623fdb3a20bdc17fe3157b79cefa7c91e5d59d459d5b6d76f4813`；CSV SHA256：`b8800414ac2c39173cb51551b73a7f6fe47faff7a6ba63449194dd1e31579765`。

基于[远端v1.15.6](https://github.com/15723209543/jitiaosai2026_qingyanbing/commit/a64b008f67f9eb3030ffb8e251329ca92e3a4022)，源模型SHA256 `2b5a15db0246ee0858e06708105d6c9a36f5b1f028439b2925cb3cabd6ad865e`。仅官方train51396张训练、valid5849张选模；无外部医学数据、valid并入训练或TTA。384整图等比补边，ImageNet归一化。AMP训练、FP32推理。ConvNeXt实际训练3轮均未采用；EfficientNetV2第1轮虽单支AUROC下降，与原ConvNeXt组合提升，按预定组合AUROC标准采用。首次提升后已停止。

训练源码、不可变上游代码、基准权重和记录已纳入版本控制：`training/20261028/`。实际历史脚本为run_training_original.py，配置/逐轮结果/重载核验分别为config.json、history.json、TRAINING_COMPLETE.json和verification.json。优化器与大图缓存继续留在本机runs/work，不推送。

预测：`python submission/code/predict.py`。计时：`python submission/cost.py pictures_100 pictures_5`。两者共用FP32、batch1、单CPU线程、禁用TF32的推理函数。训练：`python submission/code/train.py` 会以当前正式权重创建独立run，最多每支3轮、首次验证提升停止；`--resume runs/<名称>`从已有轮次边界恢复。本次正式成果不会被训练中的临时文件覆盖，完成后需核验再晋升。

`submission2/`、`submission3/`是远端保留的另两条实验，不代表本轮成果；尤其submission3的train+valid集内结果不能称独立验证。本次未启动新训练、未上传官网。旧正式目录完整备份在本机runs/20261028_github_latest/legacy_submission_before_promotion；用户原提交0a099ea保留在合并历史及backup/before-sync-20261028-0a099ea分支。

Git LFS管理PTH与safetensors。首次克隆后运行`git lfs pull`，避免把134字节指针当模型。大型原图、缓存、优化器、ZIP不提交。

## AI约定

上段为当前权威状态。未经新训练授权不要重启训练。不要把本轮材料只放在被忽略的runs/work或Codex输出目录里；可交付源码、模型、CSV和文档应在项目版本控制内。下面是历史状态，不能覆盖当前状态。

# AI交接：已完成首次再提升并停止

2026-10-28 双路线新请求（本段优先）：用户明确要求 submission2 仅改微调参数，并从正式 submission 复制代码新建 submission3，把官方 train+valid 同时用于训练与训练集内 AUROC 选模。submission3 的指标存在数据泄漏，绝不可称独立验证或与 submission2 官方验证 AUROC 横比。两实验只写代码与合成数据测试，不在本机完整训练、不动正式 PTH/CSV；旧正式 PTH 在本机仍是134字节LFS指针。详情见 docs/双路线实验_20261028.md；须只提交源码、测试、说明，原始数据与用户未跟踪PDF不提交。

2026-10-28 submission2 调参（本段优先）：用户报告 GPU 机约30轮未出现最佳权重更新，本机没有该轮日志/真实PTH，不能确认具体原因。当前代码原本已有梯度裁剪2.0，且早停8/10轮；现仅对 submission2 调小两分支学习率、分类头训练期Dropout 0.1、延长早停耐心到12轮，训练配方标识升级v3并兼容v2候选参数暖启动。旧 submission 权重/CSV和数据不动。尚未在真实数据上训练，提升未知；见 docs/训练停滞调参_20261028.md。此段完成后应提交推送代码与说明，不提交原始数据或用户未跟踪PDF。

2026-10-27 额外数据顺序纠错（本段优先）：用户明确指定**每个模型每一轮**的顺序是 `glaucoma_first/train` → `glaucoma_first_my` 三组全作训练 → 仅 `glaucoma_first/valid` 验证一次 → 根据官方验证 AUROC 更新该分支；之前“所有官方轮次结束后才训练额外数据”的段落已过时。两段训练共用当轮模型和优化器；开关为 `MY_DATA_ENABLED=False` 加一行启用的 `MY_DATA_ENABLED=True`，注释启用行即可关闭。只改代码与说明、做合成数据测试，不在本机完整训练，不动已有 PTH/CSV，也不推送额外原始数据。

2026-10-27 额外数据实验后续：用户明确要求追加阶段保持启用、调用上下各空五行，并打印当前训练数据集；完成测试后提交推送，前一段“等待批准”已被此要求取代。两套原始数据目录均通过 `.gitignore` 排除，提交时必须只暂存源码、测试与说明文档，不能提交数据、模型或用户新放入的 PDF。仍未执行真实训练。

2026-10-27 额外数据实验（未提交）：用户在 `glaucoma_first_my/` 放入带 YOLO TXT 标签的 3,209 张眼底图，并表示组委会允许此数据；许可原文未独立核验。`submission2/code/train.py` 增加独立追加阶段：先将原 `glaucoma_first/train` 完成两分支官方训练/验证，再把 `glaucoma_first_my/train`、`valid`、`test` 三部分**全部仅作额外训练**；TXT 第一项 0→青光眼阳性标签1，1→阴性标签0，后续坐标忽略。该阶段默认每分支1轮，注释 `train()` 末尾唯一的 `train_my_stage(...)` 调用即可关闭；始终只用官方 `glaucoma_first/valid` 比较 AUROC、更新最佳分支。没有运行真实训练、没有修改权重/CSV，新增原始数据与用户新放入的 PDF 均保持本地未跟踪，等待用户批准后才提交代码。需要注意项目 `AGENTS.md` 的旧比赛规则表述仍与用户新说明冲突，最终参赛使用前应由用户核对许可原文。

2026-10-27 新实验（本段优先）：远端 v1.15.1 已同步到本地；`submission/` 是用户官网 97.82 分的旧方案，原代码/CSV 均未在本实验改动。本机 `submission/model/best_model.pth` 仅为 134 字节 Git LFS 指针，不能据此在本机训练/预测；GPU 训练机须取得真实权重。新建 `submission2/`，从 `submission/` 四文件代码与 `cost.py` 复制，配方 `imagenet1k_convnext_efficientv2_rgb384_fullframe_balanced_pairauc_v2`：训练批次包含正负且每轮覆盖完整 train，少数类循环补位；focal 加 0.1 倍成对 logistic AUC 排序损失。候选 PTH 优先续训；候选不存在时只读旧 `submission/model/best_model.pth` 暖启动，先在全部官方 valid 复核，旧权重绝不覆盖。新 PTH/CSV 分别只会写到 `submission2/model/` 和 `submission2/submission.csv`；当前没有训练、没有候选 PTH/CSV，也不能宣称 AUROC 或官网分数改善。提交目录的模型、原始预训练文件、核验工作簿和 CSV 需在用户训练及验证后再补齐，不能以旧结果冒充新模型。

2026-10-26 CSV 后缀修复（本段优先）：官方 `glaucoma_first/test.csv` 的 5,855 个 `Image_name` 均以小写 `.jpg` 结尾，而磁盘图片均为大写 `.JPG`。旧 `write_submission` 直接写磁盘文件名，未来运行预测会产生不合官方名单的图片名；现改为只将输出图片名的后缀转小写，同时把输出文件后缀强制为 `.csv`。已只读核对 5,855 个新名称与官方 CSV 顺序逐行一致；25 项单元测试通过、1 项 CUDA 测试跳过。没有训练、预测或覆盖历史 `submission/submission.csv` 和 PTH；当前阶段性 PTH 仍不可用于最终预测。

2026-10-26 轮次重置（本段优先）：`submission/code/train.py` 每次启动均载入已有最佳权重，先在完整官方验证集复核 AUROC，再从本次第 1 轮训练；`CONFIG` 的 `epochs` 是本次运行轮数，当前两分支均为 1。历史最佳 AUROC 仍作更新门槛，只有超过才保存新权重；新保存的 PTH 不再存 `epoch`，但保留 AUROC、配方、来源与选模信息。旧 PTH 含 `epoch` 仍可读，仅不再以其决定起始轮次。此次只改代码和文档，未启动完整训练、未改已有 PTH/CSV。以下“需把轮数设高于历史轮次”等历史说明已被本段取代。

2026-10-26 设备修复：用户明确要求 CPU 也可训练。`submission/code/train.py` 现在先打印 CPU/GPU、PyTorch 与 CUDA 版本；有 CUDA 优先 GPU 并启用 AMP，无 CUDA 时用 CPU 且关闭 CUDA AMP/GradScaler，不再抛出“必须 GPU”。仅用极小合成批次验证 CPU 前后向，没有启动完整数据训练；已有 PTH/CSV 未改。注意工作树中 `CONFIG` 两个 `epochs` 均被用户改为 1，低于当前 PTH 记录的最佳轮次 6/3；这次设备修复保留该未提交修改，若用户希望继续训练，应先自行设定高于已完成轮次的目标值。

2026-10-26 最新目录整理（本段优先于以下历史状态）：用户要求删除旧 `submission/`，把原 `submission2/` 改名为唯一的 `submission/`；数据 `glaucoma_first/` 在同级且未改动。提交目录严格只保留附录 A 所列七项（四 Python 文件加 requirements 在 code 内），没有 README、缓存或占位文件。旧方案 `submission/submission.csv` 作为历史最高提交 CSV 保留于新目录，但它不由新模型生成。旧正式源码/权重可由 Git 历史恢复。本机已从 LFS 取得新方案真实 200,822,431 字节阶段性 PTH，安全读取确认 `training_complete=False`、`selection=None`；ConvNeXt-Tiny 最佳第 6 轮记录 AUROC 0.984011，EfficientNetV2-S 最佳第 3 轮记录 0.965275。不得强行运行预测或将此阶段性 PTH 声称为最终模型；用户在 GPU 主机续训、整体验证选模后再运行 `submission/code/predict.py`。老师截图 `Unsupported operand 118` 的根因是旧 `submission/code/utils.py` 把以 `version ...` 开头的 Git LFS 指针送入 `torch.load`；新代码会先识别指针。真实入口测试另发现阶段性 `selection=None` 曾触发 `AttributeError`，现改为明确提示训练未完成。没有修改真实 PTH、没有训练或生成新 CSV。下文早期路径/状态仅供历史溯源。

2026-10-26 旧权重读取兼容：用户的 GPU 机在 `load_resume_checkpoint` 处遇到 `Unsupported global: torch.torch_version.TorchVersion`。原因是旧 `save_checkpoint` 把 `torch.__version__` 对象写进 PTH，而新版 PyTorch 的 `weights_only=True` 默认不允许该类型。`submission2/code/utils.py` 现在使用局部 `torch.serialization.safe_globals([TorchVersion])` 继续安全加载；`predict.py` 复用相同检查点读取。没有改 `train.py` 的轮数配置、没有使用 `weights_only=False`、没有读取或改写真实 PTH。本机只有 LFS 指针；训练机实际续训仍需由用户验证。

2026-10-26 续训修复：用户在另一台 GPU 电脑运行 `submission2/code/train.py` 遇到已有 PTH 即 `FileExistsError`。已改为读取阶段性单文件检查点，校验配方/分支，逐模型先跑全部官方验证集复核 AUROC，差异超过 0.002 则拒绝覆盖；保存过的模型从最佳轮次下一轮继续，未保存的从第 1 轮开始。旧 PTH 无优化器/调度器状态，只能从最佳参数暖启动，非精确批次恢复；仅新 AUROC 提高才更新 PTH。本地仍只有 133–134 字节 LFS 指针，未下载真实模型、未启动训练。

2026-10-26 后续：已快进到远端 `af41300`，只跳过 Git LFS 实际权重下载；本机 `submission2/model/best_model.pth` 和两个原始预训练 `model.safetensors` 各为 133–134 字节指针。远端有用户上传的阶段性候选模型，不能当作训练完成。候选 `train.py` 当前不能续训且发现现有 PTH 会拒绝重启。此次仅恢复 `submission2` 训练与验证按首批/每25批/末批输出当前批次真实阳性、报阳、报对、误报、漏报；未改训练损失、选模和官方 train/valid 划分，也未下载模型或启动训练。原官方验证集不得混入训练，见 `submission2/README.md`。

2026-10-26 新请求：用户要求独立预训练优化方案并由其本人训练。当前新增 `submission2/` 四文件候选（ConvNeXt-Tiny + EfficientNetV2-S，ImageNet-1K，384 整图等比补边），未训练、未生成候选 PTH/CSV，不得声称 AUROC≥0.99。原 `submission/` 权重和 CSV 未修改；最新用户给的 `submission (4).csv` 与现有正式 CSV SHA-256 相同。当前机器正式 PTH 是 133 字节 LFS 指针，不能在本机推理，需 GPU 主机拉取完整对象。赛规允许有来源、许可、SHA-256、全文和数据说明的非医学权重；候选训练时下载并记录。详见 `docs/预训练候选方案_20261026.md`。不要自动启动训练；用户明确自行训练。原先“停止训练”约定对助手仍有效。

用户最后要求：“再训练得到更好的一次就立刻停了，我想睡觉了。”已达成。不要自行启动任何新实验，不要为了此前额度目标继续消耗额度。

正式AUROC 0.963044496140 → 0.964011481681。PTH SHA256 543e9ae5bb904ccbba75666c740a85ac4039b65fd8f84dfc851f34382551ed76；CSV SHA256 fe98cf5d7a490e02a18ce7a908727d17cb1492b617f02c68ac573780951d958e。正式文件submission/model/best_model.pth及submission/submission.csv。

最终第十轮仅训练ResNet18分类头1轮，DenseNet121沿用第六轮（本轮0轮）；主干/BN冻结，gamma1焦点损失，GPU FP32，51,396条原图训练特征，输入256。配方v10，现用代码在submission/code；原continuation_focal/code已移入本地清理备份ZIP，仍可由Git历史恢复。原特征来自第四轮缓存，复用前逐张量和文件SHA核对。未使用外部预训练或外部医学数据。

权威记录runs/20261025_focal_head/TRAINING_COMPLETE.json、POST_SELECTION_COMPLETE.json、PROMOTED.json、resnet18_history.csv；completed中DenseNet=0是保留旧训练模型，不是随机模型。POST是首次提升锁定及完整网络重载验证，不继续搜索。最终PTH的additional_head_epochs_completed为1/0，epochs_trained_each是历史整网字段，不能混计。

全部5855行CSV格式检查及100张×3轮计时一致性通过。单张本机均值0.051224秒；不一致0。材料ZIP会按file_manifest逐条SHA及CRC校验。没有上传官网或push。验证AUROC不等于官网分数。

停止证据docs/STOPPED_AFTER_IMPROVEMENT.json，实验总账docs/continuous_experiments.json，最后额度docs/quota_snapshot.json。第5、7、8、9轮未提高组合，不晋升；保留源码和检查点，不删除原图/junction目标。详见AGENTS.md。新授权前先查进程，当前无训练应在运行。

Windows推理已使用同步GPU拷贝和pin_memory=False，完整CSV复测通过。预测和cost共用score_batch，FP32、等权、无TTA。features/optimizer只保留本机，不入ZIP。不要重新运行旧轮的finalize脚本覆盖当前成果。
