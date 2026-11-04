## 2026-11-04 v1.25.2 正式融合交接

正式选定 Small 第5轮与 Tiny 对照第7轮概率各0.5，不按 test 或官网成绩选权重。完整官方 valid 5849 张 AUROC **0.9875753924850881**，高于旧正式 Small 0.9863724439657326。Small PTH SHA256 `33ce0cccaf8bdaa9d132a75c6fd25d19dc2293ae9a5955b03d2b1f2647ff7a93`；Tiny PTH SHA256 `dd4253f1ddc6a103478a43b32b21a06d9ff087567be8b95f4936e3930141a647`。两 PTH 合计309,878,282字节；本机同实现84.677ms/张，比赛指定硬件待验。SFA特征噪声独立run第5轮0.987248为最高，第6轮0.987096，低于融合；已在第6轮完整断点写入后停止我方进程，不晋升。正式模型与验证选择记录位于`submission/model/`，推理`submission/code/predict.py`、计时`submission/cost.py`都调用同一融合实现，旧Small单模型脚本更名`predict_small.py`、`cost_small.py`。训练日志、官方valid JSON、候选CSV及旧CSV备份见`项目文档/训练记录/20261104_Tiny320/`。正式CSV已于2026-11-04 16:44:10北京时间重新生成并独立核验，SHA256 `e6013be93b43e37dd6af9654a72f2bd4752f7f23213d7a703d860f02bc833236`。今日平台分数未知，报告作者98.55不属于本机。
今日另生成未晋升的实验CSV：项目文档/训练记录/20261103_低学习率重训/candidate_submission.csv，5855行独立核验通过，SHA256 50d8ab82b200e4b45776682cc1325c333e2592dc4e5c4b7b10ff51e44ed7f6fc；它与正式submission/submission.csv不同，因官方valid AUROC较低而不推荐作为最终提交。
## 2026-11-03 续训及低学习率重训

旧 run 续训第8–11轮官方独立valid AUROC分别为0.983186、0.984842、0.982625、0.982553，第11轮连续6轮无改善而早停。按用户要求，新配方从非医学ImageNet-1K原始权重第1轮重训10轮，唯一配方变化是初始学习率1e-4改为3e-5；最佳第8轮官方valid AUROC 0.981102，低于当前正式0.9863724439657326，故正式PTH和CSV均未晋升。新候选文件及完整逐轮结果见 项目文档/训练记录/20261103_低学习率重训/README.md。用户反馈的官网97.65只属于11月2日正式CSV，不代表今日候选成绩。今日Git版本v1.24.2记录实验，无模型提分声明。
**当前版本：v1.24.2；官网 97.65（用户提供，ID 29668）。**

## 2026-11-02 晚间最新状态

当前唯一提交目录 `submission/` 由今日已训练的 `submission_new/` 重命名。旧四路线各有 ZIP 与本机原目录备份，路径见 `项目文档/归档_本地/20261102_旧submission/`。用户提供官网截图：新路线 97.65（ID 29668，20:57:24；截图未含哈希），对应本地第5轮权重官方 valid AUROC 0.9863724439657326、CSV SHA256 `39412da3b861de2e1781bed85e16de91367ba40793de8deeec7056aeaeee7f30`。第6/7轮验证下降，轮次不足尚未证实。明日从 `runs/submission_new/20261102_133834` 第7轮断点同配方续训；代码已修复增加目标轮数与早停计数恢复。详见 `20261103_submission复现续训方案.md`。今天已推送 V1.23.2，目录整理不再二次推送；根目录以下旧记录仅作历史参考。
## 2026-11-01 最新用户反馈与推送限制

用户报告今日 submission2 官网 97.79，低于 10 月 31 日 97.80。今日 V1.22.3 已于 10:08 推送至远端；从现在起每天最多推送一次，因此不再二次推送。详细明日执行方案在 20261102_训练与单次推送方案.md；明日先试 submission4 全参数 512，并按官方独立 valid AUROC/速度选模，训练完成后集中推送 V1.22.4。当前 s4 今日未训练，成绩记录与方案只在本地工作树，尚未推送。

## 2026-11-01 上午交接（待提交 V1.22.3）

用户要求中午前先交一份 CSV。已与胸片任务约定青眼病先使用共享 GPU；submission2 一轮 LR3e-6 完成，官方 valid AUROC 0.9859551133640179（旧路线候选 0.9859523373289731，正式 baseline 0.9859689935392411）。已用新候选预测、独立核验并替换 submission2/submission.csv，SHA256 244d93d85fd4f4b7d99a98b614558ff410cdb4fc15815e0da5afad04bed2e480。完整权重在 项目文档/训练记录/20261101_上午短轮/submission2_candidate_epoch1.pth；原 run 在 ignored runs。GPU 0 MiB 且训练预测进程已退出，已明确交还胸片任务。submission3/4 今日尚未训练；下午训练待用户决定。新 CSV 官网分数未知，不得沿用昨日 97.80。中间位固定 22，下一提交 V1.22.3。

## 2026-10-31 Submission4 交接（V1.22.2，待训练）

`D:/青眼病题/submission4/` 已新增 512 像素 ConvNeXt-Tiny 全参数微调代码，独立非医学 ImageNet-1K 原始权重初始化；官方 train 训练、官方 valid AUROC 选模、test 仅最终预测。README/RECIPE 及三路线计划在对应目录。已用 deepStudy 环境核对依赖、官方 train/valid/test 数量 51396/5849/5855、原始权重严格加载；**未运行 GPU 训练、未生成候选权重或 CSV，未核验 100ms 速度**。明日先检查 GPU 与胸片任务，按计划串行训练 submission2、3、4，取官方独立 valid AUROC 最高且合规的两条生成 CSV。正式旧模型和两份现有 CSV 未覆盖。版本中间位固定 22，下一提交末位加一。

# AI 交接（2026-10-31，V1.22.1）

先读本目录 `使用说明.md`、`训练结果.md`、`submission2_RECIPE.md`、`submission3_RECIPE.md` 和项目 `AGENTS.md`。历史长交接见 `项目文档/历史研究记录/AI_HANDOFF_截至V1.18.3.md`，其中旧路径与待办不代表现状。物理训练机为 `D:/青眼病题`。

10 月 31 日最终交付两份已核验 CSV：`submission2/submission.csv` SHA256 `418e6ca3930e5030811d75cd815e90ed24cbf1dea56be8baffdee2c73dd41485`；`submission3/submission.csv` SHA256 `9230c3fb39e3b41d58fbed6e10d2ad5a42cc6fd971ba52db6d4c532d16769d0d`。各 5855 行，官方 test 顺序、三列表头、概率范围及0.5阈值标签正确，两份不同。用户提供的官网截图及目录对应关系：submission2（下方）97.80、ID 27513、2026-10-31 20:02:45；submission3（上方）97.79、ID 27515、2026-10-31 20:04:08。截图不含上传文件SHA256，详见 `项目文档/提交记录/20261031_两路线官网成绩.md`。

正式模型官方独立 valid AUROC 0.9859689935392411。今天最高的 submission2 是晚间 LR5e-6 full/marginAUC 第1轮 0.9859523373289731；submission3 是 LR5e-6 independent80/pairAUC 第1轮 0.9859541880190029。该s2 run续训第2轮官方valid AUROC降至0.9859329050836605；其余低学习率追加试验也更差，因此最终两份CSV不再改变。两条均未超过正式模型，故没有晋升，不得把候选覆盖为 `model/best_model.pth`。最终候选 PTH 的Git LFS副本分别在 `项目文档/训练记录/20261031_晚间追加训练/submission2/` 与 `项目文档/训练记录/20261031_候选权重/submission3/`；原 run 在各自忽略的 `runs/`。

昨日官网成绩由用户提供：submission2 97.87（ID26227），submission3 97.81（ID26230）。只属于10月30日旧CSV，平台测试反馈不用于本地AUROC选模，也不能套到今天文件。正式 `submission/` 权重与CSV保持不变。

旧 submission2/3 本地 ZIP 在 Git 忽略的 `项目文档/归档_本地/`，并作为GitHub Release资产。迁移备份ZIP仅本机保存，迁移方案见 `项目文档/修改说明/迁移与模型提升方案_20261029.md`。原始数据及junction目标只读；不得使用外部医学数据、医学权重、TTA或测试调参。修改配方须更新RECIPE并重新训练；启动GPU训练前查重复进程，中断后只以相同参数从原run `--resume`。用户已明确指定本次成绩记录提交版本为 V1.22.1；中间位22此后不可自行更改。

两份今日最终CSV已由用户上传并反馈官网分数；若继续迭代，仍只能以官方独立valid AUROC选模。

## 2026-11-02 submission_new 本机近似复现结果

依据远端 `submission_new` 与用户提供的高分报告，在本机 RTX 5060 Laptop 8GB 使用非医学 ImageNet-1K `convnext_small.fb_in1k` 初始化。官方 train 51,396 张中分层抽取 41,116 张训练；官方 valid 5,849 张只用于 AUROC 选模；test 5,855 张只用于最终预测。实际命令：`D:/anaconda3.0/envs/deepStudy/python.exe -u submission_new/code/train.py --epochs 7 --batch-size 8 --accumulation 8 --workers 2`。运行目录 `runs/submission_new/20261102_133834`，原始日志 `runs/submission_new/launch_20261102_full.out.log`。7 轮已完整结束，官方 valid AUROC 依次为 0.957677、0.977851、0.982447、0.983859、**0.9863724439657326**、0.984559、0.984456；第 5 轮最佳，模型 `submission_new/model/best_model.pth` SHA256 `33ce0cccaf8bdaa9d132a75c6fd25d19dc2293ae9a5955b03d2b1f2647ff7a93`，`training_complete=True`、`metric_scope=official_valid`。该 AUROC 高于既有正式验证基线 0.9859689935392411，但官网分数未知，不能等同报告作者的 98.55。

已用该最佳权重显式预测 `submission_new/submission.csv`，2026-11-02 15:58:32 北京时间生成，SHA256 `39412da3b861de2e1781bed85e16de91367ba40793de8deeec7056aeaeee7f30`。独立核验 5,855 行、官方 test 顺序、三列表头 `Image_name,Probability,Pred_Label`、概率范围与 0.5 标签，错误 0。无 TTA、外部医学数据、医学预训练或测试调参。该实现与报告作者仍有结构、划分、轮次和选模差异，属可运行近似复现。比赛平台分数及指定设备每张 100ms 计时尚未实测。共享 GPU 已于 16:01 确认空闲并交还胸片任务；本日不再重复启动训练。预训练文件 SHA256 `cb43de67007c925cddcddea01655191aeb805dfd9ec76eb1455e8b8de398e8a6`。

### 2026-11-02 19:36 本机速度补测

胸片任务明确释放 GPU 后，在同一台 RTX 5060 Laptop 8GB 上执行 `D:/anaconda3.0/envs/deepStudy/python.exe -u submission_new/cost.py runs/submission_new/timing_20261102/100 runs/submission_new/timing_20261102/5`。计时图片为官方 valid 的只读副本，5 张预热、100 张×3 轮；单张从读图、预处理至模型输出含 CUDA 同步，与预测共用 `score_batch`。实测平均 **0.057311 秒/张（57.311 ms/张）**，低于 100 ms 参考门槛。此结果仅代表本机与该图像集，不冒称比赛指定设备正式验收。19:36 确认无 Python 进程、GPU 显存 26 MiB，并明确归还胸片任务。此前文档“速度尚未实测”已由本段更新。
## 2026-11-03 晚：v1.25.1 Tiny 对照代码，明日待训

用户要求今晚不再训练，新增二级版本并推送代码。已准备 S23 g03 ConvNeXt-Tiny320 原报告近似复现和 ICCV 2021 独立高斯特征增强消融；两者均未产生新权重或 CSV。明日先查 GPU/胸片进程并协调，再从非医学 ImageNet-1K 权重开新 run；官方 valid 仅用于 AUROC 选模。正式 Small 权重/CSV 保持不变，Tiny 只有 AUROC 严格超过0.9863724439657326且体积/延迟合规才晋升。命令及论文边界见 `20261103_S23_Tiny320复现方案.md`。用户反馈今日低 LR 候选官网96.95，ID30606；截图不含哈希，不能用于模型选择。
