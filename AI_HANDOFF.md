# AI交接：已完成首次再提升并停止

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
