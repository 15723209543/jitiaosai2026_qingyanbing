# AI交接：已完成首次再提升并停止

2026-10-26 后续：已快进到远端 `af41300`，只跳过 Git LFS 实际权重下载；本机 `submission2/model/best_model.pth` 和两个原始预训练 `model.safetensors` 各为 133–134 字节指针。远端有用户上传的阶段性候选模型，不能当作训练完成。候选 `train.py` 当前不能续训且发现现有 PTH 会拒绝重启。此次仅恢复 `submission2` 训练与验证按首批/每25批/末批输出当前批次真实阳性、报阳、报对、误报、漏报；未改训练损失、选模和官方 train/valid 划分，也未下载模型或启动训练。原官方验证集不得混入训练，见 `submission2/README.md`。

2026-10-26 新请求：用户要求独立预训练优化方案并由其本人训练。当前新增 `submission2/` 四文件候选（ConvNeXt-Tiny + EfficientNetV2-S，ImageNet-1K，384 整图等比补边），未训练、未生成候选 PTH/CSV，不得声称 AUROC≥0.99。原 `submission/` 权重和 CSV 未修改；最新用户给的 `submission (4).csv` 与现有正式 CSV SHA-256 相同。当前机器正式 PTH 是 133 字节 LFS 指针，不能在本机推理，需 GPU 主机拉取完整对象。赛规允许有来源、许可、SHA-256、全文和数据说明的非医学权重；候选训练时下载并记录。详见 `docs/预训练候选方案_20261026.md`。不要自动启动训练；用户明确自行训练。原先“停止训练”约定对助手仍有效。

用户最后要求：“再训练得到更好的一次就立刻停了，我想睡觉了。”已达成。不要自行启动任何新实验，不要为了此前额度目标继续消耗额度。

正式AUROC 0.963044496140 → 0.964011481681。PTH SHA256 543e9ae5bb904ccbba75666c740a85ac4039b65fd8f84dfc851f34382551ed76；CSV SHA256 fe98cf5d7a490e02a18ce7a908727d17cb1492b617f02c68ac573780951d958e。正式文件submission/model/best_model.pth及submission/submission.csv。

最终第十轮仅训练ResNet18分类头1轮，DenseNet121沿用第六轮（本轮0轮）；主干/BN冻结，gamma1焦点损失，GPU FP32，51,396条原图训练特征，输入256。配方v10，现用代码在submission/code；原continuation_focal/code已移入本地清理备份ZIP，仍可由Git历史恢复。原特征来自第四轮缓存，复用前逐张量和文件SHA核对。未使用外部预训练或外部医学数据。

权威记录runs/20261025_focal_head/TRAINING_COMPLETE.json、POST_SELECTION_COMPLETE.json、PROMOTED.json、resnet18_history.csv；completed中DenseNet=0是保留旧训练模型，不是随机模型。POST是首次提升锁定及完整网络重载验证，不继续搜索。最终PTH的additional_head_epochs_completed为1/0，epochs_trained_each是历史整网字段，不能混计。

全部5855行CSV格式检查及100张×3轮计时一致性通过。单张本机均值0.051224秒；不一致0。材料ZIP会按file_manifest逐条SHA及CRC校验。没有上传官网或push。验证AUROC不等于官网分数。

停止证据docs/STOPPED_AFTER_IMPROVEMENT.json，实验总账docs/continuous_experiments.json，最后额度docs/quota_snapshot.json。第5、7、8、9轮未提高组合，不晋升；保留源码和检查点，不删除原图/junction目标。详见AGENTS.md。新授权前先查进程，当前无训练应在运行。

Windows推理已使用同步GPU拷贝和pin_memory=False，完整CSV复测通过。预测和cost共用score_batch，FP32、等权、无TTA。features/optimizer只保留本机，不入ZIP。不要重新运行旧轮的finalize脚本覆盖当前成果。
