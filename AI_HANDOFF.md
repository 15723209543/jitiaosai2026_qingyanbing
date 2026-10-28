<!-- RUN_20261028_START -->
2026-10-28 用户重新明确授权“在GitHub拉取最新模型，继续训练”，取代此前停止授权。已停止。基准AUROC 0.984618915，最终 0.985968994，增量 0.001350078。

最终采用远端原ConvNeXt参数+本轮EfficientNetV2第1轮参数。ConvNeXt虽然实际训练3轮，但三轮均未采用，不能说最终模型包含其新参数；EfficientNetV2单支AUROC下降，但与原ConvNeXt的固定等权组合AUROC提高，故按预先规定的组合选模标准采用。完整网络独立重载复测已通过。

本轮独立目录 runs/20261028_github_latest。来源 GitHub 15723209543/jitiaosai2026_qingyanbing，提交 a64b008f67f9eb3030ffb8e251329ca92e3a4022 (v1.15.6)，完整源权重SHA256 2b5a15db0246ee0858e06708105d6c9a36f5b1f028439b2925cb3cabd6ad865e，200822687字节。ConvNeXt-Tiny + EfficientNetV2-S，非医学ImageNet来源保留在PTH中。

采用最新submission2正负平衡批次+gamma1 focal+0.1成对AUC损失、分类头Dropout0.05和最新分支学习率。仅使用官方train51396张训练，valid5849张独立选模；不启用远端额外医学数据或train+valid混合实验。完整384整图等比补边，RGB、ImageNet归一化，AMP训练、FP32验证推理，无TTA。复用384无损像素缓存前校验全部原图SHA、文件名标签与抽样像素完全一致。

每轮平衡采样3316批、53055次图片访问，覆盖全部51396张唯一官方训练图；阳性图循环补位，不能把访问次数当成新增图片数。主干和分类头全部微调，梯度裁剪2.0；ConvNeXt学习率1.5e-5/1.5e-4，EfficientNetV2学习率2e-5/2e-4。固定3轮上限内使用相同低学习率，不沿用远端30轮余弦日程；这是本轮明确记录的运行配方变化。

每支最多3轮，按此前首次改善即停条件运行；只有官方验证AUROC更高才保存新全局最优。优化器从源最佳权重重新初始化，并非原训练精确续批。run_training.py --resume可从本轮已保存epoch边界恢复优化器/RNG。CPU及缓存与原图只读；同一时间只运行此GPU训练。旧正式submission和旧CSV均不覆盖。

实际日志 D:/青眼病/work/github_20261028_training.log；逐轮记录history.json，最终证据TRAINING_COMPLETE.json（存在时才算完成）。源码快照source/只作溯源，其中submission/model已恢复真实LFS对象；其余LFS文件可能仍为指针。重载权重只能weights_only=True并临时允许PyTorch自带TorchVersion类型，不用不安全反序列化。

训练结束前不得用测试分数选模。若改善并完成核验，再导出独立2026-10-28成果。没有自动向GitHub推送或向官网提交。

<!-- RUN_20261028_END -->

# AI交接：已完成首次再提升并停止

用户最后要求：“再训练得到更好的一次就立刻停了，我想睡觉了。”已达成。不要自行启动任何新实验，不要为了此前额度目标继续消耗额度。

正式AUROC 0.963044496140 → 0.964011481681。PTH SHA256 543e9ae5bb904ccbba75666c740a85ac4039b65fd8f84dfc851f34382551ed76；CSV SHA256 fe98cf5d7a490e02a18ce7a908727d17cb1492b617f02c68ac573780951d958e。正式文件submission/model/best_model.pth及submission/submission.csv。

最终第十轮仅训练ResNet18分类头1轮，DenseNet121沿用第六轮（本轮0轮）；主干/BN冻结，gamma1焦点损失，GPU FP32，51,396条原图训练特征，输入256。配方v10，代码continuation_focal/code及已同步submission/code。原特征来自第四轮缓存，复用前逐张量和文件SHA核对。未使用外部预训练或外部医学数据。

权威记录runs/20261025_focal_head/TRAINING_COMPLETE.json、POST_SELECTION_COMPLETE.json、PROMOTED.json、resnet18_history.csv；completed中DenseNet=0是保留旧训练模型，不是随机模型。POST是首次提升锁定及完整网络重载验证，不继续搜索。最终PTH的additional_head_epochs_completed为1/0，epochs_trained_each是历史整网字段，不能混计。

全部5855行CSV格式检查及100张×3轮计时一致性通过。单张本机均值0.051224秒；不一致0。材料ZIP会按file_manifest逐条SHA及CRC校验。没有上传官网或push。验证AUROC不等于官网分数。

停止证据docs/STOPPED_AFTER_IMPROVEMENT.json，实验总账docs/continuous_experiments.json，最后额度docs/quota_snapshot.json。第5、7、8、9轮未提高组合，不晋升；保留源码和检查点，不删除原图/junction目标。详见AGENTS.md。新授权前先查进程，当前无训练应在运行。

Windows推理已使用同步GPU拷贝和pin_memory=False，完整CSV复测通过。预测和cost共用score_batch，FP32、等权、无TTA。features/optimizer只保留本机，不入ZIP。不要重新运行旧轮的finalize脚本覆盖当前成果。


## 最终推理核验与限制

实际交付的predict.py与cost.py共用同一FP32/NCHW推理实现；CPU线程1、禁用TF32、batch1。5张预热、100张×3轮端到端（读取、预处理、推理、CSV写盘）均值0.127828秒/张，与5855行CSV比较不一致0张。本机复测尚未达到100毫秒/张，不能宣称已通过速度要求，官方指定设备仍待复测。训练已停止，AUROC提升已确认。未采用较慢且产生末位差异的channels-last试验；原训练权重及CSV未因此改变。
