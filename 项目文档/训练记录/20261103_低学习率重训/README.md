# 2026-11-03 续训与重新训练

正式提交版本计划 v1.24.2。今日没有候选严格超过现有官方独立 valid AUROC 0.9863724439657326，因此正式 submission/model/best_model.pth 与 submission/submission.csv 保持 11 月 2 日的权重及预测。官网 97.65 是用户提供的旧文件分数，不属于今日实验。

1. 同配方 v1 从 runs/submission_new/20261102_133834 第7轮断点续至第11轮，完整第8–11轮官方 valid AUROC：0.983186、0.984842、0.982625、0.982553；连续6轮无提升正常早停。日志：runs/submission_new/launch_20261103_resume.out.log。
2. 按用户要求，从非医学 ImageNet-1K 原始权重新开第1轮。新配方只将初始学习率由1e-4降至3e-5，代码 submission/code/train_restart.py；实际新 run runs/submission_new/20261103_135745，10轮完整结束。第1–10轮官方valid AUROC：0.935235、0.962165、0.972783、0.977784、0.978090、0.979014、0.979959、0.981102、0.979172、0.975056。候选最佳第8轮0.981102，低于正式0.9863724439657326，未晋升。
3. 候选权重 candidate_best_epoch8.pth 是新 run/best.pth 的完整副本，SHA256 b69db52b7a03b7a7bbafb13f61276fec242af440fd669462a9f276fc1265f303；使用 Git LFS 保存。原 run/last.pth 包含第10轮优化器状态，只保存在本机忽略的 runs 中。
4. candidate_submission.csv 是新候选单视图预测，供复现实验；它不是正式推荐提交，不能用测试集或官网分数反向选择模型。正式提交仍为 submission/submission.csv。候选CSV已于2026-11-03约18:01生成，独立核验5855行、官方test顺序、三列表头、概率范围、0.5阈值标签，且5855行概率均与正式CSV不同；SHA256 50d8ab82b200e4b45776682cc1325c333e2592dc4e5c4b7b10ff51e44ed7f6fc。

全程官方 valid 不参与训练，仅按 AUROC 选模。原始数据、TTA、外部医学数据/权重均未改动。
