# 2026-11-03 `submission` 近似复现续训方案

## 证据与目标

用户提供的 2026-11-02 官网截图显示 ID 29668、20:57:24、97.65；用户确认它对应当天 `submission_new`，现已重命名为 `submission`。截图没有 CSV 哈希，故该对应关系记为用户确认。当前本机 7 轮中第 5 轮官方独立 valid AUROC 0.9863724439657326 最好，第 6 轮 0.984559、第 7 轮 0.984456。**因此“轮次不足”是待验证假设，现有验证趋势不支持保证加训会提分。** 报告作者 g07 实际训练 13 轮、选中第 7 轮，官网 98.55；作者原始源码、权重、精确分组未提供，本机实现有结构、划分与选模规则差异，无法保证分数级复现。

## 明天执行

1. 开始前检查 `nvidia-smi` 和全部 Python 训练/预测进程，与胸片任务明确协调 8GB GPU 时间。先从 Git LFS/本地确认 `submission/model/best_model.pth` 和 `runs/submission_new/20261102_133834/last.pth` 完整，备份当前 `submission/submission.csv` 与最佳权重哈希。
2. 保持配方 ID `small320_erasing_ema_aux_consistency_official_valid_v1`、batch8×累积8、workers2、学习率及固定 40 轮调度不变；从原 run 第 7 轮断点续训，最大目标 25 轮：

   ```powershell
   D:/anaconda3.0/envs/deepStudy/python.exe -u submission/code/train.py --epochs 25 --batch-size 8 --accumulation 8 --workers 2 --resume runs/submission_new/20261102_133834
   ```

   代码已修复 `--epochs` 增大导致的错误拒绝续训，并恢复跨进程早停计数：当前最佳第 5 轮，断点第 7 轮，已有 2 轮无提升。每轮约 15–20 分钟，实际以日志为准。先观察到第 11–13 轮；连续 6 轮无官方 valid AUROC 改善则早停。若再变预处理、采样或结构，必须另开配方与新 run，从非医学权重重新训练，不能跨配方恢复优化器状态。
3. 仅当官方独立 valid AUROC 严格高于 **0.9863724439657326** 时晋升最佳权重。官网分数只做最终反馈，不用于调参或回选权重；测试集只生成提交 CSV。若没有更高 AUROC，继续保留今天已验证的权重与 CSV，并如实报告不能保证官网提分。
4. 晋升后用 `submission/code/predict.py --checkpoint submission/model/best_model.pth --output submission/submission.csv --workers 2` 单视图预测。独立核验 5,855 行、官方 test 顺序、三列表头、概率范围/0.5 标签、与旧 CSV 哈希差异；再运行 `submission/cost.py` 同实现测速度。记录模型体积与来源许可证，更新 RECIPE、训练结果、交接、官网提交记录。
5. 用户要求每天最多一次远端推送。11 月 2 日已获用户明确允许破例追加推送，远端目录整理和训练成绩以 **v1.24.1** 完成。用户已明确将版本中间位改为 **24**；11 月 3 日训练和 CSV 完成后集中推送一次，下一提交为 **v1.24.2**，只递增最后一位。

## 归档与目录

原四个目录各自 ZIP 在 `项目文档/归档_本地/20261102_旧submission/`，全量 CRC 校验通过；旧原目录在同处 `原目录/` 本机保存。该归档目录被 Git 忽略，不随普通 push/pull；项目根目录唯一的提交方案目录现为 `submission/`。高分报告原文在 `项目文档/历史研究记录/青光眼_真实实验与提交报告_20261101.md`。

## 2026-11-03 条件重训方案（执行前记录）

用户要求：若同配方断点续训触发早停且正式模型没有更新，则从第 1 轮开启新 run。本实验配方 ID `small320_lr3e5_ema_aux_consistency_official_valid_v2`，脚本 `submission/code/train_restart.py`。与 v1 的唯一训练配方变化是 AdamW 初始学习率及 40 轮余弦调度峰值从 `1e-4` 降至 `3e-5`；最低学习率仍为 `1e-6`。数据划分种子、41,116 张训练样本、独立官方 valid 5,849 张、模型结构、ImageNet-1K 预训练、320 像素、增强、损失、EMA、batch8×累积8、workers2 均保持 v1。理由：v1 第 5 轮后官方 valid AUROC 连续下降，降低更新幅度以检验是否能延缓退化，不保证提升。新实验从非医学 ImageNet-1K 原始权重初始化，优化器/EMA从零开始，绝不跨配方 resume；本轮时间盒目标 10 轮，若超过 17:15 不再启动新轮，18:30 前完成 CSV。

待旧 run 确认早停且胸片并行显存允许，再执行：

```powershell
D:/anaconda3.0/envs/deepStudy/python.exe -u submission/code/train_restart.py --epochs 10 --batch-size 8 --accumulation 8 --workers 2
```

新 run 的最佳候选只写到其 `runs/submission_new/<新run>/best.pth`，脚本不会覆盖正式 `submission/model/best_model.pth` 或 `submission/submission.csv`。仅在官方独立 valid AUROC 严格超过 v1 最佳 `0.9863724439657326` 时，才人工晋升并重新预测；test 和平台分数不参与选模。若无提升，保留已验证的旧 CSV，并如实标注今天没有更高分候选。完整训练日志与实际轮次待执行后填写。

### 2026-11-03 实际执行结果

v1旧run第8–11轮官方valid AUROC为0.983186、0.984842、0.982625、0.982553，连续6轮无提升在第11轮早停。v2低学习率新run runs/submission_new/20261103_135745 从ImageNet-1K原始权重完整训练10轮；第1–10轮AUROC为0.935235、0.962165、0.972783、0.977784、0.978090、0.979014、0.979959、0.981102、0.979172、0.975056。最佳第8轮0.981102低于正式v1第5轮0.9863724439657326，因此未晋升正式best_model.pth，也不覆盖正式submission.csv。候选权重已归档 项目文档/训练记录/20261103_低学习率重训/candidate_best_epoch8.pth，测试预测仅作为实验候选，不能用测试/官网反馈选模。
