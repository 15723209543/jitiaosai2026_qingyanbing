# 2026-11-04 至 11-05 通宵实验状态

时间均为北京时间。当前正式v1.26.1 Small+Tiny等权融合：完整官方valid AUROC0.9875753924850881，CSV SHA256 `e6013be93b43e37dd6af9654a72f2bd4752f7f23213d7a703d860f02bc833236`；不覆盖直到新候选完整验证、推理速度/模型体积达标。用户新冲刺目标为本地官方valid AUROC0.995、希望官网98.5；目标不代表已取得成绩。

- 18:45 从完整第6轮断点续训Tiny SFA，PID48308，原run`runs/submission_tiny_sfa/20261104_130138`，原第5轮最佳AUROC0.9872478203498175；batch4×累积16、workers1、noise0.05，同配方resume，日志`runs/submission_tiny_sfa/launch_20261104_overnight_resume.{out,err}.log`。后续完整轮次待日志核实。
- 固定0.5 Tiny基线第7轮+Tiny SFA第5轮参数平均，结果`runs/submission_soup/equal_soup_valid.json`。单Tiny soup完整官方valid AUROC0.9868980399341895；与正式Small固定概率各0.5后AUROC **0.9883693385078627**，高于正式0.9875753924850881；候选`runs/submission_soup/equal_soup_candidate.pth`，**尚未生成test CSV或晋升**。论文来源：[ICML 2022 Model soups](https://proceedings.mlr.press/v162/wortsman22a.html)。来源PTH SHA与完整指标在JSON；不使用test选择模型。
- 18:54启动Small全官方train低LR独立新run，PID15552，配方见`submission/RECIPE.md`，日志`runs/submission_small_full_finetune/launch_20261104_fulltrain.{out,err}.log`；起点Small第5轮官方valid AUROC已核对0.986372443966。真实轮次与候选待日志核实。
- 19:00额外SFA Tiny+Small验证因共享GPU仅余696MiB，安全停止仅该可中断验证进程PID42276。两条训练及胸片任务未触碰，显存余量恢复约1455MiB。该验证**没有完成、无有效AUROC**，不可引用；待至少一条训练结束且显存充足后可重新评估。

最终选择只以官方独立valid AUROC严格提高为准；test仅生成最终CSV。11月5日约17:00停止新增训练、17:30开始最终推理与核验，19:00前交付并集中推送v1.27.1。所有未完成内容须据实更新。
SFA第5轮用于soup的精确权重已备份在runs/submission_soup/sfa_source_epoch5.pth，SHA256 b30b61064fe2fa88572dc0ca4df4ecf92e5e64075898320d25fb476596bd9c63。19:00安全停止额外验证后，两条训练仍正常、共享GPU空闲约1.45GiB。


20:18：Tiny SFA续训第7轮官方valid AUROC0.987186、第8轮0.984810，连同第6轮0.987096共连续3个完整轮次未超过第5轮0.9872478203498175。核验第8轮last.pth（patience3）与第5轮best.pth完整后，仅停止我方SFA PID48308。Small全train低LR第1轮官方valid AUROC0.986029140965，低于起点0.986372443966，现第2轮仍在运行。20:19显存空闲约3.6GiB，已在安全余量内重新启动先前中断的SFA第5轮+Small完整官方valid评估PID46824，日志runs/submission_soup/sfa_small_valid_retry.{out,err}.log；未完成前不得引用其结果。胸片进程未触碰。


20:23：SFA Tiny第5轮（已按SHA归档）+正式Small固定概率各0.5，在完整官方valid5849张 AUROC **0.9886265844220017**，高于Tiny权重soup+Small 0.9883693385078627。证据runs/submission_soup/sfa_small_valid.json，尚未生成test CSV，官网未知。Small全官方train低LR第2轮官方valid AUROC0.985692315380，低于第1轮0.986029140965及起点0.986372443966；第3轮仍在进行，若无提升按脚本patience3早停。

20:50：在SFA三轮停滞并安全停止后，按先写RECIPE、后开新run的顺序，启动温和RandAugment Tiny独立实验PID19360，run `runs/submission_tiny_randaugment/20261104_205005`，日志 `runs/submission_tiny_randaugment/launch_20261104.{out,err}.log`，batch4×累积16/workers1，N1/M4；仅训练态增强、官方valid仍完整留出。共享GPU两项青眼病训练+胸片总用约6164/8151MiB，留1647MiB，胸片未触碰。此时新Tiny第1轮刚开始，无AUROC。
21:47：Small全官方train低LR实验第3轮官方valid AUROC0.985721001075，连同第1轮0.986029140965、第2轮0.985692315380均低于起点0.986372443966；脚本按patience3自动早停并退出，没有可晋升新权重。RandAugment Tiny仍在第1轮运行。显存空闲约4.37GiB后，启动已由完整官方valid证据锁定的Small+SFA Tiny第5轮候选CSV预测PID26800，输出`runs/submission_soup/sfa_small_submission.csv`，日志`runs/submission_soup/sfa_small_predict.{out,err}.log`；正式`submission/submission.csv`仍保持原版。仅生成test CSV，不用其结果选模；完成后须独立核验5855行、官方顺序及哈希。

22:17：Small+SFA Tiny第5轮固定等权候选CSV `runs/submission_soup/sfa_small_submission.csv` 完整生成并独立核验：5855行、官方test顺序、三列表头`Image_name,Probability,Pred_Label`、概率有限且[0,1]、0.5标签一致；SHA256 `dd26113aae52953c20a707e483958584696124cbf22c0cbef2409d638d9a439f`，与v1.26.1正式CSV字节不同。其选择依据仅为事先完成的官方valid AUROC0.9886265844220017；官网分数未知。正式submission/submission.csv依然未覆盖。RandAugment Tiny第1、2轮官方valid AUROC0.953778、0.973916，仍在上升，第3轮进行；GPU空闲约5.0GiB。
