# 2026-11-01 上午 submission2 短轮

北京时间约 09:42 开始，10:00 前完成训练和官方 valid；随后对官方 test 单次无 TTA 预测，10:05 生成新 CSV。与胸片任务协商先使用共享 8GB GPU，预测结束后 GPU 0 MiB，已明确交还。

- 新 run：`submission2/runs/20261101_094241_178122/`（本机 Git 忽略）。从正式 `submission/model/best_model.pth` 初始化，仅官方 train 51396 张训练；官方 valid 5849 张只评估。模式 `full`，方法 `marginauc`，BF16，学习率 `3e-6`，1 轮。
- 官方 valid AUROC：`0.9859551133640179`；本路线旧候选 `0.9859523373289731`，微小提高 `0.0000027760350448`。正式 baseline 为 `0.9859689935392411`，因此未晋升正式模型，亦不能声称平台成绩会提高。
- 本轮已按路线内 valid AUROC 选择 `candidate_epoch1.pth` 生成 `submission2/submission.csv`。候选权重 LFS 副本：`submission2_candidate_epoch1.pth`，SHA256 `f58e0b70782e1569837778e0b2e60d53c488e6d03fd6c9f25d64f269b4196cbb`。
- 新 CSV SHA256：`244d93d85fd4f4b7d99a98b614558ff410cdb4fc15815e0da5afad04bed2e480`。独立核验 5855 行、官方 test 顺序、`Image_name,Probability,Pred_Label`、概率范围及 0.5 标签。旧 CSV 备份在 `../20261101_首份提交备份/submission2_before.csv`，SHA256 `418e6ca3930e5030811d75cd815e90ed24cbf1dea56be8baffdee2c73dd41485`。
- 官网成绩未知，不能将 10 月 31 日的 97.80 套用到本轮 CSV。下午是否继续训练待用户决定；submission3 和 submission4 本轮尚未训练。
