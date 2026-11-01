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
