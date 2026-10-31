# AI 交接（2026-10-31，V1.20.2）

先读本目录 `使用说明.md`、`训练结果.md`、`submission2_RECIPE.md`、`submission3_RECIPE.md` 和项目 `AGENTS.md`。历史交接在 `项目文档/历史研究记录/AI_HANDOFF_截至V1.18.3.md`，旧路径及待办不代表现状。物理训练机路径 `D:/青眼病题`。

用户要求北京时间 2026-10-31 20:00 前交付两份今日新 CSV。已生成并独立核验：`submission2/submission.csv` SHA256 `cbaae133560c1055441832b2f7d3b130036197ffba8388a9d2d6fa4f082a8c1b`；`submission3/submission.csv` SHA256 `9230c3fb39e3b41d58fbed6e10d2ad5a42cc6fd971ba52db6d4c532d16769d0d`。各 5855 行、三列表头、官方 test 顺序、概率和标签一致；彼此不同，也与各自昨日版不同。今日平台成绩尚未知，需用户上传后反馈。

今日两条路线从正式权重开启独立低学习率 run，各完成 2 轮。submission2 full/marginAUC LR1e-5，第 1/2 轮官方独立 valid AUROC 为 0.9859329050836605 / 0.985854250757395；submission3 independent80/pairAUC LR5e-6 为 0.9859541880190029 / 0.9859162488733925。两条均取第 1 轮生成 CSV，但均低于正式基线 0.9859689935392411，因此没有晋升，不得把候选复制为 `model/best_model.pth`。两份候选 PTH 副本在 `项目文档/训练记录/20261031_候选权重/`，由 Git LFS 管理；原 run 在各自 `runs/`。昨日 CSV 快照在 `项目文档/训练记录/20261031_多轮训练/`。

10 月 30 日旧 CSV 官网成绩由用户提供：submission2 97.87（ID 26227），submission3 97.81（ID 26230）。平台测试反馈不参与本地 AUROC 选模，不能套用到今日 CSV。正式 `submission/` 权重及 CSV 未改动。

旧 submission2/3 本地 ZIP 在 Git 忽略的 `项目文档/归档_本地/`，并作为 GitHub Release 资产。迁移备份 ZIP 仅本机保存，迁移方案见 `项目文档/修改说明/迁移与模型提升方案_20261029.md`。原始数据和 junction 目标只读；不使用外部医学数据、医学权重或 TTA。修改配方须更新 RECIPE 并重训；GPU 训练前查重复进程，中断后只以相同参数从原 run `--resume`。用户指定本次提交 V1.20.2，中间位 20 不得自行变更。

本次交付含今日两份 CSV、文档和候选 PTH（Git LFS）。后续由用户上传今日两份 CSV 获取新平台分数。
