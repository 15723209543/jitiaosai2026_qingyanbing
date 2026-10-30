# AI 交接（2026-10-30，V1.20.1）

先读本目录 `使用说明.md`、`训练结果.md`、`submission2_RECIPE.md`、`submission3_RECIPE.md` 和 `AGENTS.md`。历史长交接见 `项目文档/历史研究记录/AI_HANDOFF_截至V1.18.3.md`，其中旧路径及“待完成”文字不代表当前状态。

当前物理训练机路径 `D:/青眼病题`。根目录已按用户要求整理；旧 submission2/3 的 ZIP 在 `项目文档/归档_本地/` 并作为 GitHub Release 资产。2026-10-29 迁移备份经检查包含旧仓库副本、Git bundle、未提交 diff、旧模型和预测；完整 ZIP 仅本地保存，两份相同的迁移方案只抽取一份到 `项目文档/修改说明/迁移与模型提升方案_20261029.md`。旧 submission5、PyCharm 配置、临时测试和本地日志也已本地打包并从根目录清理。

正式模型 `submission/model/best_model.pth` 保持不变，官方独立 valid AUROC 0.9859689935。新版 submission2 候选 AUROC 0.9856580776，原 submission4/5 统一训练改名为 submission3，最佳候选 AUROC 0.9859218009，均没有晋升的 `model/best_model.pth`。两份候选 CSV 都已规范名为各自目录的 `submission.csv`。候选权重原始文件在本机忽略的 `runs/`，另有 Git LFS 副本在 `项目文档/训练记录/20261030_候选权重/`，两份 CSV 已分别上传官网：submission2 97.87（ID 26227），submission3 97.81（ID 26230），由用户提供截图和对应关系；切勿把平台分数描述为独立 valid AUROC 提升。用户明确指定本次提交为 V1.20.1，中间位 20 由本次指示更新；以后不得自行改变中间位，同版本后续提交只递增末位。

不触碰胸片项目。原始数据与 junction 目标只读；官方 test 只预测，官方 valid 独立 AUROC 选模，不使用外部医学数据、医学权重、TTA。代码/配方若再改变，更新对应 RECIPE 并重新训练。推理与计时共用 `score_batch`。GPU 训练前查重复进程，中断后用原 run `--resume`。整理没有新训练，后续可从当前候选配方继续实验，但必须按实际 AUROC 晋升。
