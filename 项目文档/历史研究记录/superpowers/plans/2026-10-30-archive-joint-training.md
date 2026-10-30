# 归档与双路线训练实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 归档旧 submission2/3，合并 4/5 的训练入口，并以独立验证 AUROC 选择新的 submission2 候选，尽力在 20:00 前生成两份不同 CSV。

**Architecture:** 旧目录先做本地完整 ZIP 和 GitHub Release 资产，随后从仓库当前树移除。4/5 共用一份训练实现，以 `--mode holdout|independent` 选择抽样；双骨干以共同的融合输出和损失训练。新 submission2 从已训练且未见官方 valid 的现有模型出发，参考 ICCV 2021 DAM 的 AUC 优化思想，仅以官方 valid AUROC 晋升。

**Tech Stack:** Python, PyTorch, timm, Git LFS, GitHub Release, Windows PowerShell。

**Spec:** 用户 2026-10-30 请求及 `AGENTS.md`；早前 4/5 的 80/20 和独立 80/80 定义。

## 约束

- 版本号中间位固定 18，下次提交为 `V1.18.3`。
- 不能修改原始数据或 junction 目标；不能用测试预测选模。
- 4/5 旧运行进度必须保存。训练机已有 4/5 及胸片进程，切换前先识别并只停止本项目进程。
- 独立 80/80 的内部 AUROC 有重叠污染；只用未参与训练的官方 valid 作为最终晋升依据。
- 推理和计时共用实现；配方变更记录在 RECIPE、训练文档和 AI_HANDOFF。

## 任务

- [x] 完整打包旧 submission2/3，记录 ZIP 条目与 SHA256，并创建含两份 ZIP 的 GitHub Release。
- [x] 将 4/5 的共用抽样和联合双骨干训练收敛到一个入口，验证两模式的样本划分和独立验证。
- [ ] `--resume` 已实现但尚未在真实中断后运行验证；需后续验证精确轮次边界恢复。
- [x] 根据 ICCV 2021 DAM 文献建立 submission2 候选，在官方 train 训练、官方 valid 比较；不提升则保留原模型。
- [x] 运行训练与预测，检查两个 CSV 的官方行数、顺序、格式、SHA256 和对应权重。
- [ ] 删除仓库当前树中的旧 submission3 与旧 submission2 代码，保留 Release 归档；同步文档；提交并推送，更新 D 根目录。
