# 当前正式提交目录

`submission.csv` 已按规程位于本目录根层，列为 `Image_name,Probability,Pred_Label`，5,855 行。用户 2026-10-26 提供的 `submission (4).csv` 与它的 SHA-256 完全相同（`fe98cf5d7a490e02a18ce7a908727d17cb1492b617f02c68ac573780951d958e`），所以没有复制覆盖。它不是模型权重。

规程要求的 `code/`、`model/`、`pretrain_model/`、`licenses/`、`cost.py`、核验表格和 CSV 均在本目录。历史 `docs/`、旧 `results/`、旧轮源权重和过期 `file_manifest.json` 已经归档，不再混入现用提交目录；`code/` 中的辅助脚本仍被当前训练流程引用，不能直接删除。最终打包时只纳入核验所需文件，并重新生成与实物一致的清单。现阶段**未打包**。

当前正式方案验证 AUROC 为 0.964011，FP32，无 TTA。当前机器的 `model/best_model.pth` 是 133 字节 Git LFS 指针，不是真实权重；要运行预测/打包须先在可访问 Git LFS 的机器获取完整对象，并核对正式权重 SHA-256 `543e9ae5bb904ccbba75666c740a85ac4039b65fd8f84dfc851f34382551ed76`。当前 CSV 不依赖本机再预测；不要误用指针打包。

新实验放在相邻的 `../submission2/`。只有用户亲自训练并确认官方验证 AUROC、官网得分和合规材料后，才考虑替换本目录正式文件。
