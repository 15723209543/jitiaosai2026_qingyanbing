# 2026-10-31 多轮训练与预测交付

两份 `*_20261030_submission.csv` 是生成今日新CSV前的原版快照，已逐字节SHA256核对。正式 `submission/model/best_model.pth` 本轮不替换，因为两个新run均未超过其官方独立valid AUROC 0.9859689935。

| 路线 | 新run | LR | 第1轮官方valid AUROC | 第2轮官方valid AUROC | 选用轮次 |
|---|---|---:|---:|---:|---:|
| submission2 full marginAUC | `submission2/runs/20261031_143405_202421` | 1e-5 | 0.9859329051 | 0.9858542508 | 1 |
| submission3 independent 80/80 pairAUC | `submission3/runs/20261031_140418_260003` | 5e-6 | 0.9859541880 | 0.9859162489 | 1 |

旧版快照 SHA256：submission2 `5faee981ce5c0e7f68e2f023fd34d4bb46aaef45e13dec0a949ba2f0465b9044`；submission3 `91654a68e149cde2c3fdb82a58465a74f373dd08e6611f28c915c5e762da6156`。今日新 CSV 经独立核验：各 5855 行、官方 test 顺序、三列表头、概率与标签一致；两份不同，也与各自旧版不同。SHA256：submission2 `cbaae133560c1055441832b2f7d3b130036197ffba8388a9d2d6fa4f082a8c1b`；submission3 `9230c3fb39e3b41d58fbed6e10d2ad5a42cc6fd971ba52db6d4c532d16769d0d`。今日官网分数待用户上传。
