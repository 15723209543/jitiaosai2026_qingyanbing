# 2026-10-31 两路线多轮训练候选权重

两条路线各完成2轮；选择标准仅为未参与训练的官方 valid AUROC。两条最佳均为新run第1轮，且都低于正式模型 AUROC 0.9859689935392411，所以只保存候选，不替换 `submission/model/best_model.pth`。PTH 经源与副本 SHA256 一致校验，并交由 Git LFS 管理；克隆后运行 `git lfs pull`。

| 路线 | 候选PTH | 大小(字节) | 官方valid AUROC | SHA256 |
|---|---|---:|---:|---|
| submission2 full/marginAUC LR1e-5 | `submission2/candidate_epoch1.pth` | 200821471 | 0.9859329050836605 | `8acd1acda85d5d8d97c49c524018f1fb826c30bd2d2ae9981628f6002158e4d0` |
| submission3 independent80/pairAUC LR5e-6 | `submission3/candidate_epoch1.pth` | 200823263 | 0.9859541880190029 | `69c13717a1b3055fa89fa57a55133d699651c768178e0ae604b96b952e758e69` |

对应训练run及逐轮指标见 `项目文档/训练记录/20261031_多轮训练/README.md`，新 CSV 已生成并核验：submission2 SHA256 `cbaae133560c1055441832b2f7d3b130036197ffba8388a9d2d6fa4f082a8c1b`；submission3 SHA256 `9230c3fb39e3b41d58fbed6e10d2ad5a42cc6fd971ba52db6d4c532d16769d0d`。今日官网成绩待上传。
