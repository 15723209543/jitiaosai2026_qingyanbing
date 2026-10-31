# 2026-10-31 晚间追加训练

在V1.20.2已交付的两份CSV基础上，按用户要求继续短轮训练并与胸片任务交替使用8GB GPU。仅以未参与训练的官方valid AUROC选模，正式基线为0.9859689935392411。

| 路线 | run | 配方 | 官方valid AUROC | 决策 |
|---|---|---|---:|---|
| submission3 | `20261031_165415_744669` | independent80/80 pairAUC BF16 LR3e-6，第1轮 | 0.9859514119839582 | 低于既有s3最佳0.9859541880190029，不选 |
| submission2 | `20261031_172549_567435` | full marginAUC BF16 LR5e-6，第1轮 | 0.9859523373289731 | 高于既有s2最佳0.9859329050836605，选用 |
| submission2 | 同一run第2轮（精确resume） | full marginAUC BF16 LR5e-6 | 0.9859329050836605 | 低于该run第1轮，不选 |
| submission3 | `20261031_181138_671046` | independent80/80 pairAUC BF16 LR4e-6，第1轮 | 0.9859477106038986 | 不选 |

新s2候选PTH在 `submission2/candidate_epoch1.pth`，为Git LFS文件；原训练run见表。PTH SHA256 `ede40880f25d6bd00da24dd16537921df2e7134abfe488dd4af08f002070e67`，大小200821471字节。它仍低于正式基线，未晋升正式权重。s3继续使用 `项目文档/训练记录/20261031_候选权重/submission3/candidate_epoch1.pth`。

`submission2_v1.20.2_backup.csv` 是重新预测前原件，SHA256 `cbaae133560c1055441832b2f7d3b130036197ffba8388a9d2d6fa4f082a8c1b`。新预测经独立校验5855行、官方test顺序、表头、概率和0.5阈值标签，最终 `submission2/submission.csv` SHA256 `418e6ca3930e5030811d75cd815e90ed24cbf1dea56be8baffdee2c73dd41485`。`submission3/submission.csv` 保持V1.20.2今日最佳版本，SHA256 `9230c3fb39e3b41d58fbed6e10d2ad5a42cc6fd971ba52db6d4c532d16769d0d`。今日最终平台成绩待上传。
