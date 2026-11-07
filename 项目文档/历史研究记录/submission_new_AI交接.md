# submission_new 交接（2026-11-01）

基于用户提供的《青光眼_真实实验与提交报告_20261101.md》最高实测 98.55 的 S23-g07 方法，已写候选 `submission_new/`。注意：报告没有附原作者的完整源码、权重、精确划分与随机状态，本方案是公开参数的近似实现，**没有在真实数据上训练、没有 PTH/CSV/官网成绩，也未在比赛硬件测时**。不能将报告 98.55 或内部 selection AUROC 0.984669 宣称为本方案结果。

新候选使用 ImageNet-1K ConvNeXt-Small 320、内部官方 train 取 41,116 张、随机擦除、EMA、辅助 BCE、双视图一致性；严格仅用官方 valid AUROC 选模。`glaucoma_first_my` 不读取，test 不用于训练/选模，不进行 TTA。训练前用户须在 GPU 机安装依赖并下载公开预训练权重；默认 micro-batch2×累积32，以降低 8GB GPU 爆显存风险。首次完整训练之后单独执行 predict，正式提交前补齐权重、核验工作簿、来源许可与比赛设备耗时。

本机仅有无 PyTorch 的本地 Python 环境，已通过静态语法检查，不能声称 GPU 前后向已经实测。正式版 `submission/` 与其他历史模型、CSV 均未有意改写。详见 `submission_new/README.md` 与 `RECIPE.md`。
