# submission_new：高分报告 S23-g07 的可运行近似复现

基于用户提供的《青光眼_真实实验与提交报告_20261101.md》最高官网提交 **98.55**：单模型 ImageNet-1K ConvNeXt-Small、320 输入、训练随机擦除、EMA、辅助 BCE 和双视图一致性。该分数是报告作者的结果，**不是本目录的实测成绩**；原作者源码、原始权重、精确数据划分与随机种子没有提供，不能承诺字节级或分数级复现。

## 与当前方案对比

| 项目 | 当前正式 submission | 本实验 / 报告最高分 |
| --- | --- | --- |
| 模型 | ConvNeXt-Tiny + EfficientNetV2-S，双分支 | 单 ConvNeXt-Small |
| 图像 | 384 | 320，整图等比缩放补边 |
| 训练 | 两分支分别微调、取均值 | Small 全模型微调、EMA、随机擦除、辅助头、双视图一致性 |
| 训练样本 | 官方 train 51,396 | 官方 train 内分层取 41,116；其余 10,280 不用于本配方训练 |
| 选模 | 官方 valid AUROC | 本实现仍按官方 valid AUROC；报告作者按内部 selection 的 joint 指标 |
| 预测 | 双分支概率融合 | 单模型、单视图、0.5阈值 |

报告原作者内部选模分区与官方 valid 不同，本实现遵守项目 `AGENTS.md`：**官方 valid 只用于 AUROC 选模，不参加训练**。因此验证数字不能直接和报告中的 0.984669 比较。报告 98.55 不可通过拷贝方案自动获得；需在 GPU 机训练并官网实测。

## 训练与预测

1. 官方 `glaucoma_first/train.csv`、`valid.csv`、`test.csv` 及同名图片目录放在 `submission_new/` 的同级；亦可用环境变量 `GLAUCOMA_DATA_ROOT` 指向官方数据。绝不读取 `glaucoma_first_my`。
2. 在 CUDA GPU 环境安装 `code/requirements.txt`。首次运行 `python submission_new/code/train.py` 自动从公开的 [timm/convnext_small.fb_in1k](https://huggingface.co/timm/convnext_small.fb_in1k) 下载非医学 ImageNet-1K 权重并保存到 `pretrain_model/convnext_small.fb_in1k/model.safetensors`。无法联网时先手工取得该模型的官方 `model.safetensors` 放到相同位置。预训练来源与 Apache-2.0 许可证须在正式提交材料中写明、附全文和 SHA256。
3. 运行 `python submission_new/code/train.py`。默认 micro-batch 2×累积32（适配约8GB GPU 的保守起点）；显存充足可 `--batch-size 8 --accumulation 8` 接近报告配方。每轮完整训练 41,116 张，官方 valid 5,849 张；按 valid AUROC 提升更新唯一 `model/best_model.pth`。完整轮次状态写到同级 `runs/submission_new/<时间>/last.pth`。中断后用原目录 `--resume`，不能用新 run 覆盖旧最佳。CPU 不执行完整训练。
4. 完成验证后单独运行 `python submission_new/code/predict.py`，生成小写后缀 `submission_new/submission.csv`；训练不会改写任何历史 CSV。
5. `python submission_new/cost.py <100张目录> <5张预热目录>` 计时，必须在比赛指定设备核对平均 ≤100ms/张。未实测前不声称达标。

本目录仅是候选实验。正式提交前，按赛规补齐 `model`、`pretrain_model`、`licenses`、核验工作簿及预测 CSV；不要提交未训练的占位模型。

## 2026-11-02 新候选 submission_new

已完成 7 轮本机近似复现；官方独立 valid AUROC 最佳 0.9863724439657326（第 5 轮）。可提交 CSV 为 `submission_new/submission.csv`，5855 行已核验，SHA256 `39412da3b861de2e1781bed85e16de91367ba40793de8deeec7056aeaeee7f30`，官网分数未知。训练与报告作者的 98.55 不完全相同，详见 `submission_new/RECIPE.md`。旧路线 CSV 保留；官方平台成绩与指定硬件速度待实测。

### 2026-11-02 19:36 本机速度补测

胸片任务明确释放 GPU 后，在同一台 RTX 5060 Laptop 8GB 上执行 `D:/anaconda3.0/envs/deepStudy/python.exe -u submission_new/cost.py runs/submission_new/timing_20261102/100 runs/submission_new/timing_20261102/5`。计时图片为官方 valid 的只读副本，5 张预热、100 张×3 轮；单张从读图、预处理至模型输出含 CUDA 同步，与预测共用 `score_batch`。实测平均 **0.057311 秒/张（57.311 ms/张）**，低于 100 ms 参考门槛。此结果仅代表本机与该图像集，不冒称比赛指定设备正式验收。19:36 确认无 Python 进程、GPU 显存 26 MiB，并明确归还胸片任务。此前文档“速度尚未实测”已由本段更新。
