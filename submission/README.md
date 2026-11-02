**当前版本：v1.24.1；官网 97.65（用户提供，ID 29668）。**

# submission：ConvNeXt-Small 320 最终候选

本目录是原 `submission_new` 的重命名，作为当前唯一放在项目根目录的提交方案。旧 `submission`、`submission2`、`submission3`、`submission4` 已打包到 `项目文档/归档_本地/20261102_旧submission/`，原目录也在该本地归档中保留。用户提供的[高分报告原文](../项目文档/历史研究记录/青光眼_真实实验与提交报告_20261101.md)对应 S23 `g07_small_erasing`，报告作者官网最高 98.55；本目录仍是结构与数据划分已注明差异的**近似复现**，并非作者原始源码或权重。

## 本机已完成结果

- 非医学 ImageNet-1K `convnext_small.fb_in1k` 初始化；320 像素整图、随机擦除、EMA、辅助 BCE、双视图一致性。官方 train 内抽取 41,116 张训练，官方 valid 5,849 张仅按 AUROC 选模；无 TTA、外部医学数据、医学预训练、测试集调参。
- 2026-11-02 完成 7 轮；第 5 轮官方 valid AUROC 最高 **0.9863724439657326**。最佳权重 `model/best_model.pth` SHA256 `33ce0cccaf8bdaa9d132a75c6fd25d19dc2293ae9a5955b03d2b1f2647ff7a93`。
- `submission.csv` 已按官方 test 顺序生成并核验 5,855 行；SHA256 `39412da3b861de2e1781bed85e16de91367ba40793de8deeec7056aeaeee7f30`。用户提供官网截图：2026-11-02 20:57:24、ID 29668、得分 **97.65**；截图未包含文件哈希，文件对应关系依据用户说明。
- 本机 RTX 5060 Laptop 上，同一推理实现对 100 张×3 轮测得平均 **57.311 ms/张**；正式指定设备仍需验收。模型与预训练权重合计约 399 MB，低于比赛 500 MB 上限。

## 明天继续训练

先和胸片任务协调 8GB GPU，确认无重复进程。保持当前配方，从已完成第 7 轮的同一断点继续，目标上限 25 轮；同一 40 轮余弦调度不会因目标轮数改变。原 run 目录保留历史名称 `runs/submission_new/20261102_133834`：

```powershell
D:/anaconda3.0/envs/deepStudy/python.exe -u submission/code/train.py --epochs 25 --batch-size 8 --accumulation 8 --workers 2 --resume runs/submission_new/20261102_133834
```

每完整轮仅用独立官方 valid AUROC 严格提高来更新 `model/best_model.pth`。先重点比较至第 13 轮（报告 g07 实际运行 13 轮、选中第 7 轮）；若连续 6 轮无提升，训练代码早停。原报告与本实现的结构、划分、选模规则不同；目前第 6、7 轮的官方 valid AUROC 均低于第 5 轮，**轮次不足尚未得到证实**。如后续未提升，保留现有最佳权重与 CSV。若权重晋升，需用 `code/predict.py --checkpoint submission/model/best_model.pth --output submission/submission.csv` 重新预测并核验 5,855 行。

配方、差异、权重来源和训练细节见 [RECIPE.md](RECIPE.md)。`cost.py` 与预测共用 `score_batch`。本目录运行时需将官方 `glaucoma_first/` 放在其同级，或设置 `GLAUCOMA_DATA_ROOT`。首次拉取仓库后执行 `git lfs pull`。
