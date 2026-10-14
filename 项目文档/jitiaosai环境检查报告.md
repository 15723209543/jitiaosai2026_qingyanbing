# `jitiaosai` 环境与 2026 规程适配报告

更新日期：2026-10-14

执行约定：助手后续只做代码修改和轻量检查，由用户启动训练/预测。
`train.py` 只训练与验证，`predict.py` 单独运行时才生成提交答案。

## 1. 解释器状态

项目专用解释器：

```text
C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe
```

原学习解释器仍保留：

```text
C:\Users\LENOVO\.conda\envs\deepStudy\python.exe
```

本轮没有删除、改名、覆盖或重建 `deepStudy`。

## 2. 2026 规程推荐环境

| 项目 | 官方推荐值 |
|---|---|
| GPU | NVIDIA T4 16GB ×1 |
| CPU | 8 核 |
| 内存 | 32GB |
| 磁盘 | ≥100GB SSD |
| NVIDIA 驱动 | ≥570.117 |
| CUDA/cuDNN | CUDA 12.8 + cuDNN 9 |
| Python | 3.12 |
| PyTorch | 2.10.0 |
| torchvision | 0.25.0 |

推荐镜像：

```text
crpi-4leqgt7v1jo7rycf.cn-beijing.personal.cr.aliyuncs.com/coder02026/compete:1.0
```

## 3. 本机 jitiaosai 当前环境

| 项目 | 当前值 |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.13.0+cpu |
| torchvision | 0.28.0+cpu |
| CUDA | 不可用，当前为 CPU 版 |
| Pillow | 12.3.0 |
| NumPy | 1.26.4 |
| pandas | 2.2.3 |
| torchsummary | 已安装 |
| scikit-learn | 1.9.0，上次修改已验证可导入，用于验证指标 |
| pip check | 项目必需包可用；全环境存在无关工具的缺失依赖，见下文 |

Python 主版本符合规程；PyTorch/torchvision 比推荐版本更新，且当前为 CPU 版。本轮
没有直接降级解释器，避免破坏现有可用环境。训练、验证、预测和模型重载均已成功，说明
本项目直接依赖可用。全环境 `pip check` 会报告 alembic、httpx、pytest、uvicorn 等
与本项目无关的工具缺少 sqlalchemy、anyio 等依赖；本轮未为这些无关工具安装或删除包。
`requirements.txt` 已将正式复现目标固定为：

```text
torch==2.10.0
torchvision==0.25.0
```

初赛只上传 CSV，不上传本机环境。正式大规模训练或复赛模型验证前，应在单独环境或
官方镜像中复跑推荐版本。

## 4. 代码兼容范围

当前代码只使用稳定基础接口：

- `torch.nn.Conv2d/Linear/Dropout/CrossEntropyLoss`；
- `torch.optim.Adam`；
- `torch.utils.data.Dataset/DataLoader`；
- `torch.save/torch.load`；
- `torchvision.transforms.Resize/ToTensor/Normalize`；
- Pillow RGB 图片读取；
- pandas CSV 读写。

没有使用 PyTorch 2.13 独有编译器、算子或序列化特性。保存内容是纯 `state_dict`，
便于在官方推荐版本重新加载验证。

## 5. 2026 本地工作流

训练：

```powershell
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission/code/train.py
```

真实数据位于 `glaucoma_first/`，包含 `train.csv`、`valid.csv`、`test.csv` 和
对应的 `train/`、`valid/`、`test/` 图片目录。训练、验证 CSV 列为
`Image_name/Age/True_Label`，测试 CSV 只有 `Image_name/Age`，
训练/验证/测试规模分别为 51396/5849/5855。首轮 CPU 全量训练（1 epoch）已完成：
loss `0.49328`、训练打印 acc `0.97`、用时 `1624.53s`；验证 accuracy `0.967345`、
AUROC `0.628867`、AP `0.048873`。这些为旧首轮结果。

此后于 2026-10-13 完成三轮续训，最佳验证 AUROC `0.885387`、梯形 AUPRC `0.362978`、
AP `0.365991`、阳性召回率 `64.40%`。本次只调整入口职责，没有重新训练。

训练输入：`glaucoma_first/train.csv` 与 `glaucoma_first/train/`；验证输入：
`glaucoma_first/valid.csv` 与 `glaucoma_first/valid/`。

输出：

```text
submission/model/best_model.pth
```

预测：

```powershell
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission/code/predict.py
```

需要 `glaucoma_first/test/` 与 `submission/model/best_model.pth`。

输出：

```text
submission/results/submission.csv
```

## 6. 本轮接口验证与真实首轮结果

已使用真实数据完成以下验收：4 个 Python 源文件语法通过；`best_model.pth` 能重新加载；
2 张 3×64×64 张量的模型输出形状为 `(2,2)`；测试接口和 CSV 均为 5855 行；CSV 三列及
图片名称/顺序与官方 `test.csv` 逐字一致；`config.json` 可正常解析。

真实首轮验证集概率范围为 `0.064293`～`0.308502`；0.5 阈值下 TP=0、TN=5658、FP=0、FN=191，
sensitivity=0、specificity=1。约 97% 的准确率受类别不平衡误导，不能说明阳性检出有效。

验证阈值诊断参考：F1 阈值 `0.257520`（F1 `0.094390`、sensitivity `0.277487`、
specificity `0.844645`）；Youden 阈值 `0.190134`（sensitivity `0.790576`、
specificity `0.446271`）。提交 CSV 为 5855 个数据行，三列及测试文件名/顺序匹配正确，
但 `Pred_Label` 在默认 0.5 阈值下全为 0；连续 Probability 仍可用于 AUROC 排序评价。
实际模型文件为 54935871 bytes。模型和 CSV 当前仅为首轮流程基线，不建议视为最终竞赛方案。

## 7. 依赖安装说明

正式复现环境建议使用全新的 Python 3.12 环境，不覆盖 `deepStudy`。根据本机 GPU、
驱动和官方镜像情况安装与 CUDA 12.8 对应的 PyTorch 2.10.0；如果只做 CSV 接口检查，
CPU 版本也可以运行，但 51396 张训练图片会较慢。

环境核对命令：

```powershell
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" -c "import sys,torch,torchvision; print(sys.version); print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available())"
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" -m pip check
```

第二条命令检查解释器中的所有已安装工具，不只检查本项目。若仍保留无关 Web/打包工具，
可能显示它们的缺失依赖；判断本项目时应同时以训练、预测、模型重载和上述核心包导入为准。

## 8. 尚未完成的验证

仍需在主办方推荐的 T4/CUDA 12.8 环境测量单张端到端延迟，并通过后续实验改善阳性
检出能力。当前首轮模型和提交 CSV 仅为流程基线，不应视为最终竞赛方案。
