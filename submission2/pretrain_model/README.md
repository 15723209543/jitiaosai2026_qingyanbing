# 原始非医学预训练权重

运行 `code/train.py` 时自动从下列公开模型卡下载 `model.safetensors` 至本目录对应子目录，并将准确版本号、直链和原始文件 SHA-256 写进单个 `model/best_model.pth`。代码**不**使用任何外部眼底/医学权重或图像。

| 模型 | 原始训练数据 | 模型卡与许可 |
| --- | --- | --- |
| `convnext_tiny.fb_in1k` | ImageNet-1K | https://huggingface.co/timm/convnext_tiny.fb_in1k |
| `efficientnetv2_rw_s.ra2_in1k` | ImageNet-1K | https://huggingface.co/timm/efficientnetv2_rw_s.ra2_in1k |

两个模型卡目前标注 Apache-2.0。下载后仍须由参赛者核对该具体版本的权重许可、数据来源、重分发资格，保留原始权重和模型卡用于最终核验；不能把模型代码许可证误当作权重许可证。`code/predict.py` 不会下载或加载本目录的原始权重，只加载已微调的单个 PTH。
