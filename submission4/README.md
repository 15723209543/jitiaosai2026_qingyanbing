# Submission4：512 像素 ConvNeXt-Tiny 全参数微调

本目录是**待训练的新路线**。目前无有效 `candidate_best.pth`、`submission.csv` 或本路线的官方 valid AUROC。不得将公开仓库的成绩记为本项目成绩。

从 `D:\青眼病题` 根目录运行，先确认没有其他青眼病/胸片 GPU 训练或预测进程：

```powershell
nvidia-smi
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' } | Select-Object ProcessId,CommandLine
& 'D:\anaconda3.0\envs\deepStudy\python.exe' submission4/code/train.py --epochs 12 --batch-size 2 --accumulation 16
```

若中断，在**相同参数**下续训：

```powershell
& 'D:\anaconda3.0\envs\deepStudy\python.exe' submission4/code/train.py --epochs 12 --batch-size 2 --accumulation 16 --resume submission4/runs/<运行目录名>
```

训练结束后比较三条路线各自最佳的**官方 valid AUROC**，并核对比赛速度限制。若 submission4 入选前两名，再显式使用选中权重生成 CSV：

```powershell
& 'D:\anaconda3.0\envs\deepStudy\python.exe' submission4/code/predict.py --checkpoint submission4/runs/<运行目录名>/candidate_best.pth --output submission4/submission.csv
& 'D:\anaconda3.0\envs\deepStudy\python.exe' submission4/cost.py pictures_100 pictures_5 --checkpoint submission4/runs/<运行目录名>/candidate_best.pth
```

预测脚本只读取官方 test、保持官方 `test.csv` 顺序，输出三列 `Image_name,Probability,Pred_Label`。训练前不生成空白 CSV。详细参数和规则见 [RECIPE.md](RECIPE.md)。
