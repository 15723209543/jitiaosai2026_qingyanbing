# submission：单Tiny320 成对AUC最佳权重

**v1.26.2；官方valid 5,849张 AUROC 0.9897295956797492。** 用户提供的高分报告是方案来源，作者的官网98.55并非本机成绩。本机上一正式Small+Tiny版本官方valid为0.9875753924850881、用户反馈官网97.97。新版本官网成绩未知。

选定`model/tiny_best_model.pth`，SHA256 `0feb9619de1c5316bb2d0947bd5ea122fe9957f77d36a7bef174468f2da516c8`，第3轮EMA权重。训练从此前最高Tiny已训练权重开始，使用官方train内41,116张；官方valid仅以AUROC选模，官方test仅最终预测。没有外部医学训练数据或医学权重、TTA和测试调参。训练详情与后续未晋升实验见`RECIPE.md`及`../项目文档/训练记录/20261105_最终交付/README.md`。

正式预测`code/predict.py`与计时`cost.py`均调用`code/predict_tiny.py`/`code/measure_tiny.py`中的同一单视图Tiny `score_batch`。选定PTH 111,644,807字节；本机 RTX 5060 Laptop 使用100张验证图×3轮测得43.440ms/张，比赛指定硬件待验。`model/best_model.pth`是旧Small历史权重，正式预测不加载；保留它与旧融合记录供回退。整个`model/`内两个PTH合计仍低于500MB。

拉取后先执行`git lfs pull`，保证官方数据只读路径`glaucoma_first/`或`GLAUCOMA_DATA_ROOT`可用。然后在仓库根目录运行：

```powershell
D:/anaconda3.0/envs/deepStudy/python.exe -u submission/code/predict.py --output submission/submission.csv --workers 1
D:/anaconda3.0/envs/deepStudy/python.exe -u submission/cost.py runs/submission_new/timing_20261102/100 runs/submission_new/timing_20261102/5 --output submission/model/tiny_cost_recheck.json
```

正式CSV应有5,855行、表头`Image_name,Probability,Pred_Label`，且遵循官方test顺序；最终哈希与校验报告记录在训练记录README。官网成绩只有上传后才能获知。