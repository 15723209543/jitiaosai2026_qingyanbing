# 使用说明

训练已按首次再提升条件停止。当前正式验证AUROC 0.964011。先发submission.csv即可，它是5,855张测试图的预测，不是模型文件。最终核验保留model/best_model.pth和完整ZIP。

项目目录D:/青眼病/project。预测：`D:/青眼病/.venv/Scripts/python.exe submission/code/predict.py`。计时：`python submission/cost.py pictures_100 pictures_5`。推理FP32、batch1、固定等权、无TTA。

给人看的结果在训练结果.md，给AI的继续约定在AI_HANDOFF.md。请勿再次运行训练；如果未来重新授权，必须新建run、保存不可变起点，并核对优化器是否匹配最终权重。官方上传和指定设备复测由参赛者处理。
