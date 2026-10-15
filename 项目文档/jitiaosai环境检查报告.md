# jitiaosai 环境与 v1.3.0 适配报告

更新日期：2026-10-15。只做轻量检查，不启动训练或整批预测。

## 1. 解释器

项目解释器：`C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe`。
原学习解释器：`C:\Users\LENOVO\.conda\envs\deepStudy\python.exe`。
本轮没有安装/卸载依赖、升级/降级解释器，也没有删除、改名或覆盖deepStudy。

规程推荐Python3.12、PyTorch2.10.0、torchvision0.25.0；参考环境为T4 16GB、8核CPU、32GB内存、CUDA12.8/cuDNN9、NVIDIA驱动至少570.117。
requirements.txt继续保留官方推荐框架版本，不会因为运行代码而自动改变本机环境。

## 2. 本机环境与验证限制

历史核对的jitiaosai为Python3.12.13、PyTorch2.13.0+cpu、torchvision0.28.0+cpu、NumPy1.26.4、pandas2.2.3、Pillow12.3.0、scikit-learn1.9.0。本机当前采用CPU计算。
旧全环境pip check曾报告与本项目无关的工具缺少依赖，本次没有对无关工具做清理或安装；本轮具体接口检查见代码检查记录。

此前v1.2.0在受限沙箱执行时，scipy目录访问被拒绝，从而出现ModuleNotFoundError: scipy.sparse。
读取授权后，**同一个jitiaosai解释器**成功导入并通过回归检查，未重装SciPy。因此不能把沙箱访问失败解释为用户环境已经损坏。

## 3. 新增功能的接口

- torch.nn.LeakyReLU、Dropout2d；原有Conv2d、Linear、Dropout、CrossEntropyLoss继续保留。
- torch.optim.Adam的weight_decay参数。
- torch.optim.lr_scheduler.ReduceLROnPlateau。
- torch.save/load以及weights_only=True。
- v1.3.0新增Conv2d的padding=1及torch.nn.init.dirac_/zeros_初始化，不增加依赖。

上述均为推荐版本提供的基础接口。本轮实际回归在本机jitiaosai执行，不能冒称已经在官方2.10.0/T4环境复测。
新版pth保存由张量和基础字段构成的字典，包含state_dict、model_config、image_size、format_version；不是完整Python模型对象。旧纯state_dict仍兼容。
不再直接把新版pth整体传给model.load_state_dict，应调用项目utils.load_model。

## 4. 运行与记录

训练：
```powershell
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission/code/train.py
```

最多12轮，可能提前停止。只使用train/valid，最佳参数保存在submission/model/best_model.pth。
记录保存到项目文档/训练记录/时间目录，不在results生成训练历史。

预测：
```powershell
& "C:\Users\LENOVO\.conda\envs\jitiaosai\python.exe" submission/code/predict.py
```

只有预测入口生成submission/results/submission.csv，训练不会刷新旧答案。
旧纯权重按三层ReLU加载，v1.2.x权重按原三层配置加载；v1.3.0权重在model_config中记录extra_convs=2，恢复五层。现有三层模型不随训练默认结构改变。

## 5. 仍需用户后续确认

- 本版真实训练的验证AUROC/AUPRC/AP是否超过旧基线。
- 新CSV的平台成绩；用户反馈的79分属于修改前版本。
- 官方推荐框架/T4环境下的兼容性与单张端到端延迟≤100ms。
- CPU续训的实际耗时；轻量合成检查不能推算整套训练性能。

如需安装推荐版本，应另行确认环境方案，不覆盖原有学习解释器。
