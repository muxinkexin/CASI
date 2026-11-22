# 恢复补丁验证记录

本记录不是在用户 GPU 主机或用户真实 weights_35 上执行的结果。现有 traceback 能定位故障边界，但尚不能确定非布尔参数的具体引入来源。

## 已执行

- Python 源码编译、补丁重放、幂等性及 git diff --check。
- 模式辅助函数的 5 项单元测试：故障注入复现相同 ValueError、直接逐层设置模式、原生 bool 检查、学生可训练、冻结 BN/参数不变及第六轮更新后下一轮模式切换。
- 真实 Lite-Mono-Tiny、192×640、batch=2、CPU 合成 KITTI 输入，执行 raw=3、inhibit=7 个 epoch；EarlyRef 在完成第 1、6 个 inhibit epoch 后更新，完成第六轮更新后成功进入第七轮。
- 从 completed=6 的中间检查点恢复，并实际继续下一轮。最终编码器权重与未中断运行逐位一致，学习率一致；冻结 Pose 的参数与 BN buffers 保持不变。
- 执行恢复入口的 --check-only 以及正式训练分支。通过故障注入验证：即使配置保留 weights_init=pretrained 且原 ImageNet 文件不存在，恢复不会调用 ImageNet 初始化/下载或 load_pretrain，而是读取保存状态。
- 检查器的边界情况：保留上传配置中的 JPG 设置、completed 不符、缺失文件、NaN 权重、PNG/JPG 配置变化均会正确处理/拒绝。

这里使用的短序列是软件回归测试；不验证 KITTI 精度，也不声称复现用户原环境导致非布尔参数的真实来源。

## 进度与学习率核对

根据上传配置及日志，计算并反查：

- raw=30，completed=6 → 已保存索引 35，下一索引 36。
- inhibit 总长 30，已完成 6 → 剩余 24。
- 原调度器 completed=6 后保存的学习率约 6.944282990207195e-5。
- 下一轮学习率约 6.015697093979836e-5，终端显示 0.000060。
- 真实运行的检查点仍必须读取验证；不能用这些计算结果代替实际检查。

## 修改范围

- 逐层设置原生 training=True/False；对本项目使用的标准模块保持 BN、Dropout 和 DropPath 语义。
- 参考和 Pose 的 requires_grad 保持关闭；不全局 monkey patch PyTorch。
- 恢复初始化跳过 ImageNet 依赖，随后严格载入学生/Var/参考/Pose/优化器/RNG。
- 不修改损失公式、权重、更新周期、训练长度、seed、batch、数据或图像格式。
- 不编辑、删除或替换既有检查点。

此辅助函数不是适用于任意自定义 train() 副作用的通用替代品；限定于本项目已检查的模块结构。
