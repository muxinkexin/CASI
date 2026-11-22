# DSI 模式切换兼容修复与继续训练

## 当前结论

这次异常发生在 DSI 自抑制阶段切换参考网络模式的路径中：`model.eval()` → 递归 `train(False)` → PyTorch 检测到非原生布尔参数。标准 `eval()` 本应传入原生 `False`。仅凭这段 traceback 和 opt.json，尚不能确定非布尔参数究竟由哪个本地模块、包装或环境状态引入；不能据此认定是 NumPy、CUDA、PyTorch 版本或数据错误。

本修复针对本项目的标准 Lite-Mono、ResNet、BatchNorm、Dropout/DropPath 模块，显式给每个子模块设置原生 `True/False`，绕开出错的递归模式调用链。冻结参数继续维持 `requires_grad=False`，没有通过只禁用梯度而保留 BN 训练状态的方式掩盖问题。没有全局修改 PyTorch，也没有修改损失、超参数、参考更新周期、模型权重或优化器历史。

当前尚未访问用户 GPU 主机或实际权重。下面的检查脚本会在用户环境核验检查点，运行命令才会实际继续训练。

## 根据上传配置确定的恢复位置

| 项目 | 值 |
|---|---|
| 项目代码目录 | `/home/tzq/tzq/3/Lite-CASI` |
| 数据目录 | `/home/zy/KITTI` |
| 运行目录 | `/home/tzq/tzq/3/checkpoint/Mono_runs/kitti_litemono_dsi/mono_model_s42` |
| 图像扩展名 | JPG；`png=false`，恢复时不能添加 `--png` |
| 第一阶段长度 | 30 epochs |
| 第二阶段总长度 | 30 epochs |
| 日志所示已完成的第二阶段轮数 | 6 |
| 应检查的保存目录 | `models/weights_35` |
| 正常恢复后的 phase_epoch | 6，即第二阶段第 7 轮 |
| 正常恢复后的 checkpoint_index | 36 |
| 剩余第二阶段轮数 | 24 |
| 最终检查点 | `weights_59` |

代码顺序为：完成 epoch → 更新 EarlyRef → 保存 → 进入下一 epoch 的 set_train。你的错误位于下一 epoch 的 set_train，因此 `weights_35` 按控制流应该已保存；但必须通过实际读取确认，不能仅根据目录存在就认定完整。

不要把 `dsi_step2_epochs` 改成 24。它表示第二阶段总长度，恢复位置由保存的 completed=6 决定。不要使用 `load_weights_folder` 代替完整 DSI 恢复，也不要手动拼接来自不同轮次的模型/优化器文件。

## 1. 解压并激活原环境

将 `dsi_recovery_hotfix.zip` 下载到 `/home/tzq/tzq/3/`，随后执行：

```bash
conda activate litemono
python -m zipfile -e /home/tzq/tzq/3/dsi_recovery_hotfix.zip /home/tzq/tzq/3
cd /home/tzq/tzq/3/Lite-CASI
```

保留原环境，不需要重装 PyTorch、NumPy、CUDA 或 timm。

## 2. 应用代码补丁

```bash
python /home/tzq/tzq/3/dsi_recovery_hotfix/apply_dsi_recovery_fix.py \
    --repo /home/tzq/tzq/3/Lite-CASI --dry-run

python /home/tzq/tzq/3/dsi_recovery_hotfix/apply_dsi_recovery_fix.py \
    --repo /home/tzq/tzq/3/Lite-CASI
```

安装器会先检查全部补丁目标并编译，正式写入前在项目旁边创建 `dsi_recovery_backup_*`，备份原代码。补丁目标不匹配时停止；不会覆盖任何模型检查点、优化器、训练日志。相同版本重复安装是幂等的。

| 文件 | 修改 |
|---|---|
| `dsi_trainer.py` | 显式逐层切换模式；恢复时跳过不必要的 ImageNet 初始化和文件加载，随后完整恢复检查点 |
| `dsi_mode_compat.py` | 新增模式/冻结辅助函数 |
| `resume_dsi_training.py` | 新增配置驱动的检查与恢复入口 |
| `test_dsi_mode_compat.py` | 新增模式异常回归测试 |

阶段二恢复时仍会先重建相同网络，再加载已训练权重；代码内部的临时初始化不等于从头训练。冻结参考、Var、优化器及 RNG 从检查点恢复。原 ImageNet 文件即使已移走，也不会因为这次恢复而重新访问或下载。

## 3. 快速检查模式修复

```bash
python test_dsi_mode_compat.py
```

应看到 `Ran 5 tests` 和 `OK`。测试包含同类异常的故障注入、原生布尔状态、BatchNorm buffers 不变、学生可训练以及第 6 次更新后进入下一轮。这是模拟异常路径，不代表已经确定用户原环境中非布尔值的来源。

## 4. 检查 weights_35，但不开始训练

```bash
python resume_dsi_training.py \
    --checkpoint /home/tzq/tzq/3/checkpoint/Mono_runs/kitti_litemono_dsi/mono_model_s42/models/weights_35 \
    --expect-completed 6 --check-only
```

检查会实际读取并校验：

- `dsi_state.pth` 的 phase/format/completed、教师和初始状态；
- `encoder.pth`、`depth.pth`、`var_encoder.pth`、`var.pth`；
- `pose_encoder.pth`、`pose.pth`、`adam.pth`；
- 各状态字典的可读取性与数值有限性；
- opt.json 与 checkpoint 中的配置一致性；
- 训练/验证清单的 SHA-256；
- 参考更新时间记录、原数据路径及 CUDA 可用性。

成功时应包含：

```text
"phase": "inhibit"
"completed_inhibit_epochs": 6
"next_phase_epoch": 6
"next_checkpoint_index": 36
"remaining_inhibit_epochs": 24
"final_checkpoint_index": 59
"refresh_history": [1, 6]
"png": false
CHECK ONLY PASSED: no files changed; training has not started.
```

保存的第二阶段学习率应与日志一致，约为 `6.94428e-5`。进入下一轮后应约为 `6.01570e-5`，终端保留六位小数时显示 `0.000060`。程序读取实际保存状态，不通过这些参考值强行改写学习率。

## 5. 正式继续训练

```bash
set -o pipefail
python -u resume_dsi_training.py \
    --checkpoint /home/tzq/tzq/3/checkpoint/Mono_runs/kitti_litemono_dsi/mono_model_s42/models/weights_35 \
    --expect-completed 6 \
    2>&1 | tee /home/tzq/tzq/3/checkpoint/Mono_runs/kitti_litemono_dsi/mono_model_s42/resume_weights35.log
```

该命令从 `models/opt.json` 恢复原配置，不需要重新手写数据路径、seed、PNG/JPG 和全部训练超参数。运行前自动备份 opt.json 并保存恢复计划；`weights_35` 保留原样，接下来写 `weights_36`、`weights_37`，直到 `weights_59`。

关键判断是出现：

```text
Resumed DSI phase=inhibit, completed=6
DSI phase=inhibit phase_epoch=6 checkpoint_index=36
```

原 DSI 函数可能先打印 `DSI stage 2: reset DepthNet...`。在恢复路径中，这只是网络重建时的临时初始化，随后会加载 weights_35 的学生/Var/优化器。以紧随其后的 `Resumed ... completed=6` 和 `checkpoint_index=36` 判断恢复位置，而不是仅看初始化日志。

计时器显示新的 `time elapsed` 属于新进程的运行时长，不是重训证据；原 global_step 仍从检查点保留。原 ETA 使用累计 step 与新进程计时混合，恢复初期不宜用于判断剩余时长。

`lr_p` 打印旧 Pose 优化器的学习率，不代表冻结 Pose 正在更新；第二阶段没有执行 Pose optimizer.step。

如需后台运行，可在以上检查全部通过后使用：

```bash
nohup python -u resume_dsi_training.py \
    --checkpoint /home/tzq/tzq/3/checkpoint/Mono_runs/kitti_litemono_dsi/mono_model_s42/models/weights_35 \
    --expect-completed 6 \
    > /home/tzq/tzq/3/checkpoint/Mono_runs/kitti_litemono_dsi/mono_model_s42/resume_weights35.log 2>&1 &
```

前台和后台二选一，避免启动两份训练同时写同一日志目录。

## 6. 如果检查没有通过

### weights_35 缺少 dsi_state.pth 或文件损坏

不要把 weights_35 的文件与其他轮次混用。可以退回上一个完整的 `weights_34`：

```bash
python resume_dsi_training.py \
    --checkpoint /home/tzq/tzq/3/checkpoint/Mono_runs/kitti_litemono_dsi/mono_model_s42/models/weights_34 \
    --expect-completed 5 --check-only
```

通过后使用同一命令删除 `--check-only` 继续。这样只重跑最后一个未可靠保存的第二阶段 epoch，不会重跑第一阶段。完整 weights_35 可用时应优先使用它。

### opt.json 与检查点配置不一致

如果此前失败的恢复尝试已经覆盖 opt.json，可使用包内保存的本次上传原配置：

```bash
python resume_dsi_training.py \
    --checkpoint /home/tzq/tzq/3/checkpoint/Mono_runs/kitti_litemono_dsi/mono_model_s42/models/weights_35 \
    --opt-json /home/tzq/tzq/3/dsi_recovery_hotfix/original_opt_uploaded.json \
    --expect-completed 6 --check-only
```

确认该配置通过后，正式继续训练时也保留相同 `--opt-json`。脚本仍会逐项与检查点 config 核对，不会忽略不匹配项。

### 补丁提示无法定位目标代码

安装器会停止而不修改原文件。包内 `reference/dsi_trainer.fixed.py` 用于审阅比对，不建议在存在自定义改动时未经比较直接覆盖。应保留新的错误以及本机 dsi_trainer.py 的相关方法，定位代码差异后再应用。

## 实际验证范围

详见 VALIDATION.zh-CN.md。已在 CPU 合成输入上执行原始阶段及第二阶段前七轮，覆盖完成第六轮刷新后继续下一轮；也已从 completed=6 的检查点恢复并继续，比较最终学生权重、优化器学习率和冻结状态。真实用户 GPU 和真实 weights_35 未在本环境读取，因此以用户执行检查后的输出为准。
