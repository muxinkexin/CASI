# Training: Lite-Mono + DSI

所有命令从 Mono 仓库根目录执行。保持你已经跑通的 Python/CUDA 环境，不需要迁移到 FlexDepth 环境。
文档按 FlexDepth 的“前置条件 → 一条命令两阶段 → 分阶段运行”组织，但参数以本仓库实际实现为准。

## Prerequisites

先设置真实路径；路径中可以包含空格，后续保持双引号。

```bash
export DATA_PATH=/home/zy/KITTI
export PRETRAIN_PATH=/home/tzq/tzq/3/lite-mono-pretrain/lite-mono-pretrain.pth
```

`PRETRAIN_PATH` 必须是与你的 `--model lite-mono` 匹配的 ImageNet 编码器预训练文件，不是已在 KITTI 上训练的深度模型。所有对照使用同一个文件。
数据为 PNG 时使用下面的 `--png`；如果你已跑通的输入是 JPG，统一删除训练、预检、评估中的 `--png`，不要只改其中一部分。

```bash
python scripts/check_dsi_setup.py \
    --data_path "$DATA_PATH" --mypretrain "$PRETRAIN_PATH" \
    --model lite-mono --split eigen_zhou --batch_size 12

python -m unittest discover -s tests -v
```

预检会检查依赖及训练/验证清单中每个目标帧和相邻帧的路径。完整 CPU 合成数据检查是可选的：

```bash
python tests/smoke_dsi_pipeline.py
python tests/smoke_evaluation.py
```

它们创建并清理临时数据，不使用真实 KITTI，不输出可用于论文的精度结果。训练检查使用真实网络、192×640 输入和 batch=2；会临时写入较大的优化器检查点，请预留磁盘空间。

如果只缺少原 Lite-Mono 使用的调度器依赖，可安装原依赖：

```bash
python -m pip install 'git+https://github.com/saadnaeem-dev/pytorch-linear-warmup-cosine-annealing-warm-restarts-weight-decay'
```

其他缺失依赖应在你的原环境中按报错补齐，不要为了本补丁重新安装 CUDA 版 PyTorch。

## KITTI: original Lite-Mono baseline

原始网络和原始训练损失，30 个 epoch，保留 auto-masking。建议重新运行一个具有明确 seed 和日志的对照，而不是把来源不明的旧结果混入比较。

```bash
python train.py --dsi_stage off \
    --data_path "$DATA_PATH" --mypretrain "$PRETRAIN_PATH" \
    --model lite-mono --split eigen_zhou --png \
    --height 192 --width 640 --batch_size 12 --num_workers 8 \
    --num_epochs 30 --seed 42 --save_frequency 1 \
    --lr 0.0001 5e-6 31 0.0001 1e-5 31 \
    --log_dir ./logs/kitti_litemono --model_name baseline_s42
```

最终模型：`./logs/kitti_litemono/baseline_s42/models/weights_29/`。
`--dsi_stage off` 仍进入原 `trainer.py`；补丁只在入口给两类方法统一设置随机种子。

## KITTI: two-stage training in one command

推荐先使用与原始对照相同的 batch=12；如果两阶段训练显存不足，应同时降低两种方法的 batch，重新建立对照。不要在第二阶段才临时改变 batch 并混报实验设置。

```bash
python train.py --dsi_stage both \
    --data_path "$DATA_PATH" --mypretrain "$PRETRAIN_PATH" \
    --model lite-mono --split eigen_zhou --png \
    --height 192 --width 640 --batch_size 12 --num_workers 8 \
    --num_epochs 30 --dsi_step2_epochs 30 --dsi_early_epoch 2 \
    --dsi_beta 0.7 --dsi_mu 0.1 --dsi_gds_weight 100 \
    --dsi_refresh_start 1 --dsi_refresh_every 5 \
    --dsi_lr 0.0001 --dsi_min_lr 1e-6 --dsi_cycle 16 --dsi_gamma 0.9 \
    --seed 42 --save_frequency 1 \
    --lr 0.0001 5e-6 31 0.0001 1e-5 31 \
    --log_dir ./logs/kitti_litemono_dsi --model_name mono_model_s42
```

- 原始阶段：`weights_0` 到 `weights_29`，关闭 auto-masking、不使用 Var/DD-mask/GDS。
- 初始早期参考：`weights_2`，**编号从零开始，表示完成第三个 epoch 后的检查点**。
- 固定晚期参考及 PoseNet：`weights_29`。
- 自抑制阶段：`weights_30` 到 `weights_59`；深度网络重置为第一阶段开始前的初始状态，编码器保持相同 ImageNet 初始化，解码器保持该次运行保存的初始随机权重；不是从 `weights_29` 继续微调。
- EarlyRef 在第二阶段完成第 1、6、11、16、21、26 个 epoch 后复制学生深度网络；每次复制后继续冻结。LateRef/PoseNet 始终固定。
- 最终可用于原 Lite-Mono 推理的文件仍为 `encoder.pth` 和 `depth.pth`。

## Or train two stages separately

```bash
# Stage 1 only
python train.py --dsi_stage raw \
    --data_path "$DATA_PATH" --mypretrain "$PRETRAIN_PATH" \
    --model lite-mono --split eigen_zhou --png \
    --height 192 --width 640 --batch_size 12 --num_workers 8 \
    --num_epochs 30 --dsi_early_epoch 2 --seed 42 --save_frequency 1 \
    --lr 0.0001 5e-6 31 0.0001 1e-5 31 \
    --log_dir ./logs/kitti_litemono_dsi_split --model_name raw_s42

# Stage 2 only: explicitly select both reference checkpoints
python train.py --dsi_stage inhibit \
    --data_path "$DATA_PATH" --mypretrain "$PRETRAIN_PATH" \
    --model lite-mono --split eigen_zhou --png \
    --height 192 --width 640 --batch_size 12 --num_workers 8 \
    --num_epochs 30 --dsi_step2_epochs 30 --dsi_early_epoch 2 \
    --dsi_early_weights ./logs/kitti_litemono_dsi_split/raw_s42/models/weights_2 \
    --dsi_late_weights ./logs/kitti_litemono_dsi_split/raw_s42/models/weights_29 \
    --dsi_beta 0.7 --dsi_mu 0.1 --dsi_gds_weight 100 \
    --dsi_refresh_start 1 --dsi_refresh_every 5 \
    --dsi_lr 0.0001 --dsi_min_lr 1e-6 --dsi_cycle 16 --dsi_gamma 0.9 \
    --seed 42 --save_frequency 1 \
    --lr 0.0001 5e-6 31 0.0001 1e-5 31 \
    --log_dir ./logs/kitti_litemono_dsi_split --model_name inhibit_s42
```

第二阶段最终模型：`./logs/kitti_litemono_dsi_split/inhibit_s42/models/weights_59/`。
`--num_epochs 30` 在第二条命令中仍表示第一阶段长度和全局检查点编号偏移；实际只执行 `--dsi_step2_epochs 30`。
两种启动方式实现相同训练规则，但新进程重新设置 RNG，不能要求与一条命令方式逐位相同。

## Resume an interrupted DSI run

完整重复原 DSI 命令，只在末尾增加：

```bash
--dsi_resume ./logs/kitti_litemono_dsi/mono_model_s42/models/weights_39
```

以上片段应接在完整命令后，不能单独执行。它会从已完成的第二阶段检查点继续下一轮，而不是重新训练第二阶段。
使用相同的 `log_dir/model_name`、数据、seed、batch、模型、训练长度和超参数。补丁会校验配置及训练/验证清单哈希。
恢复内容包括 Depth/Var、优化器、当前 EarlyRef/LateRef、冻结 Pose、更新记录和随机状态。调度器按已完成轮数重放，避免序列化第三方调度器对象。
恢复粒度为 epoch；中断 epoch 尚未保存的部分需重跑。CPU 合成检查验证了恢复后继续训练的权重一致性；多进程/CUDA 训练不承诺位级确定性。
`--load_weights_folder` 不用于 DSI 断点恢复，以免误加载原优化器或覆盖初始化。

## Training-budget control

DSI 比原始 30-epoch 对照多一个训练阶段。建议增加以下“数据遍历轮数相同”对照；这并不等同于 GPU 运算量相同，因为 DSI 还运行参考和 Var 网络。

```bash
python train.py --dsi_stage off \
    --data_path "$DATA_PATH" --mypretrain "$PRETRAIN_PATH" \
    --model lite-mono --split eigen_zhou --png \
    --height 192 --width 640 --batch_size 12 --num_workers 8 \
    --num_epochs 60 --seed 42 --save_frequency 1 \
    --lr 0.0001 5e-6 31 0.0001 1e-5 31 \
    --log_dir ./logs/kitti_litemono_budget --model_name baseline60_s42
```

最终模型为 `weights_59`，保留原 31-epoch 余弦周期及其重启。报告时注明该调度设置，不把训练时长增加全部解释为 DSI 收益。
主结果稳定后，分别将 seed 改为 43、44，并同步修改输出模型名，重复 baseline 与 DSI。

## Logs and limitations

TensorBoard 关注 `dsi/keep_0`、`dsi/active_0`、`dsi/nd_0`、`dsi/gds_0`、`dsi/sigma_0` 和掩码图。keep 是 DD-mask 保留率；active 还考虑 auto-masking。长期接近全零或全一应结合图像和误差排查，不能仅凭某个固定比例判定成功。

`dsi_run.json` 记录参考权重来源与哈希、训练清单哈希、参数和更新轮次。`models/opt.json` 记录启动配置。
原仓库只提供 KITTI 数据加载器，故此次不给出实际上无法执行的 Cityscapes 命令。也不使用 FlexDepth 的 `--dy_mu`、YOLO/SDD 或上一轮提出的 CASI 改进。
