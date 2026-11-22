# Evaluation: Lite-Mono vs Lite-Mono + DSI

与训练文档一样，从仓库根目录执行。保持原 KITTI Eigen 协议：相同测试清单、GT、裁剪、深度上限、分辨率及后处理设置。不要混用 Eigen benchmark、稠密 GT、最小二乘对齐或不同尺度协议。

```bash
export DATA_PATH=/absolute/path/to/kitti_data
export SPLIT_PATH="$PWD/splits"
mkdir -p ./results
```

仓库当前的 Eigen 测试清单与 `gt_depths.npz` 都包含 697 个样本。GT 是原仓库文件，补丁不会重写。

## KITTI: median scaling (M), primary comparison

M = median(GT) / median(pred)，逐图像对齐；输出误差按图像平均。默认无翻转后处理，深度评估范围为 (0.001, 80) 米，再使用原 Lite-Mono 裁剪。

```bash
# Original Lite-Mono, final checkpoint
python evaluate_depth.py --eval_mono --png --model lite-mono \
    --data_path "$DATA_PATH" --split_path "$SPLIT_PATH" --eval_split eigen \
    --load_weights_folder ./logs/kitti_litemono/baseline_s42/models/weights_29 \
    --eval_checkpoint_tag final \
    --eval_metrics_json ./results/baseline_s42_M.json

# Lite-Mono + DSI, final checkpoint
python evaluate_depth.py --eval_mono --png --model lite-mono \
    --data_path "$DATA_PATH" --split_path "$SPLIT_PATH" --eval_split eigen \
    --load_weights_folder ./logs/kitti_litemono_dsi/mono_model_s42/models/weights_59 \
    --eval_checkpoint_tag final \
    --eval_metrics_json ./results/dsi_s42_M.json

# Equal data-pass budget control
python evaluate_depth.py --eval_mono --png --model lite-mono \
    --data_path "$DATA_PATH" --split_path "$SPLIT_PATH" --eval_split eigen \
    --load_weights_folder ./logs/kitti_litemono_budget/baseline60_s42/models/weights_59 \
    --eval_checkpoint_tag final \
    --eval_metrics_json ./results/baseline60_s42_M.json
```

JPG 数据应在三条命令中同时删除 `--png`。评估自动读取编码器检查点的输入分辨率，不依赖随意填写的 `--height/--width`。`--model` 仍必须与权重匹配。
无需加载 Var/Pose/参考网络，参数统计只计算推理用的 DepthNet。默认评估 batch=16，可用 `--eval_batch_size 4` 降低显存；最后不足一个 batch 的样本仍会评估。可加 `--no_cuda` 在 CPU 上运行，或 `--eval_profile` 单次统计 THOP MACs，避免每个 batch 重复 profiling。

## Ratio median scaling (RM), supplementary comparison

RM = median(GT/pred)，与 M 不是同一指标协议，单独成表。不要拿 DSI 的 RM 结果与 baseline 的 M 结果比较。

```bash
python evaluate_depth.py --eval_mono --png --model lite-mono --RM_scaling \
    --data_path "$DATA_PATH" --split_path "$SPLIT_PATH" --eval_split eigen \
    --load_weights_folder ./logs/kitti_litemono/baseline_s42/models/weights_29 \
    --eval_metrics_json ./results/baseline_s42_RM.json

python evaluate_depth.py --eval_mono --png --model lite-mono --RM_scaling \
    --data_path "$DATA_PATH" --split_path "$SPLIT_PATH" --eval_split eigen \
    --load_weights_folder ./logs/kitti_litemono_dsi/mono_model_s42/models/weights_59 \
    --eval_metrics_json ./results/dsi_s42_RM.json
```

## Automatic acceptance report

```bash
python scripts/compare_dsi_results.py \
    --baseline ./results/baseline_s42_M.json \
    --dsi ./results/dsi_s42_M.json \
    --output ./results/comparison_s42_M.json

python scripts/compare_dsi_results.py \
    --baseline ./results/baseline60_s42_M.json \
    --dsi ./results/dsi_s42_M.json \
    --output ./results/comparison_budget_s42_M.json
```

脚本先检查两份结果的协议、GT/清单哈希、模型、输入大小、样本次序和检查点选择规则。随后按**本实验预先约定的工程目标**判断，这些阈值不是论文保证：

| 标准 | 默认要求 | 说明 |
|---|---|---|
| Abs Rel | 相对本机 baseline 至少下降 2% | 主指标；可用 `--min_abs_rel_improvement` 在看结果之前另行规定 |
| Sq Rel | 相对恶化不超过 1% | 防止主指标改善掩盖其他退化 |
| RMSE | 相对恶化不超过 1% | 同上 |
| δ<1.25 / a1 | 绝对下降不超过 0.002 | 允许有限的统计波动 |
| 推理参数量 | 两者相同 | 两份结果均有权重推理统计时自动核对 |
| 协议可比性 | 所有检查通过 | 不通过则 `INVALID_COMPARISON`，不作优劣结论 |

`PASS` 表示满足本次整体精度目标；不等于已经证明动态场景收益、跨随机种子稳定性或论文严格复现。
`FAIL` 表示当前结果尚未满足预期；进程返回码为 2，但仍保存完整报告。
论文中的 KITTI M 协议参考为 Lite-Mono 0.107、Lite-Mono+DSI 0.103，相对下降约 3.74%。脚本另外报告是否低于 0.103，但该数值仅作参考：训练环境、预训练权重和未公开的训练细节都会影响结果。
如果你的 baseline=0.107，2% 目标对应 DSI Abs Rel≤0.10486；实际判断使用未四舍五入的 JSON 数值。

脚本还报告逐图像配对 bootstrap 的 Abs Rel 差值置信区间；上界小于零意味着这对模型在测试图像抽样下表现出一致改善。它不能代替重复训练。建议对 seed 42/43/44 分别运行配对比较，再报告均值和标准差；不要只挑最好的 seed。

## Independent dynamic/static evaluation

整体 KITTI 改善不能单独证明动态区域改善。若有独立的真实运动区域标注，可在两条评估命令中同时增加：

```bash
--eval_region_masks /absolute/path/to/kitti_dynamic_masks.npz
```

上述是附加参数片段。文件要求：NPZ 的 `data` 有 697 个二值掩码，顺序与 `test_files.txt` 一致，每张尺寸与原始 GT 一致，1=动态、0=静态。不是网络输出尺寸，也不是仅将“车辆/行人”语义类视为真实运动。使用分割标注时只能称“可移动物体区域评估”，不能称真实动态评估。

**不能使用本模型的 DD-mask 作为评估真值。** 同一图像的动态/静态区域共用全图有效 GT 上计算的尺度因子，禁止分别做 median scaling。

```bash
python scripts/compare_dsi_results.py \
    --baseline ./results/baseline_s42_M.json \
    --dsi ./results/dsi_s42_M.json \
    --require_dynamic \
    --output ./results/comparison_dynamic_s42_M.json
```

`--require_dynamic` 额外要求动态区域 Abs Rel 至少相对下降 5%。没有独立掩码时输出 `NOT_ASSESSED`，不会自动宣称动态能力达标；要求动态验收时，缺少标注也不能得到 PASS。该验收目标同样是实验预设值，不是既有实测结果。

## Result table to fill after actual training

| Run | Protocol | Abs Rel | Sq Rel | RMSE | RMSE log | a1 | Judgment |
|---|---|---|---|---|---|---|---|
| Lite-Mono, 30 epochs | M | pending | pending | pending | pending | pending | baseline |
| Lite-Mono, 60 epochs | M | pending | pending | pending | pending | pending | budget control |
| Lite-Mono + DSI | M | pending | pending | pending | pending | pending | automatic report |
| Lite-Mono | RM | pending | pending | pending | pending | pending | separate table |
| Lite-Mono + DSI | RM | pending | pending | pending | pending | pending | separate table |

当前交付完成的是代码和合成数据可执行性验证；尚无你环境中的完整 KITTI 训练结果，因此当前精度达标状态为“待验证”。

## Troubleshooting

- PNG 训练而评估报 JPG 缺失：检查是否遗漏 `--png`；此补丁已经将扩展名传入评估 Dataset。
- GT/预测/清单数量不一致：检查 `split_path/eigen`，不要借用别的 benchmark GT。程序会直接报错，避免错位评估。
- 模型权重不能载入：核对 `--model` 与编码器输入大小，不要通过忽略不匹配权重来“修复”。
- `--ext_disp_to_eval`：输入应为经 `disp_to_depth` 转换的逆深度，不是网络原始 sigmoid 输出。可结合 `--no_eval --save_pred_disps --eval_out_dir` 做离线导出。
- 长期 DD-mask 接近全零：检查早晚权重和输入是否一致、参考是否错误加载、sigma 是否数值异常；先定位，不先调测试集阈值。
- 判断与目测不同：查看未取整的 JSON 及单图误差，不以终端保留三位小数的表格代替验证。
