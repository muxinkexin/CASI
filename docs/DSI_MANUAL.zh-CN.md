# Lite-Mono + DSI 对比实验操作手册

## 1. 本次实验目标与交付边界

本次先建立 **Lite-Mono + 原始 DSI 机制**的对比方法，为后续研究提供可信基线。目标是验证两阶段训练能否在保持 Lite-Mono 推理结构不变的前提下，改善本机可复现的深度估计结果，特别是动态场景中的错误监督。

本次不加入 FlexDepth 的自适应阈值、YOLO 编码器、SDD，也不加入前一轮方案的尺度校准、统计阈值或支撑门控。FlexDepth 的 `docs/train.md`、`docs/eval.md` 用作操作文档组织形式参考。

当前状态：已实现并执行 CPU 合成数据检查；尚未在用户 GPU、真实训练集和 ImageNet 预训练权重上完成全量训练。精度是否达到目标，必须在运行后依据结果文件判断，不预填成绩。

## 2. 源码基准与实现依据

| 来源 | 固定提交 | 用途 |
|---|---|---|
| [muxinkexin/Mono](https://github.com/muxinkexin/Mono) | `128f3b041ae87bf6573a414ed4c12cff55bc8b9d` | 本次修改基底，保留已跑通环境的兼容性修改 |
| [zhangtian33/DSI-training](https://github.com/zhangtian33/DSI-training) | `15a99b6337664a546c63e406b688e168e3c95e05` | 检查公开 VarDecoder、GDS 和 RM-scaling |
| [startnew/flexdepth](https://github.com/startnew/flexdepth) | `b04ae3a52a49d70aca5300ec32d74aa210393c64` | 检查文档格式及相关公开训练实现 |

理论依据为本会话上传的 DSI 论文，重点为 ND-Loss、DD-mask、GDS-Loss、两阶段训练和评估尺度对齐。DSI 公开仓库 README 仍说明完整训练代码未公开，因此本交付是**依据论文和已公开组件实现的 Lite-Mono+DSI 基线**，不能称为作者训练代码的逐行移植或保证复现原论文数值。

## 3. 详细原理

### 3.1 原始阶段（Raw stage）

联合训练 Lite-Mono 深度网络和原有 PoseNet。输入仍为目标帧及相邻帧 `[0,-1,1]`，通过预测深度和位姿进行重投影，得到光度误差：

`r = min_source(0.85 × SSIM_error + 0.15 × L1_error)`

采用原 Lite-Mono 的均值归一化视差、边缘感知平滑和多尺度训练。DSI 的 raw 阶段关闭 auto-masking，暂不引入 ND、DD-mask 和 GDS，使早晚检查点保留其学习差异。与正常 Lite-Mono 基线的 raw 损失设置不同，不能将 raw 阶段结果当成完整 DSI 最终结果。

### 3.2 自抑制阶段（Self-inhibition stage）

从 raw 阶段选择 EarlyRef 和 LateRef，预测同一增强目标图像的视差：

`diff = abs(d_early ** beta - d_late ** beta)`

`M = float(diff < mu)`，默认 `beta=0.7, mu=0.1`。

`M=1` 保留光度监督，`M=0` 屏蔽不可靠区域。本实现使用与公开 FlexDepth 掩码实现一致的网络原始 sigmoid 视差范围，不用米制深度、不做额外中位数归一化。这一点确保它不是上一轮提出的新方法。

掩码是在“训练期间预测不稳定”意义上的可靠性判断，不是运动分割真值；反光、遮挡或静态细节也可能被屏蔽。

深度学生网络重置到 raw 阶段开始前的状态；编码器恢复相同 ImageNet 初始化，解码器恢复该次运行保存的初始随机权重。与直接加载 raw 最终模型继续微调不同。PoseNet 加载 raw 最终权重后冻结；EarlyRef/LateRef 均 `eval()` 且 `requires_grad=False`。EarlyRef 按规定轮次更新，LateRef 不更新。

### 3.3 Normal Distribution loss（ND-Loss）

增加仅训练时使用的独立 Var 网络：ResNet-18 编码器和五级解码器。输入为按 `[0,-1,1]` 顺序拼接的三帧 RGB；采用 FlexDepth 公开训练实现的三帧输入做法。VarDecoder 使用公开 DSI 组件的 sigmoid 输出，将其视为公式中的 `sigma`，不是 `sigma²`。

`L_ND = log(1 + sigma) + r² / (2 × sigma²)`

本实现取各源图像重投影误差的最小值，并以 identity reprojection 判断 auto-mask。损失为：

`L_photo = mean(M × A × L_ND)`

其中 `A` 为 auto-mask，`sigma` 下限默认 `1e-3`，避免除零。掩码乘在整个表达式外，使被屏蔽像素不再通过 `log(1+sigma)` 更新 Var 网络。按整幅图像面积取平均，未改成仅对保留像素重新归一化。

残差、sigma 和掩码严格保持 `[B,1,H,W]`，防止 `[B,H,W]` 与四维张量相乘时在不同 batch 样本之间广播。DD-mask 本身不参与反向传播。

### 3.4 Ground-contact-prior smoothness（GDS-Loss）

以 `d* = d / (mean(d)+eps)` 计算：

`L_GDS = mean(|dx d*| exp(-|dx I|))`

`        + mean([eta(1-M_edge)+M_edge] |dy d*| exp(-|dy I|))`

默认 `eta=100`；垂直边上的 `M_edge` 采用上下两个像素掩码均值，与 DSI 公开 `layers.py` 的边离散方式一致。可靠区域为普通平滑，被屏蔽区域加强垂直一致性，从相邻区域/地面接触先验获得约束。它是先验，并不保证所有物体真实满足垂直等深。

总损失保留 Lite-Mono 的尺度权重：

`L = mean_scale(L_photo + disparity_smoothness × L_GDS / 2**scale)`

输出尺度为 `0,1,2`。光度损失在全分辨率计算；GDS 在对应原生尺度计算。

### 3.5 需要明确记录的工程选择

1. 阶段二重新初始化学生，但使用已保存的相同初始权重，使初始化可追溯。
2. 阶段二沿用原仓库的 ChainedScheduler，默认初始 LR=`1e-4`、最小 LR=`1e-6`、周期=`16`、gamma=`0.9`。**该依赖的 gamma 按余弦重启周期衰减峰值，不是每个 epoch 乘 0.9。** 上传论文同时描述了逐轮衰减和余弦安排，完整作者训练代码未公开；本实现明确记录实际采用的调度器，不能将其宣称为已确认的作者完整调度细节。
3. 二值 DD-mask 上采样使用 nearest，保留二值含义；没有沿用 FlexDepth 代码中会产生软边界的 bilinear 掩码插值。
4. auto-mask 在完整 ND 表达式外部，屏蔽 identity 分支选中的像素；没有把 identity 分支误差当作重投影残差去训练 Var。
5. 仅支持原仓库已定义膨胀率的输入大小 `192×640` 或 `320×1024`；本次文档主实验为前者。
6. 当前训练限定原仓库已有的 KITTI 数据加载器、独立 ResNet PoseNet、相邻三帧和原三尺度；未实现的组合直接报错。

这些选择应随实验结果一起报告。它们不包含任何前一轮拟议新方法组件。

## 4. 修改的目标文件、具体变化与作用

| 文件 | 操作 | 具体修改 | 作用 |
|---|---|---|---|
| `train.py` | 修改 | 增加统一 seed 和 opt-in DSITrainer 选择 | 保留入口及原有环境清理逻辑；原命令走原 Trainer |
| `options.py` | 修改 | 增加两阶段、检查点、固定掩码、调度器、恢复与评估参数 | 所有新配置可由命令指定并保存 |
| `evaluate_depth.py` | 修改 | 正确传递 PNG/JPG、CPU/设备支持、M/RM、可选一次 profiling、完整计数校验和 JSON/ROI 导出 | 消除评估入口的格式及协议陷阱；产生可比较结果 |
| `dsi_trainer.py` | 新增 | 原始 Trainer 的 DSI 子类；阶段切换、重置、冻结、ND/GDS、更新、恢复 | 训练框架主体，避免大改原 `trainer.py` |
| `dsi_utils.py` | 新增 | DD-mask、ND、GDS、掩码缩放、更新时机和哈希 | 可独立验证的数学与辅助函数 |
| `networks/variance_decoder.py` | 新增 | 训练用 Var 解码器 | 配合现有 ResNet-18 构造 sigma 图 |
| `evaluation_utils.py` | 新增 | Eigen 裁剪、M/RM、指标、哈希和汇总 | 让协议可审计并支持数值测试 |
| `scripts/check_dsi_setup.py` | 新增 | 依赖和所有训练/验证相邻帧路径检查 | 正式训练前发现环境/数据问题 |
| `scripts/compare_dsi_results.py` | 新增 | 检查协议、改进幅度、辅助指标和动态 ROI | 自动判断结果是否满足预设目标 |
| `tests/test_dsi.py` | 新增 | 损失、形状、梯度、冻结基础及评估判定边界测试 | 防止隐蔽实现错误 |
| `tests/smoke_dsi_pipeline.py` | 新增 | 合成数据的真实网络训练、切换、冻结和恢复 | 验证整条训练路径可执行 |
| `tests/smoke_evaluation.py` | 新增 | 外部视差、PNG 检查点、尾 batch、M/RM 和 ROI | 验证实际评估路径 |
| `docs/train.md`, `docs/eval.md`, 本手册 | 新增 | 按 FlexDepth 文档组织的完整操作与验收说明 | 形成可重复的实验流程 |

`trainer.py`、`layers.py`、原深度编码器/解码器、数据加载器、`checkpoint_utils.py` 和预训练子项目均未修改；不移动原目录，不引入新的模型目录层级。

## 5. 安装补丁

交付包包含 `apply_litemono_dsi.py` 和 `payload.zip`，应保持在同一目录。推荐应用到你已跑通的仓库；先创建本地实验分支：

```bash
cd /absolute/path/to/Mono
git switch -c experiment/litemono-dsi

python /absolute/path/to/delivery/apply_litemono_dsi.py \
    --repo "$PWD" --dry-run

python /absolute/path/to/delivery/apply_litemono_dsi.py \
    --repo "$PWD"

git diff --stat
```

安装器先检查全部待修改原文件的 SHA-256，再执行写入；内容冲突时不会覆盖本机改动，也不会只应用一半。相同补丁再次执行为幂等操作。
原文件备份在仓库旁边的 `Mono_dsi_backup_*` 目录，控制台会打印准确位置。新文件清单记录在备份目录的 `installed_files.json`。写入异常时安装器会恢复已改文件。

如果报告原文件与基准不符，保留本机版本，在固定提交的新工作目录应用后，用差异审阅合并本机自定义改动；不要直接覆盖已跑通的工作目录。可用：

```bash
git worktree add --detach ../Mono-dsi-base 128f3b041ae87bf6573a414ed4c12cff55bc8b9d
python /absolute/path/to/delivery/apply_litemono_dsi.py --repo ../Mono-dsi-base
```

交付也附带修改后源码快照，便于查阅。快照不包含数据、预训练权重、运行检查点、`.git` 及原仓库较大的 `gt_depths.npz`；使用已有仓库应用补丁会保留你的原 GT，不需重新准备。

## 6. 正式操作顺序

1. 安装后按 [Training](train.md) 设置数据与 ImageNet 权重路径，先运行预检和数学测试。
2. 训练原始 Lite-Mono baseline；单独保存 seed、命令、日志和最终权重。
3. 运行 Lite-Mono+DSI 的一条命令两阶段训练，或明确选择早晚参考进行分阶段训练。
4. 按 [Evaluation](eval.md) 对最终检查点运行同协议 M 评估，得到两份 JSON。
5. 执行结果比较脚本。若有独立动态标注，再运行 ROI 评估和动态验收。
6. 增加训练遍历轮数对照；重要结论用多个 seed 重复。

训练期间可以使用 TensorBoard：

```bash
tensorboard --logdir ./logs
```

检查最终权重可直接供原有 `test_simple.py` 使用；该脚本及其图像输出结构没有改动。

## 7. 验证标准与是否达到期望

验收分开判断，不把“代码运行成功”替代“精度提升”：

| 层级 | 标准 | 本次交付时状态 |
|---|---|---|
| 数学实现 | 固定阈值、ND 公式/掩码梯度、GDS 方向/倍率、M/RM、batch 边界 | 已检查，具体记录见交付验证报告 |
| 训练机制 | 真正执行前后向、学生重置、冻结参数及 BN、参考更新、恢复后继续训练 | CPU 合成数据已检查 |
| 评估可执行性 | PNG 权重推理、尾 batch、M/RM、外部视差及 ROI 对齐 | CPU 合成数据已检查 |
| 本机整体性能 | Abs Rel 相对下降至少 2%，Sq Rel/RMSE 恶化≤1%，a1 下降≤0.002 | 待正式训练评估 |
| 动态区域性能 | 独立运动标注下 Abs Rel 相对下降至少 5% | 待独立标注评估 |
| 多 seed 稳定性 | 配对重复训练并报告均值/标准差，不能只挑最佳结果 | 待重复实验 |

整体和动态阈值是本次预设验收目标，并非 DSI 对任意环境的性能承诺。详细判断命令、论文数值参考和失败排查见 `docs/eval.md`。

## 8. 恢复原版本

若补丁应用于独立实验分支，可以切回原分支后按 `git status` 检查未跟踪新增文件；Git 切换分支不会自动删除未跟踪文件，需对照安装器的新增清单处理。
也可以将安装器备份中的三个原文件复制回去，并仅删除 `installed_files.json` 标记为原先不存在的新增文件。操作前保留当前实验日志与权重，避免误删结果。
