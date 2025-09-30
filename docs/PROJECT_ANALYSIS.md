# 超级马里奥兄弟 A3C 项目工作原理分析 | Working Principle Analysis

本文梳理项目模块划分、核心算法与训练/评估流程，帮助读者快速理解系统全貌。<br>This document explains the module layout, core algorithms, and training/evaluation flows to provide a complete picture of the system.

## 项目概览 | Project Overview
- 采用 PyTorch 2.1+，将 A3C 与 IMPALA/V-trace 思路结合，训练智能体通关 `SuperMarioBros`。<br>Built on PyTorch 2.1+, combining A3C with IMPALA/V-trace ideas to train an agent for `SuperMarioBros`.
- 顶层入口包括 `train.py`、`test.py` 与 `scripts/optuna_search.py`，分别覆盖训练、评估与超参搜索。<br>Top-level entry points (`train.py`, `test.py`, `scripts/optuna_search.py`) cover training, evaluation, and hyper-parameter search.
- 业务逻辑模块化布局于 `src/`：环境工厂、模型、算法、缓冲区与观测工具相互解耦。<br>Business logic is modularised under `src/` with dedicated packages for environments, models, algorithms, buffers, and monitoring utilities.

## 环境子系统 | Environment Subsystem (`src/envs/`)
- `MarioEnvConfig` 与 `MarioVectorEnvConfig` 描述单环境/向量环境的参数（关卡、动作集、帧跳、随机舞台等）。<br>`MarioEnvConfig` and `MarioVectorEnvConfig` define per-env and vector-env parameters (stage, action set, frame skip, randomised schedule, etc.).
- `create_vector_env` 支持同步或异步 `Gymnasium` 向量环境，串联帧跳、灰度化、`Resize(84×84)`、`FrameStack`、奖励塑形与录像封装。<br>`create_vector_env` builds sync or async Gymnasium vector environments chaining frame skip, grayscale, `Resize(84×84)`, `FrameStack`, reward shaping, and optional video recording.
- 通过 `ProgressInfoWrapper` 输出总进度/终止信息，`MarioRewardWrapper` 注入旗帜奖励及失败惩罚，确保奖励尺度一致。<br>`ProgressInfoWrapper` exposes progress/termination info and `MarioRewardWrapper` injects flag rewards and failure penalties to keep reward scales consistent.
- 针对 `nes_py` 的溢出问题增加运行时补丁与诊断日志，结合文件锁序列化重型初始化。<br>Runtime patches and diagnostics mitigate `nes_py` overflow issues, while file locks serialise heavy initialisation.

## 模型结构 | Model Architecture (`src/models/`)
- `MarioActorCritic` 采用 IMPALA 残差卷积骨干，输出特征再接入序列模块。<br>`MarioActorCritic` uses an IMPALA residual convolutional backbone whose features feed into a sequence module.
- 支持 `GRU`、`LSTM`、`TransformerEncoder` 或无循环结构，默认隐藏维度 512，可选 NoisyLinear 提升探索。<br>Supports `GRU`, `LSTM`, `TransformerEncoder`, or a feed-forward head with a default hidden size of 512 and optional NoisyLinear exploration.
- 模型输出 `ModelOutput`，包含策略 logits、价值估计、隐藏状态与中间特征，以便训练/评估复用。<br>Outputs are wrapped in `ModelOutput` with policy logits, value estimates, hidden state, and intermediate features shared across training and evaluation.
- 相关辅助模块位于 `layers.py` 与 `sequence.py`，提供残差块、注意力位置编码等组件。<br>Auxiliary modules (`layers.py`, `sequence.py`) supply residual blocks, positional encodings, and other building blocks.

## 算法与数据结构 | Algorithms & Data Structures
- `src/algorithms/vtrace.py` 实现 IMPALA V-trace 目标，融合行为策略与目标策略 log-prob 校正优势。<br>`src/algorithms/vtrace.py` implements IMPALA V-trace targets, blending behaviour and target policy log-probs for corrected advantages.
- `src/utils/rollout.py` 的 `RolloutBuffer` 在主机或 GPU 上缓存 `(T, N)` 轨迹，便于批量取样与截断回传。<br>The `RolloutBuffer` in `src/utils/rollout.py` stores `(T, N)` trajectories on host or GPU, enabling batched sampling and truncated BPTT.
- `src/utils/replay.py` 提供 `PrioritizedReplay`，按优势绝对值决定权重，实现混合 on/off-policy 更新。<br>`src/utils/replay.py` offers `PrioritizedReplay`, weighting samples by absolute advantage to support hybrid on/off-policy updates.
- `src/utils/schedule.py` 定义 `CosineWithWarmup` 学习率策略，`src/utils/logger.py` 封装 TensorBoard/W&B 记录。<br>`src/utils/schedule.py` defines the `CosineWithWarmup` scheduler and `src/utils/logger.py` wraps TensorBoard/W&B logging.

## 训练流程 | Training Loop (`train.py` → `run_training`)
1. **配置解析**：`parse_args` 读取训练/环境参数，`build_training_config` 组装 `TrainingConfig`。<br>**Configuration**: `parse_args` collects CLI parameters and `build_training_config` assembles the `TrainingConfig`.
2. **初始化**：创建向量环境获取空间信息，构造 `MarioActorCritic`，可选 `torch.compile` 与 AMP，配置 `AdamW` + 余弦调度。<br>**Initialisation**: create vector envs to inspect spaces, instantiate `MarioActorCritic`, optionally enable `torch.compile` and AMP, and configure `AdamW` with cosine scheduling.
3. **Rollout 收集**：`RolloutBuffer` 以 `num_steps × num_envs` 批量与环境交互，episode 结束时重置隐藏状态并累计奖励。<br>**Rollout collection**: the `RolloutBuffer` gathers `num_steps × num_envs` interactions, resets hidden states when episodes finish, and aggregates returns.
4. **目标计算**：`compute_returns` 使用 V-trace/GAE 生成价值目标与优势；启用 PER 时将样本推入优先缓存。<br>**Target computation**: `compute_returns` derives value targets and advantages via V-trace/GAE; with PER enabled, samples are pushed into the priority buffer.
5. **反向传播**：合成策略、价值、熵损失并应用梯度裁剪/累积；AMP 缩放、学习率调度与权重衰减同步执行。<br>**Backpropagation**: combine policy, value, and entropy losses, applying gradient clipping/accumulation alongside AMP scaling, schedulers, and weight decay.
6. **持久化与日志**：定期写入 TensorBoard/W&B、保存 checkpoint 及运行配置 JSON，供恢复或评估使用。<br>**Persistence & logging**: periodically log to TensorBoard/W&B, save checkpoints, and export run configuration JSON for resumption and evaluation.
7. **返回指标**：`run_training` 输出 `avg_return` 等统计，供 Optuna 等调用。<br>**Return metrics**: `run_training` returns `avg_return` and related statistics for downstream tools like Optuna.

## 评估流程 | Evaluation Flow (`test.py`)
- CLI 解析 checkpoint、关卡与渲染选项，构造单环境并启用录像。<br>CLI parameters select checkpoints, stages, and rendering options, constructing a single env with optional video recording.
- 使用贪心策略执行，维持 GRU/LSTM/Transformer 隐藏状态，记录奖励与闯关信息。<br>Runs a greedy policy, maintaining GRU/LSTM/Transformer hidden states while logging rewards and stage completion.
- 支持实时渲染与 `RecordVideo` 导出 MP4，默认保存至 `output/eval/`。<br>Supports real-time rendering and `RecordVideo` MP4 exports saved under `output/eval/` by default.

## 自动化与工具 | Automation & Tooling
- `scripts/optuna_search.py` 通过短程训练评估超参组合，输出最佳 `avg_return`。<br>`scripts/optuna_search.py` evaluates hyper-parameter candidates via short training runs and returns the best `avg_return`.
- `requirements.txt`、`environment.yml`、`Dockerfile` 覆盖 pip、conda、Docker 三种环境搭建方式。<br>`requirements.txt`, `environment.yml`, and the `Dockerfile` cover pip, conda, and Docker workflows.
- `src/utils/monitor.py`（与 CLI 开关配合）提供 CPU/GPU 监控，日志写入 TensorBoard 或 JSONL。<br>`src/utils/monitor.py`, together with CLI switches, provides CPU/GPU monitoring with outputs to TensorBoard or JSONL.

## 数据流概览 | Data Flow Summary
1. 环境返回 `(num_envs, stack, 84, 84)` 观测 -> 模型前向获取 logits/value/隐藏状态。<br>Environment yields `(num_envs, stack, 84, 84)` observations → model forward pass produces logits, values, and hidden states.
2. RolloutBuffer 累积轨迹 -> V-trace 生成目标 -> 梯度更新策略与价值头。<br>RolloutBuffer aggregates trajectories → V-trace computes targets → gradients update policy/value heads.
3. PER（如启用）额外采样高 TD-error 轨迹，混合 off-policy 更新。<br>PER (when enabled) samples high TD-error trajectories for additional off-policy updates.
4. 指标写入 TensorBoard/W&B，checkpoint 与配置同步保存供恢复。<br>Metrics stream to TensorBoard/W&B while checkpoints and configs are saved for resumption.

## 依赖与运行注意事项 | Dependencies & Runtime Notes
- 推荐 Python 3.10/3.11、PyTorch ≥ 2.1、Gymnasium 0.28，并安装 `gymnasium-super-mario-bros` 或兼容版本。<br>Recommend Python 3.10/3.11, PyTorch ≥ 2.1, Gymnasium 0.28, and `gymnasium-super-mario-bros` or a compatible variant.
- 系统需预装 FFmpeg 以录制视频；Docker 镜像已内置。<br>FFmpeg must be installed for video capture; the Docker image ships with it.
- 针对 Python 3.12 + NumPy 2.x 的 `nes_py` 溢出问题，请参考 `docs/ENV_DEBUGGING_REPORT.md`。<br>For `nes_py` overflow issues on Python 3.12 + NumPy 2.x, consult `docs/ENV_DEBUGGING_REPORT.md`.
