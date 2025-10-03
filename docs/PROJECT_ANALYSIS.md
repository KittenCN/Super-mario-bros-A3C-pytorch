# 超级马里奥兄弟 A3C 项目工作原理分析 | Working Principle Analysis

本文梳理项目模块划分、核心算法与训练/评估流程，帮助读者快速理解系统全貌。<br>This document explains the module layout, core algorithms, and training/evaluation flows to provide a complete picture of the system.

## 项目概览 | Project Overview
- 采用 PyTorch 2.1+，将 A3C 与 IMPALA/V-trace 思路结合，训练智能体通关 `SuperMarioBros`。<br>Built on PyTorch 2.1+, combining A3C with IMPALA/V-trace ideas to train an agent for `SuperMarioBros`.
- 顶层入口包括 `train.py`、`test.py` 与 `scripts/optuna_search.py`，分别覆盖训练、评估与超参搜索。<br>Top-level entry points (`train.py`, `test.py`, `scripts/optuna_search.py`) cover training, evaluation, and hyper-parameter search.
- 业务逻辑模块化布局于 `src/`：环境工厂、模型、算法、缓冲区与观测工具相互解耦。<br>Business logic is modularised under `src/` with dedicated packages for environments, models, algorithms, buffers, and monitoring utilities.

## 环境子系统 | Environment Subsystem (`src/envs/`)
- `MarioEnvConfig` 与 `MarioVectorEnvConfig` 描述单环境/向量环境的参数（关卡、动作集、帧跳、随机舞台等）。<br>`MarioEnvConfig` and `MarioVectorEnvConfig` define per-env and vector-env parameters (stage, action set, frame skip, randomised schedule, etc.).
- `create_vector_env` 支持同步或异步 `Gymnasium` 向量环境，串联帧跳、灰度化、`Resize(84×84)`、`FrameStack`、奖励塑形与录像封装。<br>`create_vector_env` builds sync or async Gymnasium vector environments chaining frame skip, grayscale, `Resize(84×84)`, `FrameStack`, reward shaping, and optional video recording.
- 通过 `ProgressInfoWrapper` 输出总进度/终止信息，`MarioRewardWrapper` 注入旗帜奖励及失败惩罚，确保奖励尺度一致；2025-10-04 起其 `get_diagnostics()` 也会缓存 `dx/scale/raw` 并驱动 `env_progress_dx`、`stagnation_envs` 等推进指标。<br>`ProgressInfoWrapper` exposes progress/termination info while `MarioRewardWrapper` now surfaces `dx/scale/raw` diagnostics that feed new training metrics such as `env_progress_dx` and `stagnation_envs`.
- 针对 `nes_py` 的溢出问题增加运行时补丁与诊断日志，结合文件锁序列化重型初始化，并在 `MARIO_SHAPING_DEBUG=1` 下限频输出 `[reward][warn] dx_missed ...` 指示缺失的 `x_pos` 数据。<br>Runtime patches and diagnostics mitigate `nes_py` overflow issues, serialise heavy initialisation via file locks, and emit throttled `[reward][warn] dx_missed ...` messages (with `MARIO_SHAPING_DEBUG=1`) when `x_pos` telemetry is missing.

## 模型结构 | Model Architecture (`src/models/`)
- `MarioActorCritic` 采用 IMPALA 残差卷积骨干，输出特征再接入序列模块。<br>`MarioActorCritic` uses an IMPALA residual convolutional backbone whose features feed into a sequence module.
- 支持 `GRU`、`LSTM`、`TransformerEncoder` 或无循环结构，默认隐藏维度 512，可选 NoisyLinear 提升探索。<br>Supports `GRU`, `LSTM`, `TransformerEncoder`, or a feed-forward head with a default hidden size of 512 and optional NoisyLinear exploration.
- 模型输出 `ModelOutput`，包含策略 logits、价值估计、隐藏状态与中间特征，以便训练/评估复用。<br>Outputs are wrapped in `ModelOutput` with policy logits, value estimates, hidden state, and intermediate features shared across training and evaluation.
- 相关辅助模块位于 `layers.py` 与 `sequence.py`，提供残差块、注意力位置编码等组件。<br>Auxiliary modules (`layers.py`, `sequence.py`) supply residual blocks, positional encodings, and other building blocks.

## 算法与数据结构 | Algorithms & Data Structures
- `src/algorithms/vtrace.py` 实现 IMPALA V-trace 目标，融合行为策略与目标策略 log-prob 校正优势。<br>`src/algorithms/vtrace.py` implements IMPALA V-trace targets, blending behaviour and target policy log-probs for corrected advantages.
- `src/utils/rollout.py` 的 `RolloutBuffer` 在主机或 GPU 上缓存 `(T, N)` 轨迹，便于批量取样与截断回传，并提供 `_scalar()`/`NUMERIC_TYPES` 辅助将 `numpy`/`array` 数据统一拉平成标量，保障指标与自适应逻辑兼容 fc_emulator 报文。<br>The `RolloutBuffer` stores `(T, N)` trajectories and now exposes `_scalar()` helpers to normalise `numpy` payloads, keeping metrics and adaptive logic compatible with fc-emulator style payloads.
- `src/utils/replay.py` 提供 `PrioritizedReplay`，按优势绝对值决定权重，实现混合 on/off-policy 更新。<br>`src/utils/replay.py` offers `PrioritizedReplay`, weighting samples by absolute advantage to support hybrid on/off-policy updates.
- `src/utils/schedule.py` 定义 `CosineWithWarmup` 学习率策略，`src/utils/logger.py` 封装 TensorBoard/W&B 记录。<br>`src/utils/schedule.py` defines the `CosineWithWarmup` scheduler and `src/utils/logger.py` wraps TensorBoard/W&B logging.

## 训练流程 | Training Loop (`train.py` → `run_training`)
1. **配置解析**：`parse_args` 读取训练/环境参数，`build_training_config` 组装 `TrainingConfig`。<br>**Configuration**: `parse_args` collects CLI parameters and `build_training_config` assembles the `TrainingConfig`.
2. **初始化**：创建向量环境获取空间信息，构造 `MarioActorCritic`，可选 `torch.compile` 与 AMP，配置 `AdamW` + 余弦调度。<br>**Initialisation**: create vector envs to inspect spaces, instantiate `MarioActorCritic`, optionally enable `torch.compile` and AMP, and configure `AdamW` with cosine scheduling.
3. **Rollout 收集**：`RolloutBuffer` 以 `num_steps × num_envs` 批量与环境交互，episode 结束时重置隐藏状态并累计奖励。<br>**Rollout collection**: the `RolloutBuffer` gathers `num_steps × num_envs` interactions, resets hidden states when episodes finish, and aggregates returns.
4. **目标计算**：`compute_returns` 使用 V-trace/GAE 生成价值目标与优势；启用 PER 时将样本推入优先缓存。<br>**Target computation**: `compute_returns` derives value targets and advantages via V-trace/GAE; with PER enabled, samples are pushed into the priority buffer.
5. **反向传播**：合成策略、价值、熵损失并应用梯度裁剪/累积；AMP 缩放、学习率调度与权重衰减同步执行。<br>**Backpropagation**: combine policy, value, and entropy losses, applying gradient clipping/accumulation alongside AMP scaling, schedulers, and weight decay.
6. **持久化与日志**：定期写入 TensorBoard/W&B 与结构化 JSONL，新增字段涵盖 `env_progress_dx`、`stagnation_envs`、`adaptive_ratio_fallback`、`wrapper_distance_weight` 等诊断，再保存 checkpoint 与运行配置 JSON 供恢复使用。<br>**Persistence & logging**: periodically log to TensorBoard/W&B and a structured JSONL feed with metrics like `env_progress_dx`, `stagnation_envs`, `adaptive_ratio_fallback`, and `wrapper_distance_weight`, then save checkpoints and run configs for resumption.
7. **返回指标**：`run_training` 输出 `avg_return` 等统计，可被 Optuna 及稳态训练脚本复用。<br>**Return metrics**: `run_training` returns `avg_return` and related statistics reused by Optuna sweeps and the stable launcher scripts.

## 评估流程 | Evaluation Flow (`test.py`)
- CLI 解析 checkpoint、关卡与渲染选项，构造单环境并启用录像。<br>CLI parameters select checkpoints, stages, and rendering options, constructing a single env with optional video recording.
- 使用贪心策略执行，维持 GRU/LSTM/Transformer 隐藏状态，记录奖励与闯关信息。<br>Runs a greedy policy, maintaining GRU/LSTM/Transformer hidden states while logging rewards and stage completion.
- 支持实时渲染与 `RecordVideo` 导出 MP4，默认保存至 `output/eval/`。<br>Supports real-time rendering and `RecordVideo` MP4 exports saved under `output/eval/` by default.

## 自动化与工具 | Automation & Tooling
- `scripts/optuna_search.py` 通过短程训练评估超参组合，输出最佳 `avg_return`。<br>`scripts/optuna_search.py` evaluates hyper-parameter candidates via short training runs and returns the best `avg_return`.
- `scripts/train_stable_sync.sh` / `scripts/train_stable_async.sh` 提供稳态训练入口：前者针对同步调试（num_envs=2），后者默认启用 `--async-env --confirm-async --overlap-collect --parent-prewarm` 并输出独立日志/保存路径，均支持 `--dry-run` 与环境变量覆盖。<br>`scripts/train_stable_sync.sh` and `scripts/train_stable_async.sh` deliver stable launchers for sync and async regimes, supporting `--dry-run` and environment overrides while defaulting to isolated log/save directories.
- `requirements.txt`、`environment.yml`、`Dockerfile` 覆盖 pip、conda、Docker 三种环境搭建方式。<br>`requirements.txt`, `environment.yml`, and the `Dockerfile` cover pip, conda, and Docker workflows.
- `src/utils/monitor.py`（与 CLI 开关配合）提供 CPU/GPU 监控，日志写入 TensorBoard 或 JSONL。<br>`src/utils/monitor.py`, together with CLI switches, provides CPU/GPU monitoring with outputs to TensorBoard or JSONL.
- HeartbeatReporter 支持结构化 JSONL 输出（`--heartbeat-path`），训练同时生成 `metrics/latest.parquet` 供数据分析。<br>HeartbeatReporter now emits JSONL heartbeats (`--heartbeat-path`) and each logging step refreshes `metrics/latest.parquet` for downstream analytics.

## 数据流概览 | Data Flow Summary
1. 环境返回 `(num_envs, stack, 84, 84)` 观测 -> 模型前向获取 logits/value/隐藏状态。<br>Environment yields `(num_envs, stack, 84, 84)` observations → model forward pass produces logits, values, and hidden states.
2. RolloutBuffer 累积轨迹 -> V-trace 生成目标 -> 梯度更新策略与价值头。<br>RolloutBuffer aggregates trajectories → V-trace computes targets → gradients update policy/value heads.
3. PER（如启用）额外采样高 TD-error 轨迹，混合 off-policy 更新。<br>PER (when enabled) samples high TD-error trajectories for additional off-policy updates.
4. 指标写入 TensorBoard/W&B，checkpoint 与配置同步保存供恢复。<br>Metrics stream to TensorBoard/W&B while checkpoints and configs are saved for resumption.

## 依赖与运行注意事项 | Dependencies & Runtime Notes
- 推荐 Python 3.10/3.11、PyTorch ≥ 2.1、Gymnasium 0.28，并安装 `gymnasium-super-mario-bros` 或兼容版本。<br>Recommend Python 3.10/3.11, PyTorch ≥ 2.1, Gymnasium 0.28, and `gymnasium-super-mario-bros` or a compatible variant.
- 系统需预装 FFmpeg 以录制视频；Docker 镜像已内置。<br>FFmpeg must be installed for video capture; the Docker image ships with it.
- 针对 Python 3.12 + NumPy 2.x 的 `nes_py` 溢出问题，请参考 `docs/ENV_DEBUGGING_REPORT.md`。<br>For `nes_py` overflow issues on Python 3.12 + NumPy 2.x, consult `docs/ENV_DEBUGGING_REPORT.md`.

## 最近修复与改进 | Recent Fixes & Improvements
- ✅ **GAE fallback bootstrap**：`compute_returns` 现正确使用 `bootstrap_value`，并由 `tests/test_training_schedule.py` 覆盖 `--no-vtrace` 路径。<br>**GAE fallback bootstrap**: `compute_returns` now honours the bootstrap value, with regression coverage in `tests/test_training_schedule.py`.
- ✅ **余弦调度对齐 update**：`CosineWithWarmup.total_steps` 基于有效 update 计算并在 JSONL/TB 中记录 `learning_rate`，确保调度曲线可观测。<br>**Cosine scheduler alignment**: `CosineWithWarmup.total_steps` tracks effective updates and logs `learning_rate` to JSONL/TensorBoard for visibility.
- ✅ **环境构建可取消**：`call_with_timeout` 支持 `cancel_event`，配合 `create_vector_env(..., cancel_event=...)` 判停，超时会立即发出取消信号。<br>**Cancellable env construction**: `call_with_timeout` propagates a `cancel_event`, and `create_vector_env(..., cancel_event=...)` honours cancellation when timeouts occur.
- ✅ **metrics JSONL 加锁**：训练主线程与监控线程共享写锁，避免并发写入交错。<br>**Locked metrics JSONL**: a shared write-lock prevents interleaved log lines between the training loop and monitor thread.
- ✅ **PER push 批量化**：`PrioritizedReplay.push` 使用向量化索引批量写入，显著降低 CPU `copy_` 开销。<br>**Vectorised PER push**: `PrioritizedReplay.push` now uses batched index writes, cutting CPU `copy_` overhead.
- ✅ **PER GPU 采样原型**：`PrioritizedReplay` 新增 `use_gpu_sampler` 开关，使用 torch `searchsorted` 在 GPU/CPU 上统一采样并保持统计。<br>**PER GPU sampler**: `PrioritizedReplay` exposes a `use_gpu_sampler` flag leveraging torch `searchsorted` for consistent GPU/CPU sampling.
- ✅ **global_step 回填流水线**：回填脚本可读取 checkpoint `global_update`，并通过 GitHub Actions (`backfill-global-step`) 手动 dry-run 校验。<br>**Backfill pipeline**: the script now reads checkpoint `global_update`, with a `backfill-global-step` GitHub Action for manual dry-run validation.
- ✅ **Heartbeat & 指标快照**：默认在 `run_dir/heartbeat.jsonl` 写入心跳状态，并同步更新 `metrics/latest.parquet` 方便可视化。<br>**Heartbeat & metrics snapshot**: heartbeats write to `run_dir/heartbeat.jsonl` while every log refreshes `metrics/latest.parquet` for visualisation.
- ✅ **学习率缩放自适应**：`AdaptiveScheduler` 新增 `lr_scale` 通道，结合正向推进比自动调整优化器学习率。<br>**Adaptive LR scaling**: `AdaptiveScheduler` now manages an `lr_scale` channel that adapts the optimiser learning rate from progress ratios.
- ✅ **PER GPU 采样原型**：`PrioritizedReplay` 新增 `use_gpu_sampler` 开关，使用 torch `searchsorted` 在 GPU/CPU 上统一采样并保持统计。<br>**PER GPU sampler**: `PrioritizedReplay` exposes a `use_gpu_sampler` flag leveraging torch `searchsorted` for consistent GPU/CPU sampling.
- ✅ **global_step 回填流水线**：回填脚本可读取 checkpoint `global_update`，并通过 GitHub Actions (`backfill-global-step`) 手动 dry-run 校验。<br>**Backfill pipeline**: the script now reads checkpoint `global_update`, with a `backfill-global-step` GitHub Action for manual dry-run validation.
