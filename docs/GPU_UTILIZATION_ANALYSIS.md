# Super Mario Bros A3C GPU 利用率分析 | GPU Utilisation Analysis

本报告归纳训练流程中的各组件职责、当前 GPU 占用表现、潜在瓶颈与优化方向，帮助团队平衡 CPU/NES 仿真与 GPU 计算。<br>This report summarises component responsibilities, current GPU utilisation behaviour, observed bottlenecks, and recommended optimisations so the team can balance CPU/NES emulation with GPU compute.

## 模块职责 | Module Responsibilities
- `train.py`：主训练循环，负责环境交互、A3C/V-trace 更新、日志与 checkpoint。<br>`train.py`: orchestrates environment interaction, A3C/V-trace updates, logging, and checkpointing.
- `src/envs/mario.py`：构建 Gymnasium 向量环境，封装帧跳、灰度、缩放、堆叠与奖励塑形，可同步/异步运行。<br>`src/envs/mario.py`: builds Gymnasium vector environments with frame skip, grayscale, resize, stacking, and shaped rewards, supporting async or sync execution.
- `src/models/policy.py`：IMPALA 风格卷积骨干 + GRU/LSTM/Transformer 序列模块，实现 Actor-Critic 双头。<br>`src/models/policy.py`: contains the IMPALA-style convolutional backbone plus GRU/LSTM/Transformer sequence modules for the actor–critic heads.
- `src/utils/rollout.py`：`RolloutBuffer` 管理 `(T, N)` 轨迹，可选 pinned memory，加速采样与训练切换。<br>`src/utils/rollout.py`: the `RolloutBuffer` manages `(T, N)` trajectories with optional pinned memory to streamline sampling-to-training transitions.
- `src/utils/replay.py`：实现优先经验回放（PER），支持混合 on/off-policy 更新。<br>`src/utils/replay.py`: implements prioritised experience replay (PER) for hybrid on/off-policy updates.
- `src/algorithms/vtrace.py`：IMPALA V-trace 目标，结合行为/目标策略 log-prob 校正优势。<br>`src/algorithms/vtrace.py`: provides the IMPALA V-trace target to mix behaviour and target policies for corrected advantages.

## 当前瓶颈观察 | Observed Bottlenecks
- **环境与数据流**：NES 仿真消耗 CPU，多进程异步 `env.step` 仍受 Python IPC 限制；GPU 需等待环境返回观测。<br>**Environment & data flow**: NES emulation is CPU-heavy; even with async workers, Python IPC limits throughput so the GPU waits for observations.
- **张量搬运**：每轮 rollout 中观测经历 env→GPU→CPU 再回 GPU 的往返复制，`obs.clone()` 等操作增加同步点。<br>**Tensor transfers**: observations bounce env→GPU→CPU→GPU within a single rollout, and calls like `obs.clone()` introduce extra synchronisation points.
- **计算批量偏小**：模型仅有中等规模，默认 `num_envs=8、rollout=64` 时反向图较小，AMP/compile 仍难以拉满 GPU。<br>**Small compute batches**: the model is mid-sized; with default `num_envs=8` and `rollout=64`, each update is lightweight, so even AMP/compile cannot fully saturate the GPU.
- **附加同步操作**：频繁调用 `nvidia-smi`、逐步构造 done mask 以及线程调度均会插入 GPU 空转。<br>**Ancillary synchronisation**: frequent `nvidia-smi` calls, iterative done-mask construction, and thread scheduling overhead add idle periods on the GPU.

## GPU 利用率不足的根因 | Root Causes of Low GPU Utilisation
1. **CPU 侧瓶颈**：多进程环境与 Python 层逻辑同步阻塞，限制可并行的环境步数。<br>**CPU-bound pipeline**: multi-process environments and Python logic block synchronously, capping the number of concurrent steps.
2. **频繁的主机/GPU 往返**：RolloutBuffer 与 PER 在 CPU/GPU 之间多次复制张量，导致流同步。<br>**Excessive host↔GPU shuttling**: the RolloutBuffer and PER repeatedly copy tensors across devices, forcing stream synchronisation.
3. **训练批次太小**：单次反向只覆盖一批 rollout，无法充分利用 GPU 并行度。<br>**Batch size too small**: each backward pass covers just one rollout batch, underutilising GPU parallelism.
4. **监控引入的阻塞**：外部命令或统计调用在默认 CUDA 流上执行，拉长等待时间。<br>**Monitoring-induced stalls**: external commands and metrics queries run on the default CUDA stream, lengthening wait times.

## 优化建议 | Recommended Optimisations

### 优先执行 | High-priority Actions
- 重构 Rollout 数据路径，使观测先写入 CPU Buffer，再统一迁移到 GPU，或保持整段 rollout 驻留 GPU 并利用异步流。<br>Restructure the rollout data path so observations land in a CPU buffer before a single transfer, or keep entire rollouts on the GPU using auxiliary streams.
- 提高环境吞吐：结合 `AsyncVectorEnv` 的 `shared_memory`/自定义 worker 或尝试 `envpool`，并为环境进程预留充足 CPU。<br>Raise environment throughput: tune `AsyncVectorEnv` shared-memory/worker options or experiment with `envpool`, while reserving enough CPU for env workers.
- 引入梯度累积或扩大 `num_envs`，增大每次反向的有效批量，提升 GPU 持续工作时间。<br>Use gradient accumulation or larger `num_envs` to increase the effective batch per backward pass and keep the GPU busy for longer.

### 结构改进 | Structural Tweaks
- 将策略采样与价值评估放到不同 CUDA stream，减轻默认流阻塞。<br>Move policy sampling and value evaluation onto separate CUDA streams to reduce default-stream blocking.
- 使用 `torch.compile(mode="reduce-overhead")` 固定输入形状，减少图构建时间。<br>Adopt `torch.compile(mode="reduce-overhead")` with fixed shapes to shrink graph-rebuild overhead.
- 在 PER 缓冲中保存 GPU 版样本，仅同步必要标量，避免反复 `.cpu()`。<br>Store GPU-resident samples inside PER and only sync essential scalars to avoid repeated `.cpu()` calls.

### 监控策略 | Monitoring Strategy
- 通过 NVML Python API 或 `torch.cuda.utilization()` 采集指标，替代频繁的 `nvidia-smi` 子进程。<br>Collect metrics via the NVML Python API or `torch.cuda.utilization()` instead of repeatedly spawning `nvidia-smi` processes.
- 记录 `SPS`、`GPU util`、`env fps`、内存占用等指标，并与训练日志对齐。<br>Track `SPS`, `GPU util`, `env fps`, and memory usage, aligning them with training logs.
- 定期运行 PyTorch Profiler，量化 env.step、张量拷贝、反向传播所占时长。<br>Run the PyTorch Profiler periodically to quantify how much time env.step, tensor transfers, and backward passes consume.

## 后续演进建议 | Future Directions
- 探索批量化 RNN 推理或将策略推理拆分到独立线程，平衡 CPU/GPU 负载。<br>Explore batched RNN inference or offload policy inference to a dedicated thread to balance CPU/GPU load.
- 研究 `TensorFloat32`、FlashAttention 等加速手段，在保持稳定性的前提下提升 Transformer 计算密度。<br>Investigate TensorFloat32 and FlashAttention-style optimisations to raise Transformer compute density while preserving stability.
- 如扩展到多 GPU，可尝试 `DistributedDataParallel`、TorchRL 或参数服务器式缓冲区以解耦采样与训练。<br>When scaling across multiple GPUs, consider `DistributedDataParallel`, TorchRL, or parameter-server-style buffers to decouple sampling from training.
