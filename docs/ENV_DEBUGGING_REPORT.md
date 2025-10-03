# AsyncVectorEnv 调试报告 | AsyncVectorEnv Debugging Report

**记录日期 | Logged on**: 2025-09-29 ～ 2025-09-30

本文汇整在构建 `AsyncVectorEnv` 时遇到的阻塞、溢出等问题，以及针对 `nes_py` 的修复尝试、实验数据与后续建议，确保团队在仓库重置后也能快速上手。<br>This report captures the blocking and overflow issues observed while constructing `AsyncVectorEnv`, the runtime patches applied to `nes_py`, experiments that were run, and suggested next steps so the team can pick up the investigation after repository resets.

## 最新变更摘要 | Latest Update (2025-10-04)

## 问题概述 | Problem Summary

### 2025-10-03 Episode 步数计数缺失与超时截断不触发
问题现象:
1. 训练早期出现大量位移 (x_pos 增长) 但从未完成 episode (`episodes_completed` 维持 0)。
2. 配置 `--episode-timeout-steps` 与 `--stagnation-limit` 均不触发截断，`avg_return` 长期 0。
3. 调试字段显示 `per_env_episode_steps` 在每个 update 后又是 0。

根因:
向量环境在后续 step 返回 batched dict（包含整批 `x_pos` 数组）而非 list[dict]，原逻辑只在 list 分支中累加 `per_env_episode_steps[i] += 1` 并应用 timeout/stagnation；batched dict 分支缺失该累加与截断标记，导致步数永远不增长、截断条件无法满足。

修复措施:
1. 在 batched dict 分支为每个 env 累加 `per_env_episode_steps`。
2. 添加 timeout/stagnation 判断与 batch 级标记 (`timeout_truncate_batch`, `stagnation_truncate_batch`)。
3. 重新计算 `done = terminated | truncated` 以反映新截断。
4. 扩展 episode-end 调试：`MARIO_EPISODE_DEBUG=1` 时输出 `[train][episode-end-debug]`，支持 batch 标记原因解析。
5. 添加 `MARIO_EPISODE_STEPS_DEBUG=1` 每个 update 打印 `[train][episode-steps-debug]`，可观察 episode 步数累积与重置点。

验证结果:
脚本前进 + 超时参数下，episode 步数在单次 episode 内递增 (0→73) 并在达阈值后触发 `timeout_truncate_batch`，同时出现 episode-end-debug 日志与非零 return（包含塑形价值）。

使用建议:
- 快速自检：设置 `--episode-timeout-steps 80 --scripted-forward-frames 400` 并导出 `MARIO_EPISODE_STEPS_DEBUG=1`，若 1~2 个 episode 内未见 steps 增长或 episode-end-debug，则再检查 env wrappers。
- 若只想最小化输出，将 `MARIO_EPISODE_STEPS_DEBUG` 关闭，仅保留在异常时打开。

相关环境变量:
- `MARIO_EPISODE_DEBUG`: 打印 episode 结束详细原因。
- `MARIO_EPISODE_STEPS_DEBUG`: 打印当前 episode 累计步数与 max_x。
- 现象：父进程构建 `AsyncVectorEnv` 时偶发超时并回退到同步环境，同时输出 “Async vector env construction timed out…”。<br>Symptom: the parent process sporadically times out while building `AsyncVectorEnv`, prints “Async vector env construction timed out…”, and falls back to the synchronous env.
- 额外现象：在 Python 3.12 + NumPy 2.x 下，单进程运行 `gym_super_mario_bros.make(...)` 会触发 `nes_py` 的 `OverflowError: Python integer 1024 out of bounds for uint8`。<br>Additional symptom: on Python 3.12 + NumPy 2.x, a single-process `gym_super_mario_bros.make(...)` call in `nes_py` raises `OverflowError: Python integer 1024 out of bounds for uint8`.
- 诊断日志显示 worker 停在 `calling_mario_make` 处，有时崩溃（未打补丁），有时静默挂起（已打补丁），推测阻塞发生在 NES 原生初始化阶段。<br>Diagnostics show workers reaching `calling_mario_make` and then either crashing (without patches) or hanging silently (with patches), indicating the stall occurs during NES native initialisation.

## 诊断入口 | Instrumentation
- `src/envs/mario.py`：<br>`src/envs/mario.py`:
  - 逐 worker 生成 `/tmp/mario_env_diag_<parentpid>_<ts>/worker_idx<pid>.log`，记录 `start`、`patch_start`、`patch_uint8_ok`、`import_*`、`calling_mario_make` 等事件。<br>Per worker diagnostic logs at `/tmp/mario_env_diag_<parentpid>_<ts>/worker_idx<pid>.log` recording events such as `start`, `patch_start`, `patch_uint8_ok`, `import_*`, and `calling_mario_make`.
  - `_patch_legacy_nes_py_uint8()` 与 `_patch_nes_py_ram_dtype()` 修复 uint8 溢出及 RAM dtype 不兼容问题。<br>`_patch_legacy_nes_py_uint8()` and `_patch_nes_py_ram_dtype()` mitigate uint8 overflows and RAM dtype mismatches.
- `_make_single_env` 在 `mario_make` 调用前获取文件锁并写入诊断日志，保证重型初始化串行化。<br>`_make_single_env` acquires a file lock and logs diagnostics before invoking `mario_make`, keeping the heavy initialisation serialised.
  - 构建 `AsyncVectorEnv` 时传入 `shared_memory=False` 并依据平台推导 `context`，避免默认共享内存导致的关闭异常。<br>When creating `AsyncVectorEnv`, pass `shared_memory=False` and derive the appropriate `context` per platform to avoid shutdown issues tied to shared memory.
- `train.py`：<br>`train.py`:
  - 添加 CLI 参数：`--parent-prewarm`、`--parent-prewarm-all`、`--worker-start-delay`、`--env-reset-timeout`、`--enable-tensorboard`。<br>Added CLI flags: `--parent-prewarm`, `--parent-prewarm-all`, `--worker-start-delay`, `--env-reset-timeout`, `--enable-tensorboard`.
  - 将父进程预热移动到构建向量环境之前，以便主进程先加载 NES 相关库。<br>Moved parent prewarm ahead of vector-env construction so the main process can preload NES libraries.

## 观测时间线 | Observed Timeline
1. 未打补丁时，单进程 `mario_make` 稳定复现 `OverflowError`，定位到 `nes_py/_rom.py`。<br>Without patches, single-process `mario_make` consistently triggers `OverflowError` originating from `nes_py/_rom.py`.
2. 加入补丁后，溢出错误明显减少，但父进程仍会在多次运行中超时。<br>After applying patches, the overflow largely disappears yet the parent still times out in multiple runs.
3. 诊断日志显示 worker 已进入 `calling_mario_make` 并偶尔拿到文件锁，但部分 worker 不再输出事件，怀疑卡在原生初始化。<br>Diagnostics show workers entering `calling_mario_make` and sometimes acquiring the lock, but some stop logging afterwards, suggesting a stall in native initialisation.
4. 尝试更换多进程 start method（`fork` → `forkserver` → `spawn`）仅带来概率改善。<br>Switching the multiprocessing start method (`fork` → `forkserver` → `spawn`) only improved reliability probabilistically.
5. `parent-prewarm-all`、`worker_start_delay`、`shared_memory=False`、文件锁串行化等手段缓解并发压力，但无法完全根除超时。<br>`parent-prewarm-all`, `worker_start_delay`, `shared_memory=False`, and file-lock serialisation reduce contention but do not eliminate timeouts entirely.

## 已完成实验 | Experiments Conducted
- **nes_py 补丁**：在 worker 与父进程预热阶段应用，显著降低 OverflowError 频率。<br>**nes_py patches**: applied during worker and parent prewarm phases, dramatically reducing OverflowError frequency.
- **预热策略**：`--parent-prewarm` 与 `--parent-prewarm-all` 快速加载 ROM/库，减轻子进程初始化压力。<br>**Prewarm strategies**: `--parent-prewarm` and `--parent-prewarm-all` preload ROMs/libraries, easing child-process initialisation cost.
- **启动延迟**：`--worker-start-delay` 以阶梯方式启动 worker，缓解“羊群效应”。<br>**Staggered start**: `--worker-start-delay` staggers worker start-up to mitigate the thundering herd effect.
- **共享内存禁用**：`shared_memory=False` 避免 `AsyncVectorEnv` 在关闭阶段出现 EBADF 异常。<br>**Shared memory disabled**: `shared_memory=False` prevents EBADF exceptions during async env shutdown.
- **文件锁串行化**：确保 `mario_make` 在同一时刻仅由一个进程执行，证明可减少超时但仍需超长等待。<br>**File-lock serialisation**: guarantees only one process runs `mario_make` at a time, reducing timeouts but still suffering from long waits.

## 遗留问题 | Outstanding Issues
- 异常仍然依赖时间与启动顺序，未完全复现/根除；需要进一步定位 NES 原生层的阻塞原因。<br>Timeouts still depend on timing and launch order; the root cause at the NES native layer needs deeper inspection.
- Windows/macOS 平台的表现未充分验证，当前结论以 Linux 为主。<br>Behaviour on Windows/macOS remains under-tested; conclusions currently focus on Linux setups.
- 预热与串行化增加构建耗时，在高并发场景中需要平衡稳定性与启动速度。<br>Prewarm and serialisation add start-up latency, so their trade-offs must be evaluated under high concurrency.

## 推荐后续动作 | Recommended Next Steps
- 使用 PyTorch Profiler 与 NVTX 标记测量 `mario_make` 周边代码的耗时，确定阻塞边界。<br>Profile sections around `mario_make` with the PyTorch Profiler and NVTX markers to pinpoint stall boundaries.
- 进一步实验 `envpool` 或其他 C++ 向量环境方案，验证能否彻底替代 Python 进程池。<br>Experiment with `envpool` or other C++ vector-env solutions to assess whether they can replace the Python process pool entirely.
- 维护 `docs/ENV_DEBUGGING_REPORT.md` 日志，记录每次新增实验的配置、结果及对稳定性的影响。<br>Keep this report updated with configuration details, results, and stability impact for every new experiment.
- 若查明 NES 原生库瓶颈，考虑为 `nes_py` 提交补丁或构建本地替代实现。<br>If the NES native bottleneck is identified, consider upstreaming patches to `nes_py` or developing a local replacement implementation.

## 增量更新 | Incremental Update (2025-10-01)
**新增 per-env 构建进度日志**：同步模式下为每个子环境打印开始/耗时，精确定位卡点；8 环境 FC emulator 测试 ~0.11s 完成。<br>**Added per-env construction logs**: start/duration printed for each env in sync mode to isolate stalls; 8-env FC emulator test ~0.11s total.

**动机 | Motivation**：之前仅有全局“Starting environment construction”，排障需要手动猜测。<br>Previously only a single global start message existed.

**影响 | Impact**：只增日志，无功能或性能回退风险。<br>Logging only, no functional/performance regression.

**验证 | Validation**：`PYTHONPATH=. python train.py --world 1 --stage 1 --num-envs 8 --total-updates 50000` 构建与训练均顺利，未再触发超时告警。<br>Run succeeded without timeout warnings.

**后续 | Next**：考虑阶段级超时 & 可选容错跳过；聚合日志为 JSON 便于统计。<br>Plan stage-level timeouts & optional fault tolerance; aggregate logs into JSON.

### 新增（2025-10-01 PM）Overlap 采集实验 | Overlap Collection Experiment
**背景**：GPU 空转主要发生在同步“采集→学习”串行之间。<br>GPU idle gaps appear between strictly serial collect→learn phases.

**实现**：在 `train.py` 引入 `--overlap-collect` 双缓冲线程：
1. 前台学习当前 rollout；
2. 后台线程并行采集下一批；
3. 迭代开始时交换缓冲。

**取舍**：
- 采用线程 + 互斥锁包裹模型前向，避免 PyTorch 非线程安全风险；牺牲部分并行度换稳定性。
- 未做 step-level 流水线，保持实现简单便于回退。

**风险**：
- 线程异常仅日志告警，需后续指标化监控。
- 模型前向仍串行，性能提升受限（后续考虑 actor 快照网络解锁无锁读）。

**验证**：CLI help 出现参数；无 flag 时路径不变；语法编译通过。等待基准数据（SPS、updates/sec）。

**下一步**：收集基线 vs overlap 模式 300~1000 updates 的耗时比与 GPU util 曲线，决定是否推进 actor-learner 分离。
