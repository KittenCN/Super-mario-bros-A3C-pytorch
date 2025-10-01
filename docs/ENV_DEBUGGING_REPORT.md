# AsyncVectorEnv 调试报告 | AsyncVectorEnv Debugging Report

**记录日期 | Logged on**: 2025-09-29 ～ 2025-09-30

本文汇整在构建 `AsyncVectorEnv` 时遇到的阻塞、溢出等问题，以及针对 `nes_py` 的修复尝试、实验数据与后续建议，确保团队在仓库重置后也能快速上手。<br>This report captures the blocking and overflow issues observed while constructing `AsyncVectorEnv`, the runtime patches applied to `nes_py`, experiments that were run, and suggested next steps so the team can pick up the investigation after repository resets.

## 最新变更摘要 | Latest Update (2025-10-01)
- **修复同步环境构建挂起**：移除同步模式下 `call_with_timeout` 对 `create_vector_env` 的包装，避免 daemon 线程中原生库初始化导致的挂起。<br>**Fixed synchronous environment construction hang**: removed `call_with_timeout` wrapping of `create_vector_env` in synchronous mode to avoid native library initialization issues in daemon threads.
- 同步环境（`SyncVectorEnv`）在主进程中直接构建，不需要超时包装；异步环境保持使用超时机制处理 worker 进程启动问题。<br>Synchronous environments (`SyncVectorEnv`) are now constructed directly in the main process without timeout wrapping; async environments continue using timeout mechanisms to handle worker process start-up issues.
- 在 `create_vector_env` 中引入严格的按序启动：每个 worker 需等待前一个 NES 构建完成后再初始化，避免并发 `mario_make` 引发竞争。<br>Serialised worker start-up in `create_vector_env` so each worker waits for the previous NES construction to finish, preventing concurrent `mario_make` contention.
- 将默认构建/重置超时设为 `max(180s, num_envs × 60s)`，兼顾大规模并发带来的累积等待。<br>Set the default construct/reset timeout to `max(180s, num_envs × 60s)` to accommodate accumulated waits in large asynchronous batches.
- 保留文件锁与诊断日志机制，便于继续分析潜在的 NES 底层阻塞。<br>Retained the file-lock and diagnostic logging mechanisms to keep visibility into potential NES-level stalls.

## 问题概述 | Problem Summary
- 现象：父进程构建 `AsyncVectorEnv` 时偶发超时并回退到同步环境，同时输出 “Async vector env construction timed out…”。<br>Symptom: the parent process sporadically times out while building `AsyncVectorEnv`, prints “Async vector env construction timed out…”, and falls back to the synchronous env.
- 额外现象：在 Python 3.12 + NumPy 2.x 下，单进程运行 `gym_super_mario_bros.make(...)` 会触发 `nes_py` 的 `OverflowError: Python integer 1024 out of bounds for uint8`。<br>Additional symptom: on Python 3.12 + NumPy 2.x, a single-process `gym_super_mario_bros.make(...)` call in `nes_py` raises `OverflowError: Python integer 1024 out of bounds for uint8`.
- 诊断日志显示 worker 停在 `calling_mario_make` 处，有时崩溃（未打补丁），有时静默挂起（已打补丁），推测阻塞发生在 NES 原生初始化阶段。<br>Diagnostics show workers reaching `calling_mario_make` and then either crashing (without patches) or hanging silently (with patches), indicating the stall occurs during NES native initialisation.

## 诊断入口 | Instrumentation

### 2025-10-01 修复：同步环境构建超时 | 2025-10-01 Fix: Synchronous Environment Construction Timeout
**问题**：使用默认同步模式（`async=False`）时，`train.py` 的 `_initialise_env` 函数对 `create_vector_env` 调用使用 `call_with_timeout` 包装，该函数在 daemon 线程中执行环境构建。当 `SyncVectorEnv` 初始化时会在主线程（实际上是 daemon 线程）中调用每个 `env_fn()`，这会触发原生库（`nes_py`、`gym_super_mario_bros`）的初始化，可能导致线程挂起。<br>**Problem**: When using the default synchronous mode (`async=False`), the `_initialise_env` function in `train.py` wrapped `create_vector_env` calls with `call_with_timeout`, which executes the environment construction in a daemon thread. When `SyncVectorEnv` initializes, it calls each `env_fn()` in the main thread (actually the daemon thread), triggering native library (`nes_py`, `gym_super_mario_bros`) initialization that can cause thread hangs.

**修复**：在同步模式下直接调用 `create_vector_env`，不使用 `call_with_timeout` 包装。仅在异步模式下保留超时机制，因为异步模式的 worker 进程在独立进程中运行，超时检测更有意义。<br>**Fix**: In synchronous mode, call `create_vector_env` directly without `call_with_timeout` wrapping. Keep the timeout mechanism only for async mode, where worker processes run in separate processes and timeout detection is more meaningful.

**代码变更**：<br>**Code change**:
```python
# train.py _initialise_env function
if env_cfg.asynchronous:
    # 异步模式：使用超时包装
    env_instance = call_with_timeout(create_vector_env, timeout, env_cfg)
else:
    # 同步模式：直接调用，避免 daemon 线程问题
    env_instance = create_vector_env(env_cfg)
```

**影响**：此修复解决了用户报告的"已连续 300s 无训练/评估进度"的挂起问题。<br>**Impact**: This fix resolves the reported "no progress for 300s" hang issue.

---

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
