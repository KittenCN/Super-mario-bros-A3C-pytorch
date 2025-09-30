# 代理更新记录 | Agent Update Log

**更新时间 | Update Date**: 2025-09-29

本文记录调试代理在 2025-09-29 会话中对仓库所做的关键修改，方便团队理解改动背景与后续操作。<br>This log summarises the key adjustments made by the debugging agent during the 2025-09-29 session so the team can understand the rationale and plan follow-up work.

## 代码改动总览 | Summary of Code Changes
- `docs/ENV_DEBUGGING_REPORT.md`：编写 `AsyncVectorEnv` 不稳定问题的完整诊断记录与推荐方案。<br>`docs/ENV_DEBUGGING_REPORT.md`: captured a comprehensive diagnostic report on `AsyncVectorEnv` instability and proposed mitigations.
- `src/envs/mario.py`：新增 worker 包装器、nes_py 运行时补丁、逐 worker 诊断日志与文件锁串行化逻辑。<br>`src/envs/mario.py`: added a worker wrapper, nes_py runtime patches, per-worker diagnostic logging, and file-lock serialisation.
- `train.py`：增补 CLI 参数（如 `--parent-prewarm-all`、`--worker-start-delay`、`--env-reset-timeout`、`--enable-tensorboard`），并调整多进程启动顺序。<br>`train.py`: introduced additional CLI flags (e.g. `--parent-prewarm-all`, `--worker-start-delay`, `--env-reset-timeout`, `--enable-tensorboard`) and refined multiprocessing start-method handling.
- `README.md`：加入与 `AsyncVectorEnv` 相关的调试提示与文档链接。<br>`README.md`: linked to the debugging report and highlighted async-vs-sync environment tips.

## 变更目的 | Purpose of the Session
- 提升环境构建的稳定性，缓解 `nes_py` 与 NumPy/Python 版本不兼容导致的阻塞或溢出。<br>Improve environment construction robustness in the presence of nes_py and NumPy/Python compatibility issues.
- 提供可追溯的诊断数据，便于后续代理或开发者继续排查。<br>Provide traceable diagnostics so future agents or developers can continue investigations efficiently.
- 将训练与监控开关参数化，方便按需开启日志功能并测试不同构建路径。<br>Expose training and monitoring toggles so logs and alternative code paths can be tested on demand.

## 额外整理 | Additional Housekeeping
- 以下 Markdown 文件已迁移至 `docs/` 目录：`PROJECT_ANALYSIS.md`、`PROJECT_OPTIMIZATION_PLAN.md`、`GPU_UTILIZATION_ANALYSIS.md`、`AGENT.md`。<br>The following Markdown files were moved into `docs/`: `PROJECT_ANALYSIS.md`, `PROJECT_OPTIMIZATION_PLAN.md`, `GPU_UTILIZATION_ANALYSIS.md`, and `AGENT.md`.
- 迁移操作的详细说明见 `docs/MOVED_FILES.md`。<br>Refer to `docs/MOVED_FILES.md` for the relocation details.

## 后续建议 | Follow-up Suggestions
- 保持 `docs/ENV_DEBUGGING_REPORT.md` 与最新实验同步更新，继续记录成功与失败案例。<br>Keep `docs/ENV_DEBUGGING_REPORT.md` aligned with future experiments, logging both successes and failures.
- 若新增 CLI 参数或调试脚本，请同步更新此日志，确保团队了解控制面板的变化。<br>When new CLI switches or debugging scripts are introduced, update this log so the team stays informed about control-surface changes.
