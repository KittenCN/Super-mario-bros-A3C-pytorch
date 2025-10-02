# 代理更新记录 | Agent Update Log

**更新时间 | Update Date**: 2025-10-02

本文记录调试代理在 2025-09-29 ~ 2025-10-02 会话中对仓库所做的关键修改，方便团队理解改动背景与后续操作。<br>This log summarises the key adjustments made by the debugging agent during the 2025-09-29 ~ 2025-10-02 sessions so the team can understand the rationale and plan follow-up work.

## 代码改动总览（2025-09-29 初始批次）| Summary of Code Changes (Initial 2025-09-29 batch)
- `docs/ENV_DEBUGGING_REPORT.md`：编写 `AsyncVectorEnv` 不稳定问题的完整诊断记录与推荐方案。<br>`docs/ENV_DEBUGGING_REPORT.md`: captured a comprehensive diagnostic report on `AsyncVectorEnv` instability and proposed mitigations.
- `src/envs/mario.py`：新增 worker 包装器、nes_py 运行时补丁、逐 worker 诊断日志与文件锁串行化逻辑。<br>`src/envs/mario.py`: added a worker wrapper, nes_py runtime patches, per-worker diagnostic logging, and file-lock serialisation.
- `train.py`：增补 CLI 参数（如 `--parent-prewarm-all`、`--worker-start-delay`、`--env-reset-timeout`、`--enable-tensorboard`），并调整多进程启动顺序。<br>`train.py`: introduced additional CLI flags (e.g. `--parent-prewarm-all`, `--worker-start-delay`, `--env-reset-timeout`, `--enable-tensorboard`) and refined multiprocessing start-method handling.
- `README.md`：加入与 `AsyncVectorEnv` 相关的调试提示与文档链接。<br>`README.md`: linked to the debugging report and highlighted async-vs-sync environment tips.

## 变更目的 | Purpose of the Session (2025-09-29)
- 提升环境构建的稳定性，缓解 `nes_py` 与 NumPy/Python 版本不兼容导致的阻塞或溢出。<br>Improve environment construction robustness in the presence of nes_py and NumPy/Python compatibility issues.
- 提供可追溯的诊断数据，便于后续代理或开发者继续排查。<br>Provide traceable diagnostics so future agents or developers can continue investigations efficiently.
- 将训练与监控开关参数化，方便按需开启日志功能并测试不同构建路径。<br>Expose training and monitoring toggles so logs and alternative code paths can be tested on demand.

## 额外整理 | Additional Housekeeping (2025-09-29)
- 以下 Markdown 文件已迁移至 `docs/` 目录：`PROJECT_ANALYSIS.md`、`PROJECT_OPTIMIZATION_PLAN.md`、`GPU_UTILIZATION_ANALYSIS.md`、`AGENT.md`。<br>The following Markdown files were moved into `docs/`: `PROJECT_ANALYSIS.md`, `PROJECT_OPTIMIZATION_PLAN.md`, `GPU_UTILIZATION_ANALYSIS.md`, and `AGENT.md`.
- 迁移操作的详细说明见 `docs/MOVED_FILES.md`。<br>Refer to `docs/MOVED_FILES.md` for the relocation details.

## 后续建议 | Follow-up Suggestions (2025-09-29)
---

## 新增改动（2025-09-30 ~ 2025-10-02）| Additional Changes (2025-09-30 ~ 2025-10-02)

### 概览 | Overview
本阶段聚焦：
- 奖励塑形（距离位移 + 动态缩放 + 诊断字段）
- 训练安全退出与原子化 checkpoint 落盘
- `torch.compile` 编译/未编译 checkpoint 互操作（灵活前缀映射）
- 递归恢复 (`trained_models/` 深度扫描) & "latest" 快照一致性
- 双缓冲重叠采集 + PER 抽样间隔 + FP16 replay 标量压缩
- 运行脚本 `run_2080ti_resume.sh` 参数扩展（REWARD_DISTANCE_WEIGHT / SCRIPTED_SEQUENCE ...）
- 首帧推进、脚本化前进 (scripted sequence / scripted forward frames / forward action probing)
- 诊断：慢 step 窗口化堆栈追踪、RAM x_pos 回退解析、stagnation 告警
- metrics schema 扩展（distance / shaping / replay priority 分布 / GPU 滑动窗口）

### 关键修复 | Key Fixes
| 问题 | 根因 | 解决 | 影响 |
|------|------|------|------|
| 训练启动 silent exit | wrapper 中属性定义缩进漂移导致导入期 NameError 被上层吞掉 | 修正缩进 & 明确异常日志 | 恢复训练可见性 |
| distance/shaping 始终 0 | batched dict info 分支未解析 dx；shaping 字段 setdefault 被覆盖 | 添加 batched 路径 & 改为直接赋值 | 产生非零 env_distance_delta_sum |
| 编译模型恢复报大量 missing/unexpected key | `_orig_mod.` 前缀不匹配 | `_flex_load_state_dict` 动态剥离/添加前缀 + `strict=False` 记录差异 | 无需强制统一保存格式 |
| 中断丢失最新进度 | 未在异常路径 snapshot | try/except 捕获 KeyboardInterrupt/Exception 落盘 latest | 提升鲁棒性 |
| global_step 历史缺失 | overlap 旧逻辑未增量 | 回填脚本 + 公式推断 + 审计字段 | 统计对齐 |

### 新增 CLI / 环境变量 | New CLI / Env Vars
训练：
- `--reward-distance-weight` / `--reward-scale-start` / `--reward-scale-final` / `--reward-scale-anneal-steps`
- `--scripted-sequence` (`ACTION:FRAMES,RIGHT+B:120`)、`--scripted-forward-frames`、`--forward-action-id`、`--probe-forward-actions`
- `--enable-ram-x-parse --ram-x-high-addr --ram-x-low-addr`
- `--overlap-collect`, `--per-sample-interval`, `--slow-step-trace-threshold`, `--slow-step-trace-window`

运行脚本扩展环境变量（自动映射到上述 CLI）：
`REWARD_DISTANCE_WEIGHT`, `REWARD_SCALE_START`, `REWARD_SCALE_FINAL`, `REWARD_SCALE_ANNEAL_STEPS`, `SCRIPTED_SEQUENCE`, `SCRIPTED_FORWARD_FRAMES`, `FORWARD_ACTION_ID`, `PROBE_FORWARD`, `ENABLE_RAM_X_PARSE`, `RAM_X_HIGH`, `RAM_X_LOW`。

### 指标扩展 | Metrics Extensions
新增字段（详见 `docs/metrics_schema.md`）：
- `env_mean_x_pos`, `env_max_x_pos`, `env_distance_delta_sum`
- `env_shaping_raw_sum`, `env_shaping_scaled_sum`, `env_shaping_last_*`, `env_shaping_parse_fail`
- `replay_priority_mean/p50/p90/p99`, `replay_last_unique_ratio`, `replay_avg_unique_ratio`
- `gpu_util_last`, `gpu_util_mean_window`
- `model_compiled`, `replay_sample_time_ms` 及拆分阶段耗时字段

### 使用建议 | Usage Tips
脚本化前进推荐：
```
RUN_NAME=bootstrap ACTION_TYPE=extended \
REWARD_DISTANCE_WEIGHT=0.05 SCRIPTED_SEQUENCE='START:8,RIGHT+B:180' \
ENABLE_RAM_X_PARSE=1 bash scripts/run_2080ti_resume.sh
```
若 shaping raw 仍为 0，检查：
1. 是否启用 RAM 解析 (fc_emulator 缺失 x_pos)；
2. distance_weight 是否过低（尝试 0.05~0.1）；
3. 是否脚本化前进帧数不足以触发 dx>0。

### 后续建议 (2025-10-02) | Follow-ups
- 为 shaping 奖励添加单元 smoke：固定脚本序列，断言 raw_sum>0。
- 引入 Parquet 聚合 (`metrics/latest.parquet`) 减少离线分析 parse 成本。
- GPU PER 原型 + 采样分布 KL 校验。
- RAM 解析失败计数阈值 → 自动降级/提示。

---
- 保持 `docs/ENV_DEBUGGING_REPORT.md` 与最新实验同步更新，继续记录成功与失败案例。<br>Keep `docs/ENV_DEBUGGING_REPORT.md` aligned with future experiments, logging both successes and failures.
- 若新增 CLI 参数或调试脚本，请同步更新此日志，确保团队了解控制面板的变化。<br>When new CLI switches or debugging scripts are introduced, update this log so the team stays informed about control-surface changes.
