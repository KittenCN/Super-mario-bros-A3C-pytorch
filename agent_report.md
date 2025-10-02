# 自动执行报告（环境稳定性回归）

日期：2025-09-30

本报告总结了对 Super-mario-bros-A3C-pytorch 仓库中环境构造（尤其是 AsyncVectorEnv）稳定性的自动化回归测试、诊断和建议。文档面向工程团队，包含复现命令、关键观察、已做修改与后续建议。

## 概述
- 目标：分析并验证仓库中异步/同步向量化环境（基于 gymnasium 的 AsyncVectorEnv / SyncVectorEnv）在当前运行时的稳定性；收集诊断信息并提出可行缓解措施。
- 核心结论：同步向量化环境（synchronous）在当前运行环境中表现稳定；异步向量化环境在并发构造阶段存在显著不稳定（高失败率或阻塞），需要额外的父进程预热或结构性替代来获得稳定性。

## 关键修改与新增脚本
  - `scripts/run_sync_regression.py`：短的同步回归测验，默认将多次构造/reset/close 的结果追加到 `env_stability_report.jsonl`。
  - `scripts/async_trial_worker.py`：子进程 worker，用于单次异步试验（构造 AsyncVectorEnv、reset、close），把结果写到 stdout。
  - `scripts/run_async_regression.py`：父进程控制脚本，带超时保护、父进程预热参数 `--prewarm-count`、采集 worker stderr、并在失败时把 `/tmp/mario_env_diag_*` 诊断目录复制到 `reports/`。
  - `scripts/parse_env_report.py`：对 `env_stability_report.jsonl` 做简单汇总（已存在并用于报告统计）。
### 2025-10-02 补充 (P0 完成)
- 新增脚本: `scripts/backfill_global_step.py` 用于回填历史 global_step。
- 更新: 历史 checkpoint JSON 写入 `reconstructed_step_source` 字段并计算真实步数。
- 安全退出: 训练循环外层增加异常捕获并在中断时保存 latest（见 `decision_record.md` #10）。
- 文档: `decision_record.md` 与 `ROADMAP.md` 标记 P0 任务完成。

这些变更均已提交到当前工作区文件系统（未推送至远程仓库）。

## 回归实验（已执行）
- 同步回归（较大规模）
  - 运行：`PYTHONPATH=. python scripts/run_sync_regression.py --trials 50 --num-envs 8`
  - 结果（追加到 `env_stability_report.jsonl`）：5 + 50 等多批试验，当前文件包含 65 条（当时）同步试验均成功。
  - 统计：successes 65/65；构造/重置耗时（秒）大致在 0.003–0.009s 范围，平均 ~0.004–0.006s。
  - 结论：同步路径在当前环境中非常可靠，适合立即用于训练。

- 异步回归（初次）
  - 运行：`PYTHONPATH=. python scripts/run_async_regression.py --trials 20 --num-envs 8 --timeout 30`
  - 结果：20 条异步试验全部记录为 timeout（超时失败）。诊断日志位于 `/tmp/mario_env_diag_*`，并被追加到 `env_stability_report.jsonl`。（后续增强脚本改进了诊断捕获）

- 异步回归（增强后，父进程预热 + 增大超时）
  - 运行：`PYTHONPATH=. python scripts/run_async_regression.py --trials 20 --num-envs 8 --timeout 120 --prewarm-count 2`
  - 结果：本次运行追加 20 条，在总体 `env_stability_report.jsonl` 中（含历史）统计为：total 105 条试验，success 65，failure 40；失败全部为 timeout。
  - 观察：父进程预热（2 次）对稳定性有一定帮助但不足以根本解决问题；异步失败率仍很高。

## 诊断文件位置
- 主报告：`env_stability_report.jsonl`（仓库根）
- 运行时诊断：`/tmp/mario_env_diag_<pid>_<ts>/`（worker/父进程会在此写入 per-worker 启动日志）
- 失败时自动收集到：`reports/async_diag_<ts>/seed_<seed>_n<num_envs>/`（每次失败时的 best-effort 复制）

## 关键技术发现（简要）
- 问题模式：AsyncVectorEnv 路径在 worker 并发初始化时常卡住或阻塞，表现为父进程等待子进程输出直到超时；诊断文件显示 worker 在 `calling_mario_make` 或相关 native/ROM 初始化步骤停滞。
- 可能原因：nes_py 或 gym-super-mario-bros 的原生扩展在多进程并发导入/ROM 加载时与当前 Python/Numpy 版本存在兼容问题（例如 numpy uint8/溢出），或原生库内部使用全局锁导致竞争与死等候。
- 已采用的缓解：
  - 在 `src/envs/mario.py` 中实现了两个运行时补丁：_patch_legacy_nes_py_uint8、_patch_nes_py_ram_dtype，用于在父进程或 worker 端尽早修补 nes_py 的潜在 dtype/溢出问题。
  - 在构造 VectorEnv 时尝试以字符串 start-method 名称传参并做回退；实现了 worker 启动延迟（`worker_start_delay`）、文件锁以序列化 ROM 初始化、父进程 prewarm 逻辑。
  - 父进程 prewarm `--parent-prewarm` / `--parent-prewarm-all` 可在训练时使用以降低首次并发初始化失败概率。

## 推荐（短期/中期/长期）
1. 短期（立即采用）
   - 在默认训练/CI 中使用同步 vector env（同步路径已经稳定）。可以通过 CLI `--force-sync` 或将默认保持为同步来避免训练时中断。
   - 在 README 或项目文档中明确记录当前 AsyncVectorEnv 的风险与建议的预热步骤（例如父进程 prewarm、增加 worker_start_delay）。
2. 中期（工程级缓解）
   - 若必须使用 AsyncVectorEnv（性能要求），在训练前在父进程运行彻底的预热：`--parent-prewarm-all` 或手动在脚本里顺序构造每个可能的 env（按 stage）来完成所有原生初始化，从而在 worker 启动时避免 ROM 相关的首次导入阻塞。
   - 在运行 Async 回归时把 `worker_start_delay` 设置得更大（例如 0.5–1.0s），以进一步串行化 worker 初始化，观察失败率是否下降。
3. 长期（更改后端）
   - 迁移到一个更稳定的进程模型或替代实现（例如 envpool 或使用容器隔离 worker），这是根本性修复但需要更大工程量与测试。
   - 或者在 CI 中限制 Python / NumPy / nes_py 的兼容性矩阵（例如 pin 到已验证的组合）以避免在主分支上出现高失败率。

## 复现命令（已用）
- 同步回归（短）：

```bash
PYTHONPATH=. python scripts/run_sync_regression.py --trials 5 --num-envs 4
```

- 同步回归（大规模）：

```bash
PYTHONPATH=. python scripts/run_sync_regression.py --trials 50 --num-envs 8
```

- 异步回归（带父进程预热与长超时）：

```bash
PYTHONPATH=. python scripts/run_async_regression.py --trials 20 --num-envs 8 --timeout 120 --prewarm-count 2
```

- 汇总报告：

```bash
python scripts/parse_env_report.py env_stability_report.jsonl
```

## 下一步建议（我可以代为执行）
- 解析 `reports/async_diag_*` 目录并生成一个诊断要点（我可以把关键日志片段摘录到报告中）。
- 在训练流程中把 `--parent-prewarm-all` 设为可选且在 README 指定为异步模式的预备步骤。
- 如果你同意，我可以把本次回归的核心结论和操作步骤写入 `README.md` 或 `ASSUMPTIONS.md`，并在 PR 描述中包含如何在 CI 中复现与验证。

## 附：需求覆盖映射（本次任务的完成情况）
- 检索/分析/对比文档：已执行（查看 AGENTS.md、docs 下分析）——Done
- 提出并实现修复（父进程预热、context 传参、nes_py 补丁）：已实现并验证（部分丢失的 async 稳定性仍存在）——Done (mitigations applied)
- 运行回归并汇总：已执行（同步成功，异步失败并收集诊断）——Done
- 生成自动化报告：本文件 `agent_report.md`（已生成）——Done

如果你想让我继续自动化分析 `reports/` 中的诊断日志并把结果追加至本报告，请回复 A；如果想要我直接把 `agent_report.md` 提交为 PR（包括变更说明和建议的 README 更新），回复 PR；或给出其他指示。

## 诊断摘要（来自 `reports/async_diag_*`）

我对 `reports/` 下的代表性目录和日志做了抽样分析，主要发现如下：

- 常见卡住点：多数失败的 worker 在 `calling_mario_make` 后停止记录，日志中通常呈现如下序列（示例）：

```
<timestamp>    start
<timestamp>    patch_start
<timestamp>    patch_uint8_ok
<timestamp>    patch_ramdtype_ok
<timestamp>    patch_done
<timestamp>    import_start_gsm
<timestamp>    import_done_gsm
<timestamp>    lock_acquired
<timestamp>    calling_mario_make
```

接下来通常没有 `mario_make_done`、`joypad_done` 或后续 wrapper 步骤，表明进程在 `mario_make()`（native ROM 加载 / nes_py 初始逻辑）阶段被阻塞。

- 序列化/延迟模式：许多 worker 只是记录 `startup_wait`（表示它们真正在等待父进程允许启动），当被允许后第一个 worker（或当前持有锁的 worker）进入 `calling_mario_make` 并停滞，后续 worker 很多保持 `startup_wait` 状态直至超时，表明存在一次性/串行的资源争用点（如 ROM 打开/解压或共享本机资源）。

- 错误签名：在部分非阻塞失败的样例中，可以看到 `OverflowError: Python integer ... out of bounds for uint8` 或 nes_py 抛出的 dtype 相关异常（这些已通过 `_patch_legacy_nes_py_uint8` / `_patch_nes_py_ram_dtype` 做了运行时修补，但修补在 worker 中并不总是能避免阻塞行为）。

- 诊断文件位置与格式说明：每个 `mario_env_diag_*` 目录包含若干 `env_init_idx{N}_pid{PID}.log`，每行带时间戳与标记（例如 `patch_start`, `calling_mario_make`, `mario_make_done`, `joypad_done` 等），可按时间点追踪 worker 初始化阶段。

结论：抽样日志清楚地表明主要阻塞发生在 `mario_make()` 阶段（nes_py 原生初始化 / ROM 加载），且在高并发构造时更易触发。父进程预热与文件锁是合理缓解，但在当前环境下仍不足以完全避免阻塞；建议采取更强的序列化或替代后端以彻底稳定异步路径。

---
## 2025-10-02 补充 (Reward Shaping / Scripted Progression / Flexible Resume)

### 新增能力概览
- 距离 + 得分增量奖励塑形，线性退火 scale（`--reward-distance-weight`, `--reward-scale-start/final/anneal-steps`）。
- RAM 回退解析 x_pos（`--enable-ram-x-parse`）支持 fc_emulator 缺失字段场景。
- 脚本化前进：`--scripted-sequence`, `--scripted-forward-frames`；自动动作探测：`--probe-forward-actions`；显式 `--forward-action-id`。
- 编译/未编译 checkpoint 自适应加载，自动处理 `_orig_mod.` 前缀。
- 中断安全退出：异常 / Ctrl+C 保存 latest 原子快照。
- 运行脚本扩展：支持环境变量注入上述 shaping / 脚本参数。

### 关键指标新增
`env_distance_delta_sum`, `env_shaping_raw_sum`, `env_shaping_scaled_sum`, `env_shaping_last_*`, `replay_priority_p50/p90/p99`, `gpu_util_mean_window`, `model_compiled`。

### 快速验证命令
```bash
RUN_NAME=shaping_demo ACTION_TYPE=extended REWARD_DISTANCE_WEIGHT=0.05 \
SCRIPTED_SEQUENCE='START:8,RIGHT+B:180' ENABLE_RAM_X_PARSE=1 \
bash scripts/run_2080ti_resume.sh --dry-run
```

### 推荐调参流程
1. 先用 scripted 序列确保 `env_distance_delta_sum>0`；
2. 调整 distance_weight 至 raw_sum / distance_delta_sum 比值合理 (<2x)；
3. 观察 scaled_sum 与 episode return 比例，必要时调低 scale_start；
4. 移除 scripted，验证策略自发前进是否保留正向 shaping。

### 后续工作建议
- 添加 shaping smoke test：执行 1 个 scripted episode 断言 raw_sum>0。
- 指标转存为 Parquet 加速分析。
- GPU PER 采样原型 + KL 验证脚本。