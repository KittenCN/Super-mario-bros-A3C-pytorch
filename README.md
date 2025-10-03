# 超级马里奥兄弟 A3C（2025 版） | Super Mario Bros A3C (2025 Edition)

现代化的 PyTorch A3C 实现，兼容 `gymnasium-super-mario-bros` 与 `gym-super-mario-bros`，结合 IMPALA 风格网络、V-trace 目标、混合精度与完善的观测工具，用于训练通过《超级马里奥兄弟》关卡的智能体。<br>A modernised PyTorch A3C implementation compatible with `gymnasium-super-mario-bros` and `gym-super-mario-bros`, featuring IMPALA-style networks, V-trace targets, mixed precision, and rich observability to train agents that clear Super Mario Bros stages.

## 核心特性 | Key Features
- **Gymnasium 向量环境**：随机关卡调度、帧堆叠/预处理、可选同步或异步执行。<br>**Gymnasium vector environments**: randomised stage schedules, frame stacking/pre-processing, and optional synchronous or asynchronous execution.
- **IMPALA + 序列建模**：残差骨干接 GRU/LSTM/Transformer，可选 NoisyLinear 提升探索。<br>**IMPALA + sequence modelling**: residual backbones feeding GRU/LSTM/Transformer heads with optional NoisyLinear exploration.
- **V-trace / GAE 结合 PER**：混合 on/off-policy 目标，支持优先经验回放。<br>**V-trace / GAE with PER**: hybrid on/off-policy targets with prioritised experience replay support.
- **PyTorch 2.1 工程栈**：`torch.compile`、AMP、余弦 warmup、梯度累积、AdamW。<br>**PyTorch 2.1 stack**: `torch.compile`, AMP, cosine warmup, gradient accumulation, and AdamW.
- **观测与监控**：TensorBoard、可选 Weights & Biases、奖励统计、周期化 checkpoint。<br>**Observability**: TensorBoard, optional Weights & Biases, reward statistics, and periodic checkpoints.
- **自动化工具**：评估脚本、Optuna 超参搜索、Docker/conda 环境、视频录制封装。<br>**Automation tooling**: evaluation scripts, Optuna hyper-parameter search, Docker/conda environments, and video capture wrappers.

## 快速开始 | Quick Start
执行以下命令创建虚拟环境、安装依赖并启动训练。<br>Run the following commands to create a virtual environment, install dependencies, and start training.

```bash
# 安装依赖 | Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 训练 | Train
python train.py --world 1 --stage 1 --num-envs 8 --total-updates 50000

# 评估（基于元数据）| Evaluate (metadata-driven)
python test.py --checkpoint trained_models/a3c_world1_stage1_latest.pt --episodes 5

# 启动 TensorBoard | Run TensorBoard
tensorboard --logdir tensorboard/a3c_super_mario_bros

# 超参搜索示例 | Hyper-parameter search example
python scripts/optuna_search.py --trials 5

# 检查 checkpoint 元信息 (模型是否编译保存 / replay 配置 / 参数统计)
python scripts/inspect_checkpoint.py --ckpt trained_models/run01/a3c_world1_stage1_latest.pt

# 关闭 torch.compile 与 overlap（调试/基线）
DISABLE_OVERLAP=1 NO_COMPILE=1 bash scripts/run_2080ti_resume.sh --dry-run

# 运行奖励塑形烟雾测试 (确保距离/塑形指标可产生非零)
pytest -k test_shaping_smoke -q
```

> **提示 | Tip**：默认依赖 `torch>=2.1`。如改用 `gymnasium-super-mario-bros`，请同步调整 `requirements.txt`。<br>Default dependency targets `torch>=2.1`. If you switch to `gymnasium-super-mario-bros`, update `requirements.txt` accordingly.

### 无元数据 JSON 的评估回退 | Evaluation Without Metadata JSON
如果遗失了与 checkpoint 同名的 `.json` 元数据文件（例如只保留了 `a3c_world1_stage1_latest.pt`），`test.py` 会：
1. 尝试从 checkpoint 内部 `config` 字段重建必要信息；
2. 若无 `config`，依据文件名模式 `a3c_world{W}_stage{S}_...` 推断 world 与 stage；
3. 写出重建后的 `*.json` 旁路文件并继续评估。
若以上步骤均失败则仍会抛出错误。你可以手动补齐一个最小 JSON（参考已存在的其他 checkpoint 元数据）。<br>
If the sidecar metadata JSON is missing, `test.py` will reconstruct it from the checkpoint payload or filename pattern and persist a new JSON. If reconstruction fails, create a minimal JSON manually referencing another checkpoint as a template.

## 配置要点 | Configuration Highlights
- CLI 参数覆盖环境、模型、优化、日志等选项，可通过 `python train.py --help` 查看完整列表。<br>CLI flags cover environment, model, optimisation, and logging options; run `python train.py --help` for the full list.
- 常用参数：`--num-envs`、`--rollout-steps`、`--recurrent-type {gru,lstm,transformer,none}`、`--entropy-beta`、`--value-coef`、`--clip-grad`、`--random-stage`、`--stage-span`、`--per`。<br>Key flags include `--num-envs`, `--rollout-steps`, `--recurrent-type {gru,lstm,transformer,none}`, `--entropy-beta`, `--value-coef`, `--clip-grad`, `--random-stage`, `--stage-span`, and `--per`.
- `--no-amp` 与 `--no-compile` 可在调试时关闭 AMP 或 `torch.compile`。<br>`--no-amp` and `--no-compile` disable AMP or `torch.compile` when debugging.
- `--wandb-project`、`--metrics-path`、`--enable-tensorboard` 控制日志输出目的地。<br>`--wandb-project`, `--metrics-path`, and `--enable-tensorboard` control logging destinations.
- 环境构建超时可用 `--env-reset-timeout` 调整；默认按环境数量线性放大。<br>Adjust `--env-reset-timeout` to tune environment construction timeouts; by default it scales with the number of envs.

## 评估与监控 | Evaluation & Monitoring
- `test.py` 支持录像与贪心评估，输出每局奖励和通关状态。<br>`test.py` performs greedy evaluation with optional video capture, reporting per-episode rewards and completion.
- Gymnasium `RecordVideo` 默认将 MP4 写入 `output/eval/`。<br>Gymnasium’s `RecordVideo` writes MP4 files to `output/eval/` by default.
- TensorBoard 记录损失、熵、学习率、奖励，W&B 可镜像同样的指标。<br>TensorBoard tracks loss, entropy, learning rate, and rewards; W&B can mirror the same metrics.
- `src/utils/monitor.py`（配合 CLI 开关）在后台采集 CPU/GPU 资源使用。<br>`src/utils/monitor.py`, together with CLI switches, gathers background CPU/GPU utilisation.

## 高级用法 | Advanced Usage
- **Optuna 搜索**：`scripts/optuna_search.py` 运行短程实验，返回最佳 `avg_return`。<br>**Optuna search**: `scripts/optuna_search.py` runs short experiments and returns the best `avg_return`.
- **Overlap 基准 (benchmark)**：`scripts/benchmark_overlap.py` 对比 `--overlap-collect` 开/关 下不同 `(num_envs, rollout_steps)` 的吞吐：
	- `steps_per_sec_mean`：丢弃 warmup 前若干比例 (`--warmup-fraction`) 后计算剩余记录中 `env_steps_per_sec` 均值。
	- `updates_per_sec_mean`：同理对 `updates_per_sec` 取均值。
	- `warmup_fraction` 目的：忽略初始 JIT / 缓冲填充 / cache 预热抖动，提升基准稳定性。
	示例：
	```bash
	python scripts/benchmark_overlap.py --num-envs 4 8 --rollout-steps 32 64 --updates 300 --warmup-fraction 0.3
	```
	输出 CSV `benchmarks/bench_overlap_*.csv`，字段含义见首行表头。
- **PER 优先级分析**：训练日志新增 `replay_priority_mean/p50/p90/p99` 与采样唯一率 `replay_avg_unique_ratio`；用于监控优先级分布是否塌缩。
- **FP16 标量存储**：设置环境变量 `MARIO_PER_FP16_SCALARS=0` 可关闭默认的 FP16 advantages/target_values 压缩（若需严格数值一致性对比）。
- **Docker**：`docker build -t mario-a3c .` 后通过 `docker run --gpus all mario-a3c` 启动训练。<br>**Docker**: build with `docker build -t mario-a3c .` then launch via `docker run --gpus all mario-a3c`.
- **Conda**：`conda env create -f environment.yml && conda activate mario-a3c`。<br>**Conda**: run `conda env create -f environment.yml && conda activate mario-a3c`.
- **配置导出**：`--config-out configs/run.json` 保存最终 `TrainingConfig` 用于复现。<br>**Config export**: use `--config-out configs/run.json` to save the effective `TrainingConfig` for reproducibility.
- **奖励塑形 (Reward Shaping)**：通过距离与得分增量提供早期信号。
	```bash
	# 基础：距离权重 + 动态缩放退火
	python train.py --world 1 --stage 1 --num-envs 8 --reward-distance-weight 0.05 \
		--reward-scale-start 0.3 --reward-scale-final 0.1 --reward-scale-anneal-steps 80000

	# 启用 RAM 回退解析 (fc_emulator 缺失 x_pos 时)
	python train.py ... --enable-ram-x-parse --ram-x-high-addr 0x006D --ram-x-low-addr 0x0086
	```
	监控指标：`env_distance_delta_sum`、`env_shaping_raw_sum`、`env_shaping_scaled_sum`（见 `docs/metrics_schema.md`）。
- **脚本化前进 / 探测 (Scripted Progression & Probing)**：降低起步阶段停滞。
	```bash
	# 简单脚本序列：按 START 8 帧后持续 RIGHT+B 180 帧
	python train.py ... --scripted-sequence 'START:8,RIGHT+B:180'

	# 指定连续前进帧数（不显式动作 id）
	python train.py ... --scripted-forward-frames 240

	# 自动探测前进动作 (每动作连续 12 帧) 并选择 dx 最大的动作 id
	python train.py ... --probe-forward-actions 12
	```
	推荐：初次跑在 extended 动作集合中结合 `--reward-distance-weight 0.05`；脚本阶段结束后正常策略接管。

### 分阶段自动训练 | Automated Phase Training
使用 `scripts/auto_phase_training.sh` 一键跑完 Phase1(bootstrap) -> Phase2(main) -> Phase3(refine)：
```bash
# 预览命令 (不真正训练)
DRY_RUN=1 bash scripts/auto_phase_training.sh

# 实际执行（默认三阶段：2000 -> 30000 -> 50000 updates）
bash scripts/auto_phase_training.sh

# 仅运行 Phase1
SKIP_PHASE2=1 SKIP_PHASE3=1 bash scripts/auto_phase_training.sh

# 强制重跑 Phase2 并追加自定义变量
FORCE_PHASE2=1 PHASE2_EXTRA="NO_COMPILE=1" bash scripts/auto_phase_training.sh
```
生成的每阶段目录下写入 `.phase_done` 记录本阶段 meta，脚本可重入并跳过已完成阶段。用户可通过 `PHASE*_RUN / PHASE*_UPDATES / PHASE*_DISTANCE_WEIGHT` 等变量覆盖，详见脚本顶部注释。

### 推进自救与奖励调度 | Progress Self-Rescue & Reward Scheduling
冷启动或早期停滞时可组合以下机制加速出现正向位移并逐步退出辅助：

| 机制 | 目的 | 关键参数 | 退出条件 |
|------|------|----------|----------|
| Scripted Forward / Sequence | 立即推进出起点 | `--scripted-forward-frames`, `--scripted-sequence` | 帧数耗尽 |
| Forward Probe | 自动选择最优前进动作 | `--probe-forward-actions` | 一次性 |
| Early Shaping Window | 放大距离奖励信号 | `--early-shaping-window`, `--early-shaping-distance-weight` | 达到窗口上限 |
| Auto Bootstrap | 在阈值前仍 0 距离时介入 | `--auto-bootstrap-threshold`, `--auto-bootstrap-frames` | 剩余帧=0 |
| Secondary Script Injection | Plateau 后二次突破 | `--secondary-script-threshold`, `--secondary-script-frames` | 剩余帧=0 |
| Distance / Penalty Anneal | 平滑衔接主策略 | `--reward-distance-weight[-final/-anneal-steps]`, `--death-penalty-*` | 退火完成 |
| Milestone Bonus | 强化持续推进 | `--milestone-interval`, `--milestone-bonus` | 可持续 (阈值递增) |
| Episode Timeout | 促使更快获得 returns | `--episode-timeout-steps` | 每局重启 |
| Adaptive Distance / Entropy | 动态平衡探索与推进 | `--adaptive-*` 系列 | 可持续 |

推荐早期配置示例（Phase1 冷启动 2k updates）：
```bash
python train.py --world 1 --stage 1 --action-type extended --num-envs 8 \
	--total-updates 2000 --rollout-steps 32 \
	--reward-distance-weight 0.05 --reward-distance-weight-final 0.02 --reward-distance-weight-anneal-steps 40000 \
	--death-penalty-start 0 --death-penalty-final -5 --death-penalty-anneal-steps 30000 \
	--scripted-forward-frames 240 --probe-forward-actions 8 \
	--early-shaping-window 60 --early-shaping-distance-weight 0.08 \
	--auto-bootstrap-threshold 40 --auto-bootstrap-frames 200 \
	--secondary-script-threshold 100 --secondary-script-frames 160 \
	--milestone-interval 300 --milestone-bonus 0.5 \
	--episode-timeout-steps 1200 --enable-ram-x-parse --metrics-path run_metrics.jsonl
```
迁移至 Phase2（主学习阶段）时可显著降低 distance_weight、禁用脚本与 early window、保留 milestone 与 timeout：
```bash
python train.py --world 1 --stage 1 --action-type extended --num-envs 8 \
	--total-updates 30000 --rollout-steps 64 \
	--reward-distance-weight 0.02 --reward-distance-weight-final 0.01 --reward-distance-weight-anneal-steps 80000 \
	--death-penalty-start -2 --death-penalty-final -6 --death-penalty-anneal-steps 60000 \
	--milestone-interval 500 --milestone-bonus 0.25 --episode-timeout-steps 2000 \
	--metrics-path run_metrics_phase2.jsonl
```
关键监控指标：
- `env_distance_delta_sum`：若连续多个 log 周期为 0，脚本/探测未生效或动作集合无推进。
- `env_positive_dx_ratio`：衡量分布式探索覆盖；低于 0.25 持续多轮可再触发脚本。
- `milestone_count`：应阶梯式上升；停滞时结合 secondary_script。
- `avg_return`：提高 Episode 终结频率（timeout）后通常开始上升。

#### 自适应调度 (Adaptive Scheduling)
可选激活一组基于推进占比 (env_positive_dx_ratio) 的动态调整：
```bash
python train.py ... \
	--adaptive-positive-dx-window 50 \
	--adaptive-distance-weight-max 0.05 --adaptive-distance-weight-min 0.005 \
	--adaptive-distance-ratio-low 0.05 --adaptive-distance-ratio-high 0.30 --adaptive-distance-lr 0.05 \
	--adaptive-entropy-beta-max 0.02 --adaptive-entropy-beta-min 0.003 \
	--adaptive-entropy-ratio-low 0.05 --adaptive-entropy-ratio-high 0.30 --adaptive-entropy-lr 0.10
```
机制说明：
- avg_ratio < low: 提高 distance_weight 与 entropy_beta 以鼓励探索 & 推进；
- avg_ratio > high: 降低两者，减少过度随机与 shaping 影响；
- 中间区: 缓慢滑动，无剧烈波动。
日志新增: `adaptive_distance_weight`, `adaptive_entropy_beta`, `adaptive_ratio_avg`。
建议：在冷启动机制已保证快速产生正向位移后再启用，避免早期震荡。


## 项目结构 | Project Structure
- `src/envs/`：环境工厂、奖励塑形、帧处理、录像封装。<br>`src/envs/`: environment factories, reward shaping, frame processing, video wrappers.
- `src/models/`：IMPALA 残差块、序列模块、NoisyLinear 组件。<br>`src/models/`: IMPALA residual blocks, sequence modules, and NoisyLinear components.
- `src/algorithms/vtrace.py`：V-trace 目标计算。<br>`src/algorithms/vtrace.py`: V-trace target computation.
- `src/utils/`：RolloutBuffer、PrioritizedReplay、CosineWithWarmup、日志工具。<br>`src/utils/`: RolloutBuffer, PrioritizedReplay, CosineWithWarmup, logging utilities.
- `train.py`：端到端训练编排（AMP、compile、PER、checkpoint）。<br>`train.py`: end-to-end training orchestration (AMP, compile, PER, checkpointing).
- `test.py`：评估脚本，支持视频录制与确定性策略执行。<br>`test.py`: evaluation script with deterministic execution and video capture.
- `scripts/`：自动化工具（如 Optuna 搜索）。<br>`scripts/`: automation helpers (e.g. Optuna search).

## 调试提示 | Debugging Notes
- 若异步环境构建出现超时或 `nes_py` 溢出，请参阅 `docs/ENV_DEBUGGING_REPORT.md`。<br>If async env construction times out or `nes_py` overflows occur, see `docs/ENV_DEBUGGING_REPORT.md`.
- 可通过 `--sync-env` 或 `--force-sync` 暂时回退到同步向量环境以验证训练流程。<br>Temporarily fall back to synchronous vector envs using `--sync-env` or `--force-sync` to validate the training loop.
- `--parent-prewarm`、`--parent-prewarm-all`、`--worker-start-delay` 有助于减少 NES 初始竞争。<br>`--parent-prewarm`, `--parent-prewarm-all`, and `--worker-start-delay` reduce NES initialisation contention.
 - 异步模式需显式确认：即使传入 `--async-env` 仍需 `--confirm-async` 或设置 `MARIO_ENABLE_ASYNC=1` 才会真正启用，以避免误用不稳定路径。
 - PER 回放：默认启用观测 uint8 压缩；`advantages`/`target_values` 使用 FP16 存储并在采样时转回 float32，若需关闭设 `MARIO_PER_FP16_SCALARS=0`。
 - 2025-10-02 修复：`PrioritizedReplay.sample` 缩进错误导致的启动中止已修正，若你在此日期前拉取代码遇到 `IndentationError` 请更新到最新版本。
 - Checkpoint 元数据现包含 `replay.per_sample_interval`，用于恢复时对齐 PER 抽样策略；所有 `.pt/.json` 采用原子写入减少半写风险。
 - 已添加 `pytest.ini` 屏蔽与项目无关的 `pkg_resources` DeprecationWarning，保持测试输出聚焦功能性失败。
 - `tests/test_shaping_smoke.py` 验证奖励塑形与距离增量管线：若 stdout 出现 `[reward][dx] first_positive_dx` 即视为最小可行；若未来需要更严格判定可改为断言 `env_shaping_raw_sum>0`。

### 2025-10-01 环境构建卡住问题修复摘要 | 2025-10-01 Env Construction Stall Fix Summary
近期在同步模式（`async=False`）下出现长时间 “已连续 XXXs 无训练进度” 告警，经排查是缺乏逐子环境构建可见性难以快速定位。此次更新：<br>Recently synchronous runs (`async=False`) emitted prolonged "no training progress" warnings, traced to missing per-env construction visibility. This update adds:

1. `create_vector_env` 为每个子环境 thunk 包装计时与进度打印（`[env] constructing env i/N ...` & 完成耗时）。<br>Per-env timing logs around each thunk.
2. 若停在某编号，可立即在诊断目录中查看对应 `env_init_idx{idx}_pid*.log` 获取阶段标记（`calling_mario_make`, `framestack_done` 等）。<br>Stalled index maps directly to its diagnostic log file.
3. 无行为/性能副作用（日志开销微小），异步与同步模式均受益。<br>No behavioural impact; negligible overhead; works for async & sync.
4. 验证：8 个 FC emulator 环境 ~0.11s 完成构建，reset 0.02s，训练恢复正常。<br>Validation: 8 FC emulator envs build in ~0.11s, reset 0.02s.

快速诊断命令 | Quick diagnose:
```bash
PYTHONPATH=. python scripts/debug_env.py
PYTHONPATH=. python train.py --world 1 --stage 1 --num-envs 8 --total-updates 50000 --force-sync
```

复现卡住时请提供：最后一条 constructing 行、对应 env_init 日志尾部、版本信息 (Python / CUDA / nes-py / gymnasium-super-mario-bros)。<br>If stalls reoccur, share: last constructing line, tail of env_init log, and version info (Python/CUDA/nes-py/gymnasium-super-mario-bros).

后续潜在改进（未实施）：mario_make 硬超时 + 可跳过策略、结构化 JSON 诊断、`--env-progress-quiet` 静默开关。<br>Future (not implemented): hard timeout + skip policy, structured JSON diagnostics, `--env-progress-quiet` flag.

### 2025-10-01 性能改进（阶段一）| 2025-10-01 Performance Improvements (Phase 1)
为缓解 GPU 空转与 Python 包装层开销，本日合入如下低风险优化（细节/取舍见 `docs/decision_record.md`）：<br>To reduce GPU idling and Python wrapper overhead, the following low-risk optimisations were merged (see `docs/decision_record.md` for rationale):

1. Fused 观测预处理：将灰度、缩放、归一化、帧堆叠合并为单一 wrapper，减少多次函数调用。<br>Fused observation preprocessing to cut multiple Python calls.
2. PER 抽样间隔：新增 `--per-sample-interval N`，允许每 N 轮才做一次 PER 样本/优先级更新。<br>`--per-sample-interval N` reduces frequency of PER sampling.
3. 监控节流：降低外部 GPU 查询频率，为后续 NVML 方案预留接口。<br>Throttled external GPU metric polling.
4. 双缓冲重叠采集（实验特性）：`--overlap-collect` 在同步向量环境下启用后台线程采集下一批 rollout，与当前批学习重叠。<br>Experimental `--overlap-collect` flag: background thread collects next rollout while learning current one (sync env only).

使用建议 | Usage tips:
```bash
# 基线运行
python train.py --world 1 --stage 1 --num-envs 8 --rollout-steps 64 --total-updates 500 --log-interval 50

# 启用重叠采集对比
python train.py --world 1 --stage 1 --num-envs 8 --rollout-steps 64 --total-updates 500 --log-interval 50 --overlap-collect

# 降低 PER 频率示例（每 4 轮）
python train.py --per --per-sample-interval 4 ...
```

基准指标（建议记录）：updates/sec、env_steps/sec、单轮耗时、GPU utilisation、PER 触发频率。<br>Recommended to log: updates/sec, env_steps/sec, per-update wall time, GPU utilisation, PER trigger frequency.

#### 自适应显存模式 | Adaptive Memory Mode
运行脚本 `scripts/run_2080ti_resume.sh` 增加 `AUTO_MEM=1` 支持在首启 OOM 时按阶梯自动回退参数组合：
1. 原始配置 `(num_envs, rollout, grad_accum)`
2. 半 rollout
3. 减少 env 数
4. 减 env + 半 rollout + 取消累积
5. 继续缩减 env / rollout
触发条件：捕获启动期标准输出中的 `CUDA out of memory`。若非 OOM 失败则立即中止并打印日志。<br>
启用示例：
```bash
AUTO_MEM=1 bash scripts/run_2080ti_resume.sh
```

若发生后台线程异常将打印 `[train][warn] background collection failed:` 警告但主循环继续；首次迭代前台采集以填充缓冲。<br>Background thread failures emit a warning but training continues; first iteration collects in foreground.

后续阶段计划：无锁 actor-learner 拆分、GPU 常驻 PER、NVML 原生监控、结构化性能基线表格。<br>Next steps: lock-free actor/learner split, GPU-resident PER, native NVML monitoring, structured performance baseline tables.

## 文档与归档策略 | Documentation Policy
完整规则见 `docs/DOCUMENTATION_POLICY.md`。核心要点：<br>See `docs/DOCUMENTATION_POLICY.md` for full rules. Summary:
- 根目录只保留最新 `README.md` 与 `AGENTS.md`；其余文档放入 `docs/`。
- 不在 `README.md` / `AGENTS.md` 内累积历史更新列表，重大变更写入 `docs/` 报告或未来的 `CHANGELOG.md`。
- 训练恢复（自动加载最新匹配 checkpoint）说明集中在 README 与文档策略文件。


## 环境要求 | Requirements & Compatibility
- 推荐 Python 3.10/3.11、PyTorch ≥ 2.1、CUDA 12（Dockerfile 提供基础镜像）。<br>Recommended Python 3.10/3.11, PyTorch ≥ 2.1, CUDA 12 (base image provided by the Dockerfile).
- 依赖 `gymnasium-super-mario-bros>=0.8.0`（首选）或 `gym-super-mario-bros>=7.4.0`，并配合 `nes-py>=8.2`。<br>Requires `gymnasium-super-mario-bros>=0.8.0` (preferred) or `gym-super-mario-bros>=7.4.0` with `nes-py>=8.2`.
- 系统需安装 FFmpeg 以生成录像。<br>FFmpeg must be installed to produce videos.
- Checkpoint 命名为 `a3c_world{W}_stage{S}_*.pt`，同名 JSON 保存运行元数据，便于评估与继续训练。<br>Checkpoints are saved as `a3c_world{W}_stage{S}_*.pt` alongside JSON metadata for evaluation or resuming.

## 致谢 | Credits
- 原始版本由 Viet Nguyen 编写，现版本参考社区对 A3C/IMPALA、可观测性实践的持续探索。<br>Originally authored by Viet Nguyen; this modern refresh draws inspiration from ongoing community experimentation around A3C/IMPALA and observability practices.
