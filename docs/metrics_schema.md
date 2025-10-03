## Metrics Schema

结构化 `metrics.jsonl` 每行一个 JSON 记录，字段含义：

核心训练: timestamp, update, global_step, loss_total/policy/value/per, entropy, learning_rate, grad_norm
回报窗口 (最近100 episodes): avg_return, recent_return_std/max/min, recent_return_p50/p90/p99
性能: env_steps_per_sec, updates_per_sec, update_time, rollout_time, learn_time
Replay (PER): replay_size, replay_capacity, replay_fill_rate, replay_last_unique_ratio, replay_avg_unique_ratio, replay_push_total
资源: resource.* (GPU/CPU/Mem) + gpu_util_last, gpu_util_mean_window (最近100快照均值)

后台监控线程 (`src/utils/monitor.py`) 也会向同一 `metrics.jsonl` 追加仅包含 `{"timestamp": ..., "monitor": {...}}` 的记录，这些行没有 `global_step/update` 字段；进行训练指标聚合时需过滤或与主线程写入区分。

新增/关联字段（2025-10-02 更新）:
- per_sample_interval: 来自 checkpoint 元数据的抽样频率策略（非 metrics 行内字段，但分析时应读取 checkpoint JSON 以还原训练语义）。
- replay_priority_mean/p50/p90/p99: 已在日志中输出，用于监控优先级分布是否塌缩。
 - replay_sample_time_ms: 单次 PER 采样耗时（毫秒，包含 CPU 采样与数据拷贝），用于评估 GPU 化潜在收益（2025-10-02 新增）。
  - replay_sample_split_prior_ms / choice_ms / weight_ms / decode_ms / tensor_ms / total_ms: 采样细分阶段耗时（仅在触发抽样轮记录，其他轮默认为 0）。
  - replay_per_sample_interval: 当前训练配置的 PER 抽样间隔（静态配置写入每条 metrics 方便外部解析）。
  - replay_gpu_sampler: 1 表示启用 `use_gpu_sampler`（torch searchsorted 路径），0 表示已回退到 CPU。（2025-10-04 新增）
  - replay_gpu_sampler_fallback: 若 GPU 采样耗时连续超过阈值触发回退，则为 1，否则为 0。
  - replay_gpu_sampler_reason: GPU 采样被禁用时记录原因（例如 fallback 触发）。
  - model_compiled: 1 表示当前运行模型是 torch.compile 产物（含 `_orig_mod`），0 表示未编译或已 unwrap（2025-10-02 新增）。
 - state_dict_load_issues.log: 若恢复时出现 missing/unexpected 键，会将完整列表追加写入该文件，仅在首次或有差异时更新；metrics 行内不重复膨胀内容。

额外 artefact：每次 log interval 还会刷新 `metrics/latest.parquet`（单行 DataFrame），方便直接用 pandas/pyarrow 快速读取最近一次指标。若 `metrics.jsonl` 超过 `--metrics-rotate-max-mb`，系统会将旧数据压缩归档 (`metrics.jsonl.<timestamp>.gz`) 并保留最近 `--metrics-rotate-retain` 个文件。

原子写保证: 自 2025-10-02 起 checkpoint `.pt` 与对应 `.json` 元数据使用临时文件 + `os.replace` 原子落盘，减少中断产生半文件风险。

兼容: 新增字段向后兼容; 消费端使用 dict.get(key, default) 处理缺失。
未来: TD 误差分布、episode length 分位数、成功率、阶段通关率。

### 距离 / Shaping 相关字段 (2025-10-02 新增)
- env_mean_x_pos: 最近一个 log 周期内各 env 最新 x_pos 的平均值。
- env_max_x_pos: 运行至当前的全局最大 x_pos（所有 env）。
- env_distance_delta_sum: 当前 log 周期内所有 env 正向位移增量 (dx>0) 的累积和。
- env_shaping_raw_sum: wrapper 中未乘动态 scale 前的 shaping raw 累积 (score_weight*score_delta + distance_weight*dx)。
- env_shaping_scaled_sum: 乘动态 scale 后 shaping 奖励累积（用于与实际 reward 对齐）。
- env_shaping_last_dx / last_raw / last_scaled / last_scale: 最近一次成功解析的 shaping 诊断值（便于调参时观察瞬时状态）。
- env_shaping_parse_fail: 本周期内解析 shaping 字段失败次数（结构或类型异常）。

解析指引: 若 env_shaping_raw_sum 始终为 0 或 env_distance_delta_sum>0 但 raw_sum=0，需检查：
1) RewardWrapper 是否被替换/禁用；
2) shaping 字段是否被底层环境覆盖；
3) batched dict info 情况是否匹配当前解析逻辑（日志中应有 debug-info 输出 keys）。

调参与建议: distance_weight 建议初期设为 0.05~0.1 以确保 raw 产生非零；scale_start/scale_final 控制整体 reward 能量，过低会导致 shaping_scaled_sum 难以显著；观察 env_distance_delta_sum 与 scaled_sum 比值评估 reward 归一化是否合理。

自动化验证: `tests/test_shaping_smoke.py` 在 CI / 本地执行中：
1. 启动单环境短程训练 (BOOTSTRAP+脚本化前进)。
2. 捕获 stdout 中的首次正位移 `[reward][dx] first_positive_dx` 或 metrics 中的非零 `env_distance_delta_sum`。
3. 作为回归守护防止后续改动导致 shaping 失效（例如 wrapper 顺序或 info 字段覆盖）。

脚本化前进与探测辅助诊断：
- 若使用 `--probe-forward-actions`，在训练 stdout 日志中应出现每个动作的 dx 结果；选择的 forward_action_id 之后应使 `env_distance_delta_sum` 与 `env_shaping_raw_sum` 上升。
- 若启用 `--scripted-sequence` 但 raw_sum 仍为 0，检查：脚本动作是否匹配扩展动作集合；RAM 解析是否启用；是否在脚本阶段之前 episode 已重置。
- `env_shaping_parse_fail` 持续递增通常意味着 RAM 地址无效或 info 结构被上游 wrapper 覆盖。

### 进阶推进自救 & 退火 (2025-10-03 新增)
- auto_bootstrap_triggered / auto_bootstrap_remaining / auto_bootstrap_action_id: 冷启动自动注入前进动作的触发与剩余帧状态。
- secondary_script_triggered / secondary_script_remaining: 二次脚本化注入（突破 plateau）触发标记与剩余帧。
- env_positive_dx_envs / env_positive_dx_ratio: 当前 log 周期内出现正向位移 (dx>0) 的环境数量 / 占比（基于 stagnation_steps==0 且 last_x>0 的近似估计）。
- milestone_count / milestone_next: 已达成的里程碑数量与下一里程碑阈值 (x_pos)。当全局 max_x >= milestone_next 时递增并可附加里程碑奖励。
- episode_timeout_steps: 配置的强制超时截断步数（>0 生效）。

里程碑奖励 (milestone bonus)：在 wrapper 外部统计全局 max_x，超过每个间隔 (milestone_interval) 时向 shaping_raw_sum 追加 bonus，并按当前 last_scale 近似转换到 scaled_sum（保持 reward 规模可监控）。

二次脚本注入触发条件：
1. 达到 secondary_script_threshold update；
2. 尚未触发 secondary_script；
3. 当前 global max_x 未超过触发时的 plateau_baseline（初次记录）。
触发后 remaining=secondary_script_frames*num_envs，actions 被强制为前进动作 id。

Early Shaping：
- 参数 early-shaping-window / early-shaping-distance-weight 允许在窗口内对 dx 奖励做差值补偿（不修改 wrapper 内部配置，保持可逆 / 简单）。

策略调参建议：
1. 冷启动：使用 scripted-forward-frames 与较高 distance_weight (0.05)。
2. 若 20~30 updates 后 max_x 未>0：提高 distance_weight 或启用 auto-bootstrap。
3. 若 max_x 停在一个 plateau（例如 40）> secondary_script_threshold：配置 secondary 脚本注入 frames（80~150）突破停滞。
4. 进展稳定后降低 distance_weight 退火到 0.01~0.02，以便价值学习接管且避免奖励爆炸。

调试顺序建议：
distance_delta_sum -> env_positive_dx_ratio -> milestone_count -> avg_return。
当 distance_delta_sum>0 且 avg_return 长期为 0，多数是 episode 未结束（timeout 可强制截断）或死亡惩罚仍为 0（退火未完成）。

### inspect_metrics.py 示例输出
运行命令：
```
python train.py --num-envs 1 --total-updates 40 --log-interval 1 --no-compile --scripted-forward-frames 64 --metrics-path tmp_metrics.jsonl
python scripts/inspect_metrics.py --path tmp_metrics.jsonl --tail 20
```
可能输出：
```
{
	"records_analyzed": 20,
	"avg_env_positive_dx_ratio_recent": 0.62,
	"last_milestone_count": 0,
	"last_milestone_next": -1,
	"secondary_triggered": 1,
	"secondary_remaining": 0,
	"auto_bootstrap_triggered": 0,
	"first_positive_distance_update": 2,
	"latest_update": 39
}
```
字段解读：
- avg_env_positive_dx_ratio_recent：最近窗口推进环境占比均值；<0.1 说明探索受阻，可提高 distance_weight / 启用 scripted。
- secondary_triggered：已触发 plateau 二次脚本；若仍无增长，调大 frames 或阈值策略。
- first_positive_distance_update：首个 distance_delta_sum>0 的 update；过晚表示 cold start 困难需加大 early shaping。

### 自适应调度字段 (Adaptive Scheduling Fields)
当启用 `--adaptive-*` 参数后，metrics 可能包含：
- adaptive_ratio_avg: 滑动窗口简单平均推进占比 (SMA)。
- adaptive_ratio_ema: 指数滑动平均 (EMA) 去抖结果，α = 2/(window+1) 或用户指定。
- adaptive_distance_weight: 最新一次 distance_weight 调整目标值（调度器内部 shadow，未必等于即时 wrapper 内真实权重；真实奖励仍受退火与 early shaping 差分影响）。
- adaptive_entropy_beta: 最新一次策略熵系数实际写回值。
- adaptive_distance_weight_effective: 训练循环为分析而记录的 shadow distance_weight（与 wrapper 原值对比可评估自适应力度）。
- adaptive_error: 自适应过程中出现的异常截断信息（不影响主循环）。

调参指引：
1. 若 adaptive_ratio_ema 长期 < low 阈值且 distance_weight 频繁触顶，说明推进极难，可提高 scripted 或扩大 dw_max。
2. 若 adaptive_ratio_ema 长期 > high 阈值且 entropy_beta 已降至下限仍波动大，可减小 dw_lr / ent_lr 或收紧上下限差距。
3. 观察 adaptive_ratio_avg 与 adaptive_ratio_ema 差值，大幅偏离说明窗口过小或环境阶段性剧烈跳变。
