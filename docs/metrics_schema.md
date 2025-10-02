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
 - model_compiled: 1 表示当前运行模型是 torch.compile 产物（含 `_orig_mod`），0 表示未编译或已 unwrap（2025-10-02 新增）。
 - state_dict_load_issues.log: 若恢复时出现 missing/unexpected 键，会将完整列表追加写入该文件，仅在首次或有差异时更新；metrics 行内不重复膨胀内容。

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

脚本化前进与探测辅助诊断：
- 若使用 `--probe-forward-actions`，在训练 stdout 日志中应出现每个动作的 dx 结果；选择的 forward_action_id 之后应使 `env_distance_delta_sum` 与 `env_shaping_raw_sum` 上升。
- 若启用 `--scripted-sequence` 但 raw_sum 仍为 0，检查：脚本动作是否匹配扩展动作集合；RAM 解析是否启用；是否在脚本阶段之前 episode 已重置。
- `env_shaping_parse_fail` 持续递增通常意味着 RAM 地址无效或 info 结构被上游 wrapper 覆盖。
