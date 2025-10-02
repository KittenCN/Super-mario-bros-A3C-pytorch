## Metrics Schema

结构化 `metrics.jsonl` 每行一个 JSON 记录，字段含义：

核心训练: timestamp, update, global_step, loss_total/policy/value/per, entropy, learning_rate, grad_norm
回报窗口 (最近100 episodes): avg_return, recent_return_std/max/min, recent_return_p50/p90/p99
性能: env_steps_per_sec, updates_per_sec, update_time, rollout_time, learn_time
Replay (PER): replay_size, replay_capacity, replay_fill_rate, replay_last_unique_ratio, replay_avg_unique_ratio, replay_push_total
资源: resource.* (GPU/CPU/Mem) + gpu_util_last, gpu_util_mean_window (最近100快照均值)

新增/关联字段（2025-10-02 更新）:
- per_sample_interval: 来自 checkpoint 元数据的抽样频率策略（非 metrics 行内字段，但分析时应读取 checkpoint JSON 以还原训练语义）。
- replay_priority_mean/p50/p90/p99: 已在日志中输出，用于监控优先级分布是否塌缩。
- replay_sample_time_ms: 单次 PER 采样耗时（毫秒，包含 CPU 采样与数据拷贝），用于评估 GPU 化潜在收益（2025-10-02 新增）。

原子写保证: 自 2025-10-02 起 checkpoint `.pt` 与对应 `.json` 元数据使用临时文件 + `os.replace` 原子落盘，减少中断产生半文件风险。

兼容: 新增字段向后兼容; 消费端使用 dict.get(key, default) 处理缺失。
未来: TD 误差分布、episode length 分位数、成功率、阶段通关率。
