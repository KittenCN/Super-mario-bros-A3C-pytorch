## Metrics Schema

结构化 `metrics.jsonl` 每行一个 JSON 记录，字段含义：

核心训练: timestamp, update, global_step, loss_total/policy/value/per, entropy, learning_rate, grad_norm
回报窗口 (最近100 episodes): avg_return, recent_return_std/max/min, recent_return_p50/p90/p99
性能: env_steps_per_sec, updates_per_sec, update_time, rollout_time, learn_time
Replay (PER): replay_size, replay_capacity, replay_fill_rate, replay_last_unique_ratio, replay_avg_unique_ratio, replay_push_total
资源: resource.* (GPU/CPU/Mem) + gpu_util_last, gpu_util_mean_window (最近100快照均值)

兼容: 新增字段向后兼容; 消费端使用 dict.get(key, default) 处理缺失。
未来: TD 误差分布、episode length 分位数、成功率、阶段通关率。
