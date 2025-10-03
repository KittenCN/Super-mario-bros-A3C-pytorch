# ROADMAP | 训练管线后续规划

> 更新时间：2025-10-04

本文件汇总：已知未修复/待提升问题、优先级评估、阶段性目标与执行顺序。结合当前代码状态（overlap 采集、PER 内存压缩、全局步数修复补丁已完成）提出下一步具体动作。

## 0. 优先级分类标准
- P0 Critical：影响结果正确性 / 训练逻辑错误 / 数据不连续。
- P1 High：显著影响效率或可观测性（性能、监控、复现困难）。
- P2 Medium：中期收益（结构优化、易用性）。
- P3 Low / Nice-to-have：长期可选增量或实验性功能。

## 1. 待办矩阵
| 编号 | 主题 | 描述 | 风险/影响 | 优先级 | 建议窗口 |
|------|------|------|-----------|--------|----------|
| P0-1 | 历史 checkpoint global_step=0 回填 | 旧 overlap 期间保存的 ckpt 步数缺失，影响统计/调度 | 学习率调度、曲线对齐失真 | P0 | 已完成（脚本支持读取 checkpoint global_update 并在 CI dry-run 校验） |
| P0-2 | 训练中断安全退出一致性 | Ctrl+C 或 OOM 后是否完整 flush / 关闭 monitor / 保存 latest | 可能丢失进度 / 文件句柄未关 | P0 | 已完成 |
| P0-3 | PER 间隔推送缺陷 | `per_sample_interval>1` 时仅在抽样轮 push，导致大量经验未入库 | 优先级样本分布失真、训练收敛失败 | P0 | 已完成（2025-10-02 拆分 push/sample 逻辑并补单测） |
| P0-4 | GAE 回退 bootstrap 错误 | 非 V-trace 路径下 `compute_returns` 最后一帧使用当前 value 代替 bootstrap value，优势/目标被系统性低估 | 关闭 V-trace 或切换 GAE 时训练结果失真 | P0 | 已完成（2025-10-04 修复并补充单测） |
| P1-1 | metrics JSONL 结构化输出 | 当前依赖 stdout + (TB 目录为空问题)，缺少稳健数值日志 | 难以离线分析 | P1 | 已完成 |
| P1-2 | TensorBoard 事件空目录调查 | 事件文件未生成 / writer 初始化逻辑验证 | 可视化受阻 | P1 | 已完成 (init 标记 + log_dir 输出) |
| P1-3 | Overlap 模式性能基线基准 | 对比 overlap=on/off steps/s、GPU 利用率 | 不知真实收益 | P1 | 已完成 (脚本) |
| P1-4 | PER 填充率与命中率监控 | 记录 size/capacity、采样重复率 | 难调优 | P1 | 已完成 (avg_unique_ratio) |
| P1-5 | 慢 step 溯源增强 | 当前仅打印耗时 > 阈值，未记录堆栈 | 定位 IO / 环锁慢点困难 | P1 | 已完成 (窗口化追踪) |
| P1-6 | 奖励分位数 / GPU 利用滑动窗口 | 增强奖励分布 & 资源趋势洞察 | 指标更全面 | P1 | 已完成 |
| P1-7 | GPU 可用性降级告警 | `cuda` 不可用时训练静默退化为 CPU，SPS < 0.3，需提示或自动降级策略 | 训练耗时无限拉长、监控误判 | P1 | 已完成（`device auto` 提示 + `MARIO_ALLOW_CPU_AUTO` 开关） |
| P1-8 | 推进统计与塑形验证 | 修复 `env_positive_dx_ratio`、新增 `env_progress_dx/stagnation_*`、塑形烟雾测试、稳态训练脚本 | 自适应调度失真、回归难定位 | P1 | 已完成 |
| P1-9 | Cosine 调度步数缩放错误 | 余弦调度 `total_steps` 以 env step 计算，但 LR scheduler 每 update 仅前进一步，实际学习率几乎不衰减 | 学习率策略失效、长期保持初始值 | P1 | 已完成（2025-10-04 对齐 update 计数并限幅 warmup） |
| P1-10 | 构建超时线程未终止 | `call_with_timeout` 线程超时后仍在后台运行，环境构建失败时可能残留卡死 | 资源泄漏、后续构建不确定性 | P1 | 已完成（2025-10-04 加入 cancel_event 支持与超时清理） |
| P2-1 | Checkpoint 回填脚本 & 自动迁移 | 单次运行修正所有 metadata JSON | 提升一致性 | P2 | 脚本已落库，待接入批量/CI |
| P2-2 | Replay FP16 优势/价值存储 | 进一步减内存 (adv/value) 约 2x | 降低内存峰值 | P2 | 已完成（默认启用 FP16 标量存储，MARIO_PER_FP16_SCALARS 可切换） |
| P2-3 | GPU 端 PER / 混合预取 | 减少 host->device 复制 | 提升吞吐 | P2 | 已完成（`use_gpu_sampler` + torch searchsorted GPU/CPU 统一采样） |
| P2-4 | Structured Heartbeat Export | 心跳输出到单独 JSONL + 进程健康指标 | 稳定性分析 | P2 | 已完成（HeartbeatReporter 支持 JSONL 输出 + CLI 路径配置） |
| P2-5 | 自动学习率/熵调度策略 | 根据最近 return/entropy 自适应 | 更快收敛 | P2 | 已完成（AdaptiveScheduler 增加 lr_scale，自适应缩放学习率与熵） |
| P2-6 | metrics JSONL 并发写入 | Monitor 线程与训练线程并发写同一 JSONL，无锁保护 | 日志行可能交错或损坏 | P2 | 已完成（2025-10-04 共享写锁隔离） |
| P2-7 | PER push 逐元素 copy | `PrioritizedReplay.push` 逐条 `copy_`，大批量 push 时 CPU 压力大 | 大批量采样阻塞训练主循环 | P2 | 已完成（2025-10-04 批量写入向量化） |
| P3-1 | 真正 Actor-Learner 多进程 | 解除 GIL 限制 + pipeline | 大型改造 | P3 | >1 周 |
| P3-2 | Transformer 记忆裁剪 / 压缩 | 长序列更高效 | 研究性 | P3 | 待定 |
| P3-3 | NVML 直接采样替换 nvidia-smi | 减少子进程开销 | 稳定微优化 | P3 | 0.5 天 |
| P3-4 | Gymnasium 迁移 & 兼容层 | 长期生态 | 减少弃用风险 | P3 | 2–3 天 |
| P3-5 | 视频生成与策略可视诊断 | 定期采样 episode 视频 | 便于调参 | P3 | 2 天 |

## 2. 立即执行 (T0, 本周内)
1. (已完成) global_step 回填脚本批量执行：`scripts/backfill_global_step.py` 可读取 checkpoint 内的 `global_update` 自动回填并在 CI 手动工作流中 dry-run 校验。
2. (已完成) 修复 GAE 回退路径：纠正 `compute_returns` 最后一帧 bootstrap 取值，并补充 `--no-vtrace`/GAE 回归测试。
3. (已完成) 修复 PER 间隔推送缺陷：`_per_step_update` 独立处理 push 与 sample，并新增单测覆盖 `per_sample_interval>1`。
4. (已完成) GPU 可用性告警：`--device auto` 无 CUDA 时阻断启动（或设置 `MARIO_ALLOW_CPU_AUTO=1` 后警告继续）。
5. (已完成) 推进统计修复：`env_progress_dx`、`stagnation_envs`、`adaptive_ratio_fallback` 等指标上线，自适应写回与 JSONL/TensorBoard 数据一致。
6. (已完成) 自动前进回退：`--auto-bootstrap-threshold/frames` 现默认由脚本注入，可在距离依旧为 0 时启动强制前进。
7. (已完成) 训练主循环异常安全：异常中断保存 latest + 清理资源。
8. (已完成) 指标结构化校验：配合 `tests/test_shaping_smoke.py` 与 `tests/test_adaptive_injection_integration.py`，确保奖励塑形与 distance_weight 回路可持续自测。

## 3. 短期 (T1, 下 1–2 周)
1. (已完成) Overlap 性能基准：脚本 `scripts/benchmark_overlap.py`。
2. (已完成) PER 指标扩展：加入 `avg_sample_unique_ratio`、`replay_push_total`。
3. (已完成) 奖励分位数 & GPU 滑动窗口利用率。
4. (已完成) FP16 replay scalar 存储：`advantages`、`target_values` 改用半精度（采样时转回 float32）。
5. (已完成) Checkpoint 迁移工具集成 CI（新增 `backfill-global-step` workflow_dispatch dry-run）。
6. (已完成) 重放采样路径 GPU 端搬运预研：`PrioritizedReplay` 支持 `use_gpu_sampler`，可在 GPU/CPU 上通过 torch 采样。
7. 修复 PER sample 函数缩进错误（已于 2025-10-02 修复并验证）。
8. (已完成) 校准余弦调度：调整 `CosineWithWarmup.total_steps` 与 update 次数对齐，并补一轮学习率衰减单测。
9. (已完成) 替换 `call_with_timeout`：支持 cancel_event，超时后触发环境构建终止并记录诊断。

## 4. 中期 (T2, 1 个月)
1. (已完成) Heartbeat JSONL + `metrics/latest.parquet` 快照输出，支撑快速健康度排查（2025-10-04）。
2. (已完成) PER push 批量化 + `use_gpu_sampler` torch 采样原型，降低 host→device 开销（2025-10-04）。
3. (进行中) Fail-fast 卡顿检测：集成 faulthandler/thread dump，自动采集慢 step 栈信息。
4. (新增) 指标历史归档：滚动压缩 metrics JSONL / Parquet，避免长期运行文件膨胀。
5. (新增) GPU 采样性能基准：记录 CPU/GPU 路径耗时，提供自动 fallback 判定。

## 5. 长期 (T3, 1–3 个月)
1. 多进程 Actor-Learner：独立进程通过共享内存 ring 或 ZeroMQ / torch.distributed 传输 (obs, action, value)。
2. 训练配置可视化面板：轻量 Web UI (FastAPI + WebSocket) 展示最新指标 / 变更操作。
3. Auto curriculum：根据通关率动态切换 stage / 难度。
4. 云/集群适配：容器化 + 分布式拓展（多 GPU 多节点）。

## 6. 风险与依赖
| 主题 | 主要风险 | 缓解策略 |
|------|----------|----------|
| 多进程 Actor-Learner | 同步一致性、梯度延迟 | 先引入队列压测，渐进替换 overlap |
| GPU PER | 原子操作 / 采样分布偏差 | 先实现 naive GPU 版本 + 单元测试 KL 对比 |
| 自适应策略 | 造成训练震荡 | 引入冷却窗口 + 上下限 |
| 回填脚本 | 错误写入历史元数据 | 备份 .bak + 幂等校验 |

## 7. 每项任务验收 (Definition of Done 简版)
- 回填脚本：执行后所有旧 JSON `global_step` 合理且新增字段 `reconstructed_step_source:"rebuild"`。
- metrics JSONL：核心与扩展指标（loss_total, avg_return, recent_return_p50/p90/p99, env_steps_per_sec, replay_fill_rate, replay_avg_unique_ratio, gpu_util_mean_window, gpu_mem_used_mb）可被外部简单 jq 解析。
- Overlap benchmark：生成 `bench_overlap_<timestamp>.csv` 含 columns: mode, num_envs, rollout, steps_per_sec, gpu_util_mean。
- PER 填充率：日志出现 `[replay][stat] fill=XX.X% unique=YY.Y% avg_unique=ZZ.Z%`。

## 8. 近期执行顺序建议
1. (P0-1) 对剩余 checkpoint (`run_tput/`, `exp_shaping1/`) 补全 global_step 与审计字段。
2. (P2-1) backfill 工具纳入 CI / 批处理脚本。
3. (P2-3) GPU 端 PER 采样原型（含 KL 验证和耗时剖析）。

## 9. 开发协作提示
- 日志前缀规范：`[train]`, `[replay]`, `[benchmark]`, `[migrate]` 保持 grep 一致。
- 新工具脚本统一放 `scripts/` 并支持 `--dry-run` 与 `-h`。
- 文档更新：完成一类任务后在 `decision_record.md` 增加一条决策摘要，并在本 ROADMAP 勾选或挪到“已完成”小节。

## 10. 已完成（近期）
- overlap 采集线程模型 + compiled unwrap
- 内存压缩/容量自适应 + PER push 拆分（2025-10-02 修复）
- global_step 在 overlap 模式下缺失的累加修复
- 自适应 AUTO_MEM 训练脚本 + 流式输出
- 细粒度 rollout 进度心跳
- PER sample 缩进语法错误修复（2025-10-02）
- FP16 标量存储启用（默认，可通过 `MARIO_PER_FP16_SCALARS=0` 关闭）
- env_progress_dx / stagnation 指标与自适应 ratio fallback 上线（2025-10-04）
- `scripts/train_stable_sync.sh` / `train_stable_async.sh` 提供稳态训练入口（2025-10-04）
- 奖励塑形/自适应注入烟雾测试 (`tests/test_shaping_smoke.py`, `tests/test_adaptive_injection_integration.py`)
- GAE fallback 修复与 `tests/test_training_schedule.py` 覆盖（2025-10-04）
- Cosine 调度按 update 计数 + 饱和处理（2025-10-04）
- 环境构建超时取消信号 (`call_with_timeout` + `cancel_event`)（2025-10-04）
- metrics JSONL 加锁避免并发写冲突（2025-10-04）
- PER push 批量化写入优化（2025-10-04）
- backfill 脚本支持 checkpoint hint + CI dry-run 工作流（2025-10-04）
- 优先经验回放 GPU 采样原型 (`use_gpu_sampler`)（2025-10-04）
- Heartbeat JSONL + `metrics/latest.parquet` 快照导出（2025-10-04）
- 自适应学习率缩放 (`adaptive_lr_scale`) + CLI 配置（2025-10-04）

## 11. 新增下一步考量（2025-10-02 更新）
1. 指标与 heartbeat 长期归档：滚动压缩/分片 `metrics.jsonl` 与 Parquet，防止长跑占用过多磁盘。
2. GPU 采样性能基准：对比 CPU/GPU `use_gpu_sampler` 耗时，并在指标中记录自动 fallback 条件。
3. Checkpoint 原子写：采用临时文件 + fsync + rename 防止中断产生半文件。
4. 训练恢复兼容性测试矩阵：针对不同 `torch` / `gymnasium` / `nes-py` 版本做最小 smoke（脚本化）。
5. 异步模式进一步隔离验证：单独最小进程示例定位 `mario_make()` 阻塞调用栈（gdb / faulthandler）。
6. GPU 可用性守卫：启动时若检测不到 CUDA，提示用户切换设备或直接拒绝长跑，避免 0.1 SPS 的无效训练。

---
如需我直接开始第 1 步“global_step 回填脚本”实现，请提出指令（例如：`实现回填脚本`）。
