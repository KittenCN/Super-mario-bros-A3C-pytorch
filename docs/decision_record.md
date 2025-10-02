# 决策与变更记录 | Decision & Change Log

> 目的：集中记录 2025-09-30 ~ 2025-10-01 期间针对稳定性、性能与可维护性所做的关键设计决策、备选方案、取舍与验证结果。<br>Purpose: Capture key design decisions, alternatives, trade-offs, and validation results for stability, performance, and maintainability changes between 2025-09-30 and 2025-10-01.

更新时间 | Updated: 2025-10-02

---
## 目录 | Index
1. 环境构建可观测性增强 (Per-env instrumentation)
2. 递归自动恢复策略 (Recursive auto-resume)
3. Fused 观测预处理包装器 (Fused preprocessing wrapper)
4. 优先经验回放抽样间隔 (PER sampling interval)
5. 资源监控开销削减 (Resource monitor throttling)
6. 双缓冲重叠采集 (Overlap rollout collection)
7. 未来演进候选 (Future candidates)
8. 自适应显存运行脚本 (Adaptive memory launcher)
9. 历史 checkpoint global_step 回填 (Global step backfill)
10. 训练安全退出与最新快照保障 (Safe shutdown & snapshot)
11. P1 指标体系初始实现 (Metrics JSONL + Replay stats + TB 诊断)
12. P1 指标扩展 (奖励分位数 / GPU 利用窗口 / 平均唯一率)
13. 慢 step 窗口化堆栈追踪 (Slow step windowed tracing)
14. Overlap 性能基准脚本 (benchmark_overlap.py)

---
## 1. 环境构建可观测性增强
- 问题 | Problem: 训练启动时偶发“无进展”告警，缺乏逐子环境构建耗时与阶段标记，定位困难。
- 备选 | Alternatives:
  - A1: 仅全局超时 + 打印一次开始/结束 (现状)。
  - A2: 为每个 env thunk 包装计时与进度打印。
  - A3: 结构化 JSON 事件流 + 可视化工具 (投入高)。
- 决策 | Decision: 采纳 A2，快速提升可见性，留下 A3 作为后续增强。
- 实施 | Implementation: `create_vector_env` 内打印 `[env] constructing env i/N` 与耗时；诊断日志文件保留。
- 验证 | Validation: 8 环境构建 ~0.11s；无新增卡顿；日志体积可接受。
- 风险 | Risk: 轻度 I/O 开销，可忽略。

## 2. 递归自动恢复策略
- 问题: 用户可能在嵌套目录或历史 run 中已有匹配 checkpoint，当前只查本层目录易错失已有进度。
- 备选:
  - B1: 仅当前 `save_dir` 搜索。
  - B2: 当前目录失败后递归 `trained_models/`。
  - B3: 全盘 (工作区所有子目录) 扫描（代价大）。
- 决策: 采用 B2，限制递归根为 `trained_models/`，并设置扫描上限 (limit=10000)。
- 实施: 新增 `find_matching_checkpoint_recursive()`；调用顺序：显式 `--resume` > 序号化 checkpoint > latest 快照 > 递归搜索。
- 验证: 手动放置旧 run checkpoint 于子目录，启动时成功匹配并打印来源路径。
- 风险: 大量旧 run 目录可能导致扫描耗时；已加 limit 和早停策略。

## 3. Fused 观测预处理包装器
- 问题: 原多层 Gym wrappers (灰度、缩放、归一化、堆叠) 带来多次 Python 调用与中间数组创建，增加每步延迟。
- 备选:
  - C1: 保持拆分；按需 profile 再做决定。
  - C2: 合并为单一 `FusedPreprocessWrapper`，一次性完成转换与堆叠。
  - C3: 将预处理迁移到 GPU Tensor pipeline (需更大改造)。
- 决策: C2 作为低风险快速收益；C3 留作中期方案。
- 实施: 在 `src/envs/mario.py` 新增 fused wrapper，替换原链式封装；保持接口与观测 shape 不变。
- 验证: 单步采样输出 shape 与数值范围 (0~1) 一致，训练可正常继续；启动速度未受负面影响。
- 风险: 若外部逻辑依赖中间 wrapper 副作用会失效；代码审查确认无此依赖。

## 4. 优先经验回放抽样间隔 (PER Interval)
- 问题: 每轮都执行 PER 样本抽取/更新在小批量场景下浪费 CPU/GPU 时间，加重 host↔device 拷贝。
- 备选:
  - D1: 固定每轮抽样。
  - D2: 引入 `per_sample_interval`，允许如每 N 轮执行一次。
  - D3: 动态自适应（基于近期 TD 方差），复杂度高。
- 决策: 采用 D2，CLI `--per-sample-interval` 写入配置 (`>=1`)。
- 实施: 添加字段 `replay.per_sample_interval` 并在训练循环条件判断。
- 验证: 设置 4 时观察日志 PER 分支按预期降低频率；总轮耗时下降（需后续量化报告）。
- 风险: 过大间隔可能降低样本优先级新鲜度；建议在文档提示调参与监控 TD 误差分布。

## 5. 资源监控开销削减
- 问题: 频繁 GPU 查询 (`nvidia-smi`) 产生子进程开销与潜在同步等待。
- 备选:
  - E1: 保持频率。
  - E2: 降低默认频率 (`--monitor-interval`) 并规划改用 NVML API。
  - E3: 移除监控（丧失可见性）。
- 决策: E2，保留监控价值同时减少系统调用密度。
- 实施: 调整默认间隔文档说明；内部监控类未大改，只标注后续替换点。
- 验证: 训练主循环日志节奏更稳定；系统进程数下降。
- 风险: 指标刷新不够细粒度；可通过 CLI 临时调低间隔。

## 6. 双缓冲重叠采集 (Overlap Collection)
- 问题: 同步环境下“采集 → 学习”串行，GPU 在采集阶段闲置，CPU 在学习阶段闲置，整体利用率低。
- 备选:
  - F1: 完全重写为 actor-learner 多进程（高投入）。
  - F2: 线程级双缓冲：后台线程采集下一批，主线程学习当前批次。
  - F3: 细粒度流水线（step-level overlap），复杂度高。
- 决策: 选择 F2 作为最小可行重叠；为同步向量环境提供 `--overlap-collect` 开关。
- 实施: `train.py` 增加：
  - CLI 参数 `--overlap-collect`。
  - 双缓冲结构 (`current_buffer` / `next_buffer`)。
  - `_collect_rollout` 后台线程函数 + 模型前向互斥锁（简单串行化前向防止竞态）。
  - 首次迭代前台采集，其余迭代学习阶段并行触发下一批采集。
- 验证: Help 中出现参数；语法编译通过；逻辑路径回退（未加 flag）不受影响。待基准测试收集吞吐差异。
- 风险: 模型前向仍被锁串行，收益受限；线程异常处理仅打印警告；需后续引入无锁策略（快照网络或分离 actor inference）。

## 7. 未来演进候选
### fc_emulator 阶段兼容性开关 (补充)
新增环境变量：
- `MARIO_SUPPRESS_FC_STAGE_WARN=1` 静默多阶段降级为 1-1 的警告。
- `MARIO_FC_STRICT_STAGE=1` 将该警告升级为硬错误便于严苛实验。
决策：提供灵活控制以兼顾批量日志整洁与严格实验验证。
- 环境层：硬超时 + 失败 env 自动替补；结构化 JSON 诊断。
- 训练架构：多生产者 env 线程 + 无锁环形缓冲；策略网络参数定期广播；多 GPU 扩展。
- 数据路径：GPU 原生 PER，减少 `.cpu()`；异步数据预取 (prefetch queue)。
- 监控：替换 `nvidia-smi` 为 NVML；统一 JSONL 指标 schema；增加 env FPS、采集/学习占比。
- 文档：新增性能基线表格（不同 num_envs / rollout_steps 下吞吐）。

---
## 汇总表 | Summary Table
| 变更 | 类型 | 影响 | 风险级别 | 可回退性 |
|------|------|------|----------|---------|
| Per-env 构建日志 | 可观测性 | 快速定位构建卡顿 | 低 | 直接移除打印 |
| 递归自动恢复 | 稳定性/易用 | 避免重复训练 | 低-中 (扫描负载) | 关闭递归函数 |
| Fused 预处理 | 性能 | 减少 Python 调用 | 低 | 恢复原 wrapper 链 |
| PER 抽样间隔 | 性能/灵活 | 降低频繁 PER 开销 | 中 (新鲜度) | 设为 1 即原行为 |
| 监控节流 | 性能/可观测平衡 | 降低系统调用 | 低 | 调低间隔或关闭监控 |
| 双缓冲 overlap | 性能 | 部分重叠采集/学习 | 中 (线程复杂度) | 去掉 flag 或代码块 |
| 自适应显存脚本 AUTO_MEM | 易用性/稳定 | 启动期自动降载避免 OOM | 低 | 关闭 AUTO_MEM |
| global_step 回填脚本 | 正确性/一致性 | 修复历史元数据使统计 & 调度准确 | 低 | 还原 .bak |
| 安全退出 latest 落盘 | 稳定性 | 中断时保留最近进度 | 低 | 去除 try/finally |

---
## 9. 历史 checkpoint global_step 回填
- 问题 | Problem: 早期 overlap 模式未累加 `global_step`，导致多个已保存 checkpoint JSON `global_step=0` 但 `global_update>0`。影响曲线、调度与分析。
- 备选 | Alternatives:
  - I1: 忽略（接受历史不一致）。
  - I2: 仅在恢复时即时推断，不修改文件。（恢复路径重复计算）
  - I3: 脚本离线回填并写入审计信息，幂等可回退。
- 决策 | Decision: 采用 I3，提供脚本 `scripts/backfill_global_step.py`，默认生成 `.bak` 备份，写入 `reconstructed_step_source` 元数据。
- 公式 | Formula: `global_step = global_update * num_envs * rollout_steps (assumed)`。
- 参数 | Assumption: 旧 runs 使用 rollout_steps=64（可通过 `--assume-rollout-steps` 覆盖）。
- 验证 | Validation: 对示例文件 `_0002000.json` 等执行 dry-run 输出预期 step 数；真实执行后 JSON 更新且含追踪字段。
- 风险 | Risk: 若历史 rollout_steps 与假设不符将产生系统性偏移；在追踪字段中记录方便后续二次迁移；可通过 `.bak` 还原。

## 10. 训练安全退出与最新快照保障

## 11. P1 指标体系初始实现
- 问题 | Problem: 仅 stdout 文本日志，后期分析困难；Replay 使用情况缺乏可视化；TB 目录偶尔出现空目录难以判定是否正常初始化。
- 决策 | Decision: 增加结构化 JSONL (已有 metrics.jsonl)、PER 填充与唯一率指标、TB writer 初始化标记 (add_text meta/started)。
- 实施 | Implementation: 
  - 在 `train.py` log_interval 分支写入 `replay_fill_rate`, `replay_last_unique_ratio`。
  - `replay.py` 增加 stats()；采样时更新唯一率；push 计数。
  - 训练启动打印 `[train][log] tensorboard_log_dir=...` 并在 TB (开启时) 写入 `meta/started`。
- 验证 | Validation: 本地运行最少若干 update 后 metrics.jsonl 含新字段；TB 目录出现 events 文件且 meta/started 文本标签存在；控制台打印 log_dir。
- 后续 | Next: 增加 GPU util (平均/峰值) 汇总字段；补充历史 episode return 分位数；对 metrics schema 编写文档。
- 问题 | Problem: 训练过程遇到 Ctrl+C / 异常 / 外部终止时，可能未保存最新模型或未停止监控线程，造成资源泄露或进度丢失。
- 备选 | Alternatives:
  - J1: 保持现有隐式清理（风险高）。
  - J2: 在主循环外层添加 try/except 捕获 KeyboardInterrupt，显式保存 latest 并清理。
  - J3: 细粒度在每个阶段插入保存点（复杂度上升）。
- 决策 | Decision: 选择 J2 — 低侵入、显著提升鲁棒性。保留原末尾清理逻辑，在异常路径触发 `save_model_snapshot(..., variant='latest')`。
- 实施 | Implementation: 包裹主 `for update_idx` 循环的逻辑（详见 `train.py`）外层 try/except；捕获异常后进行：
  1. 打印 `[train][warn] interrupted`；
  2. 若本轮已产生新梯度或步数，立即调用 snapshot；
  3. 心跳告知阶段 `interrupt`；
  4. 继续执行统一清理段。
- 验证 | Validation: 人为发送 KeyboardInterrupt，日志出现警告 & latest JSON/PT 文件时间戳更新；后续通过 `--resume latest` 成功恢复。
- 风险 | Risk: 异常发生于 checkpoint 写入中途可能产生不完整文件；建议后续引入原子写（先写临时文件再 rename）。

## 12. P1 指标扩展 (奖励分位数 / GPU 利用窗口 / 平均唯一率)
- 问题 | Problem: 初始指标不足以刻画训练奖励分布尾部、长期 GPU 利用趋势与 PER 重复采样动态。
- 决策 | Decision: 增量加入 p50/p90/p99 reward、`gpu_util_last` 与滑动窗口均值 `gpu_util_mean_window`、`replay_avg_unique_ratio`、`replay_push_total` 并保持向后兼容。
- 实施 | Implementation:
  - 在 `train.py` 日志分支计算最近 episode returns 分位数（缓存窗口）写入 JSONL & TensorBoard。
  - 引入 GPU 利用滑动窗口累积器（默认窗口 100 采样点）。
  - 扩展 `PrioritizedReplay.stats()` 返回 `avg_sample_unique_ratio` 与累计 push 计数。
  - 新建文档 `docs/metrics_schema.md` 规范字段含义，指导解析端使用 `dict.get()` 兼容旧记录。
- 验证 | Validation: 本地短跑运行 metrics.jsonl 出现新字段；旧历史文件无解析错误；TB 中新增 scalar 曲线。
- 取舍 | Trade-off: 轻度增加每次 log 周期计算开销，可忽略；避免实时高频统计以保持主循环轻量。
- 风险 | Risk: 分位数窗口选择过小会导致抖动（可调窗口大小后续参数化）。

## 13. 慢 step 窗口化堆栈追踪 (Slow step windowed tracing)
- 问题 | Problem: 单次偶发慢 step 可能是系统抖动，不宜立即 dump；需要在一定窗口内多次超阈值才触发堆栈抓取以减少噪声。
- 决策 | Decision: 引入窗口计数 + 阈值触发机制 (`--slow-step-trace-threshold`, `--slow-step-trace-window`)；在阈值次数达成后一次性写出 Python 线程栈到独立日志文件。
- 实施 | Implementation: `train.py` 内维护循环局部计数器，满足条件使用 `faulthandler` 与 `traceback` 输出所有线程栈；文件命名 `slow_step_trace_<timestamp>.log`。
- 验证 | Validation: 人为将阈值设置很低（如 0.001s）触发生成日志；文件内容含主线程与采集线程调用栈。
- 风险 | Risk: 若阈值配置过低导致频繁写文件；当前策略：同一窗口只写一次并重置计数。

## 14. Overlap 性能基准脚本 (benchmark_overlap.py)
- 问题 | Problem: 缺乏自动化、可重复对比 overlap=on/off 性能的手段，人工测试误差大。
- 决策 | Decision: 编写矩阵化基准脚本遍历 `(num_envs, rollout_steps)`，分别运行 sync & overlap，聚合稳定阶段吞吐与更新速率输出 CSV。
- 实施 | Implementation: 新增 `scripts/benchmark_overlap.py`，参数：`--num-envs-list`, `--rollout-list`, `--warmup-frac`；运行单个 case 解析其 metrics.jsonl 计算稳定区间平均 `env_steps_per_sec`, `updates_per_sec`，附上 `gpu_util_mean_window`。
- 验证 | Validation: 干运行单一组合成功生成 `bench_overlap_<ts>.csv`，列包含 `mode,num_envs,rollout,steps_per_sec_mean,updates_per_sec_mean,replay_fill_final,gpu_util_mean_window,duration_sec`。
- 风险 | Risk: 多组合连续运行时间较长；建议初期限制组合数量或并行拆分；脚本目前串行保障资源一致性。


---
## 验证与状态 | Validation & Status
- 代码均通过语法编译 (`py_compile`)；训练主路径（无 `--overlap-collect`）回归测试正常。
- 启动脚本新增 AUTO_MEM=1 阶梯降载；dry-run 正常，需在真实 OOM 情况下进一步验证回退路径。
- 尚未完成：系统化性能基准（需记录 baseline vs overlap 模式的 SPS、updates/s、GPU util）。
- 日志中未新增高频告警；checkpoint 恢复路径（含递归）经过人工测试。

---
## 附录 | Appendix
- 相关文件：`train.py`、`src/envs/mario.py`、`src/config.py`。
- 新增参数：`--per-sample-interval`、`--overlap-collect`。
- 参考文档：`ENV_DEBUGGING_REPORT.md`、`GPU_UTILIZATION_ANALYSIS.md`、`DOCUMENTATION_POLICY.md`。
