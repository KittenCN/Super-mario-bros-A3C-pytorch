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
15. PER sample 语法错误修复与 FP16 标量存储确认
16. 编译模型与 checkpoint 兼容性（延迟编译 + 灵活 state_dict 加载）
17. 奖励塑形与距离度量 (Distance reward & dynamic scaling)
18. 脚本化前进与动作探测 (Scripted progression & forward action probing)
19. RAM x_pos 回退解析与 shaping 诊断 (RAM fallback parsing)
20. 运行脚本参数扩展与原子写 (Launcher extension & atomic writes)
21. PER 间隔推送缺陷诊断 (PER interval push regression)
22. GPU 自动降级守卫 (Device auto guard)
23. 训练提示输出 (Training hints emission)

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
- 2025-10-02 更新 | Update: 脚本新增 `--update-if-lower`，允许在已有 `global_step` 小于推导值时回填；已对 `trained_models/run_balanced/` 系列 checkpoint 应用，写入 `reconstructed_step_source` 并记录新时间戳。

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

## 15. PER sample 语法错误修复与 FP16 标量存储确认
- 问题 | Problem: `src/utils/replay.py` 中 `sample` 函数的 `target_values` 与 `advantages` 两行缩进丢失，导致运行期 `IndentationError` 中止训练；同时需要确认 FP16 标量存储默认行为是否生效。
- 发现 | Discovery: 启动脚本 `run_2080ti_resume.sh` 触发训练后立即报错，日志指向行 235 缩进异常。
- 决策 | Decision: 立即修复缩进并添加注释说明；保持 FP16 (`MARIO_PER_FP16_SCALARS=1` 默认开启) 策略，采样时统一转回 float32。
- 实施 | Implementation: 恢复两行缩进，新增注释 `# 修复：此前两行意外减少缩进导致函数体语法错误`；未改动 API。确认 `PrioritizedReplay.__init__` 中 `_fp16_scalars` 环境变量逻辑与 push/sample 路径类型转换正确。
- 验证 | Validation: 静态语法检查通过（无 IndentationError）；本地 sample 路径 smoke（插入少量伪数据）返回张量 dtype 符合预期 (obs float32, actions long, target_values/advantages float32, weights float32)。
- 风险 | Risk: FP16 存储可能引入极端大/小优势截断；后续需在高方差场景监控数值稳定性（可加入单元测试对比 FP32/FP16 KL）。
- 后续 | Next: 计划添加自动化测试：填充随机数据后对比采样输出统计均值/方差在 FP32 与 FP16 模式差异阈值内 (<1e-3 相对误差)。

## 16. 编译模型与 checkpoint 兼容性（延迟编译 + 灵活 state_dict 加载）
- 问题 | Problem: 旧 checkpoint 在未编译模型下保存（键名无 `_orig_mod.` 前缀），而运行时若先执行 `torch.compile`，新模型的 `state_dict` 键空间包含 `_orig_mod.*`，直接 `load_state_dict` 会出现大量 Missing/Unexpected key 错误导致恢复失败。
- 备选 | Alternatives:
  - K1: 强制要求用户用一致版本（文档提示，不修改代码）。
  - K2: 保存与加载时始终对键进行映射（可能引入额外拷贝开销）。
  - K3: 延迟编译：恢复阶段先加载未编译模型再尝试编译；同时实现双向前缀自适应加载。
- 决策 | Decision: 采用 K3，最小化对现有保存格式的破坏，并兼容历史文件。仅在加载成功后尝试 `torch.compile`，失败则降级继续。
- 实施 | Implementation:
  1. 新增 `_flex_load_state_dict()`：检测模型和 checkpoint 各自是否带 `_orig_mod.` 前缀，必要时添加或剥离后再以 `strict=False` 加载，收集 missing/unexpected 列表用于日志。
  2. 新增 `_raw_module()`：保存 checkpoint 与 snapshot 时若模型已编译，取内部 `_orig_mod` 的 `state_dict()`，保证输出与历史未编译格式一致。
  3. `prepare_model(..., compile_now=False)` 在构造阶段接受延迟编译标志；恢复完成后再尝试编译并打印成功/失败提示。
  4. 更新 `save_checkpoint` / `save_model_snapshot` 使用 `_raw_module` 提取底层权重。
- 验证 | Validation: 人工构造已编译与未编译两种运行模式，保存 checkpoint 后在另一模式下恢复：日志仅出现少量 `flexible load issues`（或无），训练继续；单元测试（replay 相关）通过。
- 风险 | Risk: 后续若引入自定义包装器改变属性名（非 `_orig_mod`）需扩展映射；`strict=False` 可能掩盖真实结构性不兼容（通过日志截断提示缓解）。
- 后续 | Next: 计划加入单元测试：构造一个模型 -> 保存 -> 模拟编译版本加载 / 去编译版本加载，断言参数 tensor 总数与哈希一致；并在 metrics 中记录 `model_compiled=0/1` 便于实验追踪。

## 17. 奖励塑形与距离度量 (Distance reward & dynamic scaling)
- 问题 | Problem: 原始环境奖励稀疏，起步阶段随机策略难以产生正向信号；episode return 长时间贴近 0。
- 决策 | Decision: 引入 `raw_shaping = score_delta + distance_weight * dx`，并通过线性退火 scale (start→final) 输出 `scaled_shaping = raw_shaping * scale`，再与 env reward 相加；所有中间量用于诊断。
- 实施 | Implementation: `MarioRewardWrapper` 计算 dx（info.x_pos 或 RAM fallback），缓存 last_raw/last_scaled/last_dx/last_scale；metrics 增加累积和与 last_* 字段。
- CLI: `--reward-distance-weight --reward-scale-start --reward-scale-final --reward-scale-anneal-steps`。
- 验证 | Validation: 在脚本化前进 180 帧下 `env_distance_delta_sum>0` 且 `env_shaping_raw_sum` 非零；scale 随 global_step 线性插值下降。
- 风险 | Risk: distance_weight 过大可能淹没 score 差分；scale 不当会放大噪声；需监控 scaled_sum / episode_return 比值（推荐 < 0.6 初期）。

## 18. 脚本化前进与动作探测 (Scripted progression & forward action probing)
- 问题 | Problem: 初期 dx=0 导致 shaping 亦为 0；难以跨越起点惯性阶段。
- 决策 | Decision: 提供四级优先级：显式 forward_action_id > 探测 `--probe-forward-actions N` > `--scripted-sequence` > `--scripted-forward-frames`；一旦 episode 出现智能体非零策略动作或脚本耗尽即回退正常策略。
- 实施 | Implementation: 训练循环开头检查是否仍处 scripted 阶段；探测阶段对每个动作执行 N 连帧记录 dx，选择最大者；打印 `[scripted][probe] action=i dx=...` 日志。
- 验证 | Validation: 探测输出表明 extended 动作集中某组合 (RIGHT+B) dx 最大；选择后 distance 奖励开始滚动累积。
- 风险 | Risk: 过长脚本→策略依赖；探测成本 O(ActionSpace*N) 需在小 N (≤12) 与低频阶段使用。

## 19. RAM x_pos 回退解析与 shaping 诊断 (RAM fallback parsing)
- 问题 | Problem: fc_emulator 缺失 info.x_pos 导致 dx=0；distance 奖励失效。
- 决策 | Decision: 启用 `--enable-ram-x-parse` 时读取 NES RAM (0x006D 高 + 0x0086 低) 组合成位置；地址可通过 CLI 覆盖；失败计数 env_shaping_parse_fail 记录。
- 实施 | Implementation: Wrapper 缓存上一帧 x；若解析失败增加 fail 计数并不更新 dx；日志在 debug 模式输出 RAM 值。
- 验证 | Validation: 启用后在前进脚本阶段 dx>0；fail 计数保持 0；禁用开关后 raw_sum 回落 0。
- 风险 | Risk: ROM 变体地址偏移；文档提示如 dx 异常为 0 可尝试调整地址。

## 20. 运行脚本参数扩展与原子写 (Launcher extension & atomic writes)
- 问题 | Problem: 新奖励 & 脚本化 CLI 不易通过单一脚本复现；checkpoint 写入可能中途中断产生半文件。
- 决策 | Decision: `run_2080ti_resume.sh` 增加条件拼接环境变量映射；checkpoint 保存采用临时文件 + `os.replace`；中断路径写 latest 快照。
- 实施 | Implementation: 加入 `REWARD_DISTANCE_WEIGHT` 等变量；构建命令时仅在非空时追加；保存函数写 `_tmp` 后 rename；异常捕获分支调用 snapshot。
- 验证 | Validation: dry-run 展示附加 CLI；故意 Ctrl+C 后 latest.* 存在且完整；metrics 中 model_compiled 可正确反映编译状态。
- 风险 | Risk: 变量过多增加脚本复杂性；通过注释及默认留空减少心智负担。
- 2025-10-03 更新 | Update: 调整 `BOOTSTRAP=1` 默认注入参数，使 distance_weight≥0.12、scale_start=0.25、scale_final=0.12、anneal_steps=80000，并默认添加更长的 `SCRIPTED_SEQUENCE` 与 `PROBE_FORWARD=24`，以在冷启动阶段提供更强的前进信号。


## 21. PER 间隔推送缺陷诊断 (PER interval push regression)
- 问题 | Problem: 训练循环内 `per_buffer.push()` 与抽样同条件绑定，`per_sample_interval>1` 时非抽样轮的 rollout 数据被跳过，replay 实际容量远低于预期。
- 证据 | Evidence: `train.py` 第 1610 行起仅在 `update_idx % interval == 0` 分支 push；`tensorboard/a3c_super_mario_bros/20251002-125353/metrics.jsonl` 中 `replay_push_total` 始终等于单批 640；`trained_models/run_balanced` checkpoint 元数据展示 `replay_size` 长期停留在 640。
- 影响 | Impact: 优先级分布高度稀疏，`replay_fill_rate` 停滞，TD 误差更新失真，训练长时间维持 avg_return=0，SPS 也因 CPU 回放而波动。
- 决策 | Decision: 拆分 push 与 sample，确保每次 update 都写入 PER；仅在满足间隔条件时执行采样与 priority 更新；同步补充单元测试覆盖 `per_sample_interval>1`。
- 实施 | Implementation: 新增 `_per_step_update` 辅助函数集中处理 push + sample + priority 更新；日志按采样结果输出 `replay_sample_time_ms` 与细分耗时；更新 `tests/test_replay_basic.py`，通过 `_DummyModel` 验证 `per_sample_interval=3` 时 `push_total == updates * num_steps * num_envs`。
- 验证 | Validation: `pytest tests/test_replay_basic.py` 通过；`replay_push_total` 随更新累积，非抽样轮 `replay_sample_time_ms` 自动填 0。
- 后续 | Next: 修复后回归 `metrics_summary.py` 输出，观测 `replay_push_total` 与 `global_step` 同步增长；文档强调 interval 仅控制抽样频率而非写入频次。

## 22. GPU 自动降级守卫 (Device auto guard)
- 问题 | Problem: 默认 `--device auto` 在无 CUDA 环境会静默退化到 CPU，训练速度降至 <0.3 SPS，且用户不易察觉。
- 决策 | Decision: 检测 CUDA 不可用时阻止自动回退，要求显式传入 `--device cpu`，或通过环境变量允许降级。
- 实施 | Implementation: 在 `run_training` 中新增守卫；若 `MARIO_ALLOW_CPU_AUTO` 未开启且 `torch.cuda.is_available()==False`，抛出 `[train][error]` 提示；若设置允许，则打印警告后继续使用 CPU。
- 验证 | Validation: 单元测试 `tests/test_replay_basic.py` 仍通过（显式设置 `--device cpu` 场景不受影响）；本地执行 `pytest` 表明守卫未破坏现有流程。
- 后续 | Next: 需要在 README / 启动指南中提示新增环境变量；CI / 脚本应视部署环境决定是否设置 `MARIO_ALLOW_CPU_AUTO=1`。

## 23. 训练提示输出 (Training hints emission)
- 问题 | Problem: 日志虽包含指标，但缺少基于指标的即时建议，新手难以及时发现策略停滞、回放未填充等问题。
- 决策 | Decision: 在 log_interval 分支中根据关键指标输出人性化提示，如距离增量为 0、avg_return 长期为 0、回放填充率过低、GPU 利用率过低、loss 异常等。
- 实施 | Implementation: 新增 `_maybe_print_training_hints()`，根据 update 分桶避免刷屏，并在 `run_training` 写 metrics 前调用；仅当满足阈值时打印 `[train][hint] update=...` 的建议。
- 验证 | Validation: 本地短跑触发 distance_delta=0、replay_fill_rate<5% 等场景时出现提示；正常训练未触发条件时无多余输出。
- 风险 | Risk: 提示基于启发式规则，可能不适用于所有配置；后续可根据真实训练数据迭代阈值或允许用户关闭。


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
