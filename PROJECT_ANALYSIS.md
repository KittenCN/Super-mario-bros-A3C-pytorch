# Super Mario Bros A3C 项目工作原理分析（2025 版）

## 项目概览

- 采用 PyTorch 2.1+ 搭配 Gymnasium 的 A3C / IMPALA 训练框架，目标是教会智能体通关 `SuperMarioBros`。
- `train.py`、`test.py` 与 `scripts/optuna_search.py` 构成训练、评估、超参自动化的顶层入口。
- 核心逻辑拆分到 `src/` 目录：环境工厂、模型结构、算法（V-trace）、数据缓冲与监控工具均模块化封装。

## 环境系统（`src/envs/`）

- `MarioEnvConfig` / `MarioVectorEnvConfig`：描述单环境与向量化环境的所有参数（关卡、动作集合、帧跳、帧堆叠、随机舞台等）。
- `create_vector_env`：使用 `gymnasium.vector.AsyncVectorEnv`（或同步模式）构建批量环境。主要处理：
  - `FrameSkip` + 灰度化 + `ResizeObservation(84×84)` + `FrameStack`；输出张量形状 `(stack, 84, 84)`。
  - `MarioRewardWrapper` 根据得分增量塑形奖励，并在终局时附加旗帜奖励 / 失败惩罚（同时统一缩放奖励幅度）。
  - `ProgressInfoWrapper` 在 info 中提供累计进度/终止状态，便于日志统计。
  - 可选 `RecordVideo`：直接将 MP4 录像写入指定目录。
- `create_eval_env`：面向评估与录像的单环境构造函数。

## 模型结构（`src/models/`）

- `MarioActorCritic`：
  - Backbone 采用 IMPALA 风格的 `ImpalaBlock`（卷积 + 两层残差块 + max pooling）。
  - 中间 `fc` 将卷积特征映射至 `hidden_size`。
  - 支持多种序列建模方式：`GRU`、`LSTM`、`TransformerEncoder`（内置 `PositionalEncoding`）或关闭循环结构。
  - actor/critic 头可选 `NoisyLinear` 振幅噪声以提升探索能力。
- `ModelOutput` 统一携带 logits、value、隐藏状态/Cell（针对 LSTM）及中间特征，便于训练阶段复用。

## 算法与数据结构

- `src/algorithms/vtrace.py`：实现 IMPALA 提出的 V-trace 目标，结合行为策略与目标策略的 log prob 计算校正项。
- `src/utils/rollout.py`：`RolloutBuffer` 在 GPU 上缓存 `(T, N)` 维的观测、动作、reward、log_prob、value 等，支持回放序列。
- `src/utils/replay.py`：
  - `PrioritizedReplay` 实现优先经验回放（基于优势绝对值）并提供重要性采样权重。
  - `ReplaySample` 封装取样结果（obs、action、target_value、advantage、weights、indices）供额外的 value/policy 更新使用。
- `src/utils/schedule.py`：`CosineWithWarmup` 学习率策略；`src/utils/logger.py` 提供 TensorBoard + 可选 W&B 日志接口。

## 训练流程（`train.py` -> `run_training`）

1. **配置解析**：`parse_args` 接收大量现代化开关（`--random-stage`、`--per`、`--no-amp`、`--no-compile` 等），`build_training_config` 组装 `TrainingConfig`。
2. **初始化**：
   - 构造向量环境以获取动作/观测空间尺寸。
   - 创建 `MarioActorCritic`，按需启用 `torch.compile`，并设置 AMP（`GradScaler`）。
   - 使用 `AdamW` + 余弦退火（含 warmup）学习率调度。
3. **Rollout 收集**：
   - `RolloutBuffer` 收集 `num_steps × num_envs` 的轨迹。
   - 每步基于模型输出采样动作，与环境交互，并在 episode 结束时清零对应隐藏状态，记录回报。
4. **目标计算**：
   - 使用 `compute_returns` 调用 V-trace（或 GAE）得到 `vs`（state value target）与 `advantages`。
   - 若启用 PER，将展平后的样本 push 到优先级缓冲区并进行一次额外的带权更新。
5. **反向传播**：
   - `policy_loss`、`value_loss` 与熵正则共同构成主损失；再加上 PER 产生的次级损失。
   - 支持梯度累积、AMP 缩放、裁剪、学习率调度。
6. **日志与持久化**：
   - TensorBoard/W&B 记录损失、熵、学习率、近百局平均回报。
   - 定期保存 checkpoint（state dict + 优化器/调度器/Scaler 状态 + 运行配置）。
7. **返回指标**：`run_training` 输出 `avg_return` 等统计，以便 Optuna 等外部脚本直接读取。

## 评估流程（`test.py`）

- 解析 CLI -> 构造录像环境 -> 加载模型权重。
- 采用贪心策略执行；保持 GRU/LSTM/Transformer 状态，支持实时渲染与视频导出。
- 输出每一局的累积奖励与过旗状态。

## 自动化与辅助脚本

- `scripts/optuna_search.py`：
  - 通过 `train.run_training` 执行短程训练作为目标函数。
  - 搜索学习率、隐藏维度、残差块数量、循环类型、熵系数等关键超参。
- `requirements.txt`、`environment.yml`、`Dockerfile`：提供 pip、conda、Docker 三种环境搭建方式，默认包含 FFmpeg 安装。

## 数据流摘要

1. 向量化环境输出 `(N, stack, 84, 84)` -> 模型得到 logits/value/隐藏状态。
2. Rollout 写入 `RolloutBuffer`，轨迹结束后统一计算 V-trace 目标。
3. 主梯度更新结合 PER 采样的额外批次，提升样本效率与价值函数精度。
4. 指标（loss、entropy、avg_return）通过 TensorBoard 与 W&B 同步记录；checkpoint 定期保存。

## 依赖与运行注意事项

- 推荐 Python 3.11 / PyTorch ≥ 2.1，Gymnasium ≥ 0.28；若使用 `gymnasium-super-mario-bros`，确保 import 名称兼容。
- 系统需预装 `ffmpeg` 才能启用视频录制；Dockerfile 已预装。
- Optuna 搜索默认使用较短行程（500 updates），可通过 CLI 调整 `--trials`、`--storage` 等参数拓展到分布式实验。
