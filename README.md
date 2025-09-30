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
- **Docker**：`docker build -t mario-a3c .` 后通过 `docker run --gpus all mario-a3c` 启动训练。<br>**Docker**: build with `docker build -t mario-a3c .` then launch via `docker run --gpus all mario-a3c`.
- **Conda**：`conda env create -f environment.yml && conda activate mario-a3c`。<br>**Conda**: run `conda env create -f environment.yml && conda activate mario-a3c`.
- **配置导出**：`--config-out configs/run.json` 保存最终 `TrainingConfig` 用于复现。<br>**Config export**: use `--config-out configs/run.json` to save the effective `TrainingConfig` for reproducibility.

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

## 环境要求 | Requirements & Compatibility
- 推荐 Python 3.10/3.11、PyTorch ≥ 2.1、CUDA 12（Dockerfile 提供基础镜像）。<br>Recommended Python 3.10/3.11, PyTorch ≥ 2.1, CUDA 12 (base image provided by the Dockerfile).
- 依赖 `gymnasium-super-mario-bros>=0.8.0`（首选）或 `gym-super-mario-bros>=7.4.0`，并配合 `nes-py>=8.2`。<br>Requires `gymnasium-super-mario-bros>=0.8.0` (preferred) or `gym-super-mario-bros>=7.4.0` with `nes-py>=8.2`.
- 系统需安装 FFmpeg 以生成录像。<br>FFmpeg must be installed to produce videos.
- Checkpoint 命名为 `a3c_world{W}_stage{S}_*.pt`，同名 JSON 保存运行元数据，便于评估与继续训练。<br>Checkpoints are saved as `a3c_world{W}_stage{S}_*.pt` alongside JSON metadata for evaluation or resuming.

## 致谢 | Credits
- 原始版本由 Viet Nguyen 编写，现版本参考社区对 A3C/IMPALA、可观测性实践的持续探索。<br>Originally authored by Viet Nguyen; this modern refresh draws inspiration from ongoing community experimentation around A3C/IMPALA and observability practices.
