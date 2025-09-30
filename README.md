# Super Mario Bros A3C (2025 Edition)

Modernised PyTorch implementation of Asynchronous Advantage Actor-Critic (A3C) for the Super Mario Bros Gymnasium ecosystems (compatible with both `gymnasium-super-mario-bros` and the legacy `gym-super-mario-bros`), refreshed with Gymnasium vector environments, IMPALA-style networks, V-trace targets, mixed precision, and advanced tooling for monitoring and hyperparameter search.

## Key Features

- **Vectorised Gymnasium environments** with randomized stage schedules and automatic frame stacking/pre-processing.
- **IMPALA residual backbone + recurrent core** (GRU/LSTM/Transformer) with optional NoisyLinear exploration.
- **V-trace / GAE returns** blended with prioritized replay for hybrid on/off-policy training.
- **PyTorch 2.1 pipeline** featuring `torch.compile`, AMP, cosine warm-up scheduling, gradient accumulation, and AdamW optimisation.
- **Integrated observability** via TensorBoard + optional Weights & Biases, reward tracking, and periodic checkpointing.
- **Tooling** for evaluation, Optuna-based hyperparameter search, Docker/conda environments, and video capture through Gymnasium wrappers.

## Quick Start

```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Train
python train.py --world 1 --stage 1 --num-envs 8 --total-updates 50000

# Evaluate (metadata-driven)
python test.py --checkpoint trained_models/a3c_world1_stage1_latest.pt --episodes 5

# Run TensorBoard (logs written per run timestamp)
tensorboard --logdir tensorboard/a3c_super_mario_bros

# Hyperparameter search (example: 5 trials, in-process short runs)
python scripts/optuna_search.py --trials 5
```

> **Tip:** The project targets `torch>=2.1`. If you use `gymnasium-super-mario-bros` instead of `gym-super-mario-bros`, adjust `requirements.txt` accordingly.

## Configuration Overview

`train.py` exposes a comprehensive CLI; notable flags include:

- `--num-envs`, `--rollout-steps`: control environment throughput and rollout length.
- `--recurrent-type {gru,lstm,transformer,none}` and related hidden dimensions.
- `--lr`, `--entropy-beta`, `--value-coef`, `--clip-grad`: optimise loss shaping.
- `--random-stage`, `--stage-span`: build multi-stage training schedules.
- `--no-amp`, `--no-compile`: toggle mixed precision and Torch compile.
- `--per`: enable prioritized replay augmentation.
- `--wandb-project`: stream metrics to Weights & Biases.

Training checkpoints land in `trained_models/`, while TensorBoard runs live in `tensorboard/a3c_super_mario_bros/<timestamp>`.

## Evaluation & Monitoring

- `test.py` loads checkpoints, renders optional videos, and prints episodic rewards.
- Gymnasium's `RecordVideo` wrapper writes MP4s to `output/eval/` by default.
- TensorBoard dashboards track losses, entropy, learning rate, and rolling average returns.
- Optional W&B integration mirrors the same metrics for remote monitoring.

## Advanced Usage

- **Optuna search** (`scripts/optuna_search.py`): runs short ablation-style sweeps, returning best `avg_return`.
- **Docker**: `docker build -t mario-a3c .` then `docker run --gpus all mario-a3c` launches training in a CUDA 12 base image.
- **Conda**: `conda env create -f environment.yml && conda activate mario-a3c` mirrors the Python tooling stack.
- **Config export**: `--config-out configs/run.json` persists the effective `TrainingConfig` for reproducibility.

## Project Structure

- `src/envs/`: Gymnasium environment factories, reward shaping, frame processing.
- `src/models/`: IMPALA residual blocks, recurrent policy heads, positional encodings.
- `src/algorithms/vtrace.py`: V-trace targets for off-policy corrections.
- `src/utils/`: rollout buffers, prioritized replay, learning-rate schedules, logging helpers.
- `train.py`: end-to-end training orchestrator with AMP, compile, PER, and checkpointing.
- `test.py`: evaluation runner with deterministic policy execution and video capture.
- `scripts/`: automation utilities (Optuna search, etc.).

## Debugging & known issues

If you run into environment construction issues with `AsyncVectorEnv` (especially on Python 3.12 + NumPy 2.x), see `docs/ENV_DEBUGGING_REPORT.md` for detailed diagnostics, experiments and recommended workarounds (including forcing synchronous envs for long runs).

## Requirements & Compatibility

- Python 3.10/3.11, PyTorch ≥ 2.1, CUDA 12 image supplied via Dockerfile.
- `gymnasium-super-mario-bros>=0.8.0` (preferred) or `gym-super-mario-bros>=7.4.0`, plus `nes-py>=8.2`.

> Every checkpoint is saved as `a3c_world{W}_stage{S}_*.pt` alongside a JSON metadata file of the same stem. Evaluation and resumed training automatically consume this metadata, so manual overrides of immutable parameters are no longer required.
- FFmpeg must be available on the system for video logging.

## Credits

Originally authored by Viet Nguyen; modernisation inspired by the community’s continued experimentation with asynchronous RL, IMPALA, and scalable observability practices.
