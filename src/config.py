"""Configuration dataclasses for Mario A3C training pipeline."""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence, Tuple

import torch

from src.envs import MarioEnvConfig, MarioVectorEnvConfig


@dataclasses.dataclass
class ModelConfig:
    input_channels: int = 4
    action_space: int = 12
    base_channels: int = 32
    hidden_size: int = 512
    num_res_blocks: int = 3
    recurrent_type: str = "gru"  # options: gru, lstm, transformer
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.1
    use_noisy_linear: bool = False


@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float = 2.5e-4
    weight_decay: float = 0.01
    eps: float = 1e-5
    max_grad_norm: float = 0.5
    beta_entropy: float = 0.01
    value_loss_coef: float = 0.5


@dataclasses.dataclass
class SchedulerConfig:
    warmup_steps: int = 10_000
    total_steps: int = 5_000_000


@dataclasses.dataclass
class RolloutConfig:
    num_steps: int = 64
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 1.0
    use_vtrace: bool = True
    clip_rho_threshold: float = 1.0
    clip_c_threshold: float = 1.0


@dataclasses.dataclass
class ReplayBufferConfig:
    capacity: int = 200_000
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_beta_final: float = 1.0
    priority_beta_steps: int = 1_000_000
    enable: bool = True
    per_sample_interval: int = 1  # 新增：间隔多少次 update 执行一次 PER 采样（1 表示每次）


@dataclasses.dataclass
class TrainingConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    episodes: Optional[int] = None
    total_updates: int = 200_000
    eval_interval: int = 5_000
    checkpoint_interval: int = 10_000
    log_interval: int = 100
    mixed_precision: bool = True
    compile_model: bool = True
    gradient_accumulation: int = 1
    save_dir: str = "trained_models"
    log_dir: str = "tensorboard/a3c_super_mario_bros"
    resume_from: Optional[str] = None
    metrics_path: Optional[str] = None

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = dataclasses.field(default_factory=SchedulerConfig)
    rollout: RolloutConfig = dataclasses.field(default_factory=RolloutConfig)
    replay: ReplayBufferConfig = dataclasses.field(default_factory=ReplayBufferConfig)
    env: MarioVectorEnvConfig = dataclasses.field(default_factory=MarioVectorEnvConfig)


@dataclasses.dataclass
class EvaluationConfig:
    episodes: int = 5
    video_dir: str = "output"
    record_video: bool = True
    render: bool = False
    env: MarioEnvConfig = dataclasses.field(default_factory=MarioEnvConfig)


def create_default_stage_schedule(world: int, stage: int, span: int = 4) -> Sequence[Tuple[int, int]]:
    stages = []
    for offset in range(span):
        stage_idx = (stage - 1 + offset) % 4 + 1
        world_idx = world + (stage - 1 + offset) // 4
        stages.append((world_idx, stage_idx))
    return stages
