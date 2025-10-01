"""Modernised training pipeline for Super Mario Bros with A3C + V-trace."""

from __future__ import annotations

# Set multiprocessing start method at the very top to avoid fork-related
# deadlocks when native extensions or thread pools are initialised later.
import multiprocessing as _mp
# Choose a safe multiprocessing start method if one hasn't been set yet.
# Avoid forcing `fork` early as it can cause deadlocks when other threads
# or native extensions are initialised. Prefer `forkserver` then `spawn`.
try:
    current_method = _mp.get_start_method(allow_none=True)
    if current_method is None:
        for _method in ("forkserver", "spawn", "fork"):
            try:
                _mp.set_start_method(_method, force=False)
                print(f"[train] multiprocessing start method set to {_method}")
                break
            except (RuntimeError, ValueError):
                # method not available or already initialised; try next
                continue
except Exception:
    # Best-effort: if we cannot query/set the start method, continue with
    # whatever the runtime default provides.
    pass

# Suppress the legacy Gym startup notice that prints to stderr when gym is
# installed alongside gymnasium. We temporarily replace sys.stderr around the
# import of gym-related packages to avoid noisy migration messages.
import sys
import io
_stderr_orig = sys.stderr
_suppress_stderr = False
def _suppress_gym_notice(enable: bool = True):
    global _stderr_orig, _suppress_stderr
    if enable and not _suppress_stderr:
        sys.stderr = io.StringIO()
        _suppress_stderr = True
    elif not enable and _suppress_stderr:
        sys.stderr = _stderr_orig
        _suppress_stderr = False

_suppress_gym_notice(True)

import argparse
import multiprocessing as _mp

# Ensure we don't attempt to override an already-initialised start method
# later in the module; the selection above is sufficient.

import dataclasses
import json
import contextlib
import os
import math
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import time
import threading
import warnings
from torch.distributions import Categorical
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.algorithms import vtrace_returns
from src.config import (
    EvaluationConfig,
    ModelConfig,
    TrainingConfig,
    create_default_stage_schedule,
)
from src.envs.mario import MarioEnvConfig, MarioVectorEnvConfig, create_vector_env
# Import patch utilities to allow parent process to prewarm and patch nes_py
from src.envs.mario import _patch_legacy_nes_py_uint8, _patch_nes_py_ram_dtype
_suppress_gym_notice(False)
from src.models import MarioActorCritic
from src.utils import CosineWithWarmup, PrioritizedReplay, RolloutBuffer, create_logger
from src.utils.heartbeat import HeartbeatReporter
from src.utils.monitor import Monitor

try:  # optional dependency for resource monitoring
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Mario agent with modernised A3C")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action-type", type=str, default="complex", choices=["right", "simple", "complex"])
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--random-stage", action="store_true")
    parser.add_argument("--stage-span", type=int, default=4, help="Number of consecutive stages in schedule")
    parser.add_argument("--total-updates", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-res-blocks", type=int, default=3)
    parser.add_argument("--recurrent-type", type=str, default="gru", choices=["gru", "lstm", "transformer", "none"])
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--entropy-beta", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--clip-grad", type=float, default=0.5)
    parser.add_argument("--sync-env", action="store_true", help="Use synchronous vector env")
    parser.add_argument("--async-env", action="store_true", help="Force asynchronous vector env")
    parser.add_argument(
        "--confirm-async",
        action="store_true",
        help=(
            "Explicit confirmation to enable asynchronous vector env. "
            "Async mode is known to be unstable in some environments; "
            "set environment variable MARIO_ENABLE_ASYNC=1 or pass this flag to opt in."
        ),
    )
    parser.add_argument("--force-sync", action="store_true", help="Force synchronous vector env (overrides async heuristic)")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--use-noisy-linear", action="store_true")
    parser.add_argument("--log-dir", type=str, default="tensorboard/a3c_super_mario_bros")
    parser.add_argument("--save-dir", type=str, default="trained_models")
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--config-out", type=str, default=None, help="Persist effective config to JSON")
    parser.add_argument("--per", action="store_true", help="Enable prioritized replay (hybrid)")
    parser.add_argument("--per-sample-interval", type=int, default=1, help="PER 抽样更新间隔（>=1）。例如 4 表示每4次更新做一次 PER batch")
    parser.add_argument("--project", type=str, default="mario-a3c")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to run training on")
    parser.add_argument("--metrics-path", type=str, default=None, help="Path to JSONL file for periodic metrics dump")
    parser.add_argument("--env-reset-timeout", type=float, default=None, help="Timeout (s) for environment construction/reset before fallback; if unset a heuristic is used")
    parser.add_argument("--no-prewarm", action="store_true", default=False, help="Disable prewarming ROMs before training")
    parser.add_argument("--no-monitor", action="store_true", default=False, help="Disable background resource monitor")
    parser.add_argument("--monitor-interval", type=float, default=10.0, help="Monitor interval in seconds")
    parser.add_argument("--enable-tensorboard", action="store_true", default=False, help="Enable TensorBoard logging (disabled by default)")
    parser.add_argument("--parent-prewarm", action="store_true", default=False, help="Run a parent-process prewarm of the environment before launching workers")
    parser.add_argument("--parent-prewarm-all", action="store_true", default=False, help="Sequentially prewarm each env configuration in the parent before launching workers (slower, safer)")
    parser.add_argument("--worker-start-delay", type=float, default=0.2, help="Stagger delay (s) between worker initialisations when using async vector env")
    parser.add_argument("--heartbeat-interval", type=float, default=30.0, help="心跳打印间隔（秒），<=0 表示关闭")
    parser.add_argument("--heartbeat-timeout", type=float, default=300.0, help="无进展告警阈值（秒），<=0 则自动按间隔推算")
    parser.add_argument("--overlap-collect", action="store_true", help="启用采集-学习重叠：后台线程采集下一批 rollout")
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=15.0,
        help="单次环境 step 超过该秒数将发出屏幕警告，<=0 表示关闭",
    )
    return parser.parse_args(argv)


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    stage_schedule = create_default_stage_schedule(args.world, args.stage, span=args.stage_span)
    env_cfg = MarioEnvConfig(
        world=args.world,
        stage=args.stage,
        action_type=args.action_type,
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
    )
    # Default to synchronous vector env unless the user explicitly requests async
    # and confirms they accept the instability risk. Priority:
    # force_sync -> explicit async flag + confirmation/env -> sync flag -> default sync
    if getattr(args, "force_sync", False):
        async_flag = False
    elif getattr(args, "async_env", False):
        # Require explicit confirmation or environment opt-in to enable async mode
        env_allow = os.environ.get("MARIO_ENABLE_ASYNC", "0") == "1"
        if getattr(args, "confirm_async", False) or env_allow:
            async_flag = True
        else:
            print("[train] Async mode requested but not confirmed; defaulting to synchronous vector env. "
                  "To enable async, set MARIO_ENABLE_ASYNC=1 or pass --confirm-async.")
            async_flag = False
    elif getattr(args, "sync_env", False):
        async_flag = False
    else:
        async_flag = False
    vec_cfg = MarioVectorEnvConfig(
        num_envs=args.num_envs,
        asynchronous=async_flag,
        stage_schedule=tuple(stage_schedule),
        random_start_stage=args.random_stage,
        base_seed=args.seed,
        env=env_cfg,
        worker_start_delay=args.worker_start_delay,
    )
    # allow CLI to override env reset/construct timeout; if not set, scale with num_envs
    if args.env_reset_timeout is not None:
        setattr(vec_cfg, "reset_timeout", args.env_reset_timeout)
    else:
        setattr(vec_cfg, "reset_timeout", max(180.0, args.num_envs * 60.0))

    train_cfg = TrainingConfig()
    train_cfg.seed = args.seed
    train_cfg.total_updates = args.total_updates
    train_cfg.rollout = dataclasses.replace(train_cfg.rollout, num_steps=args.rollout_steps, gamma=args.gamma, tau=args.tau)
    train_cfg.optimizer = dataclasses.replace(
        train_cfg.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        beta_entropy=args.entropy_beta,
        value_loss_coef=args.value_coef,
        max_grad_norm=args.clip_grad,
    )
    train_cfg.scheduler = dataclasses.replace(train_cfg.scheduler, total_steps=args.total_updates * args.rollout_steps * args.num_envs)
    train_cfg.env = vec_cfg
    train_cfg.model = dataclasses.replace(
        train_cfg.model,
        input_channels=args.frame_stack,
        base_channels=args.base_channels,
        hidden_size=args.hidden_size,
        num_res_blocks=args.num_res_blocks,
        recurrent_type="none" if args.recurrent_type == "none" else args.recurrent_type,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        dropout=args.dropout,
        use_noisy_linear=args.use_noisy_linear,
    )
    train_cfg.mixed_precision = not args.no_amp
    train_cfg.compile_model = not args.no_compile
    train_cfg.gradient_accumulation = max(1, args.grad_accum)
    train_cfg.log_dir = args.log_dir
    train_cfg.save_dir = args.save_dir
    train_cfg.eval_interval = args.eval_interval
    train_cfg.checkpoint_interval = args.checkpoint_interval
    train_cfg.log_interval = args.log_interval
    train_cfg.resume_from = args.resume
    train_cfg.replay = dataclasses.replace(train_cfg.replay, enable=args.per)
    # 将 per_sample_interval 写入 replay 配置
    train_cfg.replay = dataclasses.replace(train_cfg.replay, per_sample_interval=max(1, getattr(args, 'per_sample_interval', 1)))
    train_cfg.device = args.device
    train_cfg.metrics_path = args.metrics_path
    return train_cfg


def prepare_model(cfg: TrainingConfig, action_space: int, device: torch.device) -> MarioActorCritic:
    model_cfg = dataclasses.replace(cfg.model, action_space=action_space, input_channels=cfg.model.input_channels)
    model = MarioActorCritic(model_cfg).to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[attr-defined]
    return model


def compute_returns(
    cfg: TrainingConfig,
    behaviour_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    dones: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gamma = cfg.rollout.gamma
    discounts = (1.0 - dones) * gamma
    if cfg.rollout.use_vtrace:
        vs, advantages = vtrace_returns(
            behaviour_log_probs=behaviour_log_probs,
            target_log_probs=target_log_probs,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            discounts=discounts,
            clip_rho_threshold=cfg.rollout.clip_rho_threshold,
            clip_c_threshold=cfg.rollout.clip_c_threshold,
        )
        return vs, advantages

    gae = torch.zeros_like(values[0])
    advantages = torch.zeros_like(values)
    vs = torch.zeros_like(values)
    next_value = bootstrap_value
    for t in reversed(range(values.shape[0])):
        next_value = values[t] if t == values.shape[0] - 1 else vs[t + 1]
        delta = rewards[t] + gamma * (1.0 - dones[t]) * next_value - values[t]
        gae = delta + gamma * cfg.rollout.tau * (1.0 - dones[t]) * gae
        advantages[t] = gae
        vs[t] = advantages[t] + values[t]
    return vs, advantages


def _checkpoint_stem(cfg: TrainingConfig) -> str:
    env_cfg = cfg.env.env
    return f"a3c_world{env_cfg.world}_stage{env_cfg.stage}"


def _serialise_stage_schedule(schedule: Sequence[Tuple[int, int]]) -> list[list[int]]:
    return [[int(world), int(stage)] for world, stage in schedule]


def _build_checkpoint_metadata(
    cfg: TrainingConfig,
    global_step: int,
    update_idx: int,
    variant: str,
) -> Dict[str, Any]:
    env_cfg = cfg.env.env
    vector_cfg = cfg.env
    metadata: Dict[str, Any] = {
        "version": 1,
        # 使用时区感知 UTC 时间避免弃用警告
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "world": env_cfg.world,
        "stage": env_cfg.stage,
        "action_type": env_cfg.action_type,
        "frame_skip": env_cfg.frame_skip,
        "frame_stack": env_cfg.frame_stack,
        "video_dir": env_cfg.video_dir,
        "record_video": env_cfg.record_video,
        "vector_env": {
            "num_envs": vector_cfg.num_envs,
            "asynchronous": vector_cfg.asynchronous,
            "stage_schedule": _serialise_stage_schedule(vector_cfg.stage_schedule),
            "random_start_stage": vector_cfg.random_start_stage,
            "base_seed": vector_cfg.base_seed,
        },
        "model": dataclasses.asdict(cfg.model),
        "save_state": {
            "global_step": int(global_step),
            "global_update": int(update_idx),
            "type": variant,
        },
    }
    return metadata


def _json_default(obj):  # noqa: D401 - 简短工具函数
    """JSON 序列化回退：转换 numpy / Path / 其它不可序列化对象。"""
    try:
        import numpy as _np  # 局部导入避免启动加重
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    # 最后兜底
    return str(obj)


def _write_metadata(base_path: Path, metadata: Dict[str, Any]) -> None:
    metadata_path = base_path.with_suffix(".json")
    payload = json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default)
    metadata_path.write_text(payload, encoding="utf-8")


def load_checkpoint_metadata(path: Path) -> Dict[str, Any]:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _metadata_matches_config(metadata: Dict[str, Any], cfg: TrainingConfig) -> bool:
    env_cfg = cfg.env.env
    if int(metadata.get("world", env_cfg.world)) != env_cfg.world:
        return False
    if int(metadata.get("stage", env_cfg.stage)) != env_cfg.stage:
        return False
    if str(metadata.get("action_type", env_cfg.action_type)) != env_cfg.action_type:
        return False
    if int(metadata.get("frame_skip", env_cfg.frame_skip)) != env_cfg.frame_skip:
        return False
    if int(metadata.get("frame_stack", env_cfg.frame_stack)) != env_cfg.frame_stack:
        return False

    vector_meta = metadata.get("vector_env", {})
    vector_cfg = cfg.env
    if int(vector_meta.get("num_envs", vector_cfg.num_envs)) != vector_cfg.num_envs:
        return False
    if bool(vector_meta.get("asynchronous", vector_cfg.asynchronous)) != vector_cfg.asynchronous:
        return False
    if bool(vector_meta.get("random_start_stage", vector_cfg.random_start_stage)) != vector_cfg.random_start_stage:
        return False
    if int(vector_meta.get("base_seed", vector_cfg.base_seed)) != vector_cfg.base_seed:
        return False
    schedule_meta = vector_meta.get("stage_schedule")
    target_schedule = _serialise_stage_schedule(vector_cfg.stage_schedule)
    if schedule_meta != target_schedule:
        return False

    model_meta = metadata.get("model")
    if not isinstance(model_meta, dict):
        return False
    model_cfg = dataclasses.asdict(cfg.model)
    for key, value in model_cfg.items():
        if model_meta.get(key) != value:
            return False

    return True


def find_matching_checkpoint(cfg: TrainingConfig) -> Optional[Tuple[Path, Dict[str, Any]]]:
    save_dir = Path(cfg.save_dir)
    if not save_dir.exists():
        return None

    matches: List[Tuple[int, int, int, Path, Dict[str, Any]]] = []
    for metadata_path in save_dir.glob("a3c_world*_stage*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _metadata_matches_config(metadata, cfg):
            continue
        checkpoint_path = metadata_path.with_suffix(".pt")
        if not checkpoint_path.exists():
            continue
        save_state = metadata.get("save_state", {})
        update_idx = int(save_state.get("global_update", 0))
        global_step = int(save_state.get("global_step", 0))
        variant = str(save_state.get("type", "checkpoint"))
        priority = 1 if variant == "checkpoint" else 0
        matches.append((update_idx, priority, global_step, checkpoint_path, metadata))

    if not matches:
        return None

    matches.sort(reverse=True)
    _, _, _, path, metadata = matches[0]
    return path, metadata


def find_matching_checkpoint_recursive(base_dir: Path, cfg: TrainingConfig, limit: int = 10000) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """Recursively search under base_dir for the best matching checkpoint.

    选择策略 | Selection strategy:
    1. 仅考虑 metadata 完全匹配当前配置 (`_metadata_matches_config`).
    2. 优先序号化 checkpoint (variant=checkpoint) 其次 latest (variant=latest)。
    3. 在同一 variant 内按 global_update / global_step 最大优先。
    4. 返回最佳 (Path, metadata)。

    参数:
        base_dir: 根目录（例如 trained_models）。
        cfg: 当前训练配置。
        limit: 最多扫描的 JSON 元数据文件数量，防止极端情况下遍历过慢。
    """
    if not base_dir.exists():
        return None
    matches: List[Tuple[int, int, int, Path, Dict[str, Any]]] = []
    count = 0
    for json_path in base_dir.rglob("a3c_world*_stage*.json"):
        count += 1
        if count > limit:
            break
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _metadata_matches_config(metadata, cfg):
            continue
        ckpt_path = json_path.with_suffix(".pt")
        if not ckpt_path.exists():
            continue
        save_state = metadata.get("save_state", {})
        update_idx = int(save_state.get("global_update", 0))
        global_step = int(save_state.get("global_step", 0))
        variant = str(save_state.get("type", "checkpoint"))
        priority = 1 if variant == "checkpoint" else 0
        matches.append((update_idx, priority, global_step, ckpt_path, metadata))
    if not matches:
        return None
    matches.sort(reverse=True)
    _, _, _, path, metadata = matches[0]
    return path, metadata


def save_checkpoint(
    cfg: TrainingConfig,
    model: MarioActorCritic,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    global_step: int,
    update_idx: int,
) -> Path:
    base_dir = Path(cfg.save_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = base_dir / f"{_checkpoint_stem(cfg)}_{update_idx:07d}.pt"
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": global_step,
        "global_update": update_idx,
        "config": dataclasses.asdict(cfg),
    }
    torch.save(payload, checkpoint_path)
    metadata = _build_checkpoint_metadata(cfg, global_step, update_idx, variant="checkpoint")
    _write_metadata(checkpoint_path, metadata)
    return checkpoint_path


def save_model_snapshot(
    cfg: TrainingConfig,
    model: MarioActorCritic,
    global_step: int,
    update_idx: int,
) -> Path:
    base_dir = Path(cfg.save_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = base_dir / f"{_checkpoint_stem(cfg)}_latest.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "global_step": global_step,
            "global_update": update_idx,
        },
        snapshot_path,
    )
    metadata = _build_checkpoint_metadata(cfg, global_step, update_idx, variant="latest")
    _write_metadata(snapshot_path, metadata)
    return snapshot_path


def maybe_save_config(cfg: TrainingConfig, path: Optional[str]):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serialisable = json.dumps(dataclasses.asdict(cfg), indent=2)
    Path(path).write_text(serialisable, encoding="utf-8")


def call_with_timeout(fn, timeout: float, *args, **kwargs):
    result: dict[str, object] = {}
    error: dict[str, BaseException] = {}

    def target():
        try:
            result["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - propagate original exception
            error["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"Call to {getattr(fn, '__name__', repr(fn))} exceeded {timeout:.1f}s")
    if error:
        raise error["error"]
    return result.get("value")


def run_training(cfg: TrainingConfig, args: argparse.Namespace) -> dict:
    # Device and performance tuning
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    hb_interval = getattr(args, "heartbeat_interval", 30.0)
    hb_timeout = getattr(args, "heartbeat_timeout", 300.0)
    if hb_timeout <= 0:
        base_interval = hb_interval if hb_interval > 0 else 30.0
        hb_timeout = max(base_interval * 4.0, 120.0)
    heartbeat = HeartbeatReporter(
        component="train",
        interval=hb_interval if hb_interval > 0 else 30.0,
        stall_timeout=hb_timeout,
        enabled=hb_interval > 0,
    )
    heartbeat.start()
    heartbeat.notify(phase="初始化", message="配置随机种子", progress=False)

    # Repro/threads
    torch.manual_seed(cfg.seed)
    # Silence known lr_scheduler ordering warning emitted in some PyTorch builds
    try:
        warnings.filterwarnings(
            "ignore",
            message=r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`.",
            category=UserWarning,
        )
    except Exception:
        pass
    # Let PyTorch choose optimal algorithms for benchmarking when input sizes are stable
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Use available CPU cores for intra-op parallelism while leaving headroom for env workers
    try:
        cpu_count = os.cpu_count() or 1
        if device.type == "cuda":
            reserve = min(cfg.env.num_envs, max(1, cpu_count // 2))
            threads = max(1, cpu_count - reserve)
        else:
            threads = max(1, cpu_count - 1)
        torch.set_num_threads(threads)
    except Exception:
        pass

    torch_threads = torch.get_num_threads()
    print(
        f"[train] device={device.type} threads={torch_threads} num_envs={cfg.env.num_envs} "
        f"rollout_steps={cfg.rollout.num_steps} grad_accum={cfg.gradient_accumulation}"
    )
    # 简单显存压力预估：obs (num_steps+1)*num_envs*frame_stack*84*84*4 bytes ~ baseline
    est_obs_mem = (cfg.rollout.num_steps + 1) * cfg.env.num_envs * cfg.model.input_channels * 84 * 84 * 4 / (1024**3)
    if device.type == "cuda" and est_obs_mem > 1.5:  # 粗略阈值
        print(f"[train][info] Estimated CPU rollout buffer size ~{est_obs_mem:.2f} GiB (pinned). Consider reducing rollout-steps or num-envs if OOM occurs.")
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if device.type == "cuda" and not alloc_conf:
        print("[train][hint] Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to mitigate fragmentation (export before running).")
    env_summary = cfg.env.env
    stage_desc = f"{env_summary.world}-{env_summary.stage}"
    print(
        f"[train] stage={stage_desc} action={env_summary.action_type} async={cfg.env.asynchronous} "
        f"total_updates={cfg.total_updates} log_interval={cfg.log_interval}"
    )
    heartbeat.notify(phase="初始化", message=f"设备 {device.type} 线程{torch_threads}", progress=False)

    # Parent-process prewarm: create and close a single env in the parent
    # to perform heavy native initialisation once before spawning workers.
    if args.parent_prewarm:
        try:
            print("[train] parent prewarming environment (single reset)...")
            # Apply nes_py compatibility patches in the parent process before
            # constructing vector envs/workers. This reduces the probability of
            # worker-side OverflowError in nes_py when using newer Python/NumPy.
            try:
                _patch_legacy_nes_py_uint8()
            except Exception:
                pass
            try:
                _patch_nes_py_ram_dtype()
            except Exception:
                pass

            pre_env_cfg = dataclasses.replace(cfg.env, num_envs=1, asynchronous=False)
            pre_env = create_vector_env(pre_env_cfg)
            pre_env.reset(seed=args.seed)
            pre_env.close()
            print("[train] parent prewarm complete")
        except Exception as exc:
            print(f"[train] parent prewarm failed: {exc}")
            heartbeat.notify(phase="env", message=f"父进程预热失败: {exc}", progress=False)

    # Optionally prewarm each worker configuration sequentially in the parent.
    # This is slower but avoids a thundering herd of first-time imports in the
    # worker processes which can lead to long blocking during vector env
    # construction. Use --parent-prewarm-all to enable.
    if args.parent_prewarm_all:
        try:
            print("[train] parent prewarming each worker configuration sequentially...")
            for idx in range(cfg.env.num_envs):
                try:
                    # Build a single-env cfg for the current stage/seed
                    single_cfg = dataclasses.replace(cfg.env, num_envs=1, asynchronous=False)
                    # rotate stage schedule to pick per-index stage deterministically
                    # this mirrors create_vector_env stage selection and therefore
                    # warms the same code paths.
                    # Apply patches before each prewarm to be extra safe.
                    try:
                        _patch_legacy_nes_py_uint8()
                    except Exception:
                        pass
                    try:
                        _patch_nes_py_ram_dtype()
                    except Exception:
                        pass

                    pre_env = create_vector_env(single_cfg)
                    pre_env.reset(seed=cfg.seed + idx)
                    pre_env.close()
                except Exception as exc:
                    print(f"[train] parent prewarm for worker {idx} failed: {exc}")
                    heartbeat.notify(phase="env", message=f"worker {idx} 预热失败: {exc}", progress=False)
            print("[train] parent prewarm-all complete")
            heartbeat.notify(phase="env", message="全量预热完成", progress=False)
        except Exception as exc:
            print(f"[train] parent prewarm-all failed: {exc}")
            heartbeat.notify(phase="env", message=f"全量预热失败: {exc}", progress=False)

    def _initialise_env(env_cfg: MarioVectorEnvConfig):
        timeout = getattr(env_cfg, "reset_timeout", 60.0)
        start_construct = time.time()
        print(f"[train] Starting environment construction with timeout={timeout}s...")
        try:
            env_instance = call_with_timeout(create_vector_env, timeout, env_cfg)
            construct_time = time.time() - start_construct
            print(f"[train] Environment constructed in {construct_time:.2f}s")
        except TimeoutError:
            construct_time = time.time() - start_construct
            print(f"[train][error] Environment construction timed out after {construct_time:.2f}s")
            raise RuntimeError("Vector env construction timed out")

        start_reset = time.time()
        print("[train] Resetting environment...")
        try:
            obs, info = call_with_timeout(env_instance.reset, timeout, seed=cfg.seed)
            reset_time = time.time() - start_reset
            print(f"[train] Environment reset in {reset_time:.2f}s")
            return env_instance, obs, info, env_cfg
        except TimeoutError as err:
            reset_time = time.time() - start_reset
            print(f"[train][error] Environment reset timed out after {reset_time:.2f}s")
            raise RuntimeError("Vector env reset timed out") from err

    env, obs_np, _, effective_env_cfg = _initialise_env(cfg.env)
    cfg.env = effective_env_cfg
    heartbeat.notify(phase="env", message=f"环境已就绪 async={cfg.env.asynchronous}", progress=False)
    heartbeat.heartbeat_now()

    # Honor force-sync CLI flag
    if getattr(args, "force_sync", False):
        cfg.env = dataclasses.replace(cfg.env, asynchronous=False)

    # Prewarm ROMs / envs to reduce first-time construction cost (can be disabled)
    if not args.no_prewarm:
        try:
            print("[train] prewarming environment (single reset)...")
            pre_env_cfg = dataclasses.replace(cfg.env, num_envs=1, asynchronous=False)
            pre_env = create_vector_env(pre_env_cfg)
            pre_env.reset(seed=cfg.seed)
            pre_env.close()
            print("[train] prewarm complete")
            heartbeat.notify(phase="env", message="父进程预热完成", progress=False)
        except Exception as exc:
            print(f"[train] prewarm failed: {exc}")
            heartbeat.notify(phase="env", message=f"预热失败: {exc}", progress=False)

    # Start background resource monitor will be created after logger/metrics are set up

    def _to_cpu_tensor(array, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        tensor = torch.as_tensor(array, dtype=dtype)
        if device.type == "cuda":
            tensor = tensor.pin_memory()
        return tensor

    obs_cpu = _to_cpu_tensor(obs_np)
    obs_gpu = obs_cpu.to(device, non_blocking=(device.type == "cuda"))

    action_space = env.single_action_space.n
    cfg.model = dataclasses.replace(cfg.model, action_space=action_space, input_channels=env.single_observation_space.shape[0])
    model = prepare_model(cfg, action_space, device)
    # Single-GPU / CPU path: model is already moved to device in prepare_model
    model.train()

    optimizer = AdamW(model.parameters(), lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay, eps=1e-5)
    schedule_fn = CosineWithWarmup(cfg.scheduler.warmup_steps, cfg.scheduler.total_steps)
    scheduler = LambdaLR(optimizer, lr_lambda=schedule_fn)
    # Use new torch.amp API where available
    try:
        scaler = torch.amp.GradScaler(enabled=cfg.mixed_precision)  # type: ignore[attr-defined]
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)

    resume_path: Optional[Path] = None
    resume_metadata: Optional[Dict[str, Any]] = None
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        resume_metadata = load_checkpoint_metadata(resume_path)
        if not _metadata_matches_config(resume_metadata, cfg):
            raise ValueError("Resume checkpoint metadata is incompatible with current configuration.")
    else:
        # 自动探测逻辑：优先编号最高的序号化 checkpoint，其次 latest 快照
        found = find_matching_checkpoint(cfg)
        if found is not None:
            resume_path, resume_metadata = found
            print(f"[train] auto-resume matched checkpoint: {resume_path.name}")
        else:
            # 尝试 latest 快照
            latest = Path(cfg.save_dir) / f"{_checkpoint_stem(cfg)}_latest.pt"
            meta = latest.with_suffix('.json')
            if latest.exists() and meta.exists():
                try:
                    meta_obj = json.loads(meta.read_text(encoding='utf-8'))
                    if _metadata_matches_config(meta_obj, cfg):
                        resume_path = latest
                        resume_metadata = meta_obj
                        print(f"[train] auto-resume using latest snapshot: {latest.name}")
                except Exception:
                    pass
        # 若仍未找到，扩展搜索整个 trained_models 根目录
        if resume_path is None:
            root_dir = Path('trained_models')
            if root_dir.exists():
                recursive = find_matching_checkpoint_recursive(root_dir, cfg)
                if recursive is not None:
                    resume_path, resume_metadata = recursive
                    try:
                        rel = resume_path.relative_to(root_dir)
                        print(f"[train] recursive-resume found checkpoint under trained_models/: {rel}")
                    except Exception:
                        print(f"[train] recursive-resume found checkpoint: {resume_path}")
        if resume_path is None:
            print("[train] no matching checkpoint found (current save_dir & recursive search) – starting fresh training")

    writer, wandb_run, run_dir = create_logger(
        cfg.log_dir,
        project=args.wandb_project,
        resume=bool(resume_path),
        enable_tb=bool(args.enable_tensorboard),
    )
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    maybe_save_config(cfg, args.config_out)
    heartbeat.notify(phase="日志", message="日志与保存目录已就绪", progress=False)

    metrics_file = Path(cfg.metrics_path).expanduser() if cfg.metrics_path else Path(run_dir) / "metrics.jsonl"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    # Start background resource monitor (safe start)
    monitor = None
    if not args.no_monitor:
        try:
            monitor = Monitor(writer, wandb_run, metrics_file, interval=args.monitor_interval)
            monitor.start()
            heartbeat.notify(phase="监控", message="资源监控已启动", progress=False)
        except Exception as exc:
            print(f"[train] Monitor failed to start: {exc}")
            monitor = None
            heartbeat.notify(phase="监控", message="资源监控启动失败", progress=False)

    step_timeout = getattr(args, "step_timeout", 0.0)
    if step_timeout is not None and step_timeout <= 0:
        step_timeout = None
    slow_step_cooldown = None
    if step_timeout is not None:
        slow_step_cooldown = max(step_timeout * 0.5, 5.0)
    last_slow_step_warn = 0.0

    rollout = RolloutBuffer(cfg.rollout.num_steps, cfg.env.num_envs, env.single_observation_space.shape, device, pin_memory=(device.type == "cuda"))
    per_buffer = None
    if cfg.replay.enable:
        per_buffer = PrioritizedReplay(
            capacity=cfg.replay.capacity,
            alpha=cfg.replay.priority_alpha,
            beta_start=cfg.replay.priority_beta,
            beta_final=cfg.replay.priority_beta_final,
            beta_steps=cfg.replay.priority_beta_steps,
            device=device,
        )
    hidden_state, cell_state = model.initial_state(cfg.env.num_envs, device)
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    global_update = 0
    last_resource_log = 0.0
    episode_returns = np.zeros(cfg.env.num_envs, dtype=np.float32)
    completed_returns: list[float] = []

    last_grad_norm = 0.0
    last_log_time = time.time()
    last_log_step = global_step
    last_log_update = global_update

    if resume_path is not None:
        print(f"[train] Loading checkpoint for resume: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        missing_warn: List[str] = []
        try:
            model.load_state_dict(checkpoint["model"])
        except Exception as e:
            raise RuntimeError(f"Failed to load model state_dict from {resume_path}: {e}")
        for key, loader in (
            ("optimizer", optimizer.load_state_dict),
            ("scheduler", scheduler.load_state_dict),
            ("scaler", scaler.load_state_dict),
        ):
            state_obj = checkpoint.get(key)
            if state_obj is None:
                missing_warn.append(key)
                continue
            try:
                loader(state_obj)  # type: ignore[arg-type]
            except Exception as e:  # noqa: BLE001
                missing_warn.append(f"{key} (error: {e})")
        global_step = int(checkpoint.get("global_step", 0))
        global_update = int(checkpoint.get("global_update", 0))
        if missing_warn:
            print("[train][warn] Partial resume: missing/failed -> " + ", ".join(missing_warn))
        print(f"[train] Resume state: update={global_update} step={global_step}")
        heartbeat.notify(phase="恢复", message=f"加载检查点 {resume_path.name}", progress=False)

    last_log_step = global_step
    last_log_update = global_update
    last_log_time = time.time()

    overlap_enabled = bool(getattr(args, "overlap_collect", False)) and not cfg.env.asynchronous
    # 只在同步向量环境下尝试重叠，异步环境不启用（避免复杂性）
    if overlap_enabled:
        print("[train] overlap collection enabled (double-buffer thread model)")
        # 已知问题：torch.compile 产生的包装在新线程前向会触发 FX tracing 警告/失败。
        # 优先尝试直接 unwrap 编译包装的原始模块 (._orig_mod)；若不可得则禁用 overlap。
        if cfg.compile_model:
            if hasattr(model, "_orig_mod"):
                try:
                    orig = getattr(model, "_orig_mod")
                    # 保持设备与训练模式
                    orig.to(device)
                    orig.train()
                    model = orig
                    cfg.compile_model = False
                    print("[train][info] unwrap compiled model -> 原始模块用于 overlap 线程前向，避免 FX 追踪冲突。")
                except Exception as e:  # pragma: no cover
                    print(f"[train][warn] unwrap 编译模型失败: {e}，禁用 overlap 以保证稳定。")
                    overlap_enabled = False
            else:
                print("[train][warn] 未发现 _orig_mod，无法安全与 torch.compile 并用；禁用 overlap 模式。")
                overlap_enabled = False
    model_lock = threading.Lock() if overlap_enabled else None

    # 用于存放后台线程结果
    class _CollectResult:
        __slots__ = ("obs_cpu", "obs_gpu", "hidden", "cell", "rollout", "episode_returns")

        def __init__(self):
            self.obs_cpu = None
            self.obs_gpu = None
            self.hidden = None
            self.cell = None
            self.rollout = None
            self.episode_returns = None

    def _collect_rollout(buffer: RolloutBuffer, start_obs_cpu: torch.Tensor, start_hidden, start_cell, ep_returns: np.ndarray, update_idx_hint: int, progress_heartbeat: bool = False):
        local_hidden = start_hidden
        local_cell = start_cell
        obs_local_cpu = start_obs_cpu
        if local_hidden is not None:
            buffer.set_initial_hidden(local_hidden.detach(), local_cell.detach() if local_cell is not None else None)
        else:
            buffer.set_initial_hidden(None, None)
        for step in range(cfg.rollout.num_steps):
            with model_lock or contextlib.nullcontext():
                # inference_mode 防止 autograd 追踪，减少线程间潜在状态交叉
                with torch.inference_mode():
                    with torch.amp.autocast(device_type=device.type, enabled=cfg.mixed_precision):
                        out = model(obs_local_cpu.to(device, non_blocking=(device.type == "cuda")), local_hidden, local_cell)
            dist = Categorical(logits=out.logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            values = out.value
            actions_cpu = actions.detach().cpu()
            log_probs_cpu = log_probs.detach().cpu()
            values_cpu = values.detach().cpu()
            step_start = time.time()
            next_obs_np, reward_np, terminated, truncated, _infos = env.step(actions_cpu.numpy())
            done = np.logical_or(terminated, truncated)
            reward_cpu = _to_cpu_tensor(reward_np, dtype=torch.float32)
            done_cpu = _to_cpu_tensor(done.astype(np.float32), dtype=torch.float32)
            buffer.insert(step, obs_local_cpu, actions_cpu, log_probs_cpu, reward_cpu, values_cpu, done_cpu)
            obs_local_cpu = _to_cpu_tensor(next_obs_np)
            ep_returns += reward_np
            if done.any():
                for idx_env, flag in enumerate(done):
                    if flag:
                        completed_returns.append(float(ep_returns[idx_env]))
                        ep_returns[idx_env] = 0.0
                if len(completed_returns) > 1000:
                    completed_returns[:] = completed_returns[-1000:]
            if out.hidden_state is not None:
                local_hidden = out.hidden_state.detach()
                if done.any():
                    mask = torch.as_tensor(done, dtype=torch.bool, device=local_hidden.device)
                    local_hidden[:, mask] = 0.0
                if out.cell_state is not None:
                    local_cell = out.cell_state.detach()
                    if done.any():
                        mask = torch.as_tensor(done, dtype=torch.bool, device=local_cell.device)
                        local_cell[:, mask] = 0.0
                else:
                    local_cell = None
            else:
                local_hidden = None
                local_cell = None
            if progress_heartbeat and (step + 1) % 8 == 0:
                heartbeat.notify(global_step=-1, phase="rollout", message=f"(bg) 采样 {step+1}/{cfg.rollout.num_steps}")
        buffer.set_next_obs(obs_local_cpu)
        return obs_local_cpu, local_hidden, local_cell, ep_returns

    # 双缓冲：当前 buffer / 下一个 buffer
    current_buffer = rollout
    next_buffer = RolloutBuffer(cfg.rollout.num_steps, cfg.env.num_envs, env.single_observation_space.shape, device, pin_memory=(device.type == "cuda")) if overlap_enabled else None
    pending_thread = None
    pending_result = _CollectResult() if overlap_enabled else None

    def _start_background_collection(start_obs_cpu, hidden_state, cell_state, ep_returns, next_update_idx):
        assert overlap_enabled and next_buffer is not None and pending_result is not None
        next_buffer.reset()
        pending_result.obs_cpu = None
        def _runner():
            try:
                obs_cpu_final, h_final, c_final, ep_ret = _collect_rollout(next_buffer, start_obs_cpu, hidden_state, cell_state, ep_returns, next_update_idx, progress_heartbeat=False)
                pending_result.obs_cpu = obs_cpu_final
                pending_result.hidden = h_final
                pending_result.cell = c_final
                pending_result.episode_returns = ep_ret
            except Exception as e:  # pragma: no cover - 仅日志
                print(f"[train][warn] background collection failed: {e}")
        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        return t

    for update_idx in range(global_update, cfg.total_updates):
        update_wall_start = time.time()
        heartbeat.notify(
            update_idx=update_idx,
            global_step=global_step,
            phase="rollout",
            message=f"采集第{update_idx}轮数据",
        )
        if not overlap_enabled:
            # 原始同步采集逻辑保留
            rollout.reset()
            if hidden_state is not None:
                rollout.set_initial_hidden(hidden_state.detach(), cell_state.detach() if cell_state is not None else None)
            else:
                rollout.set_initial_hidden(None, None)
            for step in range(cfg.rollout.num_steps):
                # 采集阶段不需要梯度，使用 inference_mode 防止构建计算图，降低显存占用
                with torch.inference_mode():
                    with torch.amp.autocast(device_type=device.type, enabled=cfg.mixed_precision):
                        output = model(obs_gpu, hidden_state, cell_state)
                        logits = output.logits
                        values = output.value
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                actions_cpu = actions.detach().cpu()
                log_probs_cpu = log_probs.detach().cpu()
                values_cpu = values.detach().cpu()
                step_start_time = time.time()
                next_obs_np, reward_np, terminated, truncated, infos = env.step(actions_cpu.numpy())
                step_duration = time.time() - step_start_time
                if step_timeout is not None and step_duration > step_timeout:
                    now_warn = time.time()
                    if slow_step_cooldown is None or now_warn - last_slow_step_warn >= slow_step_cooldown:
                        print(f"[train][warn] env.step 耗时 {step_duration:.1f}s (阈值 {step_timeout:.1f}s)，可能存在卡顿。")
                        heartbeat.notify(phase="rollout", message=f"env.step {step_duration:.1f}s", progress=False)
                        last_slow_step_warn = now_warn
                done = np.logical_or(terminated, truncated)
                reward_cpu = _to_cpu_tensor(reward_np, dtype=torch.float32)
                done_cpu = _to_cpu_tensor(done.astype(np.float32), dtype=torch.float32)
                rollout.insert(step, obs_cpu, actions_cpu, log_probs_cpu, reward_cpu, values_cpu, done_cpu)
                obs_cpu = _to_cpu_tensor(next_obs_np)
                obs_gpu = obs_cpu.to(device, non_blocking=(device.type == "cuda"))
                episode_returns += reward_np
                if done.any():
                    for idx_env, flag in enumerate(done):
                        if flag:
                            completed_returns.append(float(episode_returns[idx_env]))
                            episode_returns[idx_env] = 0.0
                    if len(completed_returns) > 1000:
                        completed_returns = completed_returns[-1000:]
                if output.hidden_state is not None:
                    hidden_state = output.hidden_state.detach()
                    if done.any():
                        mask = torch.as_tensor(done, dtype=torch.bool, device=device)
                        hidden_state[:, mask] = 0.0
                    if output.cell_state is not None:
                        cell_state = output.cell_state.detach()
                        if done.any():
                            mask = torch.as_tensor(done, dtype=torch.bool, device=device)
                            cell_state[:, mask] = 0.0
                    else:
                        cell_state = None
                else:
                    hidden_state = None
                    cell_state = None
                # 主动释放局部变量引用，帮助 Python 更快回收（避免长生存期列表）
                del output, logits, values, dist, actions, log_probs
                global_step += cfg.env.num_envs
                if (step + 1) % 8 == 0 or step + 1 == cfg.rollout.num_steps:
                    heartbeat.notify(global_step=global_step, phase="rollout", message=f"采样进度 {step + 1}/{cfg.rollout.num_steps}")
            rollout.set_next_obs(obs_cpu)
        else:
            # overlap 模式
            if update_idx == global_update:
                # 首次需要前台采集以获得初始缓冲
                current_buffer.reset()
                obs_cpu, hidden_state, cell_state, episode_returns = _collect_rollout(current_buffer, obs_cpu, hidden_state, cell_state, episode_returns, update_idx, progress_heartbeat=True)
                obs_gpu = obs_cpu.to(device, non_blocking=(device.type == "cuda"))
            else:
                # 等待后台线程完成上一个 next_buffer
                if pending_thread is not None:
                    pending_thread.join()
                # 将 next_buffer 变为当前
                current_buffer, next_buffer = next_buffer, current_buffer  # type: ignore
                rollout = current_buffer  # 保持后续变量引用一致
                obs_cpu = pending_result.obs_cpu  # type: ignore
                obs_gpu = obs_cpu.to(device, non_blocking=(device.type == "cuda"))
                hidden_state = pending_result.hidden
                cell_state = pending_result.cell
                if pending_result.episode_returns is not None:
                    episode_returns = pending_result.episode_returns
            # 在学习阶段开始前启动下一次后台采集（除非最后一次）
            if update_idx + 1 < cfg.total_updates:
                pending_thread = _start_background_collection(obs_cpu, hidden_state, cell_state, episode_returns, update_idx + 1)

        rollout_duration = time.time() - update_wall_start
        heartbeat.notify(
            update_idx=update_idx,
            global_step=global_step,
            phase="learn",
            message="回放采样完成，开始计算回报",
        )
        learn_start = time.time()

        with torch.no_grad():
            bootstrap = model(obs_gpu, hidden_state, cell_state).value.detach()

        sequences = rollout.get_sequences(device)

        with torch.amp.autocast(device_type=device.type, enabled=cfg.mixed_precision):
            target_output = model(
                sequences["obs"],
                sequences["initial_hidden"].hidden,
                sequences["initial_hidden"].cell,
            )
            target_logits = target_output.logits
            target_values = target_output.value

        target_dist = Categorical(logits=target_logits)
        target_log_probs = target_dist.log_prob(sequences["actions"])
        entropy = target_dist.entropy().mean()

        vs, advantages = compute_returns(
            cfg,
            behaviour_log_probs=sequences["behaviour_log_probs"],
            target_log_probs=target_log_probs,
            rewards=sequences["rewards"],
            values=target_values,
            bootstrap_value=bootstrap,
            dones=sequences["dones"],
        )

        per_loss = torch.tensor(0.0, device=device)
        if per_buffer is not None and (update_idx % cfg.replay.per_sample_interval == 0):
            obs_flat = sequences["obs"].reshape(-1, *sequences["obs"].shape[2:])
            actions_flat = sequences["actions"].reshape(-1)
            vs_flat = vs.detach().reshape(-1, 1)
            adv_flat = advantages.detach().reshape(-1, 1)
            per_buffer.push(obs_flat, actions_flat, vs_flat, adv_flat)
            per_sample = per_buffer.sample(cfg.rollout.batch_size)
            if per_sample is not None:
                with torch.amp.autocast(device_type=device.type, enabled=cfg.mixed_precision):
                    per_output = model(per_sample.observations, None, None)
                    per_dist = Categorical(logits=per_output.logits)
                    per_values = per_output.value.squeeze(-1)
                per_log_probs = per_dist.log_prob(per_sample.actions)
                per_policy_loss = -(per_log_probs * per_sample.advantages.detach() * per_sample.weights).mean()
                td_error_raw = per_sample.target_values - per_values
                per_value_loss = (td_error_raw.pow(2) * per_sample.weights).mean()
                per_loss = per_policy_loss + cfg.optimizer.value_loss_coef * per_value_loss
                per_buffer.update_priorities(per_sample.indices, td_error_raw.detach().abs())

        policy_loss = -(advantages.detach() * target_log_probs).mean()
        value_loss = F.mse_loss(target_values, vs.detach())
        total_loss = policy_loss + cfg.optimizer.value_loss_coef * value_loss - cfg.optimizer.beta_entropy * entropy + per_loss
        total_loss = total_loss / cfg.gradient_accumulation

        scaler.scale(total_loss).backward()

        if (update_idx + 1) % cfg.gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)
            last_grad_norm = float(grad_norm_tensor)
            # Execute optimizer step via scaler then advance scheduler only if optimizer succeeded
            try:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                try:
                    scheduler.step()
                except Exception:
                    pass
            except Exception:
                # If optimizer step failed, still zero grads to avoid stale grads
                try:
                    optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass

        learn_duration = time.time() - learn_start
        total_update_time = time.time() - update_wall_start
        heartbeat.notify(
            update_idx=update_idx,
            global_step=global_step,
            phase="log",
            message=f"单轮耗时 {total_update_time:.1f}s",
        )

        if update_idx % cfg.log_interval == 0:
            loss_total = float(total_loss.item())
            loss_policy = float(policy_loss.item())
            loss_value = float(value_loss.item())
            loss_per = float(per_loss.item())
            entropy_val = float(entropy.item())
            lr_value = float(scheduler.get_last_lr()[0])
            recent_returns = completed_returns[-100:] if completed_returns else []
            avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            return_std = float(np.std(recent_returns)) if recent_returns else 0.0
            return_max = float(np.max(recent_returns)) if recent_returns else 0.0
            return_min = float(np.min(recent_returns)) if recent_returns else 0.0
            now = time.time()
            updates_since_last = max(update_idx - last_log_update, 1)
            steps_since_last = max(global_step - last_log_step, cfg.env.num_envs)
            elapsed = max(now - last_log_time, 1e-6)
            updates_per_sec = updates_since_last / elapsed
            env_steps_per_sec = steps_since_last / elapsed

            writer.add_scalar("Loss/total", loss_total, global_step)
            writer.add_scalar("Loss/policy", loss_policy, global_step)
            writer.add_scalar("Loss/value", loss_value, global_step)
            writer.add_scalar("Loss/per", loss_per, global_step)
            writer.add_scalar("Policy/entropy", entropy_val, global_step)
            writer.add_scalar("LearningRate", lr_value, global_step)
            writer.add_scalar("Reward/avg_return", avg_return, global_step)
            writer.add_scalar("Reward/recent_std", return_std, global_step)
            writer.add_scalar("Grad/global_norm", last_grad_norm, global_step)

            print(
                f"[train] update={update_idx} step={global_step} avg_return={avg_return:.2f} "
                f"loss={loss_total:.4f} lr={lr_value:.2e} grad={last_grad_norm:.3f} "
                f"steps/s={env_steps_per_sec:.1f} upd/s={updates_per_sec:.2f} "
                f"update={total_update_time:.1f}s rollout={rollout_duration:.1f}s learn={learn_duration:.1f}s"
            )

            metrics_entry = {
                "timestamp": time.time(),
                "update": update_idx,
                "global_step": global_step,
                "loss_total": loss_total,
                "loss_policy": loss_policy,
                "loss_value": loss_value,
                "loss_per": loss_per,
                "entropy": entropy_val,
                "learning_rate": lr_value,
                "avg_return": avg_return,
                "recent_return_std": return_std,
                "recent_return_max": return_max,
                "recent_return_min": return_min,
                "episodes_completed": len(completed_returns),
                "grad_norm": last_grad_norm,
                "env_steps_per_sec": env_steps_per_sec,
                "updates_per_sec": updates_per_sec,
                "update_time": total_update_time,
                "rollout_time": rollout_duration,
                "learn_time": learn_duration,
            }

            if wandb_run is not None:
                wandb_run.log({**metrics_entry})

            # Resource monitoring: GPU/CPU/memory
            def _get_resource_stats():
                stats = {}
                # GPU stats via torch
                try:
                    if torch.cuda.is_available():
                        stats["gpu_count"] = torch.cuda.device_count()
                        for gi in range(torch.cuda.device_count()):
                            stats[f"gpu_{gi}_mem_alloc_bytes"] = float(torch.cuda.memory_allocated(gi))
                            stats[f"gpu_{gi}_mem_reserved_bytes"] = float(torch.cuda.memory_reserved(gi))
                except Exception:
                    pass

                # GPU utilization and memory via nvidia-smi (if available)
                try:
                    cmd = [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ]
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                    if res.returncode == 0 and res.stdout.strip():
                        lines = [l for l in res.stdout.strip().splitlines() if l.strip()]
                        for idx, line in enumerate(lines):
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) >= 3:
                                util, mem_used, mem_total = parts[:3]
                                stats[f"gpu_{idx}_util_pct"] = float(util)
                                stats[f"gpu_{idx}_mem_used_mb"] = float(mem_used)
                                stats[f"gpu_{idx}_mem_total_mb"] = float(mem_total)
                except Exception:
                    pass

                # Process and system stats via psutil if available
                collected = False
                if psutil is not None:
                    try:
                        p = psutil.Process()
                        stats["proc_cpu_percent"] = float(p.cpu_percent(interval=None))
                        stats["proc_mem_rss_bytes"] = float(p.memory_info().rss)
                        stats["system_cpu_percent"] = float(psutil.cpu_percent(interval=None))
                        stats["system_mem_percent"] = float(psutil.virtual_memory().percent)
                        collected = True
                    except Exception:
                        pass
                if not collected:
                    try:
                        load1, load5, load15 = os.getloadavg()
                        stats["load1"] = float(load1)
                    except Exception:
                        pass

                return stats

            stats = {}
            current_time = time.time()
            if current_time - last_resource_log >= 60.0:
                stats = _get_resource_stats()
                last_resource_log = current_time
                for name, val in stats.items():
                    try:
                        writer.add_scalar(f"Resource/{name}", float(val), global_step)
                    except Exception:
                        pass
                if wandb_run is not None and stats:
                    logd = dict(stats)
                    logd["global_step"] = global_step
                    wandb_run.log(logd)

            # 降低 nvidia-smi 调用频率：仅当距离上次记录 >=120s 再写入 GPU util（否则仅使用 torch.cuda.memory 信息）
            if stats:
                metrics_entry["resource"] = stats

            try:
                with metrics_file.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(metrics_entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

            # 训练健康度检查：长时间没有完成 episode 可能意味着仍在使用 Dummy env（历史 bug）或奖励逻辑异常
            if update_idx > 0 and len(completed_returns) == 0 and update_idx % (cfg.log_interval * 10) == 0:
                print("[train][warn] No completed episodes yet – verify environments are real (not Dummy) and reward/done signals are propagating.")

            last_log_time = now
            last_log_step = global_step
            last_log_update = update_idx

        if cfg.checkpoint_interval and update_idx % cfg.checkpoint_interval == 0 and update_idx > 0:
            checkpoint_path = save_checkpoint(
                cfg,
                model,
                optimizer,
                scheduler,
                scaler,
                global_step,
                update_idx,
            )
            print(f"[train] checkpoint saved -> {checkpoint_path}")
            heartbeat.notify(phase="checkpoint", message=f"保存 {checkpoint_path.name}", progress=False)

        if cfg.eval_interval and update_idx % cfg.eval_interval == 0 and update_idx > 0:
            save_model_snapshot(cfg, model, global_step, update_idx)
            print("[train] latest snapshot refreshed")
            heartbeat.notify(phase="checkpoint", message="最新快照已更新", progress=False)

    # Robust cleanup: ensure resources are closed even if some fail
    try:
        # Prefer graceful close, but attempt terminate if available to ensure workers are cleaned up
        try:
            env.close()
        except Exception:
            try:
                env.close(terminate=True)  # type: ignore[arg-type]
            except Exception:
                pass
    except Exception:
        pass
    try:
        if monitor is not None:
            monitor.stop()
    except Exception:
        pass
    try:
        if 'writer' in locals() and writer is not None:
            writer.close()
    except Exception:
        pass
    try:
        if 'wandb_run' in locals() and wandb_run is not None:
            wandb_run.finish()
    except Exception:
        pass
    # Allow subprocesses and file descriptors to settle and run a GC cycle
    try:
        time.sleep(0.1)
    except Exception:
        pass
    try:
        import gc

        gc.collect()
    except Exception:
        pass

    heartbeat.notify(phase="完成", message="训练循环结束，整理指标", progress=False)
    heartbeat.heartbeat_now()
    heartbeat.stop(timeout=2.0)

    avg_return = float(np.mean(completed_returns[-100:])) if completed_returns else 0.0
    return {
        "global_step": global_step,
        "total_updates": cfg.total_updates,
        "final_lr": scheduler.get_last_lr()[0],
        "avg_return": avg_return,
    }


def train() -> None:
    args = parse_args()
    cfg = build_training_config(args)
    run_training(cfg, args)


if __name__ == "__main__":
    train()
