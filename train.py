"""Modernised training pipeline for Super Mario Bros with A3C + V-trace."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Optional

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import time
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
from src.envs import MarioEnvConfig, MarioVectorEnvConfig, create_vector_env
from src.models import MarioActorCritic
from src.utils import CosineWithWarmup, PrioritizedReplay, RolloutBuffer, create_logger


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
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--use-noisy-linear", action="store_true")
    parser.add_argument("--log-dir", type=str, default="tensorboard/a3c_super_mario_bros")
    parser.add_argument("--save-dir", type=str, default="trained_models")
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--config-out", type=str, default=None, help="Persist effective config to JSON")
    parser.add_argument("--per", action="store_true", help="Enable prioritized replay (hybrid)")
    parser.add_argument("--project", type=str, default="mario-a3c")
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
    vec_cfg = MarioVectorEnvConfig(
        num_envs=args.num_envs,
        asynchronous=not args.sync_env,
        stage_schedule=tuple(stage_schedule),
        random_start_stage=args.random_stage,
        base_seed=args.seed,
        env=env_cfg,
    )

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


def maybe_save_config(cfg: TrainingConfig, path: Optional[str]):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serialisable = json.dumps(dataclasses.asdict(cfg), indent=2)
    Path(path).write_text(serialisable, encoding="utf-8")


def run_training(cfg: TrainingConfig, args: argparse.Namespace) -> dict:
    # Device and performance tuning
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    # Repro/threads
    torch.manual_seed(cfg.seed)
    # Let PyTorch choose optimal algorithms for benchmarking when input sizes are stable
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Use available CPU cores for intra-op parallelism
    try:
        cpu_count = os.cpu_count() or 1
        torch.set_num_threads(max(1, cpu_count // 2))
    except Exception:
        pass

    env = create_vector_env(cfg.env)
    obs_np, _ = env.reset(seed=cfg.seed)
    # Move observations to device using pinned memory and non-blocking copy for better throughput
    obs = torch.from_numpy(obs_np).pin_memory().to(device, non_blocking=True, dtype=torch.float32)

    action_space = env.single_action_space.n
    cfg.model = dataclasses.replace(cfg.model, action_space=action_space, input_channels=env.single_observation_space.shape[0])
    model = prepare_model(cfg, action_space, device)
    # Single-GPU / CPU path: model is already moved to device in prepare_model
    model.train()

    optimizer = AdamW(model.parameters(), lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay, eps=1e-5)
    schedule_fn = CosineWithWarmup(cfg.scheduler.warmup_steps, cfg.scheduler.total_steps)
    scheduler = LambdaLR(optimizer, lr_lambda=schedule_fn)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)

    writer, wandb_run = create_logger(cfg.log_dir, project=args.wandb_project, resume=bool(cfg.resume_from))
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    maybe_save_config(cfg, args.config_out)

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
    episode_returns = np.zeros(cfg.env.num_envs, dtype=np.float32)
    completed_returns: list[float] = []

    if cfg.resume_from:
        checkpoint = torch.load(cfg.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        global_step = checkpoint.get("global_step", 0)
        global_update = checkpoint.get("global_update", 0)

    for update_idx in range(global_update, cfg.total_updates):
        rollout.reset()
        if hidden_state is not None:
            rollout.set_initial_hidden(hidden_state.detach(), cell_state.detach() if cell_state is not None else None)
        else:
            rollout.set_initial_hidden(None, None)

        for step in range(cfg.rollout.num_steps):
            obs_tensor = obs.clone()
            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
                output = model(obs_tensor, hidden_state, cell_state)
                logits = output.logits
                values = output.value

            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            next_obs_np, reward_np, terminated, truncated, infos = env.step(actions.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            # Use pinned memory + non-blocking copy for numerical arrays
            reward = torch.from_numpy(reward_np).pin_memory().to(device, non_blocking=True, dtype=torch.float32)
            done_tensor = torch.from_numpy(done.astype(np.float32)).pin_memory().to(device, non_blocking=True)

            rollout.insert(step, obs_tensor, actions, log_probs, reward, values, done_tensor)

            obs = torch.from_numpy(next_obs_np).pin_memory().to(device, non_blocking=True, dtype=torch.float32)
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
                    mask = torch.tensor(done, dtype=torch.bool, device=device)
                    hidden_state[:, mask] = 0.0
                if output.cell_state is not None:
                    cell_state = output.cell_state.detach()
                    if done.any():
                        mask = torch.tensor(done, dtype=torch.bool, device=device)
                        cell_state[:, mask] = 0.0
                else:
                    cell_state = None
            else:
                hidden_state = None
                cell_state = None

            global_step += cfg.env.num_envs

        rollout.set_next_obs(obs)

        with torch.no_grad():
            bootstrap = model(obs, hidden_state, cell_state).value.detach()

        sequences = rollout.get_sequences()

        with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
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
        if per_buffer is not None:
            obs_flat = sequences["obs"].reshape(-1, *sequences["obs"].shape[2:])
            actions_flat = sequences["actions"].reshape(-1)
            vs_flat = vs.detach().reshape(-1, 1)
            adv_flat = advantages.detach().reshape(-1, 1)
            per_buffer.push(obs_flat, actions_flat, vs_flat, adv_flat)
            per_sample = per_buffer.sample(cfg.rollout.batch_size)
            if per_sample is not None:
                with torch.cuda.amp.autocast(enabled=cfg.mixed_precision):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if update_idx % cfg.log_interval == 0:
            writer.add_scalar("Loss/total", total_loss.item(), global_step)
            writer.add_scalar("Loss/policy", policy_loss.item(), global_step)
            writer.add_scalar("Loss/value", value_loss.item(), global_step)
            writer.add_scalar("Loss/per", per_loss.item(), global_step)
            writer.add_scalar("Policy/entropy", entropy.item(), global_step)
            writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], global_step)
            avg_return = float(np.mean(completed_returns[-100:])) if completed_returns else 0.0
            writer.add_scalar("Reward/avg_return", avg_return, global_step)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "loss_total": total_loss.item(),
                        "loss_policy": policy_loss.item(),
                        "loss_value": value_loss.item(),
                        "loss_per": per_loss.item(),
                        "entropy": entropy.item(),
                        "lr": scheduler.get_last_lr()[0],
                        "avg_return": avg_return,
                        "global_step": global_step,
                    }
                )

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
                try:
                    import psutil

                    p = psutil.Process()
                    stats["proc_cpu_percent"] = float(p.cpu_percent(interval=None))
                    stats["proc_mem_rss_bytes"] = float(p.memory_info().rss)
                    stats["system_cpu_percent"] = float(psutil.cpu_percent(interval=None))
                    stats["system_mem_percent"] = float(psutil.virtual_memory().percent)
                except Exception:
                    try:
                        load1, load5, load15 = os.getloadavg()
                        stats["load1"] = float(load1)
                    except Exception:
                        pass

                return stats

            stats = _get_resource_stats()
            for name, val in stats.items():
                try:
                    writer.add_scalar(f"Resource/{name}", float(val), global_step)
                except Exception:
                    pass
            if wandb_run is not None and stats:
                logd = dict(stats)
                logd["global_step"] = global_step
                wandb_run.log(logd)

        if cfg.checkpoint_interval and update_idx % cfg.checkpoint_interval == 0 and update_idx > 0:
            checkpoint_path = Path(cfg.save_dir) / f"mario_a3c_{update_idx:07d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "global_step": global_step,
                    "global_update": update_idx,
                    "config": dataclasses.asdict(cfg),
                },
                checkpoint_path,
            )

        if cfg.eval_interval and update_idx % cfg.eval_interval == 0 and update_idx > 0:
            torch.save(model.state_dict(), Path(cfg.save_dir) / "latest_model.pt")

    env.close()
    writer.close()
    if wandb_run is not None:
        wandb_run.finish()

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
