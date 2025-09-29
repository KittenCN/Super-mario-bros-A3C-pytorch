"""Evaluate trained Mario policies."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.distributions import Categorical

from src.config import EvaluationConfig, ModelConfig
from src.envs import MarioEnvConfig, create_eval_env
from src.models import MarioActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Mario policy")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action-type", type=str, default="complex", choices=["right", "simple", "complex"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video-dir", type=str, default="output/eval")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--recurrent-type", type=str, default="gru", choices=["gru", "lstm", "transformer", "none"])
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-noisy-linear", action="store_true")
    return parser.parse_args()


def build_eval_config(args: argparse.Namespace) -> EvaluationConfig:
    env_cfg = MarioEnvConfig(
        world=args.world,
        stage=args.stage,
        action_type=args.action_type,
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
        record_video=True,
        video_dir=args.video_dir,
    )
    return EvaluationConfig(episodes=args.episodes, video_dir=args.video_dir, render=args.render, env=env_cfg)


def load_model(checkpoint_path: str, action_space: int, cfg: argparse.Namespace, device: torch.device) -> MarioActorCritic:
    model_cfg = ModelConfig(
        input_channels=cfg.frame_stack,
        action_space=action_space,
        hidden_size=cfg.hidden_size,
        recurrent_type="none" if cfg.recurrent_type == "none" else cfg.recurrent_type,
        transformer_layers=cfg.transformer_layers,
        transformer_heads=cfg.transformer_heads,
        dropout=cfg.dropout,
        use_noisy_linear=cfg.use_noisy_linear,
    )
    model = MarioActorCritic(model_cfg).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        model.load_state_dict(state_dict["model"])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate(model: MarioActorCritic, cfg: EvaluationConfig, device: torch.device):
    env = create_eval_env(cfg.env, seed=cfg.episodes)
    total_rewards = []
    Path(cfg.video_dir).mkdir(parents=True, exist_ok=True)
    for episode in range(cfg.episodes):
        obs, info = env.reset()
        obs = torch.from_numpy(obs).to(device, dtype=torch.float32)
        hidden_state, cell_state = model.initial_state(1, device)
        done = False
        cumulative_reward = 0.0
        while not done:
            with torch.no_grad():
                output = model(obs.unsqueeze(0), hidden_state, cell_state)
                dist = Categorical(logits=output.logits.squeeze(0))
                action = dist.probs.argmax().item()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            cumulative_reward += reward
            obs = torch.from_numpy(next_obs).to(device, dtype=torch.float32)
            if output.hidden_state is not None:
                hidden_state = output.hidden_state.detach()
                if output.cell_state is not None:
                    cell_state = output.cell_state.detach()
            if cfg.render:
                env.render()
        total_rewards.append(cumulative_reward)
        print(f"Episode {episode + 1}: reward={cumulative_reward:.2f}, flag={info.get('flag_get', False)}")

    print(f"Average reward over {cfg.episodes} episodes: {np.mean(total_rewards):.2f}")
    env.close()


def main():
    args = parse_args()
    eval_cfg = build_eval_config(args)
    device = torch.device(args.device)
    env = create_eval_env(eval_cfg.env)
    action_space = env.action_space.n
    env.close()

    model = load_model(args.checkpoint, action_space, args, device)
    evaluate(model, eval_cfg, device)


if __name__ == "__main__":
    main()
