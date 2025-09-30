"""Evaluate trained Mario policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.distributions import Categorical

from src.config import EvaluationConfig, ModelConfig
from src.envs import MarioEnvConfig, create_eval_env
from src.models import MarioActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Mario policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory to store evaluation videos (defaults to metadata)")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording even if metadata enables it")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _metadata_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".json")


def load_checkpoint_metadata(checkpoint_path: Path) -> Dict:
    metadata_path = _metadata_path(checkpoint_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_eval_config(args: argparse.Namespace, metadata: Dict) -> EvaluationConfig:
    video_dir = args.video_dir or metadata.get("video_dir", "output/eval")
    record_video = not args.no_video
    env_cfg = MarioEnvConfig(
        world=int(metadata["world"]),
        stage=int(metadata["stage"]),
        action_type=str(metadata.get("action_type", "complex")),
        frame_skip=int(metadata.get("frame_skip", 4)),
        frame_stack=int(metadata.get("frame_stack", 4)),
        record_video=record_video,
        video_dir=video_dir,
    )
    return EvaluationConfig(
        episodes=args.episodes,
        video_dir=video_dir,
        record_video=record_video,
        render=args.render,
        env=env_cfg,
    )


def load_model(checkpoint_path: Path, metadata: Dict, device: torch.device) -> MarioActorCritic:
    model_data = metadata.get("model")
    if not isinstance(model_data, dict):
        raise ValueError("Checkpoint metadata is missing required 'model' configuration.")
    model_cfg = ModelConfig(**model_data)
    model = MarioActorCritic(model_cfg).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        model_state = state_dict["model"]
    else:
        model_state = state_dict
    if any(key.startswith("_orig_mod.") for key in model_state.keys()):
        model_state = {
            key.split("_orig_mod.", 1)[1]: value
            for key, value in model_state.items()
            if key.startswith("_orig_mod.")
        }
    model.load_state_dict(model_state)
    model.eval()
    return model


def evaluate(model: MarioActorCritic, cfg: EvaluationConfig, metadata: Dict, device: torch.device):
    env = create_eval_env(cfg.env, seed=metadata.get("vector_env", {}).get("base_seed"))
    total_rewards = []
    if cfg.record_video:
        Path(cfg.video_dir).mkdir(parents=True, exist_ok=True)
    expected_actions = metadata.get("model", {}).get("action_space")
    if expected_actions is not None and hasattr(env.action_space, "n"):
        if env.action_space.n != int(expected_actions):
            raise ValueError(
                f"Action space mismatch: metadata expects {expected_actions}, environment reports {env.action_space.n}."
            )
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
        world = metadata.get("world")
        stage = metadata.get("stage")
        print(
            f"Episode {episode + 1} (World {world} Stage {stage}): reward={cumulative_reward:.2f}, flag={info.get('flag_get', False)}"
        )

    print(f"Average reward over {cfg.episodes} episodes: {np.mean(total_rewards):.2f}")
    env.close()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    metadata = load_checkpoint_metadata(checkpoint_path)
    eval_cfg = build_eval_config(args, metadata)
    device = torch.device(args.device)
    model = load_model(checkpoint_path, metadata, device)
    evaluate(model, eval_cfg, metadata, device)


if __name__ == "__main__":
    main()
