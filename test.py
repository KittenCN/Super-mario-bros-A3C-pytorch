"""Evaluate trained Mario policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def _infer_from_filename(path: Path) -> Optional[Dict]:
    # Example names: a3c_world1_stage1_latest.pt or a3c_world1_stage1_0000100.pt
    stem = path.stem
    # crude parse
    # split by '_' expecting tokens like a3c, world1, stage1, ...
    parts = stem.split('_')
    world = None
    stage = None
    for p in parts:
        if p.startswith('world'):
            try:
                world = int(p.replace('world',''))
            except Exception:
                pass
        if p.startswith('stage'):
            try:
                stage = int(p.replace('stage',''))
            except Exception:
                pass
    if world is None or stage is None:
        return None
    return {
        "world": world,
        "stage": stage,
        "action_type": "complex",
        "frame_skip": 4,
        "frame_stack": 4,
        "video_dir": "output/videos",
        "record_video": False,
        "vector_env": {
            "num_envs": 1,
            "asynchronous": False,
            "stage_schedule": [[world, stage]],
            "random_start_stage": False,
            "base_seed": 42,
        },
        # minimal model fields (will be overwritten if real config found)
        "model": {
            "action_space": None,
            "input_channels": 4,
            "base_channels": 32,
            "hidden_size": 512,
            "num_res_blocks": 3,
            "recurrent_type": "gru",
            "transformer_layers": 2,
            "transformer_heads": 4,
            "dropout": 0.1,
            "use_noisy_linear": False,
        },
        "save_state": {"global_step": 0, "global_update": 0, "type": "latest"},
    }


def load_checkpoint_metadata(checkpoint_path: Path) -> Dict:
    metadata_path = _metadata_path(checkpoint_path)
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    # Fallback: inspect checkpoint payload to reconstruct minimal metadata
    try:
        payload = torch.load(checkpoint_path, map_location='cpu')
    except Exception:
        payload = {}
    meta_from_ckpt = None
    if isinstance(payload, dict):
        # training pipeline saved 'config' or we can introspect partial fields
        if 'config' in payload and isinstance(payload['config'], dict):
            meta_from_ckpt = payload['config'].get('env') or None
        # try to build unified metadata when full training config present
        if 'config' in payload:
            cfg_all = payload['config']
            try:
                env_cfg = cfg_all.get('env', {})
                model_cfg = cfg_all.get('model', {})
                reconstructed = {
                    "world": env_cfg.get('env', {}).get('world', 1) if isinstance(env_cfg.get('env'), dict) else cfg_all.get('env', {}).get('env', {}).get('world', 1),
                    "stage": env_cfg.get('env', {}).get('stage', 1) if isinstance(env_cfg.get('env'), dict) else cfg_all.get('env', {}).get('env', {}).get('stage', 1),
                    "action_type": env_cfg.get('env', {}).get('action_type', 'complex') if isinstance(env_cfg.get('env'), dict) else 'complex',
                    "frame_skip": env_cfg.get('env', {}).get('frame_skip', 4) if isinstance(env_cfg.get('env'), dict) else 4,
                    "frame_stack": env_cfg.get('env', {}).get('frame_stack', 4) if isinstance(env_cfg.get('env'), dict) else 4,
                    "video_dir": env_cfg.get('env', {}).get('video_dir', 'output/videos') if isinstance(env_cfg.get('env'), dict) else 'output/videos',
                    "record_video": env_cfg.get('env', {}).get('record_video', False) if isinstance(env_cfg.get('env'), dict) else False,
                    "vector_env": {
                        "num_envs": env_cfg.get('num_envs', 1),
                        "asynchronous": env_cfg.get('asynchronous', False),
                        "stage_schedule": env_cfg.get('stage_schedule', [[1,1]]),
                        "random_start_stage": env_cfg.get('random_start_stage', False),
                        "base_seed": env_cfg.get('base_seed', 42),
                    },
                    "model": model_cfg,
                    "save_state": {
                        "global_step": payload.get('global_step', 0),
                        "global_update": payload.get('global_update', 0),
                        "type": 'latest'
                    },
                }
                meta_from_ckpt = reconstructed
            except Exception:
                pass
    if meta_from_ckpt is None:
        meta_from_ckpt = _infer_from_filename(checkpoint_path)
    if meta_from_ckpt is None:
        raise FileNotFoundError(f"Metadata file not found and unable to reconstruct: {metadata_path}")
    # Write a sidecar so后续评估可重用
    try:
        metadata_path.write_text(json.dumps(meta_from_ckpt, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"[eval] Reconstructed metadata written to {metadata_path}")
    except Exception:
        pass
    return meta_from_ckpt


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

    # Load raw payload first to allow inferring action space before constructing model
    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model" in payload:
        model_state_candidate = payload["model"]
    else:
        model_state_candidate = payload

    # Infer action_space & hidden_size if missing (metadata reconstruction path)
    inferred_action = None if model_data.get("action_space") in (None, 0) else model_data.get("action_space")
    inferred_hidden = None
    if isinstance(model_state_candidate, dict):
        for k, v in model_state_candidate.items():
            if hasattr(v, "shape") and len(v.shape) == 2:
                if k.endswith("actor_head.weight") and inferred_action is None:
                    inferred_action = int(v.shape[0])
                    inferred_hidden = int(v.shape[1])
                elif k.endswith("critic_head.weight") and inferred_hidden is None:
                    inferred_hidden = int(v.shape[1])
                elif k.endswith("fc.0.weight") and inferred_hidden is None:
                    inferred_hidden = int(v.shape[0])
    # Fallback action space from temporary env if still unknown
    if inferred_action is None:
        try:
            tmp_env_cfg = MarioEnvConfig(
                world=int(metadata.get("world", 1)),
                stage=int(metadata.get("stage", 1)),
                action_type=str(metadata.get("action_type", "complex")),
                frame_skip=int(metadata.get("frame_skip", 4)),
                frame_stack=int(metadata.get("frame_stack", 4)),
            )
            tmp_env = create_eval_env(tmp_env_cfg, seed=None)
            if hasattr(tmp_env.action_space, "n"):
                inferred_action = int(tmp_env.action_space.n)
            tmp_env.close()
        except Exception:
            pass
    if inferred_action is not None:
        model_data["action_space"] = inferred_action
    if inferred_hidden is not None:
        model_data["hidden_size"] = inferred_hidden
    if model_data.get("action_space") in (None, 0):
        raise ValueError("Unable to infer action_space for model reconstruction.")

    model_cfg = ModelConfig(**model_data)
    model = MarioActorCritic(model_cfg).to(device)
    model_state = model_state_candidate
    if isinstance(model_state_candidate, dict) and any(key.startswith("_orig_mod.") for key in model_state_candidate.keys()):
        model_state = {
            key.split("_orig_mod.", 1)[1]: value
            for key, value in model_state_candidate.items()
            if key.startswith("_orig_mod.")
        }
    if isinstance(model_state, dict):
        model.load_state_dict(model_state, strict=False)
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
