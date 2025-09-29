"""Mario environment factories leveraging Gymnasium vector APIs."""

from __future__ import annotations

import dataclasses
import itertools
import os
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from nes_py.wrappers import JoypadSpace

try:  # pragma: no cover - prefer maintained Gymnasium fork
    from gymnasium_super_mario_bros import make as mario_make  # type: ignore
    from gymnasium_super_mario_bros.actions import (  # type: ignore
        RIGHT_ONLY,
        SIMPLE_MOVEMENT,
        COMPLEX_MOVEMENT,
    )
except ImportError:
    try:
        from gym_super_mario_bros import make as mario_make  # type: ignore
        from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT  # type: ignore
    except ImportError as err:  # pragma: no cover
        raise RuntimeError(
            "Install gymnasium-super-mario-bros >= 0.8.0"
        ) from err


def _patch_legacy_nes_py_uint8() -> None:
    """Patch nes_py ROM helpers to avoid numpy uint8 overflow on Python 3.12 / NumPy 2."""

    try:  # pragma: no cover - defensive for alternate forks
        from nes_py import _rom as nes_rom  # type: ignore
    except ImportError:  # pragma: no cover
        return

    rom_cls = getattr(nes_rom, "NESRom", None)
    if rom_cls is None or getattr(rom_cls, "_uint8_patch_applied", False):  # pragma: no cover
        return

    def _wrap_property(prop_name: str) -> None:
        prop = getattr(rom_cls, prop_name, None)
        if isinstance(prop, property) and prop.fget is not None:
            original = prop.fget

            def patched(self, _orig=original):
                value = _orig(self)
                if hasattr(value, "item"):
                    return int(value.item())
                return int(value)

            setattr(rom_cls, prop_name, property(patched))

    for name in ("prg_rom_size", "chr_rom_size", "mapper", "submapper"):
        _wrap_property(name)

    setattr(rom_cls, "_uint8_patch_applied", True)


_patch_legacy_nes_py_uint8()

from .wrappers import MarioRewardWrapper, ProgressInfoWrapper, RewardConfig, TransformObservation, TransformReward


_ACTION_SPACES = {
    "right": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
}


@dataclasses.dataclass
class MarioEnvConfig:
    """Configuration for a single Super Mario Bros environment."""

    world: int = 1
    stage: int = 1
    action_type: str = "complex"
    frame_skip: int = 4
    frame_stack: int = 4
    render_mode: Optional[str] = None
    record_video: bool = False
    video_dir: str = "output/videos"
    reward_config: RewardConfig = dataclasses.field(default_factory=RewardConfig)

    def env_id(self) -> str:
        return f"SuperMarioBros-{self.world}-{self.stage}-v3"


@dataclasses.dataclass
class MarioVectorEnvConfig:
    """Configuration for vectorised Mario environments."""

    num_envs: int = 8
    asynchronous: bool = True
    stage_schedule: Sequence[Tuple[int, int]] = ((1, 1),)
    random_start_stage: bool = False
    base_seed: int = 1234
    observation_normalize: bool = True
    compile_model: bool = True
    use_amp: bool = True

    env: MarioEnvConfig = dataclasses.field(default_factory=MarioEnvConfig)


def _select_actions(action_type: str) -> Sequence[Tuple[str, ...]]:
    if action_type not in _ACTION_SPACES:
        valid = ", ".join(_ACTION_SPACES.keys())
        raise ValueError(f"Unknown action_type '{action_type}'. Valid: {valid}")
    return _ACTION_SPACES[action_type]


def list_available_stages(max_world: int = 8, max_stage: int = 4) -> List[Tuple[int, int]]:
    """List all available world-stage combinations."""

    return [(world, stage) for world in range(1, max_world + 1) for stage in range(1, max_stage + 1)]


def _make_single_env(config: MarioEnvConfig, seed: Optional[int] = None):
    actions = _select_actions(config.action_type)

    def thunk():
        env = mario_make(
            config.env_id(),
            apply_api_compatibility=True,
            render_mode=config.render_mode,
        )
        env = JoypadSpace(env, actions)
        env = ProgressInfoWrapper(env)
        env = MarioRewardWrapper(env, config.reward_config)
        env = gym.wrappers.FrameSkip(env, config.frame_skip)
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = TransformObservation(env, lambda obs: obs.astype(np.float32))
        env = TransformReward(env, float)
        env = gym.wrappers.FrameStack(env, config.frame_stack, new_axis=0)
        env = TransformObservation(env, lambda obs: np.asarray(obs, dtype=np.float32) / 255.0)
        if config.record_video:
            os.makedirs(config.video_dir, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=config.video_dir,
                episode_trigger=lambda episode_id: True,
                name_prefix=f"world{config.world}_stage{config.stage}",
            )
        if seed is not None:
            env.reset(seed=seed)
        return env

    return thunk


def create_vector_env(config: MarioVectorEnvConfig):
    """Create a vectorised Mario environment."""

    if not config.stage_schedule:
        raise ValueError("stage_schedule must contain at least one stage")

    stage_list = list(config.stage_schedule)
    if not stage_list:
        raise ValueError("stage_schedule must contain at least one stage")

    stage_cycle: Iterable[Tuple[int, int]] = itertools.cycle(stage_list)

    env_fns = []
    for idx in range(config.num_envs):
        if config.random_start_stage:
            world, stage = random.choice(stage_list)
        else:
            world, stage = next(stage_cycle)
        stage_env_cfg = dataclasses.replace(config.env, world=world, stage=stage)
        seed = config.base_seed + idx
        env_fns.append(_make_single_env(stage_env_cfg, seed=seed))

    vector_cls = AsyncVectorEnv if config.asynchronous else SyncVectorEnv
    vec_env = vector_cls(env_fns)
    return vec_env


def create_eval_env(config: MarioEnvConfig, seed: Optional[int] = None):
    """Create a single evaluation environment with rendering enabled."""

    eval_cfg = dataclasses.replace(config, render_mode="rgb_array")
    return _make_single_env(eval_cfg, seed=seed)()
