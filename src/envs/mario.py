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

_MARIO_PACKAGE = None

try:  # pragma: no cover - prefer Gymnasium fork when available
    from gymnasium_super_mario_bros import make as mario_make  # type: ignore
    from gymnasium_super_mario_bros.actions import (  # type: ignore
        RIGHT_ONLY,
        SIMPLE_MOVEMENT,
        COMPLEX_MOVEMENT,
    )
    _MARIO_PACKAGE = "gymnasium-super-mario-bros"
except ImportError:
    try:
        from gym_super_mario_bros import make as mario_make  # type: ignore
        from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT  # type: ignore
        _MARIO_PACKAGE = "gym-super-mario-bros"
    except ImportError as err:  # pragma: no cover
        raise RuntimeError(
            "Install gymnasium-super-mario-bros>=0.8.0 or gym-super-mario-bros>=7.4.0"
        ) from err


def _patch_legacy_nes_py_uint8() -> None:
    """Patch nes_py ROM helpers to avoid numpy uint8 overflow on Python 3.12 / NumPy 2."""

    try:  # pragma: no cover - defensive for alternate forks
        from nes_py import _rom as nes_rom  # type: ignore
    except ImportError:  # pragma: no cover
        return

    import inspect

    # Try to find the ROM class under a couple of likely names across nes_py versions
    rom_cls = getattr(nes_rom, "NESRom", None) or getattr(nes_rom, "ROM", None)
    if rom_cls is None or getattr(rom_cls, "_uint8_patch_applied", False):  # pragma: no cover
        return

    def _to_int(value):
        # Convert numpy scalars or arrays to a Python int when possible
        try:
            if hasattr(value, "item"):
                value = value.item()
        except Exception:
            pass
        try:
            return int(value)
        except Exception:
            # Fallback: return original value if cannot convert
            return value

    def _wrap_name(name: str) -> None:
        # Try to get descriptor from the class dict first to avoid triggering
        # instance descriptor logic.
        orig = None
        if name in rom_cls.__dict__:
            orig = rom_cls.__dict__[name]
        else:
            orig = getattr(rom_cls, name, None)

        # If it's a property object, wrap its fget
        if isinstance(orig, property) and orig.fget is not None:
            original = orig.fget

            def patched(self, _orig=original):
                value = _orig(self)
                return _to_int(value)

            setattr(rom_cls, name, property(patched))
            return

        # If it's a function on the class, convert to a property that calls it
        if inspect.isfunction(orig) or inspect.ismethod(orig):
            original = orig

            def patched(self, _orig=original):
                value = _orig(self)
                return _to_int(value)

            setattr(rom_cls, name, property(patched))
            return

        # If it's a simple attribute (e.g., descriptor), try to wrap via getattr
        try:
            sample = getattr(rom_cls, name)
            if isinstance(sample, (int,)):
                return
        except Exception:
            pass

    # Patch known numeric ROM metadata that may be numpy uint8 on older nes_py
    for name in ("prg_rom_size", "chr_rom_size", "mapper", "submapper", "prg_rom_start", "prg_rom_stop"):
        try:
            _wrap_name(name)
        except Exception:
            # Don't fail hard if a name isn't present or wrapping fails
            continue

    setattr(rom_cls, "_uint8_patch_applied", True)


_patch_legacy_nes_py_uint8()


def _patch_nes_py_ram_dtype() -> None:
    """Patch nes_py NESEnv to expose RAM as a wider integer dtype to avoid uint8 overflow."""

    try:
        import nes_py.nes_env as nes_env_module  # type: ignore
    except Exception:
        return

    ne_env_cls = getattr(nes_env_module, "NESEnv", None)
    if ne_env_cls is None or getattr(ne_env_cls, "_ram_dtype_patch_applied", False):
        return

    # Only patch if the original method exists
    orig_ram_buffer = getattr(ne_env_cls, "_ram_buffer", None)
    if orig_ram_buffer is None:
        return

    def _patched_ram_buffer(self):
        # call original to get the uint8 memory view, then cast to int64
        arr = orig_ram_buffer(self)
        try:
            # Ensure we return a numpy array with a safe signed integer dtype
            return arr.astype(np.int64)
        except Exception:
            # Fallback: return original array if astype fails
            return arr

    setattr(ne_env_cls, "_ram_buffer", _patched_ram_buffer)
    setattr(ne_env_cls, "_ram_dtype_patch_applied", True)


_patch_nes_py_ram_dtype()

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
