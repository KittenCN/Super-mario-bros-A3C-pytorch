"""Mario environment factories leveraging Gymnasium vector APIs."""

from __future__ import annotations

import dataclasses
import itertools
import tempfile
from pathlib import Path
import time
import os
import random
from typing import Iterable, List, Optional, Sequence, Tuple

import sys, io

# Temporarily silence stderr during imports that may emit Gym migration notices
_saved_stderr = sys.stderr
try:
    sys.stderr = io.StringIO()
    import gymnasium as gym
    import numpy as np
    from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
    import errno
finally:
    sys.stderr = _saved_stderr

# Defensive wrapper: some platforms/conditions can cause AsyncVectorEnv to
# raise OSError(9) when its destructor attempts to close already-closed
# pipe handles. Wrap close_extras to ignore EBADF so that cleanup does not
# escalate into an unhandled exception during process teardown.
try:
    if hasattr(AsyncVectorEnv, "close_extras"):
        _orig_close_extras = AsyncVectorEnv.close_extras

        def _safe_close_extras(self, *args, **kwargs):
            try:
                return _orig_close_extras(self, *args, **kwargs)
            except OSError as e:
                if getattr(e, "errno", None) == errno.EBADF:
                    # Already closed FD — ignore, as worker cleanup may have raced
                    return
                raise

        AsyncVectorEnv.close_extras = _safe_close_extras
except Exception:
    # If wrapping fails, don't prevent module import — fallback to original behaviour
    pass
import multiprocessing

_MARIO_PACKAGE = None

# pragma: no cover - prefer Gymnasium fork when available
# Some legacy packages print migration notices to stderr when imported.
# Temporarily silence stderr while attempting to import these optional
# dependencies so the startup does not emit noisy migration messages.
import sys, io

_saved_stderr = sys.stderr
try:
    sys.stderr = io.StringIO()
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
finally:
    sys.stderr = _saved_stderr

# Import nes_py wrappers after suppressing stderr earlier so any transitively
# imported packages that may import legacy `gym` don't emit migration notices
try:
    from nes_py.wrappers import JoypadSpace
except Exception:
    # If nes_py not available, we'll fail later when creating envs; keep import
    # errors non-fatal at import time to allow parts of the package to be used
    JoypadSpace = None  # type: ignore


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
    reset_timeout: float = 60.0
    # Delay in seconds between starting subsequent worker initialisation
    worker_start_delay: float = 0.2

    env: MarioEnvConfig = dataclasses.field(default_factory=MarioEnvConfig)


def _select_actions(action_type: str) -> Sequence[Tuple[str, ...]]:
    if action_type not in _ACTION_SPACES:
        valid = ", ".join(_ACTION_SPACES.keys())
        raise ValueError(f"Unknown action_type '{action_type}'. Valid: {valid}")
    return _ACTION_SPACES[action_type]


def list_available_stages(max_world: int = 8, max_stage: int = 4) -> List[Tuple[int, int]]:
    """List all available world-stage combinations."""

    return [(world, stage) for world in range(1, max_world + 1) for stage in range(1, max_stage + 1)]


def _make_single_env(config: MarioEnvConfig, seed: Optional[int] = None, diag_dir: Optional[str] = None, idx: Optional[int] = None):
    actions = _select_actions(config.action_type)
    # Dummy environment returned when creating a single instance in the main
    # process for space inspection. Real NES env will be created in worker
    # processes where the process name is not 'MainProcess'.
    class _DummyEnv(gym.Env):
        def __init__(self, obs_shape, n_actions):
            super().__init__()
            self.observation_space = gym.spaces.Box(0.0, 1.0, shape=obs_shape, dtype=np.float32)
            self.action_space = gym.spaces.Discrete(n_actions)

        def reset(self, seed=None, options=None):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, {}

        def step(self, action):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, False, False, {}

    def thunk():
        # If we are in the main process, return a lightweight dummy env to
        # avoid performing heavy native initialization (ROM loading, C++ lib
        # setup). Worker processes will construct the real env.
        if multiprocessing.current_process().name == "MainProcess":
            obs_shape = (config.frame_stack, 84, 84)
            return _DummyEnv(obs_shape, len(actions))
        # Prepare a simple diagnostics file so worker startup progress can be
        # inspected by the parent if construction times out.
        try:
            _diag_base = Path(diag_dir) if diag_dir is not None else Path(tempfile.gettempdir())
            _diag_base.mkdir(parents=True, exist_ok=True)
            _diag_file = _diag_base / f"env_init_idx{idx}_pid{os.getpid()}.log"
        except Exception:
            _diag_file = None

        def _diag(msg: str):
            try:
                if _diag_file is not None:
                    with open(_diag_file, "a", encoding="utf-8") as f:
                        f.write(f"{time.time():.6f}\t{msg}\n")
            except Exception:
                pass

        _diag("start")
        # Apply nes_py runtime patches early in the worker so any subsequent
        # imports that rely on nes_py get the patched behaviour (avoid uint8 overflow)
        try:
            _diag("patch_start")
            try:
                _patch_legacy_nes_py_uint8()
                _diag("patch_uint8_ok")
            except Exception as e:
                _diag(f"patch_uint8_error:{repr(e)}")
            try:
                _patch_nes_py_ram_dtype()
                _diag("patch_ramdtype_ok")
            except Exception as e:
                _diag(f"patch_ramdtype_error:{repr(e)}")
            _diag("patch_done")
        except Exception:
            pass
        # Eagerly import modules that may block during first-use and log timings
        try:
            _diag("import_start_gsm")
            import importlib

            importlib.import_module("gym_super_mario_bros")
            _diag("import_done_gsm")
        except Exception as e:
            _diag(f"import_gsm_error:{repr(e)}")
        try:
            _diag("calling_mario_make")
            env = mario_make(
                config.env_id(),
                apply_api_compatibility=True,
                render_mode=config.render_mode,
            )
            _diag("mario_make_done")
        except Exception as e:
            import traceback

            _diag(f"error:{repr(e)}")
            try:
                if _diag_file is not None:
                    with open(_diag_file, "a", encoding="utf-8") as f:
                        f.write(traceback.format_exc())
            except Exception:
                pass
            raise
        _diag("mario_make_done")
        env = JoypadSpace(env, actions)
        _diag("joypad_done")
        env = ProgressInfoWrapper(env)
        _diag("progress_wrapper_done")
        env = MarioRewardWrapper(env, config.reward_config)
        _diag("reward_wrapper_done")
        env = gym.wrappers.FrameSkip(env, config.frame_skip)
        _diag("frameskip_done")
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
        _diag("grayscale_done")
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        _diag("resize_done")
        env = TransformObservation(env, lambda obs: obs.astype(np.float32))
        _diag("transform_obs_dtype_done")
        env = TransformReward(env, float)
        _diag("transform_reward_done")
        env = gym.wrappers.FrameStack(env, config.frame_stack, new_axis=0)
        _diag("framestack_done")
        env = TransformObservation(env, lambda obs: np.asarray(obs, dtype=np.float32) / 255.0)
        _diag("transform_obs_scale_done")
        if config.record_video:
            os.makedirs(config.video_dir, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=config.video_dir,
                episode_trigger=lambda episode_id: True,
                name_prefix=f"world{config.world}_stage{config.stage}",
            )
            _diag("recordvideo_done")
        # Worker processes will reset as needed; don't reset here.
        _diag("done")
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
    # Diagnostic directory to capture per-worker init progress
    diag_dir = str(Path(tempfile.gettempdir()) / f"mario_env_diag_{os.getpid()}_{int(time.time())}")
    print(f"[env] diagnostic dir: {diag_dir}")
    for idx in range(config.num_envs):
        if config.random_start_stage:
            world, stage = random.choice(stage_list)
        else:
            world, stage = next(stage_cycle)
        stage_env_cfg = dataclasses.replace(config.env, world=world, stage=stage)
        seed = config.base_seed + idx
        # Wrap thunk to stagger worker startup when running asynchronously.
        if config.asynchronous and getattr(config, "worker_start_delay", 0.0) > 0.0:
            delay = float(config.worker_start_delay) * float(idx)

            def _thunk_with_delay(fn=_make_single_env(stage_env_cfg, seed=seed, diag_dir=diag_dir, idx=idx), d=delay):
                try:
                    if d > 0:
                        time.sleep(d)
                except Exception:
                    pass
                return fn()

            env_fns.append(_thunk_with_delay)
        else:
            env_fns.append(_make_single_env(stage_env_cfg, seed=seed, diag_dir=diag_dir, idx=idx))

    vector_cls = AsyncVectorEnv if config.asynchronous else SyncVectorEnv
    # Create a file to capture vector env construction timing / exceptions
    try:
        _create_diag = Path(diag_dir) / f"create_vector_env_pid{os.getpid()}.log"
        with open(_create_diag, "a", encoding="utf-8") as f:
            f.write(f"{time.time():.6f}\tbefore_construct\n")
    except Exception:
        _create_diag = None

    try:
        # If using AsyncVectorEnv, force it to use the same multiprocessing
        # start method as the parent and disable shared_memory. Passing an
        # explicit context here prevents the vector implementation from
        # accidentally using a different start method and avoids races when
        # creating shared memory segments on some platforms.
        if vector_cls is AsyncVectorEnv:
            try:
                ctx = multiprocessing.get_start_method()
            except Exception:
                ctx = None
            # AsyncVectorEnv accepts a 'context' name (str) in newer gym versions
            vec_env = vector_cls(env_fns, shared_memory=False, context=ctx)
        else:
            vec_env = vector_cls(env_fns)
        try:
            if _create_diag is not None:
                with open(_create_diag, "a", encoding="utf-8") as f:
                    f.write(f"{time.time():.6f}\tafter_construct\n")
        except Exception:
            pass
    except Exception as e:
        try:
            if _create_diag is not None:
                with open(_create_diag, "a", encoding="utf-8") as f:
                    f.write(f"{time.time():.6f}\terror:{repr(e)}\n")
                    import traceback

                    f.write(traceback.format_exc())
        except Exception:
            pass
        raise

    class ManagedVectorEnv:
        """Wrapper around VectorEnv that enforces stronger lifecycle management.

        It forwards attribute access to the underlying vector env, but overrides
        close() to attempt a graceful close and fall back to a forced terminate
        if necessary. Also implements context manager and a destructor that
        best-effort cleans up worker resources.
        """

        def __init__(self, inner):
            self._inner = inner
            self._closed = False

        def __getattr__(self, item):
            return getattr(self._inner, item)

        def close(self, terminate: bool = False):
            if self._closed:
                return
            # If caller requests forced termination, try that first
            if terminate:
                try:
                    # Some VectorEnv implementations accept terminate kw
                    self._inner.close(terminate=True)  # type: ignore[arg-type]
                except Exception:
                    try:
                        self._inner.close()
                    except Exception:
                        pass
                self._closed = True
                return

            # Otherwise attempt graceful close then fallback to forced terminate
            try:
                self._inner.close()
                self._closed = True
            except Exception:
                try:
                    self._inner.close(terminate=True)  # type: ignore[arg-type]
                except Exception:
                    pass
                self._closed = True

        def terminate(self):
            try:
                self._inner.close(terminate=True)  # type: ignore[arg-type]
            except Exception:
                try:
                    self._inner.close()
                except Exception:
                    pass
            finally:
                self._closed = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            try:
                self.close()
            except Exception:
                try:
                    self.terminate()
                except Exception:
                    pass

        def __del__(self):
            try:
                # best-effort cleanup in destructor
                if not self._closed:
                    try:
                        self._inner.close()
                    except Exception:
                        try:
                            self._inner.close(terminate=True)  # type: ignore[arg-type]
                        except Exception:
                            pass
            except Exception:
                pass

    return ManagedVectorEnv(vec_env)


def create_eval_env(config: MarioEnvConfig, seed: Optional[int] = None):
    """Create a single evaluation environment with rendering enabled."""

    eval_cfg = dataclasses.replace(config, render_mode="rgb_array")
    return _make_single_env(eval_cfg, seed=seed)()
