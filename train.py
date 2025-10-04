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
                continue
except Exception as e:
    print(f"[train][start-method][warn] failed to set start method: {e}")

# Suppress the legacy Gym startup notice that prints to stderr when gym is
# installed alongside gymnasium. We temporarily replace sys.stderr around the
# import of gym-related packages to avoid noisy migration messages.
import io
import sys

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

import argparse  # noqa: E402
import contextlib  # noqa: E402

# Ensure we don't attempt to override an already-initialised start method
# later in the module; the selection above is sufficient.
import dataclasses  # noqa: E402
import faulthandler  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as _mp  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: E402
import gzip  # noqa: E402
import shutil  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.distributions import Categorical  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402

from src.algorithms import vtrace_returns  # noqa: E402
from src.config import TrainingConfig, create_default_stage_schedule  # noqa: E402

# Import patch utilities to allow parent process to prewarm and patch nes_py
from src.envs.mario import (  # noqa: E402
    MarioEnvConfig,
    MarioVectorEnvConfig,
    _patch_legacy_nes_py_uint8,
    _patch_nes_py_ram_dtype,
    create_vector_env,
)

_suppress_gym_notice(False)
from src.models import MarioActorCritic  # noqa: E402
from src.utils import (  # noqa: E402
    CosineWithWarmup,
    PrioritizedReplay,
    RolloutBuffer,
    create_logger,
)
from src.utils.adaptive import AdaptiveConfig, AdaptiveScheduler  # noqa: E402
from src.utils.heartbeat import HeartbeatReporter  # noqa: E402
from src.utils.monitor import Monitor  # noqa: E402
from src.utils.metrics_export import rotate_metrics_file, write_latest_metrics  # noqa: E402

try:  # optional dependency for resource monitoring
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore


NUMERIC_TYPES = (int, float, np.integer, np.floating)


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        try:
            value = value.reshape(-1)[0]
        except Exception:
            try:
                value = value.item()
            except Exception:
                return value
    if hasattr(value, "item") and not isinstance(value, (bytes, bytearray)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Mario agent with modernised A3C"
    )
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument(
        "--action-type",
        type=str,
        default="complex",
        choices=["right", "simple", "complex", "extended"],
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--random-stage", action="store_true")
    parser.add_argument(
        "--stage-span",
        type=int,
        default=4,
        help="Number of consecutive stages in schedule",
    )
    parser.add_argument("--total-updates", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-res-blocks", type=int, default=3)
    parser.add_argument(
        "--recurrent-type",
        type=str,
        default="gru",
        choices=["gru", "lstm", "transformer", "none"],
    )
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
    parser.add_argument(
        "--sync-env", action="store_true", help="Use synchronous vector env"
    )
    parser.add_argument(
        "--async-env", action="store_true", help="Force asynchronous vector env"
    )
    parser.add_argument(
        "--confirm-async",
        action="store_true",
        help=(
            "Explicit confirmation to enable asynchronous vector env. "
            "Async mode is known to be unstable in some environments; "
            "set environment variable MARIO_ENABLE_ASYNC=1 or pass this flag to opt in."
        ),
    )
    parser.add_argument(
        "--force-sync",
        action="store_true",
        help="Force synchronous vector env (overrides async heuristic)",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--use-noisy-linear", action="store_true")
    parser.add_argument(
        "--log-dir", type=str, default="tensorboard/a3c_super_mario_bros"
    )
    parser.add_argument("--save-dir", type=str, default="trained_models")
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    # 放宽恢复匹配：默认开启；可通过 --no-resume-relax-match 关闭
    parser.add_argument(
        "--resume-relax-match",
        dest="resume_relax_match",
        action="store_true",
        help="在严格匹配找不到时，放宽为仅匹配 world/stage/action/frame_* 与 model 配置",
    )
    parser.add_argument(
        "--no-resume-relax-match",
        dest="resume_relax_match",
        action="store_false",
        help="禁用放宽匹配，仅严格匹配 metadata",
    )
    parser.set_defaults(resume_relax_match=False)
    parser.add_argument(
        "--config-out", type=str, default=None, help="Persist effective config to JSON"
    )
    parser.add_argument(
        "--per", action="store_true", help="Enable prioritized replay (hybrid)"
    )
    parser.add_argument(
        "--per-sample-interval",
        type=int,
        default=1,
        help="PER 抽样更新间隔（>=1）。例如 4 表示每4次更新做一次 PER batch",
    )
    parser.add_argument(
        "--per-gpu-sample",
        action="store_true",
        help="在支持时启用 GPU 侧优先级采样以减少 host->device 开销",
    )
    parser.add_argument(
        "--per-gpu-sample-fallback-ms",
        type=float,
        default=0.0,
        help="GPU 采样耗时超过该毫秒（连续多次）时自动回退到 CPU，0 表示禁用",
    )
    parser.add_argument("--project", type=str, default="mario-a3c")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run training on",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Path to JSONL file for periodic metrics dump",
    )
    parser.add_argument(
        "--episodes-event-path",
        type=str,
        default=None,
        help="可选：单独写入 episode_end 事件 JSONL（默认与 metrics 混写在同一文件）",
    )
    parser.add_argument(
        "--metrics-rotate-max-mb",
        type=float,
        default=0.0,
        help="超过该大小(MB) 时滚动压缩 metrics JSONL，0 表示关闭",
    )
    parser.add_argument(
        "--metrics-rotate-retain",
        type=int,
        default=5,
        help="保留的 metrics 压缩文件数量",
    )
    parser.add_argument(
        "--env-reset-timeout",
        type=float,
        default=None,
        help="Timeout (s) for environment construction/reset before fallback; if unset a heuristic is used",
    )
    parser.add_argument(
        "--no-prewarm",
        action="store_true",
        default=False,
        help="Disable prewarming ROMs before training",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        default=False,
        help="Disable background resource monitor",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=10.0,
        help="Monitor interval in seconds",
    )
    parser.add_argument(
        "--enable-tensorboard",
        action="store_true",
        default=False,
        help="Enable TensorBoard logging (disabled by default)",
    )
    parser.add_argument(
        "--parent-prewarm",
        action="store_true",
        default=False,
        help="Run a parent-process prewarm of the environment before launching workers",
    )
    parser.add_argument(
        "--parent-prewarm-all",
        action="store_true",
        default=False,
        help="Sequentially prewarm each env configuration in the parent before launching workers (slower, safer)",
    )
    parser.add_argument(
        "--worker-start-delay",
        type=float,
        default=0.2,
        help="Stagger delay (s) between worker initialisations when using async vector env",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=30.0,
        help="心跳打印间隔（秒），<=0 表示关闭",
    )
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=300.0,
        help="无进展告警阈值（秒），<=0 则自动按间隔推算",
    )
    parser.add_argument(
        "--heartbeat-path",
        type=str,
        default=None,
        help="Heartbeat JSONL 输出路径，未指定则写入 run_dir/heartbeat.jsonl",
    )
    parser.add_argument(
        "--stall-dump-dir",
        type=str,
        default=None,
        help="心跳检测到 stall 时将堆栈写入该目录，默认 run_dir/stall_dumps",
    )
    parser.add_argument(
        "--overlap-collect",
        action="store_true",
        help="启用采集-学习重叠：后台线程采集下一批 rollout",
    )
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=15.0,
        help="单次环境 step 超过该秒数将发出屏幕警告，<=0 表示关闭",
    )
    # 进度细粒度控制
    parser.add_argument(
        "--rollout-progress-interval",
        type=int,
        default=8,
        help="常规更新阶段 rollout 步进进度打印/心跳的步数间隔 (>=1)",
    )
    parser.add_argument(
        "--rollout-progress-warmup-updates",
        type=int,
        default=2,
        help="前多少个 update 使用 warmup 细粒度间隔 (>=0)",
    )
    parser.add_argument(
        "--rollout-progress-warmup-interval",
        type=int,
        default=1,
        help="warmup 阶段的步进进度间隔 (>=1)，默认每步打印",
    )
    parser.add_argument(
        "--slow-step-trace-threshold",
        type=int,
        default=5,
        help="在窗口内累计达到该慢 step 次数触发栈追踪",
    )
    parser.add_argument(
        "--slow-step-trace-window",
        type=float,
        default=120.0,
        help="慢 step 统计时间窗口 (秒)",
    )
    # Reward shaping / environment control
    parser.add_argument(
        "--reward-distance-weight", type=float, default=1.0 / 80.0, help="位移奖励权重"
    )
    parser.add_argument(
        "--reward-distance-weight-final",
        type=float,
        default=None,
        help="位移奖励权重最终值 (线性退火)，未指定则不退火",
    )
    parser.add_argument(
        "--reward-distance-weight-anneal-steps",
        type=int,
        default=0,
        help="位移奖励权重退火步数 (wrapper 内 step)，0 不启用",
    )
    parser.add_argument(
        "--reward-scale-start", type=float, default=0.2, help="动态缩放起始值"
    )
    parser.add_argument(
        "--reward-scale-final", type=float, default=0.1, help="动态缩放最终值"
    )
    parser.add_argument(
        "--reward-scale-anneal-steps",
        type=int,
        default=50_000,
        help="缩放线性退火步数 (env steps)",
    )
    parser.add_argument(
        "--death-penalty-start",
        type=float,
        default=None,
        help="死亡惩罚退火起始值 (通常为 0，未指定则使用固定 death_penalty)",
    )
    parser.add_argument(
        "--death-penalty-final",
        type=float,
        default=None,
        help="死亡惩罚退火最终值 (通常为负数)",
    )
    parser.add_argument(
        "--death-penalty-anneal-steps",
        type=int,
        default=0,
        help="死亡惩罚退火步数 (wrapper 内 step 计数)，0 不启用",
    )
    parser.add_argument(
        "--auto-start-frames",
        type=int,
        default=0,
        help="开局自动发送 START(+RIGHT) 的帧数，0 禁用",
    )
    parser.add_argument(
        "--stagnation-warn-updates",
        type=int,
        default=20,
        help="连续若干 update 无 max_x 提升触发警告，<=0 关闭",
    )
    parser.add_argument(
        "--stagnation-limit",
        type=int,
        default=400,
        help="若某 env 连续无前进达到该步数则触发截断（与内部 stagnation_limit 对应），默认400",
    )
    parser.add_argument(
        "--scripted-forward-frames",
        type=int,
        default=0,
        help="训练起始阶段前 N*env_num 帧强制使用前进动作 (RIGHT 或 RIGHT+B)，用于 warmup 推进，0 表示关闭",
    )
    parser.add_argument(
        "--forward-action-id",
        type=int,
        default=None,
        help="显式指定用于 scripted-forward 的离散动作 id；不指定则自动猜测 (优先含 RIGHT)",
    )
    parser.add_argument(
        "--probe-forward-actions",
        type=int,
        default=0,
        help=">0 时训练前对每个动作连续执行指定步数以探测前进效果",
    )
    parser.add_argument(
        "--enable-ram-x-parse",
        action="store_true",
        help="启用 RAM 解析 x_pos 回退 (MarioRewardWrapper)",
    )
    parser.add_argument(
        "--ram-x-high-addr",
        type=lambda x: int(x, 0),
        default=0x006D,
        help="水平位置高字节 RAM 地址 (hex 或十进制)",
    )
    parser.add_argument(
        "--ram-x-low-addr",
        type=lambda x: int(x, 0),
        default=0x0086,
        help="水平位置低字节 RAM 地址 (hex 或十进制)",
    )
    parser.add_argument(
        "--scripted-sequence",
        type=str,
        default=None,
        help="动作脚本: 逗号分隔 'NAME[:frames]'，NAME 为组合别名 (START,RIGHT,RIGHT+B,RIGHT+A,RIGHT+A+B,START+RIGHT,START+RIGHT+B)。未指定 frames 默认1",
    )
    parser.add_argument(
        "--auto-bootstrap-threshold",
        type=int,
        default=0,
        help="若 >0 且在该 update 前距离增量始终为 0，则自动触发前进动作注入",
    )
    parser.add_argument(
        "--auto-bootstrap-frames",
        type=int,
        default=0,
        help="自动前进触发后强制执行的总帧数 (按 env step)，需与阈值搭配使用",
    )
    parser.add_argument(
        "--auto-bootstrap-action-id",
        type=int,
        default=None,
        help="自动前进阶段使用的动作 id，如未指定则尽量推断包含 RIGHT 的动作",
    )
    # Early shaping 强化窗口
    parser.add_argument(
        "--early-shaping-window",
        type=int,
        default=0,
        help=">0 表示前 N updates 使用更强 distance_weight 覆盖 (early-shaping)，之后恢复退火逻辑",
    )
    parser.add_argument(
        "--early-shaping-distance-weight",
        type=float,
        default=None,
        help="early 窗口内 distance_weight 覆盖值；未提供则自动取当前 distance_weight * 2",
    )
    # 二次脚本注入（突破停滞）
    parser.add_argument(
        "--secondary-script-threshold",
        type=int,
        default=0,
        help=">0 时若 max_x 在该 update 之后仍未超过 plateau 值，执行二次短脚本注入",
    )
    parser.add_argument(
        "--secondary-script-frames",
        type=int,
        default=0,
        help="二次脚本注入帧数 (按 env step，总帧=frames*num_envs)",
    )
    parser.add_argument(
        "--secondary-script-forward-action-id",
        type=int,
        default=None,
        help="二次脚本使用的动作 id，未指定自动解析",
    )
    # 里程碑奖励与强制截断
    parser.add_argument(
        "--milestone-interval",
        type=int,
        default=0,
        help=">0 时每当全局 max_x 超过上一个里程碑+interval 给予额外 shaping 奖励一次",
    )
    parser.add_argument(
        "--milestone-bonus",
        type=float,
        default=0.0,
        help="里程碑奖励数值 (添加到 shaping_raw_sum & scaled_sum 前的 raw)",
    )
    parser.add_argument(
        "--episode-timeout-steps",
        type=int,
        default=0,
        help=">0 时对单 env episode 达到该步数强制截断以产生回报",
    )
    # 自适应调度：基于最近窗口 env_positive_dx_ratio 调整 entropy 与 distance 权重
    parser.add_argument(
        "--adaptive-positive-dx-window",
        type=int,
        default=50,
        help="计算正向位移比例的滑动窗口更新数，用于自适应调度 (>=10)",
    )
    parser.add_argument(
        "--adaptive-distance-weight-max",
        type=float,
        default=None,
        help="自适应 distance_weight 上限，未设则不启用调度",
    )
    parser.add_argument(
        "--adaptive-distance-weight-min",
        type=float,
        default=None,
        help="自适应 distance_weight 下限 (ratio 达到 high 阈值后趋近)",
    )
    parser.add_argument(
        "--adaptive-distance-ratio-low",
        type=float,
        default=0.05,
        help="正向位移比例低阈值 (<低阈值提升 distance_weight)",
    )
    parser.add_argument(
        "--adaptive-distance-ratio-high",
        type=float,
        default=0.30,
        help="正向位移比例高阈值 (>高阈值衰减 distance_weight)",
    )
    parser.add_argument(
        "--adaptive-distance-lr",
        type=float,
        default=0.05,
        help="distance_weight 自适应更新步长 (相对插值系数)",
    )
    parser.add_argument(
        "--adaptive-entropy-beta-max",
        type=float,
        default=None,
        help="自适应 entropy_beta 上限；启用需同时设下限",
    )
    parser.add_argument(
        "--adaptive-entropy-beta-min",
        type=float,
        default=None,
        help="自适应 entropy_beta 下限",
    )
    parser.add_argument(
        "--adaptive-entropy-ratio-low",
        type=float,
        default=0.05,
        help="正向位移比例低阈值 (<低阈值=>提高 entropy_beta)",
    )
    parser.add_argument(
        "--adaptive-entropy-ratio-high",
        type=float,
        default=0.30,
        help="正向位移比例高阈值 => 降低 entropy_beta",
    )
    parser.add_argument(
        "--adaptive-entropy-lr",
        type=float,
        default=0.10,
        help="entropy_beta 自适应更新步长 (相对插值系数)",
    )
    parser.add_argument(
        "--adaptive-lr-scale-max",
        type=float,
        default=None,
        help="自适应学习率缩放上限（需同时设置下限）",
    )
    parser.add_argument(
        "--adaptive-lr-scale-min",
        type=float,
        default=None,
        help="自适应学习率缩放下限",
    )
    parser.add_argument(
        "--adaptive-lr-scale-lr",
        type=float,
        default=0.05,
        help="学习率缩放自适应插值步长",
    )
    return parser.parse_args(argv)


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    stage_schedule = create_default_stage_schedule(
        args.world, args.stage, span=args.stage_span
    )
    env_cfg = MarioEnvConfig(
        world=args.world,
        stage=args.stage,
        action_type=args.action_type,
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
    )
    # 注入奖励配置
    try:
        env_cfg.reward_config.distance_weight = float(args.reward_distance_weight)
        if getattr(args, "reward_distance_weight_final", None) is not None:
            env_cfg.reward_config.distance_weight_final = float(
                args.reward_distance_weight_final
            )
            env_cfg.reward_config.distance_weight_anneal_steps = int(
                getattr(args, "reward_distance_weight_anneal_steps", 0) or 0
            )
        env_cfg.reward_config.scale_start = float(args.reward_scale_start)
        env_cfg.reward_config.scale_final = float(args.reward_scale_final)
        env_cfg.reward_config.scale_anneal_steps = int(args.reward_scale_anneal_steps)
        if (
            getattr(args, "death_penalty_start", None) is not None
            and getattr(args, "death_penalty_final", None) is not None
        ):
            env_cfg.reward_config.death_penalty_start = float(args.death_penalty_start)
            env_cfg.reward_config.death_penalty_final = float(args.death_penalty_final)
            env_cfg.reward_config.death_penalty_anneal_steps = int(
                getattr(args, "death_penalty_anneal_steps", 0) or 0
            )
    except Exception:
        pass
    # 记录 auto-start 配置（扩展字段）
    setattr(
        env_cfg, "auto_start_frames", max(0, int(getattr(args, "auto_start_frames", 0)))
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
            print(
                "[train] Async mode requested but not confirmed; defaulting to synchronous vector env. "
                "To enable async, set MARIO_ENABLE_ASYNC=1 or pass --confirm-async."
            )
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
    train_cfg.rollout = dataclasses.replace(
        train_cfg.rollout, num_steps=args.rollout_steps, gamma=args.gamma, tau=args.tau
    )
    train_cfg.optimizer = dataclasses.replace(
        train_cfg.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        beta_entropy=args.entropy_beta,
        value_loss_coef=args.value_coef,
        max_grad_norm=args.clip_grad,
    )
    grad_accum = max(1, args.grad_accum)
    effective_updates = max(1, math.ceil(args.total_updates / grad_accum))
    warmup_steps = min(train_cfg.scheduler.warmup_steps, effective_updates)
    train_cfg.scheduler = dataclasses.replace(
        train_cfg.scheduler,
        warmup_steps=warmup_steps,
        total_steps=effective_updates,
    )
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
    train_cfg.gradient_accumulation = grad_accum
    train_cfg.log_dir = args.log_dir
    train_cfg.save_dir = args.save_dir
    train_cfg.eval_interval = args.eval_interval
    train_cfg.checkpoint_interval = args.checkpoint_interval
    train_cfg.log_interval = args.log_interval
    train_cfg.resume_from = args.resume
    train_cfg.replay = dataclasses.replace(train_cfg.replay, enable=args.per)
    # 将 per_sample_interval 写入 replay 配置
    train_cfg.replay = dataclasses.replace(
        train_cfg.replay,
        per_sample_interval=max(1, getattr(args, "per_sample_interval", 1)),
    )
    train_cfg.replay = dataclasses.replace(
        train_cfg.replay,
        use_gpu_sampler=bool(getattr(args, "per_gpu_sample", False)),
        gpu_sampler_fallback_ms=max(0.0, float(getattr(args, "per_gpu_sample_fallback_ms", 0.0))),
    )
    train_cfg.device = args.device
    train_cfg.metrics_path = args.metrics_path
    setattr(
        train_cfg,
        "metrics_rotate_max_mb",
        max(0.0, float(getattr(args, "metrics_rotate_max_mb", 0.0))),
    )
    setattr(
        train_cfg,
        "metrics_rotate_retain",
        max(0, int(getattr(args, "metrics_rotate_retain", 5))),
    )
    # 记录自适应范围到 cfg 以便后续引用（存入 model/optimizer config 简单方案：附加属性）
    setattr(
        train_cfg,
        "adaptive_cfg",
        AdaptiveConfig(
            window=max(10, int(getattr(args, "adaptive_positive_dx_window", 50))),
            dw_max=getattr(args, "adaptive_distance_weight_max", None),
            dw_min=getattr(args, "adaptive_distance_weight_min", None),
            dw_low=getattr(args, "adaptive_distance_ratio_low", 0.05),
            dw_high=getattr(args, "adaptive_distance_ratio_high", 0.30),
            dw_lr=getattr(args, "adaptive_distance_lr", 0.05),
            ent_max=getattr(args, "adaptive_entropy_beta_max", None),
            ent_min=getattr(args, "adaptive_entropy_beta_min", None),
            ent_low=getattr(args, "adaptive_entropy_ratio_low", 0.05),
            ent_high=getattr(args, "adaptive_entropy_ratio_high", 0.30),
            ent_lr=getattr(args, "adaptive_entropy_lr", 0.10),
            lr_scale_max=getattr(args, "adaptive_lr_scale_max", None),
            lr_scale_min=getattr(args, "adaptive_lr_scale_min", None),
            lr_lr=getattr(args, "adaptive_lr_scale_lr", 0.05),
        ),
    )
    setattr(train_cfg, "heartbeat_path", args.heartbeat_path)
    setattr(train_cfg, "stall_dump_dir", getattr(args, "stall_dump_dir", None))
    return train_cfg


def prepare_model(
    cfg: TrainingConfig,
    action_space: int,
    device: torch.device,
    compile_now: bool = True,
) -> MarioActorCritic:
    """构造并可选编译模型。

    当需要从 checkpoint 恢复时，先不编译（compile_now=False），待权重加载完再尝试 torch.compile，
    避免 `_orig_mod.` 前缀差异导致加载失败。
    """
    model_cfg = dataclasses.replace(
        cfg.model, action_space=action_space, input_channels=cfg.model.input_channels
    )
    model = MarioActorCritic(model_cfg).to(device)
    if compile_now and cfg.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover
            print(f"[train][warn] torch.compile 失败，继续使用未编译模型: {e}")
    return model


def _raw_module(mod: nn.Module) -> nn.Module:
    """若是 torch.compile OptimizedModule，返回其 `_orig_mod`。"""
    if hasattr(mod, "_orig_mod"):
        try:
            return getattr(mod, "_orig_mod")
        except Exception:
            return mod
    return mod


def _flex_load_state_dict(model: nn.Module, state: dict) -> list[str]:
    """灵活加载 state_dict，自动适配是否带 `_orig_mod.` 前缀。返回问题键列表。"""
    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    model_prefixed = any(k.startswith("_orig_mod.") for k in model_keys)
    state_prefixed = any(k.startswith("_orig_mod.") for k in state_keys)
    adj = state
    if model_prefixed and not state_prefixed:
        adj = {f"_orig_mod.{k}": v for k, v in state.items()}
    elif not model_prefixed and state_prefixed:
        adj = {
            k[len("_orig_mod.") :]: v
            for k, v in state.items()
            if k.startswith("_orig_mod.")
        }
    missing, unexpected = model.load_state_dict(adj, strict=False)
    issues: list[str] = list(missing)
    issues.extend([f"unexpected:{u}" for u in unexpected])
    return issues


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

    advantages = torch.zeros_like(values)
    vs = torch.zeros_like(values)
    gae = torch.zeros_like(bootstrap_value)
    next_value = bootstrap_value
    lambda_coef = cfg.rollout.tau
    for t in reversed(range(values.shape[0])):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * non_terminal * next_value - values[t]
        gae = delta + gamma * lambda_coef * non_terminal * gae
        advantages[t] = gae
        vs[t] = advantages[t] + values[t]
        next_value = values[t]
    return vs, advantages


def _maybe_print_training_hints(update_idx: int, metrics: Dict[str, Any]) -> None:
    """Emit human-friendly training hints based on logged metrics."""

    if update_idx < 0:
        return

    bucket = max(0, update_idx // 50)
    hint_history = getattr(run_training, "_hint_history", None)  # type: ignore[attr-defined]
    if hint_history is None:
        hint_history = set()
        setattr(run_training, "_hint_history", hint_history)  # type: ignore[attr-defined]

    hints: list[str] = []
    avg_return = float(metrics.get("avg_return", 0.0) or 0.0)
    distance_delta = float(metrics.get("env_distance_delta_sum", 0) or 0)
    shaping_raw = float(metrics.get("env_shaping_raw_sum", 0.0) or 0.0)
    shaping_scaled = float(metrics.get("env_shaping_scaled_sum", 0.0) or 0.0)
    replay_fill = float(metrics.get("replay_fill_rate", 0.0) or 0.0)
    gpu_util = float(metrics.get("gpu_util_mean_window", -1.0) or -1.0)
    loss_total = float(metrics.get("loss_total", 0.0) or 0.0)
    global_step_value = int(metrics.get("global_step", 0) or 0)
    episodes_completed = int(metrics.get("episodes_completed", 0) or 0)
    train_device = str(metrics.get("train_device", ""))
    num_envs = int(metrics.get("num_envs", 0) or 0)

    if distance_delta == 0 and shaping_raw == 0 and global_step_value > 0:
        hints.append(
            "距离增量为 0，考虑提高 reward_distance_weight 或启用 scripted-forward 或设置 BOOTSTRAP=1。"
        )

    # 允许通过环境变量降低测试阈值（例如小规模快速试验）
    import os as _os  # 局部导入防止顶层污染
    test_lower = _os.environ.get("MARIO_HINT_TEST_LOWER_THRESH")
    try:
        test_lower_val = int(test_lower) if test_lower is not None else None
    except Exception:
        test_lower_val = None
    threshold_update = 200 if not test_lower_val else max(1, test_lower_val)

    if avg_return <= 0.0 and update_idx >= threshold_update:
        if shaping_scaled > 1.0 and episodes_completed == 0:
            hints.append(
                "已检测到推进/塑形奖励，但原生奖励仍为 0；建议缩短 episode 超时或增加死亡惩罚以促使环境结束。"
            )
            # 详细诊断：打印关键指标快照，帮助定位是 done 信号缺失还是超时/死亡惩罚问题
            try:
                diag = {
                    "env_shaping_scaled_sum": shaping_scaled,
                    "env_shaping_raw_sum": shaping_raw,
                    "env_distance_delta_sum": distance_delta,
                    "env_positive_dx_ratio": metrics.get("env_positive_dx_ratio"),
                    "episodes_completed": episodes_completed,
                    "global_step": metrics.get("global_step"),
                    "wrapper_distance_weight": metrics.get("wrapper_distance_weight"),
                    "env_shaping_last_dx": metrics.get("env_shaping_last_dx"),
                    "env_shaping_last_scale": metrics.get("env_shaping_last_scale"),
                }
                diag_items = " ".join(f"{k}={v}" for k, v in diag.items())
                print(f"[train][hint-diagn] update={update_idx} {diag_items}")
            except Exception:
                pass
        elif shaping_scaled <= 1.0:
            hints.append("avg_return 长期为 0，可检查奖励塑形或动作脚本是否生效。")

    if replay_fill < 0.05 and update_idx >= 100:
        hints.append(
            "经验回放填充率 <5%，建议延长预热或提高 per_sample_interval 以外的 push 频率。"
        )

    if (
        train_device == "cuda"
        and gpu_util >= 0
        and gpu_util < 5.0
        and update_idx >= max(200, num_envs * 10)
    ):
        hints.append(
            "GPU 利用率较低（长期 <5%），可增加 num_envs/rollout 或确认数据加载是否受 CPU 限制。"
        )

    if loss_total > 2.5 and update_idx >= 50:
        hints.append("loss_total 偏高，可检查学习率或梯度截断设置。")

    if not hints:
        return

    hint_key = (bucket, tuple(hints))
    if hint_key in hint_history:
        return
    hint_history.add(hint_key)

    joined = " | ".join(hints)
    print(f"[train][hint] update={update_idx} {joined}")


def _per_step_update(
    per_buffer: Optional[PrioritizedReplay],
    model: MarioActorCritic,
    sequences: Dict[str, torch.Tensor],
    vs: torch.Tensor,
    advantages: torch.Tensor,
    cfg: TrainingConfig,
    device: torch.device,
    update_idx: int,
    *,
    mixed_precision: bool,
    batch_size: int,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Handle PER push + optional sample/update for一个训练 step."""

    metrics: Dict[str, Any] = {"sampled": False, "sample_time_ms": 0.0, "timings": {}}
    per_loss = torch.tensor(0.0, device=device)

    if per_buffer is None:
        return per_loss, metrics

    obs = sequences.get("obs")
    actions = sequences.get("actions")
    if obs is None or actions is None:
        return per_loss, metrics

    obs_flat = obs.reshape(-1, *obs.shape[2:])
    actions_flat = actions.reshape(-1)
    vs_flat = vs.detach().reshape(-1, 1)
    adv_flat = advantages.detach().reshape(-1, 1)

    per_buffer.push(obs_flat, actions_flat, vs_flat, adv_flat)

    interval = max(1, int(getattr(cfg.replay, "per_sample_interval", 1)))
    if update_idx % interval != 0:
        return per_loss, metrics

    per_sample_start = time.time()
    per_sample, per_timings = per_buffer.sample_detailed(batch_size)
    metrics["sample_time_ms"] = (time.time() - per_sample_start) * 1000.0
    metrics["timings"] = dict(per_timings)
    fallback_ms = float(getattr(cfg.replay, "gpu_sampler_fallback_ms", 0.0))
    if fallback_ms > 0 and per_buffer.using_gpu_sampler:
        total_ms = per_timings.get("total", metrics["sample_time_ms"])
        if per_buffer.register_sample_time(total_ms, fallback_ms):
            metrics["gpu_sampler_fallback"] = True
        else:
            metrics["gpu_sampler_fallback"] = False
    else:
        metrics["gpu_sampler_fallback"] = False

    if per_sample is None:
        return per_loss, metrics

    metrics["sampled"] = True

    with torch.amp.autocast(device_type=device.type, enabled=mixed_precision):
        per_output = model(per_sample.observations, None, None)
        per_dist = Categorical(logits=per_output.logits)
        per_values = per_output.value.squeeze(-1)

    per_log_probs = per_dist.log_prob(per_sample.actions)
    per_policy_loss = -(
        per_log_probs * per_sample.advantages.detach() * per_sample.weights
    ).mean()
    td_error_raw = per_sample.target_values - per_values
    per_value_loss = (td_error_raw.pow(2) * per_sample.weights).mean()
    per_loss = per_policy_loss + cfg.optimizer.value_loss_coef * per_value_loss
    per_buffer.update_priorities(per_sample.indices, td_error_raw.detach().abs())

    return per_loss, metrics


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
        "replay": dataclasses.asdict(
            cfg.replay
        ),  # 记录回放配置（含 per_sample_interval）
        "save_state": {
            "global_step": int(global_step),
            "global_update": int(update_idx),
            "type": variant,
            # reason 字段可在异常/中断补写或保留默认 normal
            "reason": "normal",
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
    tmp_path = metadata_path.with_suffix(
        metadata_path.suffix + f".tmp_{os.getpid()}_{int(time.time()*1000)}"
    )
    try:
        with open(tmp_path, "w", encoding="utf-8") as fp:
            fp.write(payload)
            fp.flush()
            try:
                os.fsync(fp.fileno())  # 尽力刷盘
            except Exception:
                pass
        os.replace(tmp_path, metadata_path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


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
    if (
        bool(vector_meta.get("asynchronous", vector_cfg.asynchronous))
        != vector_cfg.asynchronous
    ):
        return False
    if (
        bool(vector_meta.get("random_start_stage", vector_cfg.random_start_stage))
        != vector_cfg.random_start_stage
    ):
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


def _metadata_relaxed_matches_config(metadata: Dict[str, Any], cfg: TrainingConfig) -> bool:
    """放宽版本的匹配：
    仅比较以下关键维度，忽略 vector_env（num_envs/async/base_seed/schedule）。
      - world, stage, action_type, frame_skip, frame_stack
      - model 配置完全一致
    """
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
    model_meta = metadata.get("model")
    if not isinstance(model_meta, dict):
        return False
    model_cfg = dataclasses.asdict(cfg.model)
    for key, value in model_cfg.items():
        if model_meta.get(key) != value:
            return False
    return True


def find_matching_checkpoint(
    cfg: TrainingConfig,
) -> Optional[Tuple[Path, Dict[str, Any]]]:
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


def find_matching_checkpoint_recursive(
    base_dir: Path, cfg: TrainingConfig, limit: int = 10000
) -> Optional[Tuple[Path, Dict[str, Any]]]:
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


def find_matching_checkpoint_recursive_relaxed(
    base_dir: Path, cfg: TrainingConfig, limit: int = 10000
) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """递归搜索（放宽匹配）。选择策略同严格版。"""
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
        if not _metadata_relaxed_matches_config(metadata, cfg):
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
    base_model_for_save = _raw_module(model)
    payload = {
        "model": base_model_for_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": global_step,
        "global_update": update_idx,
        "config": dataclasses.asdict(cfg),
    }
    tmp_ckpt = checkpoint_path.with_suffix(
        checkpoint_path.suffix + f".tmp_{os.getpid()}_{int(time.time()*1000)}"
    )
    try:
        torch.save(payload, tmp_ckpt)
        try:
            with open(tmp_ckpt, "rb") as fp:
                os.fsync(fp.fileno())
        except Exception:
            pass
        os.replace(tmp_ckpt, checkpoint_path)
    finally:
        try:
            if tmp_ckpt.exists():
                tmp_ckpt.unlink()
        except Exception:
            pass
    metadata = _build_checkpoint_metadata(
        cfg, global_step, update_idx, variant="checkpoint"
    )
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
    base_model_for_save = _raw_module(model)
    snap_payload = {
        "model": base_model_for_save.state_dict(),
        "global_step": global_step,
        "global_update": update_idx,
        "config": dataclasses.asdict(cfg),  # latest 同样带 config
    }
    tmp_snap = snapshot_path.with_suffix(
        snapshot_path.suffix + f".tmp_{os.getpid()}_{int(time.time()*1000)}"
    )
    try:
        torch.save(snap_payload, tmp_snap)
        try:
            with open(tmp_snap, "rb") as fp:
                os.fsync(fp.fileno())
        except Exception:
            pass
        os.replace(tmp_snap, snapshot_path)
    finally:
        try:
            if tmp_snap.exists():
                tmp_snap.unlink()
        except Exception:
            pass
    metadata = _build_checkpoint_metadata(
        cfg, global_step, update_idx, variant="latest"
    )
    _write_metadata(snapshot_path, metadata)
    return snapshot_path


def maybe_save_config(cfg: TrainingConfig, path: Optional[str]):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serialisable = json.dumps(dataclasses.asdict(cfg), indent=2)
    Path(path).write_text(serialisable, encoding="utf-8")


def call_with_timeout(
    fn,
    timeout: float,
    *args,
    cancel_event: Optional[threading.Event] = None,
    **kwargs,
):
    if timeout is not None and timeout <= 0:
        return fn(*args, **kwargs)

    result: dict[str, object] = {}
    error: dict[str, BaseException] = {}
    finished = threading.Event()
    cancel_event = cancel_event or threading.Event()

    def target():
        try:
            result["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - propagate original exception
            error["error"] = exc
        finally:
            finished.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    completed = finished.wait(timeout)
    if not completed:
        cancel_event.set()
        thread.join(0.1)
        raise TimeoutError(
            f"Call to {getattr(fn, '__name__', repr(fn))} exceeded {timeout:.1f}s"
        )
    thread.join()
    if error:
        raise error["error"]
    return result.get("value")


def run_training(cfg: TrainingConfig, args: argparse.Namespace) -> dict:
    # Device and performance tuning
    if cfg.device == "auto":
        allow_cpu_auto = os.environ.get("MARIO_ALLOW_CPU_AUTO", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            message = "未检测到可用的 CUDA 设备；请显式传入 --device cpu (或设置 MARIO_ALLOW_CPU_AUTO=1 允许自动回退)"
            if not allow_cpu_auto:
                raise SystemExit(f"[train][error] {message}")
            print(f"[train][warn] {message}")
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    hb_interval = getattr(args, "heartbeat_interval", 30.0)
    hb_timeout = getattr(args, "heartbeat_timeout", 300.0)
    if hb_timeout <= 0:
        base_interval = hb_interval if hb_interval > 0 else 30.0
        hb_timeout = max(base_interval * 4.0, 120.0)
    heartbeat_path_cfg = getattr(cfg, "heartbeat_path", None)
    heartbeat_log_path = (
        Path(str(heartbeat_path_cfg)).expanduser()
        if heartbeat_path_cfg
        else None
    )
    heartbeat = HeartbeatReporter(
        component="train",
        interval=hb_interval if hb_interval > 0 else 30.0,
        stall_timeout=hb_timeout,
        enabled=hb_interval > 0,
        log_path=heartbeat_log_path,
    )
    heartbeat.start()
    heartbeat.notify(phase="初始化", message="配置随机种子", progress=False)

    auto_bootstrap_threshold = max(
        0, int(getattr(args, "auto_bootstrap_threshold", 0) or 0)
    )
    auto_bootstrap_frames = max(0, int(getattr(args, "auto_bootstrap_frames", 0) or 0))
    auto_bootstrap_action_override = getattr(args, "auto_bootstrap_action_id", None)
    auto_bootstrap_state = {
        "remaining": 0,
        "triggered": False,
        "action_id": (
            int(auto_bootstrap_action_override)
            if auto_bootstrap_action_override is not None
            else None
        ),
        "announced_done": False,
        "announced_start": False,
    }
    # Early shaping 窗口状态
    early_shaping_window = max(0, int(getattr(args, "early_shaping_window", 0) or 0))
    early_shaping_distance_weight = getattr(args, "early_shaping_distance_weight", None)
    # Secondary script 注入状态
    secondary_script_threshold = max(
        0, int(getattr(args, "secondary_script_threshold", 0) or 0)
    )
    secondary_script_frames = max(
        0, int(getattr(args, "secondary_script_frames", 0) or 0)
    )
    secondary_script_action_id = getattr(
        args, "secondary_script_forward_action_id", None
    )
    secondary_script_state = {"remaining": 0, "triggered": False, "plateau_baseline": 0}
    milestone_interval = max(0, int(getattr(args, "milestone_interval", 0) or 0))
    milestone_bonus = float(getattr(args, "milestone_bonus", 0.0) or 0.0)
    episode_timeout_steps = max(0, int(getattr(args, "episode_timeout_steps", 0) or 0))
    milestone_state = {
        "next": milestone_interval if milestone_interval > 0 else None,
        "count": 0,
    }
    # 追踪每个 env 当前 episode step 数
    per_env_episode_steps = np.zeros((cfg.env.num_envs,), dtype=np.int64)

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
    est_obs_mem = (
        (cfg.rollout.num_steps + 1)
        * cfg.env.num_envs
        * cfg.model.input_channels
        * 84
        * 84
        * 4
        / (1024**3)
    )
    if device.type == "cuda" and est_obs_mem > 1.5:  # 粗略阈值
        print(
            f"[train][info] Estimated CPU rollout buffer size ~{est_obs_mem:.2f} GiB (pinned). Consider reducing rollout-steps or num-envs if OOM occurs."
        )
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if device.type == "cuda" and not alloc_conf:
        print(
            "[train][hint] Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to mitigate fragmentation (export before running)."
        )
    env_summary = cfg.env.env
    stage_desc = f"{env_summary.world}-{env_summary.stage}"
    print(
        f"[train] stage={stage_desc} action={env_summary.action_type} async={cfg.env.asynchronous} "
        f"total_updates={cfg.total_updates} log_interval={cfg.log_interval}"
    )
    heartbeat.notify(
        phase="初始化",
        message=f"设备 {device.type} 线程{torch_threads}",
        progress=False,
    )

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
            heartbeat.notify(
                phase="env", message=f"父进程预热失败: {exc}", progress=False
            )

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
                    single_cfg = dataclasses.replace(
                        cfg.env, num_envs=1, asynchronous=False
                    )
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
                    heartbeat.notify(
                        phase="env",
                        message=f"worker {idx} 预热失败: {exc}",
                        progress=False,
                    )
            print("[train] parent prewarm-all complete")
            heartbeat.notify(phase="env", message="全量预热完成", progress=False)
        except Exception as exc:
            print(f"[train] parent prewarm-all failed: {exc}")
            heartbeat.notify(
                phase="env", message=f"全量预热失败: {exc}", progress=False
            )

    def _initialise_env(env_cfg: MarioVectorEnvConfig):
        timeout = getattr(env_cfg, "reset_timeout", 60.0)
        start_construct = time.time()
        print(f"[train] Starting environment construction with timeout={timeout}s...")
        env_cancel_event = threading.Event() if not env_cfg.asynchronous else None

        def _construct_env():
            return create_vector_env(env_cfg, cancel_event=env_cancel_event)

        try:
            env_instance = call_with_timeout(
                _construct_env, timeout, cancel_event=env_cancel_event
            )
            construct_time = time.time() - start_construct
            print(f"[train] Environment constructed in {construct_time:.2f}s")
        except TimeoutError as err:
            construct_time = time.time() - start_construct
            print(
                f"[train][error] Environment construction timed out after {construct_time:.2f}s"
            )
            env_cancel_event.set()
            raise RuntimeError("Vector env construction timed out") from err
        except Exception:
            env_cancel_event.set()
            raise

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
    heartbeat.notify(
        phase="env", message=f"环境已就绪 async={cfg.env.asynchronous}", progress=False
    )
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
    cfg.model = dataclasses.replace(
        cfg.model,
        action_space=action_space,
        input_channels=env.single_observation_space.shape[0],
    )

    # 前进动作自动探测（仅在用户指定 probe-forward-actions >0 时执行）
    forward_probe_result = None
    if getattr(args, "probe_forward_actions", 0) > 0:
        try:
            single = None
            if (
                hasattr(env, "envs")
                and isinstance(env.envs, (list, tuple))
                and env.envs
            ):
                single = env.envs[0]
            if single is not None:
                n_actions = int(getattr(single.action_space, "n", 0))
                per = max(1, int(args.probe_forward_actions))
                distances = []
                # (removed unused numpy import previously triggering F401)

                def _extract_x(inf: dict):
                    if not isinstance(inf, dict):
                        return 0
                    x = inf.get("x_pos") or inf.get("progress")
                    metrics = inf.get("metrics") if isinstance(inf, dict) else None
                    if x is None and isinstance(metrics, dict):
                        xv = metrics.get("mario_x") or metrics.get("x_pos")
                        if xv is not None:
                            try:
                                if hasattr(xv, "item"):
                                    xv = xv.item()
                                elif hasattr(xv, "__len__") and len(xv) > 0:
                                    xv = xv[0]
                                x = int(xv)
                            except Exception:
                                x = None
                    try:
                        return int(x) if x is not None else 0
                    except Exception:
                        return 0

                # reset 基准
                single.reset()
                for aid in range(n_actions):
                    single.reset()
                    start_obs, start_info = single.reset()
                    start_x = _extract_x(start_info)
                    last_x = start_x
                    for _ in range(per):
                        ob2, r2, term2, trunc2, inf2 = single.step(aid)
                        last_x = _extract_x(inf2)
                        if term2 or trunc2:
                            break
                    distances.append((aid, max(0, last_x - start_x), last_x))
                distances.sort(key=lambda t: t[1], reverse=True)
                if distances:
                    forward_probe_result = distances
                    print(
                        f"[train][probe] 动作探测结果 top: {distances[:min(8,len(distances))]}"
                    )
                    best = distances[0]
                    if best[1] > 0 and getattr(args, "forward_action_id", None) is None:
                        setattr(args, "forward_action_id", best[0])
                        print(
                            f"[train][probe] 自动选择 forward_action_id={best[0]} (Δx={best[1]})"
                        )
        except Exception as _exc:
            print(f"[train][probe][warn] 动作探测失败: {_exc}")

    def _resolve_forward_action_id() -> int:
        if getattr(args, "forward_action_id", None) is not None:
            return int(args.forward_action_id)
        if hasattr(run_training, "_forward_action_id") and getattr(run_training, "_forward_action_id") is not None:  # type: ignore[attr-defined]
            return int(getattr(run_training, "_forward_action_id"))  # type: ignore[attr-defined]
        candidates = [1, 2, 3, 4, 0]
        space_n = (
            env.action_space.n
            if hasattr(env.action_space, "n")
            else max(candidates) + 1
        )
        fid = next((c for c in candidates if c < space_n), 0)
        setattr(run_training, "_forward_action_id", fid)  # type: ignore[attr-defined]
        return fid

    def _apply_auto_bootstrap_actions(actions_tensor: torch.Tensor) -> torch.Tensor:
        nonlocal auto_bootstrap_state
        remaining = auto_bootstrap_state["remaining"]
        if remaining <= 0:
            return actions_tensor
        action_id = auto_bootstrap_state["action_id"]
        if action_id is None:
            try:
                action_id = _resolve_forward_action_id()
            except Exception:
                action_id = 0
            auto_bootstrap_state["action_id"] = action_id
        auto_bootstrap_state["remaining"] = max(0, remaining - actions_tensor.numel())
        if auto_bootstrap_state["remaining"] <= 0 and not auto_bootstrap_state.get(
            "announced_done", False
        ):
            print("[train][auto-bootstrap] 自动前进阶段结束，恢复策略动作。")
            auto_bootstrap_state["announced_done"] = True
        return torch.full_like(actions_tensor, int(action_id))

    model = prepare_model(cfg, action_space, device, compile_now=False)
    # Single-GPU / CPU path: model is already moved to device in prepare_model
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        eps=1e-5,
    )
    schedule_fn = CosineWithWarmup(
        cfg.scheduler.warmup_steps, cfg.scheduler.total_steps
    )
    scheduler = LambdaLR(optimizer, lr_lambda=schedule_fn)
    # Use new torch.amp API where available
    try:
        scaler = torch.amp.GradScaler(enabled=cfg.mixed_precision)  # type: ignore[attr-defined]
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)

    def _refresh_effective_lr() -> float:
        try:
            base_lr_value = float(scheduler.get_last_lr()[0])
        except Exception:
            base_lr_value = float(cfg.optimizer.learning_rate)
        adaptive_scale = float(globals().get("adaptive_lr_scale_val", 1.0))
        effective = base_lr_value * adaptive_scale
        try:
            for group in optimizer.param_groups:
                group["lr"] = effective
        except Exception:
            pass
        globals()["adaptive_effective_lr"] = effective
        return effective

    _refresh_effective_lr()

    resume_path: Optional[Path] = None
    resume_metadata: Optional[Dict[str, Any]] = None
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        resume_metadata = load_checkpoint_metadata(resume_path)
        if not _metadata_matches_config(resume_metadata, cfg):
            raise ValueError(
                f"Resume 失败：{resume_path} 的 metadata 与当前配置不兼容。\n"
                f"请确认 world/stage/action/frame_* 与训练时一致；或移除 --resume 以启用自动在 {Path('trained_models').resolve()} 下递归匹配。"
            )
    else:
        # 自动探测逻辑：优先编号最高的序号化 checkpoint，其次 latest 快照
        found = find_matching_checkpoint(cfg)
        if found is not None:
            resume_path, resume_metadata = found
            print(f"[train] auto-resume matched checkpoint: {resume_path.name}")
        else:
            # 尝试 latest 快照
            latest = Path(cfg.save_dir) / f"{_checkpoint_stem(cfg)}_latest.pt"
            meta = latest.with_suffix(".json")
            if latest.exists() and meta.exists():
                try:
                    meta_obj = json.loads(meta.read_text(encoding="utf-8"))
                    if _metadata_matches_config(meta_obj, cfg):
                        resume_path = latest
                        resume_metadata = meta_obj
                        print(
                            f"[train] auto-resume using latest snapshot: {latest.name}"
                        )
                except Exception:
                    pass
        # 若仍未找到，扩展搜索整个 trained_models 根目录（严格匹配）
        if resume_path is None:
            root_dir = Path("trained_models")
            if root_dir.exists():
                recursive = find_matching_checkpoint_recursive(root_dir, cfg)
                if recursive is not None:
                    resume_path, resume_metadata = recursive
                    try:
                        rel = resume_path.relative_to(root_dir)
                        print(
                            f"[train] recursive-resume found checkpoint under trained_models/: {rel}"
                        )
                    except Exception:
                        print(
                            f"[train] recursive-resume found checkpoint: {resume_path}"
                        )
        # 若严格匹配仍未找到，且允许放宽匹配，则进行放宽匹配
        allow_relax = bool(getattr(args, "resume_relax_match", True))
        if resume_path is None and allow_relax:
            root_dir = Path("trained_models")
            if root_dir.exists():
                relaxed = find_matching_checkpoint_recursive_relaxed(root_dir, cfg)
                if relaxed is not None:
                    resume_path, resume_metadata = relaxed
                    try:
                        rel = resume_path.relative_to(root_dir)
                        print(
                            f"[train][resume] 使用放宽匹配在 trained_models/ 下找到: {rel}"
                        )
                    except Exception:
                        print(
                            f"[train][resume] 使用放宽匹配找到: {resume_path}"
                        )
                    print(
                        "[train][resume][hint] 已忽略 vector_env 差异（num_envs/async/base_seed/schedule），"
                        "仅匹配 world/stage/action/frame_* 与 model；若不希望此行为，可添加 --no-resume-relax-match。"
                    )
        if resume_path is None:
            print(
                "[train] no matching checkpoint found (current save_dir & recursive search) – starting fresh training"
            )
            try:
                env_cfg = cfg.env.env
                print(
                    f"[train][hint] 当前配置: world={env_cfg.world} stage={env_cfg.stage} action={env_cfg.action_type} "
                    f"frame_stack={env_cfg.frame_stack} frame_skip={env_cfg.frame_skip}; 若已存在历史模型但向量环境配置不同，可使用 --resume-relax-match（默认开启）。"
                )
            except Exception:
                pass

    writer, wandb_run, run_dir = create_logger(
        cfg.log_dir,
        project=args.wandb_project,
        resume=bool(resume_path),
        enable_tb=bool(args.enable_tensorboard),
    )
    try:
        print(
            f"[train][log] tensorboard_log_dir={getattr(writer,'log_dir', getattr(writer,'_base',run_dir))}"
        )
        if args.enable_tensorboard:
            writer.add_text("meta/started", datetime.now(UTC).isoformat(), 0)
            try:
                writer.add_scalar("meta/init", 1, 0)
            except Exception:
                pass
    except Exception:
        pass
    if heartbeat_log_path is None and hb_interval > 0:
        try:
            heartbeat.set_log_path(Path(run_dir) / "heartbeat.jsonl")
        except Exception:
            pass
    stall_dump_dir_cfg = getattr(cfg, "stall_dump_dir", None)
    if hb_interval > 0:
        try:
            if stall_dump_dir_cfg:
                heartbeat.set_stall_dump_dir(Path(stall_dump_dir_cfg).expanduser())
            else:
                heartbeat.set_stall_dump_dir(Path(run_dir) / "stall_dumps")
        except Exception:
            pass
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    maybe_save_config(cfg, args.config_out)
    heartbeat.notify(phase="日志", message="日志与保存目录已就绪", progress=False)

    metrics_file = (
        Path(cfg.metrics_path).expanduser()
        if cfg.metrics_path
        else Path(run_dir) / "metrics.jsonl"
    )
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_lock = threading.Lock()
    # episode_end 事件独立文件（若未指定则复用 metrics_file）
    episodes_event_file: Path | None = None
    try:
        if getattr(args, "episodes_event_path", None):
            episodes_event_file = Path(args.episodes_event_path).expanduser()
            episodes_event_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            episodes_event_file = metrics_file
    except Exception:
        episodes_event_file = metrics_file

    # Start background resource monitor (safe start)
    monitor = None
    if not args.no_monitor:
        try:
            monitor = Monitor(
                writer,
                wandb_run,
                metrics_file,
                interval=args.monitor_interval,
                metrics_lock=metrics_lock,
            )
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
    slow_step_events: list[float] = []
    slow_trace_threshold = getattr(args, "slow_step_trace_threshold", 5)
    slow_trace_window = getattr(args, "slow_step_trace_window", 120.0)

    rollout = RolloutBuffer(
        cfg.rollout.num_steps,
        cfg.env.num_envs,
        env.single_observation_space.shape,
        device,
        pin_memory=(device.type == "cuda"),
    )
    per_buffer = None
    if cfg.replay.enable:
        per_buffer = PrioritizedReplay(
            capacity=cfg.replay.capacity,
            alpha=cfg.replay.priority_alpha,
            beta_start=cfg.replay.priority_beta,
            beta_final=cfg.replay.priority_beta_final,
            beta_steps=cfg.replay.priority_beta_steps,
            device=device,
            use_gpu_sampler=cfg.replay.use_gpu_sampler,
        )
    hidden_state, cell_state = model.initial_state(cfg.env.num_envs, device)
    optimizer.zero_grad(set_to_none=True)

    # 启用 RAM 解析 fallback：遍历底层 envs 尝试设置属性
    if getattr(args, "enable_ram_x_parse", False):
        try:
            if hasattr(env, "envs"):
                for _e in env.envs:
                    # 逐层 unwrap 找到 MarioRewardWrapper
                    cur = _e
                    for _ in range(6):  # 防止过深循环
                        if cur is None:
                            break
                        name = cur.__class__.__name__.lower()
                        if name.startswith("mariorewardwrapper"):
                            try:
                                cur.enable_ram_x_parse = True
                                cur.ram_addr_high = int(
                                    getattr(args, "ram_x_high_addr")
                                )
                                cur.ram_addr_low = int(getattr(args, "ram_x_low_addr"))
                                print(
                                    f"[train][ram-x] enabled on wrapper (high=0x{cur.ram_addr_high:04X} low=0x{cur.ram_addr_low:04X})"
                                )
                            except Exception:
                                pass
                            break
                        cur = getattr(cur, "env", None)
        except Exception as _exc:
            print(f"[train][ram-x][warn] enable failed: {_exc}")

    global_step = 0
    global_update = 0
    last_resource_log = 0.0
    episode_returns = np.zeros(cfg.env.num_envs, dtype=np.float32)
    completed_returns: list[float] = []
    # 追加：episode 长度与结束原因（仅保留最近 1000 条避免内存膨胀）
    completed_episode_lengths: list[int] = []
    completed_episode_reasons: list[str] = []  # e.g. terminated;truncated;timeout_truncate_batch
    # 追踪每个 env 的最新 x_pos 与累计最大值，便于日志/诊断
    env_last_x = np.zeros(cfg.env.num_envs, dtype=np.int32)
    env_max_x = np.zeros(cfg.env.num_envs, dtype=np.int32)
    env_prev_step_x = np.zeros(cfg.env.num_envs, dtype=np.int32)
    env_progress_dx = np.zeros(cfg.env.num_envs, dtype=np.int32)
    distance_delta_sum = 0  # 本 update 内累计正向位移
    shaping_raw_sum = 0.0
    shaping_scaled_sum = 0.0
    # 额外诊断：记录最近一次 shaping 信息 & 解析失败次数
    shaping_last_dx = 0.0
    shaping_last_scale = 0.0
    shaping_last_raw = 0.0
    shaping_last_scaled = 0.0
    shaping_parse_fail = 0
    action_hist = np.zeros(64, dtype=np.int64)  # 假设动作空间 <64
    stagnation_steps = np.zeros(cfg.env.num_envs, dtype=np.int32)
    stagnation_limit = max(1, int(getattr(args, "stagnation_limit", 400) or 400))
    prev_global_max_x = 0
    stagnation_update_counter = 0
    stagnation_warn_updates = max(0, int(getattr(args, "stagnation_warn_updates", 0)))
    auto_start_frames = int(getattr(cfg.env.env, "auto_start_frames", 0))
    auto_start_sequence = [
        0
    ] * auto_start_frames  # 暂以动作 0 代表 START/NOOP，可后续替换具体 start 动作

    last_grad_norm = 0.0
    last_log_time = time.time()
    last_log_step = global_step
    last_log_update = global_update

    globals()["adaptive_lr_scale_val"] = 1.0
    globals()["adaptive_effective_lr"] = float(cfg.optimizer.learning_rate)

    if resume_path is not None:
        print(f"[train] Loading checkpoint for resume: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        missing_warn: List[str] = []
        try:
            issues = _flex_load_state_dict(model, checkpoint["model"])
            if issues:
                # 控制台只输出前若干，避免噪声
                print(
                    f"[train][warn] flexible load issues: {issues[:8]}{'...' if len(issues)>8 else ''}"
                )
                # 写入详细列表到独立日志文件（幂等追加）
                try:
                    issues_path = Path(run_dir) / "state_dict_load_issues.log"
                    with issues_path.open("a", encoding="utf-8") as fp:
                        fp.write(
                            f"# {datetime.now(UTC).isoformat()} resume={resume_path}\n"
                        )
                        for it in issues:
                            fp.write(it + "\n")
                except Exception as _werr:  # noqa: BLE001
                    print(f"[train][warn] failed to write issues log: {_werr}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model state_dict from {resume_path}: {e}"
            )
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
            print(
                "[train][warn] Partial resume: missing/failed -> "
                + ", ".join(missing_warn)
            )
        print(f"[train] Resume state: update={global_update} step={global_step}")
        heartbeat.notify(
            phase="恢复", message=f"加载检查点 {resume_path.name}", progress=False
        )

    last_log_step = global_step
    last_log_update = global_update
    last_log_time = time.time()

    # 加载完成后再尝试编译
    if cfg.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            print("[train][info] model compiled after weight load")
        except Exception as e:  # pragma: no cover
            print(f"[train][warn] post-load compile 失败，将继续使用未编译模型: {e}")
    overlap_enabled = (
        bool(getattr(args, "overlap_collect", False)) and not cfg.env.asynchronous
    )
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
                    print(
                        "[train][info] unwrap compiled model -> 原始模块用于 overlap 线程前向，避免 FX 追踪冲突。"
                    )
                except Exception as e:  # pragma: no cover
                    print(
                        f"[train][warn] unwrap 编译模型失败: {e}，禁用 overlap 以保证稳定。"
                    )
                    overlap_enabled = False
            else:
                print(
                    "[train][warn] 未发现 _orig_mod，无法安全与 torch.compile 并用；禁用 overlap 模式。"
                )
                overlap_enabled = False
    model_lock = threading.Lock() if overlap_enabled else None

    # 用于存放后台线程结果
    class _CollectResult:
        __slots__ = (
            "obs_cpu",
            "obs_gpu",
            "hidden",
            "cell",
            "rollout",
            "episode_returns",
            "steps",
        )

        def __init__(self):
            self.obs_cpu = None
            self.obs_gpu = None
            self.hidden = None
            self.cell = None
            self.rollout = None
            self.episode_returns = None
            self.steps = 0  # 该批实际采集的环境步数 (= num_envs * rollout_steps)

    def _collect_rollout(
        buffer: RolloutBuffer,
        start_obs_cpu: torch.Tensor,
        start_hidden,
        start_cell,
        ep_returns: np.ndarray,
        update_idx_hint: int,
        progress_heartbeat: bool = False,
        step_interval: int = 8,
    ):
        local_hidden = start_hidden
        local_cell = start_cell
        obs_local_cpu = start_obs_cpu
        if local_hidden is not None:
            buffer.set_initial_hidden(
                local_hidden.detach(),
                local_cell.detach() if local_cell is not None else None,
            )
        else:
            buffer.set_initial_hidden(None, None)
        for step in range(cfg.rollout.num_steps):
            with model_lock or contextlib.nullcontext():
                # 使用 no_grad 而非 inference_mode，避免生成 inference tensor 被后续保存进计算图时报错
                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type=device.type, enabled=cfg.mixed_precision
                    ):
                        out = model(
                            obs_local_cpu.to(
                                device, non_blocking=(device.type == "cuda")
                            ),
                            local_hidden,
                            local_cell,
                        )
            dist = Categorical(logits=out.logits)
            actions = dist.sample()
            if auto_bootstrap_state["remaining"] > 0:
                actions = _apply_auto_bootstrap_actions(actions)
            log_probs = dist.log_prob(actions)
            values = out.value
            actions_cpu = actions.detach().cpu()
            log_probs_cpu = log_probs.detach().cpu()
            values_cpu = values.detach().cpu()
            next_obs_np, reward_np, terminated, truncated, _infos = env.step(
                actions_cpu.numpy()
            )
            done = np.logical_or(terminated, truncated)
            reward_cpu = _to_cpu_tensor(reward_np, dtype=torch.float32)
            done_cpu = _to_cpu_tensor(done.astype(np.float32), dtype=torch.float32)
            buffer.insert(
                step,
                obs_local_cpu,
                actions_cpu,
                log_probs_cpu,
                reward_cpu,
                values_cpu,
                done_cpu,
            )
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
                # detach + clone 产生普通张量，避免 inference tensor backward 限制
                local_hidden = out.hidden_state.detach().clone()
                if done.any():
                    mask = torch.as_tensor(
                        done, dtype=torch.bool, device=local_hidden.device
                    )
                    local_hidden[:, mask] = 0.0
                if out.cell_state is not None:
                    local_cell = out.cell_state.detach().clone()
                    if done.any():
                        mask = torch.as_tensor(
                            done, dtype=torch.bool, device=local_cell.device
                        )
                        local_cell[:, mask] = 0.0
                else:
                    local_cell = None
            else:
                local_hidden = None
                local_cell = None
            if progress_heartbeat and (
                (step + 1) % step_interval == 0 or step + 1 == cfg.rollout.num_steps
            ):
                heartbeat.notify(
                    global_step=-1,
                    phase="rollout",
                    message=f"(bg) 采样 {step+1}/{cfg.rollout.num_steps}",
                )
        buffer.set_next_obs(obs_local_cpu)
        steps_collected = cfg.rollout.num_steps * cfg.env.num_envs
        return obs_local_cpu, local_hidden, local_cell, ep_returns, steps_collected

    # 双缓冲：当前 buffer / 下一个 buffer
    current_buffer = rollout
    next_buffer = (
        RolloutBuffer(
            cfg.rollout.num_steps,
            cfg.env.num_envs,
            env.single_observation_space.shape,
            device,
            pin_memory=(device.type == "cuda"),
        )
        if overlap_enabled
        else None
    )
    pending_thread = None
    pending_result = _CollectResult() if overlap_enabled else None

    def _start_background_collection(
        start_obs_cpu,
        hidden_state,
        cell_state,
        ep_returns,
        next_update_idx,
        step_interval: int,
    ):
        assert (
            overlap_enabled and next_buffer is not None and pending_result is not None
        )
        next_buffer.reset()
        pending_result.obs_cpu = None

        def _runner():
            try:
                obs_cpu_final, h_final, c_final, ep_ret, steps_collected = (
                    _collect_rollout(
                        next_buffer,
                        start_obs_cpu,
                        hidden_state,
                        cell_state,
                        ep_returns,
                        next_update_idx,
                        progress_heartbeat=False,
                        step_interval=step_interval,
                    )
                )
                pending_result.obs_cpu = obs_cpu_final
                pending_result.hidden = h_final
                pending_result.cell = c_final
                pending_result.episode_returns = ep_ret
                pending_result.steps = steps_collected
            except Exception as e:  # pragma: no cover - 仅日志
                print(f"[train][warn] background collection failed: {e}")

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        return t

    # 进度输出策略参数
    base_interval = max(1, getattr(args, "rollout_progress_interval", 8))
    warmup_updates = max(0, getattr(args, "rollout_progress_warmup_updates", 2))
    warmup_interval = max(1, getattr(args, "rollout_progress_warmup_interval", 1))

    try:
        for update_idx in range(global_update, cfg.total_updates):
            update_wall_start = time.time()
            heartbeat.notify(
                update_idx=update_idx,
                global_step=global_step,
                phase="rollout",
                message=f"采集第{update_idx}轮数据",
            )
            # 当前 update 使用的步进进度间隔（前 warmup_updates 个使用 warmup_interval）
            current_interval = (
                warmup_interval
                if (update_idx - global_update) < warmup_updates
                else base_interval
            )
            if not overlap_enabled:
                # 原始同步采集逻辑保留
                rollout.reset()
                if hidden_state is not None:
                    rollout.set_initial_hidden(
                        hidden_state.detach(),
                        cell_state.detach() if cell_state is not None else None,
                    )
                else:
                    rollout.set_initial_hidden(None, None)
                for step in range(cfg.rollout.num_steps):
                    # 采集阶段不需要梯度：使用 no_grad 避免构建计算图，同时避免 inference_mode 的 backward 限制
                    with torch.no_grad():
                        with torch.amp.autocast(
                            device_type=device.type, enabled=cfg.mixed_precision
                        ):
                            output = model(obs_gpu, hidden_state, cell_state)
                            logits = output.logits
                            values = output.value
                    dist = Categorical(logits=logits)
                    actions = dist.sample()
                    # 若配置 auto-start，在前若干全局 step 强制覆盖动作（仅第一轮）
                    if (
                        auto_start_frames > 0
                        and update_idx == global_update
                        and global_step < auto_start_frames * cfg.env.num_envs
                    ):
                        actions = torch.zeros_like(actions)
                    # Scripted forward warmup：在指定帧数内强制使用前进动作
                    if (
                        getattr(args, "scripted_forward_frames", 0) > 0
                        and global_step
                        < args.scripted_forward_frames * cfg.env.num_envs
                    ):
                        try:
                            # 推断 forward 动作 id（缓存）
                            if not hasattr(run_training, "_forward_action_id") or run_training._forward_action_id is None:  # type: ignore[attr-defined]
                                # 自动猜测：env.action_space.n 与 action_type 组合无直接暴露，借助 metrics 中已统计的 action_hist shape 或假设 id 小
                                # 简单策略：若用户传入 forward_action_id 则直接用；否则使用 1 或 2 中含 RIGHT 概率较高，再 fallback 0
                                faid = getattr(args, "forward_action_id", None)
                                if faid is None:
                                    # 假设 0 可能是 NOOP；优先尝试 1，再 2，再 3，再 0
                                    candidates = [1, 2, 3, 4, 0]
                                else:
                                    candidates = [int(faid)]
                                # 约束到动作空间范围
                                space_n = (
                                    env.action_space.n
                                    if hasattr(env.action_space, "n")
                                    else max(candidates) + 1
                                )
                                run_training._forward_action_id = next((c for c in candidates if c < space_n), 0)  # type: ignore[attr-defined]
                            fid = int(run_training._forward_action_id)  # type: ignore[attr-defined]
                            actions = torch.full_like(actions, fid)
                        except Exception:
                            pass
                    # Scripted sequence 覆盖（优先级高于 forward_frames）
                    if getattr(args, "scripted_sequence", None):
                        # 预处理脚本为 (action_id, remaining_frames) 队列，缓存到 run_training._scripted_seq
                        if not hasattr(run_training, "_scripted_seq"):
                            seq_spec = str(args.scripted_sequence).split(",")
                            parsed = []
                            # 获取动作组合列表（如果 fc wrapper 暴露 _action_combos）
                            action_combos = (
                                getattr(env.envs[0], "_action_combos", None)
                                if hasattr(env, "envs") and env.envs
                                else None
                            )

                            def _combo_to_id(name: str):
                                if action_combos is None:
                                    return None
                                target = tuple(
                                    part.strip().upper()
                                    for part in name.split("+")
                                    if part.strip()
                                )
                                for i, c in enumerate(action_combos):
                                    if tuple(c) == target:
                                        return i
                                return None

                            for token in seq_spec:
                                token = token.strip()
                                if not token:
                                    continue
                                if ":" in token:
                                    nm, fr = token.split(":", 1)
                                    try:
                                        frn = max(1, int(fr))
                                    except Exception:
                                        frn = 1
                                else:
                                    nm, frn = token, 1
                                aid = _combo_to_id(nm)
                                if aid is None:
                                    continue
                                parsed.append([aid, frn])
                            if parsed:
                                run_training._scripted_seq = parsed  # type: ignore[attr-defined]
                                run_training._scripted_seq_idx = 0  # type: ignore[attr-defined]
                        # 应用脚本（每 global_step 减少一帧）
                        if hasattr(run_training, "_scripted_seq"):
                            seq = run_training._scripted_seq  # type: ignore[attr-defined]
                            if seq:
                                cur = seq[0]
                                aid, remain = cur
                                actions = torch.full_like(actions, int(aid))
                                cur[1] -= 1
                                if cur[1] <= 0:
                                    seq.pop(0)
                    # Secondary scripted injection 优先级最高（突破 plateau）
                    if secondary_script_state["remaining"] > 0:
                        sid = secondary_script_action_id
                        if sid is None:
                            try:
                                sid = _resolve_forward_action_id()
                            except Exception:
                                sid = 0
                        take = min(actions.numel(), secondary_script_state["remaining"])
                        actions.view(-1)[:take] = int(sid)
                        secondary_script_state["remaining"] = max(
                            0, secondary_script_state["remaining"] - take
                        )
                        if secondary_script_state[
                            "remaining"
                        ] == 0 and not secondary_script_state.get("announced_done"):
                            print(
                                "[train][secondary-script] 二次脚本注入结束，恢复策略动作。"
                            )
                            secondary_script_state["announced_done"] = True
                    elif auto_bootstrap_state["remaining"] > 0:
                        actions = _apply_auto_bootstrap_actions(actions)
                    log_probs = dist.log_prob(actions)
                    actions_cpu = actions.detach().cpu()
                    # 动作频次统计
                    try:
                        vals, counts = np.unique(
                            actions_cpu.numpy(), return_counts=True
                        )
                        action_hist[vals] += counts
                    except Exception:
                        pass
                    log_probs_cpu = log_probs.detach().cpu()
                    values_cpu = values.detach().cpu()
                    step_start_time = time.time()
                    next_obs_np, reward_np, terminated, truncated, infos = env.step(
                        actions_cpu.numpy()
                    )
                    # 前若干步调试：打印原始 info 结构（仅 env0）
                    if global_step < 20 * cfg.env.num_envs:
                        try:
                            if (
                                isinstance(infos, (list, tuple))
                                and infos
                                and isinstance(infos[0], dict)
                            ):
                                keys0 = list(infos[0].keys())
                                metrics0 = (
                                    list(infos[0].get("metrics", {}).keys())
                                    if isinstance(infos[0].get("metrics"), dict)
                                    else None
                                )
                                print(
                                    f"[train][debug-info] gstep={global_step} keys={keys0} metrics_keys={metrics0}"
                                )
                            elif isinstance(infos, dict):
                                keys0 = list(infos.keys())
                                mk = (
                                    list(infos.get("metrics", {}).keys())
                                    if isinstance(infos.get("metrics"), dict)
                                    else None
                                )
                                print(
                                    f"[train][debug-info] gstep={global_step} dict-keys={keys0} metrics_keys={mk}"
                                )
                        except Exception:
                            pass
                    step_duration = time.time() - step_start_time
                    if step_timeout is not None and step_duration > step_timeout:
                        now_warn = time.time()
                        if (
                            slow_step_cooldown is None
                            or now_warn - last_slow_step_warn >= slow_step_cooldown
                        ):
                            print(
                                f"[train][warn] env.step 耗时 {step_duration:.1f}s (阈值 {step_timeout:.1f}s)，可能存在卡顿。"
                            )
                            heartbeat.notify(
                                phase="rollout",
                                message=f"env.step {step_duration:.1f}s",
                                progress=False,
                            )
                            last_slow_step_warn = now_warn
                        slow_step_events.append(now_warn)
                        # 仅保留窗口内事件
                        slow_step_events = [
                            t
                            for t in slow_step_events
                            if now_warn - t <= slow_trace_window
                        ]
                        if len(slow_step_events) >= slow_trace_threshold:
                            trace_path = (
                                Path(run_dir) / f"slow_step_trace_{int(now_warn)}.log"
                            )
                            try:
                                import faulthandler
                                import traceback as _tb

                                with trace_path.open("w", encoding="utf-8") as fp:
                                    fp.write(
                                        f"# Slow step trace trigger at {datetime.now(UTC).isoformat()}\n"
                                    )
                                    faulthandler.dump_traceback(file=fp)
                                    for tid, frame in sys._current_frames().items():  # type: ignore[attr-defined]
                                        fp.write(f"\n# Thread {tid}\n")
                                        fp.write("".join(_tb.format_stack(frame)))
                                print(
                                    f"[train][trace] slow step threshold reached -> {trace_path}"
                                )
                            except Exception as _exc:  # noqa: BLE001
                                print(
                                    f"[train][trace][error] stack dump failed: {_exc}"
                                )
                            slow_step_events.clear()
                    done = np.logical_or(terminated, truncated)
                    # 提取 x_pos 进度（infos 可能是 list 或 tuple）
                    try:
                        if isinstance(infos, (list, tuple)):
                            for i_env, inf in enumerate(infos):
                                if not isinstance(inf, dict):
                                    continue
                                x_val = inf.get("x_pos") or inf.get("progress")
                                shaping_inf = (
                                    inf.get("shaping")
                                    if isinstance(inf, dict)
                                    else None
                                )
                                if x_val is None:
                                    metrics = (
                                        inf.get("metrics")
                                        if isinstance(inf, dict)
                                        else None
                                    )
                                    if metrics and isinstance(metrics, dict):
                                        x_raw = metrics.get("mario_x") or metrics.get(
                                            "x_pos"
                                        )
                                        if x_raw is not None:
                                            try:
                                                if hasattr(x_raw, "item"):
                                                    x_raw = x_raw.item()
                                                elif (
                                                    hasattr(x_raw, "__len__")
                                                    and len(x_raw) > 0
                                                ):
                                                    x_raw = x_raw[0]
                                                x_val = int(x_raw)
                                            except Exception:
                                                x_val = None
                                if isinstance(x_val, NUMERIC_TYPES):
                                    env_last_x[i_env] = int(x_val)
                                    if env_last_x[i_env] > env_max_x[i_env]:
                                        env_max_x[i_env] = env_last_x[i_env]
                                        # 里程碑检测（基于全局 max）
                                        if (
                                            milestone_interval > 0
                                            and milestone_state["next"] is not None
                                        ):
                                            global_candidate = int(env_max_x.max())
                                            while (
                                                milestone_state["next"] is not None
                                                and global_candidate
                                                >= milestone_state["next"]
                                            ):
                                                if milestone_bonus != 0.0:
                                                    shaping_raw_sum += milestone_bonus
                                                    shaping_scaled_sum += (
                                                        milestone_bonus
                                                        * max(1.0, shaping_last_scale)
                                                    )
                                                milestone_state["count"] += 1
                                                milestone_state[
                                                    "next"
                                                ] += milestone_interval
                                    # 正向位移统计
                                    dx = env_last_x[i_env] - env_prev_step_x[i_env]
                                    # 如果之前始终为 0，允许第一次正值直接加入 (dx = current_x)
                                    if (
                                        env_prev_step_x[i_env] == 0
                                        and env_last_x[i_env] > 0
                                        and dx == env_last_x[i_env]
                                    ):
                                        pass  # dx 已是完整跳跃
                                    if dx > 0:
                                        distance_delta_sum += dx
                                        env_progress_dx[i_env] += int(dx)
                                        stagnation_steps[i_env] = 0
                                    else:
                                        stagnation_steps[i_env] += 1
                                    env_prev_step_x[i_env] = env_last_x[i_env]
                                    # Episode 步数 + 超时截断
                                    per_env_episode_steps[i_env] += 1
                                    if (
                                        episode_timeout_steps > 0
                                        and per_env_episode_steps[i_env] >= episode_timeout_steps
                                    ):
                                        try:
                                            if isinstance(infos[i_env], dict):
                                                infos[i_env]["timeout_truncate"] = True
                                            truncated[i_env] = True
                                        except Exception:
                                            pass
                                    # shaping 奖励原始/缩放累积
                                    if isinstance(shaping_inf, dict):
                                        try:
                                            # Early shaping：在窗口内动态放大 distance 权重（通过 dx 与 last dx scale 额外加成）
                                            dx_candidate = _scalar(
                                                shaping_inf.get("dx")
                                            )
                                            if (
                                                early_shaping_window > 0
                                                and update_idx < early_shaping_window
                                                and dx_candidate
                                            ):
                                                # 推断期望覆盖权重
                                                target_w = early_shaping_distance_weight
                                                if target_w is None:
                                                    # 默认乘以 2
                                                    target_w = (
                                                        float(
                                                            getattr(
                                                                cfg.env.env.reward_config,
                                                                "distance_weight",
                                                                1.0,
                                                            )
                                                        )
                                                        * 2.0
                                                    )
                                                # 原始已加成权重为 cfg.env.env.reward_config.distance_weight (或其退火后的瞬时)；差值补偿
                                                base_w = float(
                                                    getattr(
                                                        cfg.env.env.reward_config,
                                                        "distance_weight",
                                                        1.0,
                                                    )
                                                )
                                                dx_local = float(
                                                    _scalar(dx_candidate) or 0.0
                                                )
                                                if dx_local > 0 and target_w > base_w:
                                                    extra = (
                                                        (target_w - base_w)
                                                        * dx_local
                                                        * float(
                                                            _scalar(
                                                                shaping_inf.get(
                                                                    "scale", 1.0
                                                                )
                                                            )
                                                            or 1.0
                                                        )
                                                    )
                                                    shaping_scaled_sum += extra
                                                    shaping_last_scaled += (
                                                        0.0  # 不覆盖最后刻度，仅累积
                                                    )
                                            raw_v = _scalar(shaping_inf.get("raw"))
                                            scaled_v = _scalar(shaping_inf.get("scaled"))
                                            scale_v = _scalar(shaping_inf.get("scale"))
                                            dx_v = _scalar(shaping_inf.get("dx"))
                                            if isinstance(raw_v, NUMERIC_TYPES):
                                                shaping_raw_sum += float(raw_v)
                                                shaping_last_raw = float(raw_v)
                                            if isinstance(scaled_v, NUMERIC_TYPES):
                                                shaping_scaled_sum += float(scaled_v)
                                                shaping_last_scaled = float(scaled_v)
                                            if isinstance(scale_v, NUMERIC_TYPES):
                                                shaping_last_scale = float(scale_v)
                                            if isinstance(dx_v, NUMERIC_TYPES):
                                                shaping_last_dx = float(dx_v)
                                        except Exception:
                                            shaping_parse_fail += 1
                                            pass
                                    # 早停截断（无进度过久）：标记 truncated=True
                                    if stagnation_steps[i_env] >= stagnation_limit:
                                        try:
                                            if isinstance(infos[i_env], dict):
                                                infos[i_env]["stagnation_truncate"] = True
                                            truncated[i_env] = True
                                        except Exception:
                                            pass
                        elif isinstance(infos, dict):  # 向量 env 可能直接给 dict
                            # 单一 dict：尝试提取 batched 形式 (x_pos: array)
                            x_array = infos.get("x_pos")
                            if x_array is not None:
                                try:
                                    arr = np.asarray(x_array).astype(int)
                                    if arr.ndim == 0:
                                        # 标量 -> 仅更新 env0（其余视为无进展）
                                        arr = np.full(
                                            (cfg.env.num_envs,), int(arr), dtype=int
                                        )
                                    if arr.shape[0] == cfg.env.num_envs:
                                        for i_env in range(cfg.env.num_envs):
                                            last = int(arr[i_env])
                                            env_last_x[i_env] = last
                                            if last > env_max_x[i_env]:
                                                env_max_x[i_env] = last
                                            dx = last - env_prev_step_x[i_env]
                                            if (
                                                env_prev_step_x[i_env] == 0
                                                and last > 0
                                                and dx == last
                                            ):
                                                # 首次整体跳跃
                                                pass
                                            if dx > 0:
                                                distance_delta_sum += dx
                                                env_progress_dx[i_env] += int(dx)
                                                stagnation_steps[i_env] = 0
                                            else:
                                                stagnation_steps[i_env] += 1
                                            env_prev_step_x[i_env] = last
                                            # 计数 episode 步数（batched dict 分支原先缺失）
                                            per_env_episode_steps[i_env] += 1
                                            # 应用超时截断
                                            if (
                                                episode_timeout_steps > 0
                                                and per_env_episode_steps[i_env] >= episode_timeout_steps
                                            ):
                                                # 标记全局 truncated：batched dict 没有逐 env info，使用列表存储标记
                                                try:
                                                    if not isinstance(truncated, (list, tuple)):
                                                        # truncated 可能是 ndarray；直接 in-place 修改
                                                        truncated[i_env] = True  # type: ignore[index]
                                                except Exception:
                                                    pass
                                                # 记录 batched 超时标记（便于后续 episode-end debug 提示）
                                                try:
                                                    tk = "timeout_truncate_batch"
                                                    if tk not in infos:
                                                        infos[tk] = [False] * cfg.env.num_envs
                                                    infos[tk][i_env] = True  # type: ignore[index]
                                                except Exception:
                                                    pass
                                            # 停滞截断
                                            if stagnation_steps[i_env] >= stagnation_limit:
                                                try:
                                                    if not isinstance(truncated, (list, tuple)):
                                                        truncated[i_env] = True  # type: ignore[index]
                                                except Exception:
                                                    pass
                                                try:
                                                    sk = "stagnation_truncate_batch"
                                                    if sk not in infos:
                                                        infos[sk] = [False] * cfg.env.num_envs
                                                    infos[sk][i_env] = True  # type: ignore[index]
                                                except Exception:
                                                    pass
                                except Exception:
                                    pass
                            # batched shaping: 可能是 list/tuple，每个元素为 dict
                            shaping_batch = infos.get("shaping")
                            if (
                                isinstance(shaping_batch, (list, tuple))
                                and len(shaping_batch) == cfg.env.num_envs
                            ):
                                for _i, sh in enumerate(shaping_batch):
                                    if not isinstance(sh, dict):
                                        continue
                                    try:
                                        raw_v = _scalar(sh.get("raw"))
                                        scaled_v = _scalar(sh.get("scaled"))
                                        scale_v = _scalar(sh.get("scale"))
                                        dx_v = _scalar(sh.get("dx"))
                                        if isinstance(raw_v, NUMERIC_TYPES):
                                            shaping_raw_sum += float(raw_v)
                                            shaping_last_raw = float(raw_v)
                                        if isinstance(scaled_v, NUMERIC_TYPES):
                                            shaping_scaled_sum += float(scaled_v)
                                            shaping_last_scaled = float(scaled_v)
                                        if isinstance(scale_v, NUMERIC_TYPES):
                                            shaping_last_scale = float(scale_v)
                                        if isinstance(dx_v, NUMERIC_TYPES):
                                            shaping_last_dx = float(dx_v)
                                    except Exception:
                                        shaping_parse_fail += 1
                            elif isinstance(shaping_batch, dict):
                                try:
                                    raw_arr = np.asarray(
                                        shaping_batch.get("raw"), dtype=float
                                    )
                                except Exception:
                                    raw_arr = None
                                try:
                                    scaled_arr = np.asarray(
                                        shaping_batch.get("scaled"), dtype=float
                                    )
                                except Exception:
                                    scaled_arr = None
                                try:
                                    scale_arr = np.asarray(
                                        shaping_batch.get("scale"), dtype=float
                                    )
                                except Exception:
                                    scale_arr = None
                                try:
                                    dx_arr = np.asarray(
                                        shaping_batch.get("dx"), dtype=float
                                    )
                                except Exception:
                                    dx_arr = None
                                for idx_env in range(cfg.env.num_envs):
                                    try:
                                        raw_v = _scalar(
                                            raw_arr[idx_env]
                                            if raw_arr is not None
                                            and raw_arr.size > idx_env
                                            else None
                                        )
                                        scaled_v = _scalar(
                                            scaled_arr[idx_env]
                                            if scaled_arr is not None
                                            and scaled_arr.size > idx_env
                                            else None
                                        )
                                        scale_v = _scalar(
                                            scale_arr[idx_env]
                                            if scale_arr is not None
                                            and scale_arr.size > idx_env
                                            else None
                                        )
                                        dx_v = _scalar(
                                            dx_arr[idx_env]
                                            if dx_arr is not None
                                            and dx_arr.size > idx_env
                                            else None
                                        )
                                        if isinstance(raw_v, NUMERIC_TYPES):
                                            shaping_raw_sum += float(raw_v)
                                            shaping_last_raw = float(raw_v)
                                        if isinstance(scaled_v, NUMERIC_TYPES):
                                            shaping_scaled_sum += float(scaled_v)
                                            shaping_last_scaled = float(scaled_v)
                                        if isinstance(scale_v, NUMERIC_TYPES):
                                            shaping_last_scale = float(scale_v)
                                        if isinstance(dx_v, NUMERIC_TYPES):
                                            shaping_last_dx = float(dx_v)
                                    except Exception:
                                        shaping_parse_fail += 1
                    except Exception:
                        pass
                    # 重新计算 done 以包含可能刚刚标记的 timeout/stagnation 截断
                    done = np.logical_or(terminated, truncated)
                    reward_cpu = _to_cpu_tensor(reward_np, dtype=torch.float32)
                    done_cpu = _to_cpu_tensor(done.astype(np.float32), dtype=torch.float32)
                    # 额外调试：打印每 update 的 episode steps & 截断原因（受环境变量控制）
                    try:
                        import os as _os_es
                        if _os_es.environ.get("MARIO_EPISODE_STEPS_DEBUG", "0").lower() in {"1", "true", "on"}:
                            if (global_step // cfg.env.num_envs) % cfg.rollout.num_steps == 0:  # 粗略在每 rollout 末尾打印一次
                                print(f"[train][episode-steps-debug] update={update_idx} steps={per_env_episode_steps.tolist()} max_x={env_max_x.tolist()}")
                    except Exception:
                        pass
                    # Episode end debug (before resetting returns) when enabled
                    try:
                        import os as _os_dbg
                        if done.any() and _os_dbg.environ.get("MARIO_EPISODE_DEBUG", "0").lower() in {"1", "true", "on"}:
                            # infos 可能是 list；逐 env 打印终止原因与 return（未清零前）
                            for i_env, flag in enumerate(done):
                                if not flag:
                                    continue
                                reason_flags: list[str] = []
                                try:
                                    if isinstance(infos, (list, tuple)):
                                        source_info = infos[i_env] if len(infos) > i_env else {}
                                        if isinstance(source_info, dict):
                                            for rk in ("flag_get", "timeout_truncate", "stagnation_truncate", "terminated", "truncated"):
                                                if source_info.get(rk):
                                                    reason_flags.append(rk)
                                    elif isinstance(infos, dict):
                                        # 尝试从 batched 标记中解析
                                        for rk in ("timeout_truncate_batch", "stagnation_truncate_batch"):
                                            arr_flags = infos.get(rk)
                                            try:
                                                if arr_flags and arr_flags[i_env]:
                                                    reason_flags.append(rk)
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                                reason_str = ",".join(reason_flags) if reason_flags else "unknown"
                                try:
                                    ep_ret_val = float(episode_returns[i_env])
                                except Exception:
                                    ep_ret_val = 0.0
                                print(
                                    f"[train][episode-end-debug] update={update_idx} gstep={global_step} env={i_env} return={ep_ret_val:.2f} x_pos={env_last_x[i_env]} terminated={bool(terminated[i_env])} truncated={bool(truncated[i_env])} reasons={reason_str}"
                                )
                    except Exception:
                        pass
                    rollout.insert(
                        step,
                        obs_cpu,
                        actions_cpu,
                        log_probs_cpu,
                        reward_cpu,
                        values_cpu,
                        done_cpu,
                    )
                    obs_cpu = _to_cpu_tensor(next_obs_np)
                    obs_gpu = obs_cpu.to(device, non_blocking=(device.type == "cuda"))
                    episode_returns += reward_np
                    if done.any():
                        for idx_env, flag in enumerate(done):
                            if flag:
                                # 汇总结束原因（利用前面 episode-end-debug 已解析的 flags 再次构造）
                                reason_tags: list[str] = []
                                try:
                                    if bool(terminated[idx_env]):
                                        reason_tags.append("terminated")
                                    if bool(truncated[idx_env]):
                                        reason_tags.append("truncated")
                                    # info 结构中可能包含详细标志
                                    inf_obj = infos[idx_env] if isinstance(infos, (list, tuple)) and len(infos) > idx_env else None
                                    if isinstance(inf_obj, dict):
                                        for rk in (
                                            "flag_get",
                                            "timeout_truncate",
                                            "stagnation_truncate",
                                            "timeout_truncate_batch",
                                            "stagnation_truncate_batch",
                                            "dead",
                                        ):
                                            if inf_obj.get(rk):
                                                # 统一 death 标记
                                                if rk == "dead":
                                                    reason_tags.append("death")
                                                else:
                                                    reason_tags.append(rk)
                                except Exception:
                                    pass
                                # 写入 episode_end 事件行（结构化单行 JSON）
                                try:  # 防御性，避免训练中断
                                    ep_len_val = int(per_env_episode_steps[idx_env])
                                    ep_ret_val = float(episode_returns[idx_env])
                                    event_obj = {
                                        "timestamp": time.time(),
                                        "event": "episode_end",
                                        "update": update_idx,
                                        "global_step": global_step,
                                        "env": int(idx_env),
                                        "episode_length": ep_len_val,
                                        "episode_return": ep_ret_val,
                                        "reasons": sorted(set(reason_tags)) or ["unknown"],
                                    }
                                    line = json.dumps(event_obj, ensure_ascii=False)
                                    target_file = episodes_event_file if episodes_event_file else metrics_file  # type: ignore[name-defined]
                                    if target_file is not None:
                                        if metrics_lock is not None:
                                            with metrics_lock:
                                                with target_file.open("a", encoding="utf-8") as fp:  # type: ignore[attr-defined]
                                                    fp.write(line + "\n")
                                        else:
                                            with target_file.open("a", encoding="utf-8") as fp:  # type: ignore[attr-defined]
                                                fp.write(line + "\n")
                                except Exception:
                                    pass
                                completed_returns.append(float(episode_returns[idx_env]))
                                completed_episode_lengths.append(int(per_env_episode_steps[idx_env]))
                                completed_episode_reasons.append(";".join(sorted(set(reason_tags))) or "unknown")
                                episode_returns[idx_env] = 0.0
                                per_env_episode_steps[idx_env] = 0
                        if len(completed_returns) > 1000:
                            completed_returns = completed_returns[-1000:]
                            completed_episode_lengths = completed_episode_lengths[-1000:]
                            completed_episode_reasons = completed_episode_reasons[-1000:]
                    if output.hidden_state is not None:
                        hidden_state = output.hidden_state.detach().clone()
                        if done.any():
                            mask = torch.as_tensor(
                                done, dtype=torch.bool, device=device
                            )
                            hidden_state[:, mask] = 0.0
                        if output.cell_state is not None:
                            cell_state = output.cell_state.detach().clone()
                            if done.any():
                                mask = torch.as_tensor(
                                    done, dtype=torch.bool, device=device
                                )
                                cell_state[:, mask] = 0.0
                        else:
                            cell_state = None
                    else:
                        hidden_state = None
                        cell_state = None
                    # 主动释放局部变量引用，帮助 Python 更快回收（避免长生存期列表）
                    del output, logits, values, dist, actions, log_probs
                    global_step += cfg.env.num_envs
                    if (
                        step + 1
                    ) % current_interval == 0 or step + 1 == cfg.rollout.num_steps:
                        heartbeat.notify(
                            global_step=global_step,
                            phase="rollout",
                            message=f"采样进度 {step + 1}/{cfg.rollout.num_steps}",
                        )
                rollout.set_next_obs(obs_cpu)
            else:
                # overlap 模式
                if update_idx == global_update:
                    # 首次需要前台采集以获得初始缓冲
                    current_buffer.reset()
                    (
                        obs_cpu,
                        hidden_state,
                        cell_state,
                        episode_returns,
                        steps_collected,
                    ) = _collect_rollout(
                        current_buffer,
                        obs_cpu,
                        hidden_state,
                        cell_state,
                        episode_returns,
                        update_idx,
                        progress_heartbeat=True,
                        step_interval=current_interval,
                    )
                    obs_gpu = obs_cpu.to(device, non_blocking=(device.type == "cuda"))
                    global_step += steps_collected  # 首次采集完成后累加步数
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
                    # 累计后台线程采集的步数
                    global_step += getattr(
                        pending_result,
                        "steps",
                        cfg.rollout.num_steps * cfg.env.num_envs,
                    )
                # 在学习阶段开始前启动下一次后台采集（除非最后一次）
                if update_idx + 1 < cfg.total_updates:
                    pending_thread = _start_background_collection(
                        obs_cpu,
                        hidden_state,
                        cell_state,
                        episode_returns,
                        update_idx + 1,
                        step_interval=current_interval,
                    )

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

            with torch.amp.autocast(
                device_type=device.type, enabled=cfg.mixed_precision
            ):
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

            if per_buffer is not None:
                per_loss, per_metrics = _per_step_update(
                    per_buffer=per_buffer,
                    model=model,
                    sequences=sequences,
                    vs=vs,
                    advantages=advantages,
                    cfg=cfg,
                    device=device,
                    update_idx=update_idx,
                    mixed_precision=cfg.mixed_precision,
                    batch_size=cfg.rollout.batch_size,
                )
            else:
                per_loss = torch.tensor(0.0, device=device)
                per_metrics = {"sampled": False, "sample_time_ms": 0.0, "timings": {}}

            policy_loss = -(advantages.detach() * target_log_probs).mean()
            value_loss = F.mse_loss(target_values, vs.detach())
            total_loss = (
                policy_loss
                + cfg.optimizer.value_loss_coef * value_loss
                - cfg.optimizer.beta_entropy * entropy
                + per_loss
            )
            total_loss = total_loss / cfg.gradient_accumulation

            scaler.scale(total_loss).backward()

            if (update_idx + 1) % cfg.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optimizer.max_grad_norm
                )
                last_grad_norm = float(grad_norm_tensor)
                # Execute optimizer step via scaler then advance scheduler only if optimizer succeeded
                try:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                except Exception:
                    # If optimizer step failed, still zero grads to avoid stale grads
                    try:
                        optimizer.zero_grad(set_to_none=True)
                    except Exception:
                        pass
                else:
                    try:
                        scheduler.step()
                    except Exception:
                        pass
                    finally:
                        _refresh_effective_lr()

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
                try:
                    base_lr_current = float(scheduler.get_last_lr()[0])
                except Exception:
                    base_lr_current = float(cfg.optimizer.learning_rate)
                lr_scale = float(globals().get("adaptive_lr_scale_val", 1.0))
                lr_value = base_lr_current * lr_scale
                globals()["adaptive_effective_lr"] = lr_value
                recent_returns = completed_returns[-100:] if completed_returns else []
                recent_episode_lengths = completed_episode_lengths[-100:] if completed_episode_lengths else []
                avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
                return_std = float(np.std(recent_returns)) if recent_returns else 0.0
                return_max = float(np.max(recent_returns)) if recent_returns else 0.0
                return_min = float(np.min(recent_returns)) if recent_returns else 0.0
                p50 = (
                    float(np.percentile(recent_returns, 50)) if recent_returns else 0.0
                )
                p90 = (
                    float(np.percentile(recent_returns, 90)) if recent_returns else 0.0
                )
                p99 = (
                    float(np.percentile(recent_returns, 99)) if recent_returns else 0.0
                )
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
                # 进度统计
                try:
                    mean_x = float(np.mean(env_last_x))
                    max_x = int(np.max(env_max_x))
                    writer.add_scalar("Env/mean_x_pos", mean_x, global_step)
                    writer.add_scalar("Env/max_x_pos", max_x, global_step)
                    writer.add_scalar(
                        "Env/distance_delta_sum", float(distance_delta_sum), global_step
                    )
                    writer.add_scalar(
                        "Env/shaping_raw_sum", float(shaping_raw_sum), global_step
                    )
                    writer.add_scalar(
                        "Env/shaping_scaled_sum", float(shaping_scaled_sum), global_step
                    )
                except Exception:
                    pass
                writer.add_scalar("Reward/avg_return", avg_return, global_step)
                writer.add_scalar("Reward/p50", p50, global_step)
                writer.add_scalar("Reward/p90", p90, global_step)
                writer.add_scalar("Reward/p99", p99, global_step)
                writer.add_scalar("Reward/recent_std", return_std, global_step)
                writer.add_scalar("Grad/global_norm", last_grad_norm, global_step)

                print(
                    f"[train] update={update_idx} step={global_step} avg_return={avg_return:.2f} "
                    f"loss={loss_total:.4f} lr={lr_value:.2e} grad={last_grad_norm:.3f} "
                    f"last_dx={shaping_last_dx:.2f} last_scale={shaping_last_scale:.3f} "
                    f"steps/s={env_steps_per_sec:.1f} upd/s={updates_per_sec:.2f} "
                    f"update={total_update_time:.1f}s rollout={rollout_duration:.1f}s learn={learn_duration:.1f}s"
                )
                if update_idx == global_update:
                    try:
                        print(
                            f"[train][debug] first_log distance_delta_sum={distance_delta_sum} env_last_x={env_last_x.tolist()} env_prev_step_x={env_prev_step_x.tolist()}"
                        )
                    except Exception:
                        pass

                metrics_entry = {
                    "timestamp": time.time(),
                    "update": update_idx,
                    "global_step": global_step,
                    "model_compiled": 1 if hasattr(model, "_orig_mod") else 0,
                    "train_device": device.type,
                    "num_envs": cfg.env.num_envs,
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
                    "recent_return_p50": p50,
                    "recent_return_p90": p90,
                    "recent_return_p99": p99,
                    "episodes_completed": len(completed_returns),
                    # Episode 长度分布（最近窗口）
                    "episode_length_mean": float(np.mean(recent_episode_lengths)) if recent_episode_lengths else 0.0,
                    "episode_length_p50": float(np.percentile(recent_episode_lengths, 50)) if recent_episode_lengths else 0.0,
                    "episode_length_p90": float(np.percentile(recent_episode_lengths, 90)) if recent_episode_lengths else 0.0,
                    "episode_length_p99": float(np.percentile(recent_episode_lengths, 99)) if recent_episode_lengths else 0.0,
                    "grad_norm": last_grad_norm,
                    "env_steps_per_sec": env_steps_per_sec,
                    "updates_per_sec": updates_per_sec,
                    "update_time": total_update_time,
                    "rollout_time": rollout_duration,
                    "learn_time": learn_duration,
                    "env_mean_x_pos": (
                        float(np.mean(env_last_x)) if env_last_x.size else 0.0
                    ),
                    "env_max_x_pos": int(np.max(env_max_x)) if env_max_x.size else 0,
                    "env_distance_delta_sum": int(distance_delta_sum),
                    "env_shaping_raw_sum": float(shaping_raw_sum),
                    "env_shaping_scaled_sum": float(shaping_scaled_sum),
                    "env_shaping_last_dx": float(shaping_last_dx),
                    "env_shaping_last_scale": float(shaping_last_scale),
                    "env_shaping_last_raw": float(shaping_last_raw),
                    "env_shaping_last_scaled": float(shaping_last_scaled),
                    "env_shaping_parse_fail": int(shaping_parse_fail),
                }
                # 估算本 update 内有正向增量的 env 数量，并驱动自适应调度
                progress_ratio = 0.0
                fallback_ratio = None
                progress_envs = 0
                progress_dx_sum = int(np.sum(env_progress_dx))
                # 活动（未终止）episode 当前步数统计（全部 env，因为每个 env 都在运行一个 episode）
                try:
                    active_steps_arr = per_env_episode_steps
                    if getattr(active_steps_arr, "size", 0) > 0:
                        metrics_entry["episode_active_steps_mean"] = float(np.mean(active_steps_arr))
                        metrics_entry["episode_active_steps_max"] = int(np.max(active_steps_arr))
                        metrics_entry["episode_active_steps_p50"] = float(np.percentile(active_steps_arr, 50))
                except Exception:
                    pass
                # 结束原因直方图（最近 200 条 episode）
                try:
                    if completed_episode_reasons:
                        recent_reasons = completed_episode_reasons[-200:]
                        reason_counts: Dict[str, int] = {}
                        for r in recent_reasons:
                            for part in r.split(";"):
                                reason_counts[part] = reason_counts.get(part, 0) + 1
                        total_r = float(sum(reason_counts.values())) or 1.0
                        # 限制最多写入 12 个类别（避免字段爆炸）
                        for rk, rv in list(reason_counts.items())[:12]:
                            metrics_entry[f"episode_end_reason_{rk}"] = int(rv)
                        metrics_entry["episode_end_timeout_ratio"] = float(reason_counts.get("timeout_truncate_batch", 0)) / total_r
                        metrics_entry["episode_end_stagnation_ratio"] = float(reason_counts.get("stagnation_truncate_batch", 0)) / total_r
                        metrics_entry["episode_end_flag_ratio"] = float(reason_counts.get("flag_get", 0)) / total_r
                        metrics_entry["episode_end_death_ratio"] = float(reason_counts.get("death", 0)) / total_r
                except Exception:
                    pass
                try:
                    env_count = int(getattr(env_last_x, "size", 0))
                    if env_count:
                        progress_envs = int(np.count_nonzero(env_progress_dx > 0))
                    metrics_entry["env_positive_dx_envs"] = progress_envs
                    if env_count:
                        progress_ratio = float(progress_envs) / float(env_count)
                        metrics_entry["stagnation_envs"] = int(
                            np.count_nonzero(stagnation_steps > 0)
                        )
                        metrics_entry["stagnation_mean_steps"] = float(
                            np.mean(stagnation_steps)
                        )
                    else:
                        metrics_entry["stagnation_envs"] = 0
                        metrics_entry["stagnation_mean_steps"] = 0.0
                    if progress_ratio <= 0.0 and distance_delta_sum > 0:
                        denom = float(env_count) if env_count else max(
                            1.0, float(cfg.env.num_envs)
                        )
                        base_value = float(
                            progress_dx_sum if progress_dx_sum > 0 else distance_delta_sum
                        )
                        fallback_ratio = min(1.0, base_value / denom)
                        metrics_entry["adaptive_ratio_fallback"] = fallback_ratio
                        progress_ratio = fallback_ratio
                    metrics_entry["env_positive_dx_ratio"] = progress_ratio
                    # 自适应调度：使用 AdaptiveScheduler（全局单例）
                    try:
                        if "adaptive_scheduler_obj" not in globals():
                            adaptive_cfg = getattr(cfg, "adaptive_cfg", None)
                            if isinstance(adaptive_cfg, AdaptiveConfig):
                                base_dw = 0.0
                                try:
                                    base_dw = float(
                                        getattr(
                                            getattr(cfg.env, "env"),
                                            "reward_config",
                                        ).distance_weight
                                    )  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                                globals()["adaptive_scheduler_obj"] = AdaptiveScheduler(
                                    adaptive_cfg,
                                    base_dw,
                                    cfg.optimizer.beta_entropy,
                                    globals().get("adaptive_lr_scale_val", 1.0),
                                )
                        sched = globals().get("adaptive_scheduler_obj")
                        if sched and sched.enabled():
                            new_dw, new_ent, new_lr_scale, adapt_metrics = sched.step(
                                progress_ratio
                            )
                            metrics_entry.update(adapt_metrics)
                            if new_ent is not None:
                                cfg.optimizer = dataclasses.replace(
                                    cfg.optimizer, beta_entropy=float(new_ent)
                                )
                            if new_dw is not None:
                                metrics_entry["adaptive_distance_weight_effective"] = float(
                                    new_dw
                                )
                                # 写回实际 reward wrapper（所有 env）以使 shaping 立即生效
                                try:
                                    if hasattr(env, "envs"):
                                        for _e in env.envs:  # type: ignore[attr-defined]
                                            cur = _e
                                            for _ in range(8):  # unwrap 深度限制
                                                if cur is None:
                                                    break
                                                if cur.__class__.__name__.lower().startswith("mariorewardwrapper"):
                                                    try:
                                                        cur.set_distance_weight(float(new_dw))  # type: ignore[attr-defined]
                                                        metrics_entry.setdefault(
                                                            "wrapper_distance_weight",
                                                            float(new_dw),
                                                        )
                                                    except Exception:
                                                        pass
                                                    break
                                                cur = getattr(cur, "env", None)
                                except Exception:
                                    pass
                            else:
                                # 即使本轮没有调度更新，也尝试记录一个 representative wrapper 的当前值（一次即可）
                                if "wrapper_distance_weight" not in metrics_entry:
                                    try:
                                        if hasattr(env, "envs") and env.envs:  # type: ignore[attr-defined]
                                            cur = env.envs[0]
                                            for _ in range(8):
                                                if cur is None:
                                                    break
                                                if cur.__class__.__name__.lower().startswith("mariorewardwrapper") and hasattr(cur, "get_distance_weight"):
                                                    metrics_entry["wrapper_distance_weight"] = float(
                                                        cur.get_distance_weight()
                                                    )  # type: ignore[attr-defined]
                                                    break
                                                cur = getattr(cur, "env", None)
                                    except Exception:
                                        pass
                            if new_lr_scale is not None:
                                globals()["adaptive_lr_scale_val"] = float(new_lr_scale)
                                _refresh_effective_lr()
                    except Exception as _adapt_e:
                        metrics_entry["adaptive_error"] = str(_adapt_e)[:120]
                except Exception:
                    pass
                # Plateau 检测：若 secondary_script 尚未触发，满足 threshold 且 max_x 未超过 baseline
                try:
                    if (
                        secondary_script_threshold > 0
                        and secondary_script_frames > 0
                        and not secondary_script_state["triggered"]
                        and update_idx >= secondary_script_threshold
                    ):
                        current_global_max = (
                            int(env_max_x.max()) if env_max_x.size else 0
                        )
                        if secondary_script_state["plateau_baseline"] == 0:
                            secondary_script_state["plateau_baseline"] = (
                                current_global_max
                            )
                        # 若当前 max 仍不大于 baseline（允许等于）则触发
                        if (
                            current_global_max
                            <= secondary_script_state["plateau_baseline"]
                        ):
                            secondary_script_state["remaining"] = (
                                secondary_script_frames * cfg.env.num_envs
                            )
                            secondary_script_state["triggered"] = True
                            print(
                                f"[train][secondary-script] 触发二次脚本注入 baseline={secondary_script_state['plateau_baseline']} frames={secondary_script_frames}"
                            )
                            metrics_entry["secondary_script_triggered"] = 1
                        else:
                            metrics_entry["secondary_script_progress_surpassed"] = 1
                    if secondary_script_state["triggered"]:
                        metrics_entry["secondary_script_remaining"] = int(
                            secondary_script_state["remaining"]
                        )
                except Exception:
                    pass
                # 追加 auto-bootstrap 状态与诊断字段，便于外部快速判断是否已触发/剩余帧数
                try:
                    metrics_entry["auto_bootstrap_triggered"] = (
                        1 if auto_bootstrap_state.get("triggered") else 0
                    )
                    metrics_entry["auto_bootstrap_remaining"] = int(
                        auto_bootstrap_state.get("remaining", 0)
                    )
                    if auto_bootstrap_state.get("action_id") is not None:
                        metrics_entry["auto_bootstrap_action_id"] = int(
                            auto_bootstrap_state.get("action_id")
                        )
                except Exception:
                    pass
                # 若累计距离仍为 0，输出一次额外诊断（每 50 updates 最多一次，避免刷屏）
                try:
                    if (
                        int(metrics_entry.get("env_distance_delta_sum", 0) or 0) == 0
                        and update_idx > 0
                        and update_idx % 50 == 0
                    ):
                        # 采样前若干 env 的最近 x / max / prev_step 差异
                        sample_n = (
                            min(4, env_last_x.shape[0])
                            if hasattr(env_last_x, "shape")
                            else 0
                        )
                        if sample_n > 0:
                            last_sample = [int(env_last_x[i]) for i in range(sample_n)]
                            prev_sample = [
                                int(env_prev_step_x[i]) for i in range(sample_n)
                            ]
                            max_sample = [int(env_max_x[i]) for i in range(sample_n)]
                            print(
                                f"[train][diag] update={update_idx} zero-distance persists last_x={last_sample} prev_x={prev_sample} max_x={max_sample} "
                                f"raw_sum={metrics_entry.get('env_shaping_raw_sum')} scaled_sum={metrics_entry.get('env_shaping_scaled_sum')} "
                                f"last_dx={metrics_entry.get('env_shaping_last_dx')} scale={metrics_entry.get('env_shaping_last_scale')}"
                            )
                except Exception:
                    pass
                # 里程碑与超时统计写入
                try:
                    if milestone_interval > 0:
                        metrics_entry["milestone_count"] = int(milestone_state["count"])
                        metrics_entry["milestone_next"] = (
                            int(milestone_state["next"])
                            if milestone_state["next"] is not None
                            else -1
                        )
                    if episode_timeout_steps > 0:
                        metrics_entry["episode_timeout_steps"] = int(
                            episode_timeout_steps
                        )
                except Exception:
                    pass
                # 若存在 forward 探测结果，仅在首个 log 写入一次（前 8 个）
                if forward_probe_result is not None and update_idx == global_update:
                    for i, (aid, dx_probe, last_x_probe) in enumerate(
                        forward_probe_result[:8]
                    ):
                        metrics_entry[f"probe_action_{i}_id"] = aid
                        metrics_entry[f"probe_action_{i}_dx"] = dx_probe
                        metrics_entry[f"probe_action_{i}_last_x"] = last_x_probe
                    metrics_entry["probe_forward_action_selected"] = int(
                        getattr(args, "forward_action_id", -1)
                        if getattr(args, "forward_action_id", None) is not None
                        else -1
                    )
                # 动作直方图（写入前 16 个 bins）
                try:
                    total_actions = int(action_hist.sum())
                    if total_actions > 0:
                        for aid in range(min(16, action_hist.shape[0])):
                            if action_hist[aid] > 0:
                                metrics_entry[f"action_freq_{aid}"] = (
                                    float(action_hist[aid]) / total_actions
                                )
                except Exception:
                    pass
                # 如果当前轮进行了 PER 抽样，添加采样耗时指标
                if per_buffer is not None:
                    sampled = bool(per_metrics.get("sampled"))
                    sample_time = (
                        float(per_metrics.get("sample_time_ms", 0.0))
                        if sampled
                        else 0.0
                    )
                    metrics_entry["replay_sample_time_ms"] = sample_time
                    timings = per_metrics.get("timings", {}) if sampled else {}
                    for k in ("prior", "choice", "weight", "decode", "tensor", "total"):
                        metrics_entry[f"replay_sample_split_{k}_ms"] = (
                            float(timings.get(k, 0.0)) if sampled else 0.0
                        )
                # 静态配置：per_sample_interval 直接写入，便于外部解析
                metrics_entry["replay_per_sample_interval"] = (
                    int(cfg.replay.per_sample_interval) if per_buffer is not None else 0
                )
                stats_snapshot = per_buffer.stats() if per_buffer is not None else {}
                metrics_entry["replay_gpu_sampler"] = (
                    1 if (per_buffer is not None and per_buffer.using_gpu_sampler) else 0
                )
                metrics_entry["replay_gpu_sampler_fallback"] = (
                    1 if bool(per_metrics.get("gpu_sampler_fallback")) else 0
                )
                if stats_snapshot.get("gpu_sampler_reason"):
                    metrics_entry["replay_gpu_sampler_reason"] = stats_snapshot[
                        "gpu_sampler_reason"
                    ]

                # PER stats (if enabled)
                if per_buffer is not None:
                    rstats = per_buffer.stats()
                    metrics_entry.update(
                        {
                            "replay_size": rstats["size"],
                            "replay_capacity": rstats["capacity"],
                            "replay_fill_rate": rstats["fill_rate"],
                            "replay_last_unique_ratio": rstats[
                                "last_sample_unique_ratio"
                            ],
                            "replay_avg_unique_ratio": rstats.get(
                                "avg_sample_unique_ratio", 0.0
                            ),
                            "replay_push_total": rstats.get("push_total", 0),
                            "replay_priority_mean": rstats.get("priority_mean", 0.0),
                            "replay_priority_p50": rstats.get("priority_p50", 0.0),
                            "replay_priority_p90": rstats.get("priority_p90", 0.0),
                            "replay_priority_p99": rstats.get("priority_p99", 0.0),
                        }
                    )
                    try:
                        writer.add_scalar(
                            "Replay/fill_rate", rstats["fill_rate"], global_step
                        )
                        writer.add_scalar(
                            "Replay/last_unique_ratio",
                            rstats["last_sample_unique_ratio"],
                            global_step,
                        )
                        if "avg_sample_unique_ratio" in rstats:
                            writer.add_scalar(
                                "Replay/avg_unique_ratio",
                                rstats["avg_sample_unique_ratio"],
                                global_step,
                            )
                        writer.add_scalar("Replay/size", rstats["size"], global_step)
                        if "priority_mean" in rstats:
                            writer.add_scalar(
                                "Replay/priority_mean",
                                rstats["priority_mean"],
                                global_step,
                            )
                            writer.add_scalar(
                                "Replay/priority_p90",
                                rstats["priority_p90"],
                                global_step,
                            )
                    except Exception:
                        pass

                if wandb_run is not None:
                    wandb_run.log({**metrics_entry})
                # 每次 log 后清空按 update 统计的累计量（下一次 update 重新积累）
                distance_delta_sum = 0
                shaping_raw_sum = 0.0
                shaping_scaled_sum = 0.0
                shaping_last_dx = 0.0
                shaping_last_scale = 0.0
                shaping_last_raw = 0.0
                shaping_last_scaled = 0.0
                shaping_parse_fail = 0
                action_hist[:] = 0
                env_progress_dx[:] = 0
                stagnation_steps[:] = 0
                env_prev_step_x[:] = env_last_x
                # 更新级别停滞检测（基于全局 max）
                try:
                    global_max_now = int(env_max_x.max())
                    if global_max_now > prev_global_max_x:
                        prev_global_max_x = global_max_now
                        stagnation_update_counter = 0
                    else:
                        stagnation_update_counter += 1
                        if (
                            stagnation_warn_updates > 0
                            and stagnation_update_counter >= stagnation_warn_updates
                        ):
                            print(
                                f"[train][warn] Progress stagnation: no global max_x improvement for {stagnation_update_counter} updates (max_x={prev_global_max_x})."
                            )
                            stagnation_update_counter = 0  # 重置避免刷屏
                except Exception:
                    pass

                # Resource monitoring: GPU/CPU/memory
                def _get_resource_stats():
                    stats = {}
                    # GPU stats via torch
                    try:
                        if torch.cuda.is_available():
                            stats["gpu_count"] = torch.cuda.device_count()
                            for gi in range(torch.cuda.device_count()):
                                stats[f"gpu_{gi}_mem_alloc_bytes"] = float(
                                    torch.cuda.memory_allocated(gi)
                                )
                                stats[f"gpu_{gi}_mem_reserved_bytes"] = float(
                                    torch.cuda.memory_reserved(gi)
                                )
                    except Exception:
                        pass

                    # GPU utilization and memory via nvidia-smi (if available)
                    try:
                        cmd = [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits",
                        ]
                        res = subprocess.run(
                            cmd, capture_output=True, text=True, timeout=2
                        )
                        if res.returncode == 0 and res.stdout.strip():
                            lines = [
                                line_var for line_var in res.stdout.strip().splitlines() if line_var.strip()
                            ]
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
                            stats["proc_cpu_percent"] = float(
                                p.cpu_percent(interval=None)
                            )
                            stats["proc_mem_rss_bytes"] = float(p.memory_info().rss)
                            stats["system_cpu_percent"] = float(
                                psutil.cpu_percent(interval=None)
                            )
                            stats["system_mem_percent"] = float(
                                psutil.virtual_memory().percent
                            )
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
                            writer.add_scalar(
                                f"Resource/{name}", float(val), global_step
                            )
                        except Exception:
                            pass
                    if wandb_run is not None and stats:
                        logd = dict(stats)
                        logd["global_step"] = global_step
                        wandb_run.log(logd)

                # 降低 nvidia-smi 调用频率：仅当距离上次记录 >=120s 再写入 GPU util（否则仅使用 torch.cuda.memory 信息）
                if stats:
                    metrics_entry["resource"] = stats
                    # GPU 利用率窗口聚合
                    if not hasattr(run_training, "_gpu_util_hist"):
                        run_training._gpu_util_hist = []  # type: ignore[attr-defined]
                    gpu_hist: list[float] = run_training._gpu_util_hist  # type: ignore[attr-defined]
                    util_vals = [v for k, v in stats.items() if k.endswith("_util_pct")]
                    if util_vals:
                        util_mean_snapshot = float(np.mean(util_vals))
                        gpu_hist.append(util_mean_snapshot)
                        if len(gpu_hist) > 100:
                            del gpu_hist[: len(gpu_hist) - 100]
                        metrics_entry["gpu_util_last"] = util_mean_snapshot
                        metrics_entry["gpu_util_mean_window"] = float(np.mean(gpu_hist))
                        try:
                            writer.add_scalar(
                                "Resource/gpu_util_last",
                                util_mean_snapshot,
                                global_step,
                            )
                            writer.add_scalar(
                                "Resource/gpu_util_mean_window",
                                float(np.mean(gpu_hist)),
                                global_step,
                            )
                        except Exception:
                            pass

                if (
                    not auto_bootstrap_state["triggered"]
                    and auto_bootstrap_threshold > 0
                    and auto_bootstrap_frames > 0
                    and update_idx >= auto_bootstrap_threshold
                    and int(metrics_entry.get("env_distance_delta_sum", 0) or 0) <= 0
                ):
                    auto_bootstrap_state["remaining"] = (
                        auto_bootstrap_frames * cfg.env.num_envs
                    )
                    auto_bootstrap_state["triggered"] = True
                    auto_bootstrap_state["announced_done"] = False
                    if not auto_bootstrap_state.get("announced_start", False):
                        print(
                            "[train][auto-bootstrap] 距离增量为 0，自动注入前进动作以帮助突破冷启动。"
                        )
                        auto_bootstrap_state["announced_start"] = True

                try:
                    _maybe_print_training_hints(update_idx, metrics_entry)
                except Exception:
                    pass

                try:
                    with metrics_lock:
                        with metrics_file.open("a", encoding="utf-8") as fp:
                            fp.write(json.dumps(metrics_entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                try:
                    write_latest_metrics(
                        metrics_entry, metrics_file.with_name("latest.parquet")
                    )
                except Exception:
                    pass
                try:
                    rotate_metrics_file(
                        metrics_file,
                        getattr(cfg, "metrics_rotate_max_mb", 0.0),
                        getattr(cfg, "metrics_rotate_retain", 5),
                    )
                except Exception:
                    pass

                # 训练健康度检查：长时间没有完成 episode 可能意味着仍在使用 Dummy env（历史 bug）或奖励逻辑异常
                if (
                    update_idx > 0
                    and len(completed_returns) == 0
                    and update_idx % (cfg.log_interval * 10) == 0
                ):
                    print(
                        "[train][warn] No completed episodes yet – verify environments are real (not Dummy) and reward/done signals are propagating."
                    )

                last_log_time = now
                last_log_step = global_step
                last_log_update = update_idx

            if (
                cfg.checkpoint_interval
                and update_idx % cfg.checkpoint_interval == 0
                and update_idx > 0
            ):
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
                heartbeat.notify(
                    phase="checkpoint",
                    message=f"保存 {checkpoint_path.name}",
                    progress=False,
                )

            if (
                cfg.eval_interval
                and update_idx % cfg.eval_interval == 0
                and update_idx > 0
            ):
                save_model_snapshot(cfg, model, global_step, update_idx)
                print("[train] latest snapshot refreshed")
                heartbeat.notify(
                    phase="checkpoint", message="最新快照已更新", progress=False
                )
        # 首次 update 完成后做一次 debug 打印（仅一次）
        if update_idx == global_update:
            try:
                mean_x_dbg = float(np.mean(env_last_x))
                print(
                    f"[train][debug] first_update_post_learn global_step={global_step} mean_x={mean_x_dbg:.1f} max_x={int(env_max_x.max())} distance_delta_sum={distance_delta_sum}"
                )
            except Exception:
                pass
    except KeyboardInterrupt:
        print(
            "[train][warn] interrupted by user, attempting graceful shutdown & latest snapshot save"
        )
        heartbeat.notify(
            phase="interrupt", message="收到中断信号，保存最新模型", progress=False
        )
        try:
            _safe_update = int(locals().get("update_idx", globals().get("global_update", 0)))
            snap_path = save_model_snapshot(cfg, model, global_step, _safe_update)  # type: ignore[name-defined]
            # 更新 metadata reason=interrupt
            try:
                meta_file = snap_path.with_suffix(".json")
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    meta["save_state"]["reason"] = "interrupt"
                    meta_file.write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001
            print(f"[train][error] failed to save latest snapshot on interrupt: {exc}")
    except Exception as exc:  # pragma: no cover - 异常捕获仅日志
        print(f"[train][error] unexpected exception: {exc}")
        heartbeat.notify(phase="error", message=str(exc), progress=False)
        try:
            _safe_update = int(locals().get("update_idx", globals().get("global_update", 0)))
            snap_path = save_model_snapshot(cfg, model, global_step, _safe_update)  # type: ignore[name-defined]
            try:
                meta_file = snap_path.with_suffix(".json")
                if meta_file.exists():
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    meta["save_state"]["reason"] = "exception"
                    meta_file.write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
            except Exception:
                pass
            print("[train][info] latest snapshot saved after exception")
        except Exception as exc2:  # noqa: BLE001
            print(f"[train][error] failed to save snapshot after exception: {exc2}")

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
        if "writer" in locals() and writer is not None:
            writer.close()
    except Exception:
        pass
    try:
        if "wandb_run" in locals() and wandb_run is not None:
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
