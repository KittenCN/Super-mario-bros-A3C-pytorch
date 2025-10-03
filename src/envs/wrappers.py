"""Custom Gymnasium wrappers for Super Mario Bros environments."""

from __future__ import annotations

import dataclasses
import io
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Generic, Protocol

# Temporarily silence stderr during imports that may emit Gym migration notices
_saved_stderr = sys.stderr
try:
    sys.stderr = io.StringIO()
    import gymnasium as gym
    import numpy as np
finally:
    sys.stderr = _saved_stderr


def _coerce_scalar(value: Any) -> Any:
    """尽可能将 numpy 数组 / 序列压缩为标量，失败则返回原值。"""

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
            pass
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _coerce_scalar(value[0])
    return value


@dataclasses.dataclass
class RewardConfig:
    """Configuration for reward shaping."""

    score_weight: float = 1.0 / 40.0
    # 位置推进奖励权重：对 (x_pos_t - x_pos_{t-1}) 的正向增量给予加成
    distance_weight: float = 1.0 / 80.0
    # 若需要对 distance_weight 做线性退火，可指定最终值与步数（按 wrapper step 计数）
    distance_weight_final: float | None = None
    distance_weight_anneal_steps: int = 0
    flag_reward: float = 5.0
    death_penalty: float = -5.0
    # 同理可对 death_penalty 做从 start -> final 的线性插值（通常从 0 降到负值，减小早期学习噪声）
    death_penalty_start: float | None = None
    death_penalty_final: float | None = None
    death_penalty_anneal_steps: int = 0
    # 原始全局缩放（保持向后兼容，如果未启用动态调度即使用该值）
    scale: float = 0.1
    # 动态缩放：线性插值 (0 -> anneal_steps) 从 scale_start 到 scale_final
    scale_start: float = 0.2
    scale_final: float = 0.1
    scale_anneal_steps: int = 50_000  # 全局 env.step 计数（非 updates）


class MarioRewardWrapper(gym.Wrapper):
    """Reward shaping wrapper.

    新增：
    - set_distance_weight(): 允许训练循环/自适应调度器在运行期安全更新 distance_weight。
    - get_diagnostics(): 返回最近一次 step 缓存的 shaping 诊断，便于测试/监控。
    """

    def __init__(self, env: gym.Env, config: Optional[RewardConfig] = None) -> None:
        super().__init__(env)
        self.config = config or RewardConfig()
        self._prev_score = 0
        self._prev_x = 0
        self._step_counter = 0  # wrapper 自身统计的 step 数，用于动态缩放
        # RAM 解析相关（训练脚本可在创建后通过属性启用）
        self.enable_ram_x_parse: bool = False
        self.ram_addr_high: int = 0x006D  # NES SMB 常见地址 (高字节)
        self.ram_addr_low: int = 0x0086  # NES SMB 常见地址 (低字节)
        self._ram_failure: int = 0
        self._ram_success: int = 0
        self._ram_last_x: int = 0
        self._first_progress_pending: bool = (
            True  # 首次读取进度时允许把 current_x 记作正增量
        )
        # 调试输出计数（需放在 __init__ 内，否则模块导入时引用 self 抛 NameError 导致静默退出）
        self._ram_debug_prints: int = 0  # 限制调试输出次数
        # 最近一次 step 的诊断快照（避免直接引用 info 被外部修改）
        self._last_diag: Dict[str, Any] = {}
        self._dx_missed_warn: int = 0

    # ---- Runtime adaptive setters ----
    def set_distance_weight(self, new_weight: float) -> None:
        """Update distance_weight at runtime (用于 AdaptiveScheduler 注入)。"""
        try:
            if not isinstance(new_weight, (int, float)):
                return
            if new_weight < 0:
                return
            self.config.distance_weight = float(new_weight)
        except Exception:  # pragma: no cover - 防御性
            pass

    def get_distance_weight(self) -> float:
        return float(self.config.distance_weight)

    def get_diagnostics(self) -> Dict[str, Any]:  # 供测试/调试
        return dict(self._last_diag)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        observation, info = self.env.reset(seed=seed, options=options)
        # fc_emulator backend 可能将信息放在 metrics 子字典中
        metrics = info.get("metrics") if isinstance(info, dict) else None
        if metrics and isinstance(metrics, dict):
            # score 可能以 numpy array 形式存在，如 array([0])
            raw_score = _coerce_scalar(metrics.get("score"))
            if raw_score is not None:
                info["score"] = raw_score
        score_initial = _coerce_scalar(info.get("score", 0))
        self._prev_score = float(score_initial) if isinstance(score_initial, (int, float)) else 0.0
        # 记录初始 x_pos（优先 metrics 再回退 info）
        metrics_x = (
            _coerce_scalar(metrics.get("mario_x") or metrics.get("x_pos"))
            if metrics and isinstance(metrics, dict)
            else None
        )
        info_x = _coerce_scalar(info.get("x_pos")) if isinstance(info, dict) else None
        if isinstance(metrics_x, (int, float)):
            x_pos = int(metrics_x)
            if isinstance(info, dict):
                info["x_pos"] = x_pos
        elif isinstance(info_x, (int, float)):
            x_pos = int(info_x)
        else:
            x_pos = 0
        # 直接记录初始 x 但不视为已消耗首次增量，待下一步 step 时若有提升将 dx=当前值
        self._prev_x = x_pos
        self._first_progress_pending = True
        self._dx_missed_warn = 0
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        metrics = info.get("metrics") if isinstance(info, dict) else None
        if metrics and isinstance(metrics, dict):
            if "score" not in info:
                raw_score = _coerce_scalar(metrics.get("score"))
                if raw_score is not None:
                    info["score"] = raw_score
            metrics_x = _coerce_scalar(metrics.get("mario_x") or metrics.get("x_pos"))
        else:
            metrics_x = None
        score = _coerce_scalar(info.get("score", 0)) or 0
        score_delta = float(score) - float(self._prev_score)
        shaped_reward = reward + self.config.score_weight * score_delta
        self._prev_score = float(score)

        # 前向位移奖励（仅对正向增量，避免倒退产生负 shaping 干扰）
        current_x = _coerce_scalar(info.get("x_pos", None))
        if isinstance(metrics_x, (int, float)):
            current_x = float(metrics_x)
        if not isinstance(current_x, (int, float)) and self.enable_ram_x_parse:
            # 尝试基于 RAM fallback
            try:
                ram_buf = None
                base = getattr(self.env, "unwrapped", self.env)
                if hasattr(base, "_ram_buffer") and callable(
                    getattr(base, "_ram_buffer")
                ):
                    ram_buf = base._ram_buffer()
                elif hasattr(base, "ram"):
                    ram_buf = base.ram
                if ram_buf is not None:
                    arr = np.asarray(ram_buf)
                    hi = (
                        int(arr[self.ram_addr_high])
                        if self.ram_addr_high < len(arr)
                        else 0
                    )
                    lo = (
                        int(arr[self.ram_addr_low])
                        if self.ram_addr_low < len(arr)
                        else 0
                    )
                    parsed_x = (hi << 8) | lo
                    current_x = parsed_x
                    self._ram_success += 1
                    if self._ram_debug_prints < 10:
                        print(
                            f"[reward][ram-debug] step={self._step_counter} hi=0x{hi:02X} lo=0x{lo:02X} parsed_x={parsed_x} prev_x={self._prev_x}"
                        )
                        self._ram_debug_prints += 1
                else:
                    self._ram_failure += 1
            except Exception:
                self._ram_failure += 1
        if isinstance(current_x, (int, float)):
            current_x = float(current_x)
            if self._first_progress_pending and current_x > float(self._prev_x):
                # 视为首次整体位移增量
                dx = current_x
                self._first_progress_pending = False
            else:
                dx = current_x - float(self._prev_x)
            if dx > 0:
                shaped_reward += self.config.distance_weight * dx
            self._prev_x = current_x
            self._ram_last_x = (
                int(current_x)
                if isinstance(current_x, (int, float))
                else self._ram_last_x
            )
        else:
            dx = 0.0
            try:
                if (
                    current_x is None
                    and self.config.distance_weight > 0
                    and self._dx_missed_warn < 5
                    and self._step_counter >= 5
                ):
                    import os

                    if os.environ.get("MARIO_SHAPING_DEBUG", "0") in {
                        "1",
                        "true",
                        "on",
                    }:
                        print(
                            "[reward][warn] dx_missed: no x_pos data "
                            f"step={self._step_counter} distance_weight={self.config.distance_weight} "
                            f"ram_enabled={self.enable_ram_x_parse} ram_fail={self._ram_failure}"
                        )
                        self._dx_missed_warn += 1
            except Exception:  # pragma: no cover
                pass
        # 诊断：若外部统计出现推进但本地 dx 始终 0，便于定位 (通过环境变量开关避免噪音)
        try:
            if dx == 0.0 and isinstance(current_x, (int, float)) and self.config.distance_weight > 0 and self._step_counter < 5:
                import os

                if os.environ.get("MARIO_SHAPING_DEBUG", "0") in {"1", "true", "on"}:
                    print(
                        f"[reward][debug] dx_zero current_x={current_x} prev_x={self._prev_x} first_pending={self._first_progress_pending} step={self._step_counter}"
                    )
        except Exception:  # pragma: no cover
            pass

        # 动态缩放系数
        self._step_counter += 1
        if self.config.scale_anneal_steps > 0:
            t = min(1.0, self._step_counter / float(self.config.scale_anneal_steps))
            dyn_scale = (
                self.config.scale_start
                + (self.config.scale_final - self.config.scale_start) * t
            )
        else:
            dyn_scale = self.config.scale
        # 若用户未修改默认 scale_start/scale_final，则保持与 scale 一致
        if self.config.scale_start == self.config.scale_final:
            dyn_scale = self.config.scale_start

        # 距离权重动态退火（可选）
        if (
            self.config.distance_weight_final is not None
            and self.config.distance_weight_anneal_steps > 0
        ):
            t_dw = min(
                1.0,
                self._step_counter / float(self.config.distance_weight_anneal_steps),
            )
            eff_distance_weight = (
                self.config.distance_weight
                + (self.config.distance_weight_final - self.config.distance_weight)
                * t_dw
            )
        else:
            eff_distance_weight = self.config.distance_weight

        # 将上面使用的 shaped_reward 中 distance_weight（对 dx>0 时添加）调整：我们之前已用 distance_weight 计算在 shaped_reward 里；
        # 为避免重算，这里只在 dx>0 且有差异时补差值（差分法），避免复制逻辑。
        try:
            if (
                eff_distance_weight != self.config.distance_weight
                and isinstance(dx, (int, float))
                and dx > 0
            ):
                delta_extra = (eff_distance_weight - self.config.distance_weight) * dx
                shaped_reward += delta_extra
        except Exception:
            pass
        raw_before_scale = shaped_reward
        shaped_reward *= dyn_scale

        # 诊断信息（不覆盖已有 key）
        try:
            shaping = info.setdefault("shaping", {}) if isinstance(info, dict) else None
            diag_local: Dict[str, Any] = {}
            if isinstance(shaping, dict):
                shaping["dx"] = dx
                shaping["score_delta"] = score_delta
                shaping["raw"] = float(raw_before_scale)
                shaping["scale"] = float(dyn_scale)
                shaping["scaled"] = float(shaped_reward)
                diag_local = {
                    "dx": dx,
                    "score_delta": score_delta,
                    "raw": float(raw_before_scale),
                    "scale": float(dyn_scale),
                    "scaled": float(shaped_reward),
                }
                if self.enable_ram_x_parse:
                    ram_d = {
                        "success": self._ram_success,
                        "failure": self._ram_failure,
                        "last_x": self._ram_last_x,
                    }
                    shaping["ram_parse"] = ram_d
                    diag_local["ram_parse"] = ram_d
            self._last_diag = diag_local
        except Exception:  # pragma: no cover
            pass

        # 调试：首次出现正向位移时打印一次（不依赖 RAM debug 次数）
        if dx > 0 and getattr(self, "_printed_first_dx", False) is False:
            print(
                f"[reward][dx] first_positive_dx step={self._step_counter} dx={dx:.2f} distance_weight={self.config.distance_weight}"
            )
            setattr(self, "_printed_first_dx", True)

        if terminated or truncated:
            # 动态 death_penalty 退火（仅 episode 终止/截断时应用）
            if info.get("flag_get"):
                shaped_reward += self.config.flag_reward
            else:
                # 计算当前退火 death
                dp_base = self.config.death_penalty
                if (
                    self.config.death_penalty_start is not None
                    and self.config.death_penalty_final is not None
                    and self.config.death_penalty_anneal_steps > 0
                ):
                    t_dp = min(
                        1.0,
                        self._step_counter
                        / float(self.config.death_penalty_anneal_steps),
                    )
                    dp_eff = (
                        self.config.death_penalty_start
                        + (
                            self.config.death_penalty_final
                            - self.config.death_penalty_start
                        )
                        * t_dp
                    )
                else:
                    dp_eff = dp_base
                shaped_reward += dp_eff
            # 可选：终止调试输出
            try:
                import os
                if os.environ.get("MARIO_EPISODE_TERMINATION_DEBUG", "0") in {"1","true","on"}:
                    print(
                        f"[reward][episode-end] step={self._step_counter} terminated={terminated} truncated={truncated} flag={bool(info.get('flag_get'))} reward_post={shaped_reward:.3f} dx_last={self._last_diag.get('dx')} raw={self._last_diag.get('raw')} scale={self._last_diag.get('scale')}"
                    )
            except Exception:  # pragma: no cover
                pass

        # 注意：末尾不再再次乘 self.config.scale（已通过动态 dyn_scale 应用）
        return observation, shaped_reward, terminated, truncated, info


class ProgressInfoWrapper(gym.Wrapper):
    """Augment info dict with progress statistics."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._max_distance = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        observation, info = self.env.reset(seed=seed, options=options)
        metrics = info.get("metrics") if isinstance(info, dict) else None
        x_pos = info.get("x_pos", 0)
        if metrics and isinstance(metrics, dict):
            raw_x = metrics.get("mario_x") or metrics.get("x_pos")
            if raw_x is not None:
                try:
                    if hasattr(raw_x, "item"):
                        raw_x = raw_x.item()
                    elif hasattr(raw_x, "__len__") and len(raw_x) > 0:
                        raw_x = raw_x[0]
                except Exception:
                    pass
                x_pos = int(raw_x)
                info.setdefault("x_pos", x_pos)
        self._max_distance = x_pos
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        metrics = info.get("metrics") if isinstance(info, dict) else None
        distance = info.get("x_pos", 0)
        if metrics and isinstance(metrics, dict):
            raw_x = metrics.get("mario_x") or metrics.get("x_pos")
            if raw_x is not None:
                try:
                    if hasattr(raw_x, "item"):
                        raw_x = raw_x.item()
                    elif hasattr(raw_x, "__len__") and len(raw_x) > 0:
                        raw_x = raw_x[0]
                except Exception:
                    pass
                distance = int(raw_x)
                info.setdefault("x_pos", distance)
        if distance > self._max_distance:
            self._max_distance = distance
        info["progress"] = self._max_distance
        info["terminated"] = terminated
        info["truncated"] = truncated
        return observation, reward, terminated, truncated, info


class TransformObservation(gym.ObservationWrapper):
    """Apply arbitrary transformation to observation."""

    def __init__(self, env: gym.Env, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        super().__init__(env)
        self._fn = fn
        assert isinstance(self.observation_space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            low=self._fn(self.observation_space.low),
            high=self._fn(self.observation_space.high),
            dtype=self._fn(np.asarray(self.observation_space.low)).dtype,
        )

    def observation(self, observation):
        return self._fn(observation)


class TransformReward(gym.Wrapper):
    """Apply transformation to rewards."""

    def __init__(self, env: gym.Env, fn: Callable[[float], float]) -> None:
        super().__init__(env)
        self._fn = fn

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self._fn(reward), terminated, truncated, info


__all__ = [
    "MarioRewardWrapper",
    "ProgressInfoWrapper",
    "RewardConfig",
    "TransformObservation",
    "TransformReward",
]
