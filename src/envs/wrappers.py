"""Custom Gymnasium wrappers for Super Mario Bros environments."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional

import sys, io

# Temporarily silence stderr during imports that may emit Gym migration notices
_saved_stderr = sys.stderr
try:
    sys.stderr = io.StringIO()
    import gymnasium as gym
    import numpy as np
finally:
    sys.stderr = _saved_stderr


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
    """Reward shaping to encourage progress and penalize failure."""

    def __init__(self, env: gym.Env, config: Optional[RewardConfig] = None) -> None:
        super().__init__(env)
        self.config = config or RewardConfig()
        self._prev_score = 0
        self._prev_x = 0
        self._step_counter = 0  # wrapper 自身统计的 step 数，用于动态缩放
        # RAM 解析相关（训练脚本可在创建后通过属性启用）
        self.enable_ram_x_parse: bool = False
        self.ram_addr_high: int = 0x006D  # NES SMB 常见地址 (高字节)
        self.ram_addr_low: int = 0x0086   # NES SMB 常见地址 (低字节)
        self._ram_failure: int = 0
        self._ram_success: int = 0
        self._ram_last_x: int = 0
        self._first_progress_pending: bool = True  # 首次读取进度时允许把 current_x 记作正增量
        # 调试输出计数（需放在 __init__ 内，否则模块导入时引用 self 抛 NameError 导致静默退出）
        self._ram_debug_prints: int = 0  # 限制调试输出次数

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        observation, info = self.env.reset(seed=seed, options=options)
        # fc_emulator backend 可能将信息放在 metrics 子字典中
        metrics = info.get("metrics") if isinstance(info, dict) else None
        if metrics and isinstance(metrics, dict):
            # score 可能以 numpy array 形式存在，如 array([0])
            raw_score = metrics.get("score")
            if raw_score is not None:
                try:
                    if hasattr(raw_score, "item"):
                        raw_score = raw_score.item()
                    elif hasattr(raw_score, "__len__") and len(raw_score) > 0:
                        raw_score = raw_score[0]
                except Exception:
                    pass
                info.setdefault("score", raw_score)
        self._prev_score = info.get("score", 0)
        # 记录初始 x_pos（如果可得）
        x_pos = 0
        if isinstance(info, dict):
            if "x_pos" in info:
                try:
                    x_pos = int(info["x_pos"])
                except Exception:
                    pass
            elif metrics and isinstance(metrics, dict):
                raw_x = metrics.get("mario_x") or metrics.get("x_pos")
                if raw_x is not None:
                    try:
                        if hasattr(raw_x, "item"):
                            raw_x = raw_x.item()
                        elif hasattr(raw_x, "__len__") and len(raw_x) > 0:
                            raw_x = raw_x[0]
                        x_pos = int(raw_x)
                    except Exception:
                        pass
        # 直接记录初始 x 但不视为已消耗首次增量，待下一步 step 时若有提升将 dx=当前值
        self._prev_x = x_pos
        self._first_progress_pending = True
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        metrics = info.get("metrics") if isinstance(info, dict) else None
        if metrics and isinstance(metrics, dict) and "score" not in info:
            raw_score = metrics.get("score")
            if raw_score is not None:
                try:
                    if hasattr(raw_score, "item"):
                        raw_score = raw_score.item()
                    elif hasattr(raw_score, "__len__") and len(raw_score) > 0:
                        raw_score = raw_score[0]
                except Exception:
                    pass
                info["score"] = raw_score
        score = info.get("score", 0)
        score_delta = score - self._prev_score
        shaped_reward = reward + self.config.score_weight * score_delta
        self._prev_score = score

        # 前向位移奖励（仅对正向增量，避免倒退产生负 shaping 干扰）
        current_x = info.get("x_pos", None)
        if current_x is None and metrics and isinstance(metrics, dict):
            raw_x = metrics.get("mario_x") or metrics.get("x_pos")
            if raw_x is not None:
                try:
                    if hasattr(raw_x, "item"):
                        raw_x = raw_x.item()
                    elif hasattr(raw_x, "__len__") and len(raw_x) > 0:
                        raw_x = raw_x[0]
                    current_x = int(raw_x)
                except Exception:
                    current_x = None
        if not isinstance(current_x, (int, float)) and self.enable_ram_x_parse:
            # 尝试基于 RAM fallback
            try:
                ram_buf = None
                base = getattr(self.env, 'unwrapped', self.env)
                if hasattr(base, '_ram_buffer') and callable(getattr(base, '_ram_buffer')):
                    ram_buf = base._ram_buffer()
                elif hasattr(base, 'ram'):
                    ram_buf = base.ram
                if ram_buf is not None:
                    arr = np.asarray(ram_buf)
                    hi = int(arr[self.ram_addr_high]) if self.ram_addr_high < len(arr) else 0
                    lo = int(arr[self.ram_addr_low]) if self.ram_addr_low < len(arr) else 0
                    parsed_x = (hi << 8) | lo
                    current_x = parsed_x
                    self._ram_success += 1
                    if self._ram_debug_prints < 10:
                        print(f"[reward][ram-debug] step={self._step_counter} hi=0x{hi:02X} lo=0x{lo:02X} parsed_x={parsed_x} prev_x={self._prev_x}")
                        self._ram_debug_prints += 1
                else:
                    self._ram_failure += 1
            except Exception:
                self._ram_failure += 1
        if isinstance(current_x, (int, float)):
            if self._first_progress_pending and float(current_x) > float(self._prev_x):
                # 视为首次整体位移增量
                dx = float(current_x)
                self._first_progress_pending = False
            else:
                dx = float(current_x) - float(self._prev_x)
            if dx > 0:
                shaped_reward += self.config.distance_weight * dx
            self._prev_x = float(current_x)
            self._ram_last_x = int(current_x) if isinstance(current_x, (int, float)) else self._ram_last_x
        else:
            dx = 0.0

        # 动态缩放系数
        self._step_counter += 1
        if self.config.scale_anneal_steps > 0:
            t = min(1.0, self._step_counter / float(self.config.scale_anneal_steps))
            dyn_scale = self.config.scale_start + (self.config.scale_final - self.config.scale_start) * t
        else:
            dyn_scale = self.config.scale
        # 若用户未修改默认 scale_start/scale_final，则保持与 scale 一致
        if self.config.scale_start == self.config.scale_final:
            dyn_scale = self.config.scale_start

        # 距离权重动态退火（可选）
        if self.config.distance_weight_final is not None and self.config.distance_weight_anneal_steps > 0:
            t_dw = min(1.0, self._step_counter / float(self.config.distance_weight_anneal_steps))
            eff_distance_weight = self.config.distance_weight + (self.config.distance_weight_final - self.config.distance_weight) * t_dw
        else:
            eff_distance_weight = self.config.distance_weight

        # 将上面使用的 shaped_reward 中 distance_weight（对 dx>0 时添加）调整：我们之前已用 distance_weight 计算在 shaped_reward 里；
        # 为避免重算，这里只在 dx>0 且有差异时补差值（差分法），避免复制逻辑。
        try:
            if eff_distance_weight != self.config.distance_weight and isinstance(dx, (int, float)) and dx > 0:
                delta_extra = (eff_distance_weight - self.config.distance_weight) * dx
                shaped_reward += delta_extra
        except Exception:
            pass
        raw_before_scale = shaped_reward
        shaped_reward *= dyn_scale

        # 诊断信息（不覆盖已有 key）
        try:
            shaping = info.setdefault("shaping", {}) if isinstance(info, dict) else None
            if isinstance(shaping, dict):
                # 直接覆盖，保证日志/metrics 始终反映最新一次 step 的诊断值
                shaping["dx"] = dx
                shaping["score_delta"] = score_delta
                shaping["raw"] = float(raw_before_scale)
                shaping["scale"] = float(dyn_scale)
                shaping["scaled"] = float(shaped_reward)
                if self.enable_ram_x_parse:
                    shaping["ram_parse"] = {
                        "success": self._ram_success,
                        "failure": self._ram_failure,
                        "last_x": self._ram_last_x,
                    }
        except Exception:
            pass

        # 调试：首次出现正向位移时打印一次（不依赖 RAM debug 次数）
        if dx > 0 and getattr(self, "_printed_first_dx", False) is False:
            print(f"[reward][dx] first_positive_dx step={self._step_counter} dx={dx:.2f} distance_weight={self.config.distance_weight}")
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
                    t_dp = min(1.0, self._step_counter / float(self.config.death_penalty_anneal_steps))
                    dp_eff = self.config.death_penalty_start + (self.config.death_penalty_final - self.config.death_penalty_start) * t_dp
                else:
                    dp_eff = dp_base
                shaped_reward += dp_eff

        # 注意：末尾不再再次乘 self.config.scale（已通过动态 dyn_scale 应用）
        return observation, shaped_reward, terminated, truncated, info


class ProgressInfoWrapper(gym.Wrapper):
    """Augment info dict with progress statistics."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._max_distance = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
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

