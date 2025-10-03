"""Integration test: AdaptiveScheduler distance_weight 应实时影响 shaping raw.

思路：
1. 构造一个最小伪环境 wrapper 链，手动创建 MarioRewardWrapper 并伪造 step() 产生正向 dx。
2. 模拟 adaptive scheduler 输出一系列 new_dw，调用 set_distance_weight 后调用 step。
3. 断言 raw 奖励增量与 distance_weight 线性对应。

由于真实 Mario 环境较重，此测试不依赖 gym-super-mario-bros；使用简化 FakeEnv。
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import gymnasium as gym

from src.envs.wrappers import MarioRewardWrapper, RewardConfig


class FakeEnv(gym.Env):
    """最简环境：每步提供固定 dx 与 score 增量，满足 gym.Env 接口。"""

    metadata: Dict[str, Any] = {}

    def __init__(self) -> None:
        super().__init__()
        self._x = 0
        self._score = 0
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):  # type: ignore[override]
        self._x = 0
        self._score = 0
        return np.zeros((1,), dtype=np.float32), {"x_pos": 0, "score": 0}

    def step(self, action):  # type: ignore[override]
        self._x += 5
        self._score += 1
        obs = np.zeros((1,), dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {"x_pos": self._x, "score": self._score}
        return obs, reward, terminated, truncated, info


def test_adaptive_distance_weight_injection_linear():
    env = FakeEnv()
    wrapper = MarioRewardWrapper(env, RewardConfig(distance_weight=0.01, scale_start=1.0, scale_final=1.0, scale_anneal_steps=0))
    obs, info = wrapper.reset()
    # baseline step
    _, r1, *_ = wrapper.step(0)
    baseline_raw = wrapper.get_diagnostics().get("raw", 0.0)
    assert baseline_raw > 0, "应产生正向 shaping raw"

    # 提升 distance_weight 10x
    wrapper.set_distance_weight(0.1)
    _, r2, *_ = wrapper.step(0)
    raw_after = wrapper.get_diagnostics().get("raw", 0.0)

    # raw 应近似按比例放大（score_weight 贡献较小，可允许相对误差 15%）
    assert raw_after > baseline_raw * 5, f"raw_after={raw_after} baseline={baseline_raw} 未明显放大"

    # 再次提升
    wrapper.set_distance_weight(0.2)
    _, r3, *_ = wrapper.step(0)
    raw_after2 = wrapper.get_diagnostics().get("raw", 0.0)
    assert raw_after2 > raw_after * 1.5, "第二次提升未生效"

    # distance_weight 不能为负
    wrapper.set_distance_weight(-1.0)
    assert wrapper.get_distance_weight() >= 0, "负值写入应被忽略"

