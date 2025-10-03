"""单环境 shaping 冒烟测试：模拟 fc_emulator 聚合格式确保 raw > 0.

流程：
1. 使用 FakeFcEnv + MarioRewardWrapper 产生含 metrics 的 info。
2. 构造与 fc_emulator 类似的 batched info 结构（dict + array）。
3. 复用训练循环中的解析逻辑，累计 shaping_raw_sum 与 dx 指标。
4. 断言 shaping_raw_sum、distance_delta_sum 与进度占比均为正，避免回归。
"""

from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import numpy as np

from src.envs.wrappers import MarioRewardWrapper, RewardConfig


class FakeFcEnv(gym.Env):
    """最小化 fc_emulator 风格的环境：仅返回 metrics.x_pos/score。"""

    metadata: Dict[str, Any] = {}

    def __init__(self) -> None:
        super().__init__()
        self._x = 0
        self._score = 0
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(  # type: ignore[override]
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ):
        self._x = 0
        self._score = 0
        obs = np.zeros((1,), dtype=np.float32)
        info = {"metrics": {"mario_x": self._x, "score": self._score}}
        return obs, info

    def step(self, action):  # type: ignore[override]
        self._x += 8
        self._score += 1
        obs = np.zeros((1,), dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {"metrics": {"mario_x": self._x, "score": self._score}}
        return obs, reward, terminated, truncated, info


def test_single_env_fc_shaping_smoke():
    wrapper = MarioRewardWrapper(
        FakeFcEnv(),
        RewardConfig(
            distance_weight=0.05,
            scale_start=1.0,
            scale_final=1.0,
            scale_anneal_steps=0,
        ),
    )
    wrapper.reset()
    _, _, _, _, info = wrapper.step(0)
    shaping = info.get("shaping")
    assert isinstance(shaping, dict), "wrapper 应写入 shaping 诊断"
    assert shaping.get("raw", 0.0) > 0.0, "shaping raw 应为正"

    x_raw = info.get("x_pos")
    if not isinstance(x_raw, (int, np.integer)):
        x_raw = shaping.get("dx", 0)
    fc_infos = {
        "x_pos": np.array([int(x_raw)], dtype=np.int32),
        "shaping": [shaping],
    }
    env_last_x = np.zeros(1, dtype=np.int32)
    env_prev_step_x = np.zeros(1, dtype=np.int32)
    env_progress_dx = np.zeros(1, dtype=np.int32)
    stagnation_steps = np.zeros(1, dtype=np.int32)
    distance_delta_sum = 0
    shaping_raw_sum = 0.0
    shaping_scaled_sum = 0.0

    x_array = fc_infos.get("x_pos")
    assert x_array is not None
    arr = np.asarray(x_array).astype(int)
    for i_env in range(arr.shape[0]):
        last = int(arr[i_env])
        env_last_x[i_env] = last
        dx = last - env_prev_step_x[i_env]
        if env_prev_step_x[i_env] == 0 and last > 0 and dx == last:
            pass
        if dx > 0:
            distance_delta_sum += dx
            env_progress_dx[i_env] += int(dx)
            stagnation_steps[i_env] = 0
        else:
            stagnation_steps[i_env] += 1
        env_prev_step_x[i_env] = last

    shaping_batch = fc_infos.get("shaping")
    assert isinstance(shaping_batch, list) and len(shaping_batch) == 1
    for sh in shaping_batch:
        raw_v = sh.get("raw")
        scaled_v = sh.get("scaled")
        if isinstance(raw_v, (int, float)):
            shaping_raw_sum += float(raw_v)
        if isinstance(scaled_v, (int, float)):
            shaping_scaled_sum += float(scaled_v)

    assert distance_delta_sum > 0, "应检测到正向位移"
    assert env_progress_dx[0] > 0, "env_progress_dx 累计应>0"
    assert shaping_raw_sum > 0.0, "shaping_raw_sum 累计应为正"
    assert shaping_scaled_sum > 0.0, "shaping_scaled_sum 累计应为正"
    progress_ratio = float(np.count_nonzero(env_progress_dx > 0)) / 1.0
    assert progress_ratio > 0.0, "进度占比应大于 0"
