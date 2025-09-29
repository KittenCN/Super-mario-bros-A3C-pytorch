"""Custom Gymnasium wrappers for Super Mario Bros environments."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np


@dataclasses.dataclass
class RewardConfig:
    """Configuration for reward shaping."""

    score_weight: float = 1.0 / 40.0
    flag_reward: float = 5.0
    death_penalty: float = -5.0
    scale: float = 0.1


class MarioRewardWrapper(gym.Wrapper):
    """Reward shaping to encourage progress and penalize failure."""

    def __init__(self, env: gym.Env, config: Optional[RewardConfig] = None) -> None:
        super().__init__(env)
        self.config = config or RewardConfig()
        self._prev_score = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._prev_score = info.get("score", 0)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        score = info.get("score", 0)
        shaped_reward = reward + self.config.score_weight * (score - self._prev_score)
        self._prev_score = score

        if terminated or truncated:
            if info.get("flag_get"):
                shaped_reward += self.config.flag_reward
            else:
                shaped_reward += self.config.death_penalty

        shaped_reward *= self.config.scale
        return observation, shaped_reward, terminated, truncated, info


class ProgressInfoWrapper(gym.Wrapper):
    """Augment info dict with progress statistics."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._max_distance = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._max_distance = info.get("x_pos", 0)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        distance = info.get("x_pos", 0)
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

