"""Rollout buffer for vectorised on-policy data collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class HiddenState:
    hidden: Optional[torch.Tensor]
    cell: Optional[torch.Tensor]


class RolloutBuffer:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        device: torch.device,
        pin_memory: bool = True,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.target_device = device
        storage_device = torch.device("cpu")
        pin = pin_memory and device.type == "cuda"

        def _alloc(shape: Tuple[int, ...], dtype: torch.dtype):
            tensor = torch.empty(shape, device=storage_device, dtype=dtype)
            if pin:
                tensor = tensor.pin_memory()
            return tensor

        self.obs = _alloc((num_steps + 1, num_envs, *obs_shape), torch.float32)
        self.actions = _alloc((num_steps, num_envs), torch.long)
        self.rewards = _alloc((num_steps, num_envs), torch.float32)
        self.dones = _alloc((num_steps, num_envs), torch.float32)
        self.behaviour_log_probs = _alloc((num_steps, num_envs), torch.float32)
        self.values = _alloc((num_steps, num_envs), torch.float32)
        self.initial_hidden: HiddenState = HiddenState(None, None)

    def to(self, device: torch.device):
        self.target_device = device
        return self

    def reset(self):
        for tensor in (
            self.obs,
            self.actions,
            self.rewards,
            self.dones,
            self.behaviour_log_probs,
            self.values,
        ):
            tensor.zero_()
        self.initial_hidden = HiddenState(None, None)

    def set_initial_hidden(
        self, hidden: Optional[torch.Tensor], cell: Optional[torch.Tensor]
    ):
        hidden_detached = hidden.detach().cpu() if hidden is not None else None
        cell_detached = cell.detach().cpu() if cell is not None else None
        self.initial_hidden = HiddenState(hidden_detached, cell_detached)

    def insert(
        self,
        step: int,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        self.obs[step].copy_(obs)
        self.actions[step].copy_(action)
        self.behaviour_log_probs[step].copy_(log_prob)
        self.rewards[step].copy_(reward)
        self.values[step].copy_(value)
        self.dones[step].copy_(done)

    def set_next_obs(self, obs: torch.Tensor):
        self.obs[-1].copy_(obs)

    def get_sequences(
        self, device: Optional[torch.device] = None, non_blocking: bool = True
    ):
        data = {
            "obs": self.obs[:-1],
            "next_obs": self.obs[-1],
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "behaviour_log_probs": self.behaviour_log_probs,
            "values": self.values,
            "initial_hidden": self.initial_hidden,
        }

        if device is None:
            return data

        def _to(t: torch.Tensor) -> torch.Tensor:
            kwargs = {"non_blocking": non_blocking} if t.is_pinned() else {}
            return t.to(device, **kwargs)

        initial_hidden = self.initial_hidden
        hidden = (
            _to(initial_hidden.hidden) if initial_hidden.hidden is not None else None
        )
        cell = _to(initial_hidden.cell) if initial_hidden.cell is not None else None

        return {
            "obs": _to(data["obs"]),
            "next_obs": _to(data["next_obs"]),
            "actions": _to(data["actions"]),
            "rewards": _to(data["rewards"]),
            "dones": _to(data["dones"]),
            "behaviour_log_probs": _to(data["behaviour_log_probs"]),
            "values": _to(data["values"]),
            "initial_hidden": HiddenState(hidden=hidden, cell=cell),
        }
