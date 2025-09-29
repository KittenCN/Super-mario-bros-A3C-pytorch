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
    def __init__(self, num_steps: int, num_envs: int, obs_shape: Tuple[int, ...], device: torch.device, pin_memory: bool = True) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        # allocate on CPU using pinned memory to speed up host->device copies
        cpu_device = torch.device("cpu")
        pin = pin_memory if device.type == "cuda" else False
        self.obs = torch.empty((num_steps + 1, num_envs, *obs_shape), device=cpu_device, dtype=torch.float32).pin_memory() if pin else torch.empty((num_steps + 1, num_envs, *obs_shape), device=cpu_device, dtype=torch.float32)
        self.actions = torch.empty((num_steps, num_envs), device=cpu_device, dtype=torch.long).pin_memory() if pin else torch.empty((num_steps, num_envs), device=cpu_device, dtype=torch.long)
        self.rewards = torch.empty((num_steps, num_envs), device=cpu_device, dtype=torch.float32).pin_memory() if pin else torch.empty((num_steps, num_envs), device=cpu_device, dtype=torch.float32)
        self.dones = torch.empty((num_steps, num_envs), device=cpu_device, dtype=torch.float32).pin_memory() if pin else torch.empty((num_steps, num_envs), device=cpu_device, dtype=torch.float32)
        self.behaviour_log_probs = torch.empty((num_steps, num_envs), device=cpu_device, dtype=torch.float32).pin_memory() if pin else torch.empty((num_steps, num_envs), device=cpu_device, dtype=torch.float32)
        self.values = torch.empty((num_steps, num_envs), device=cpu_device, dtype=torch.float32).pin_memory() if pin else torch.empty((num_steps, num_envs), device=cpu_device, dtype=torch.float32)
        self.initial_hidden: HiddenState = HiddenState(None, None)

    def to(self, device: torch.device):
        self.device = device
        for attr in ("obs", "actions", "rewards", "dones", "behaviour_log_probs", "values"):
            tensor = getattr(self, attr)
            # use non_blocking if the tensor is pinned in CPU memory
            kwargs = {"non_blocking": True} if tensor.is_pinned() and device.type == "cuda" else {}
            setattr(self, attr, tensor.to(device, **kwargs))
        if self.initial_hidden.hidden is not None:
            self.initial_hidden = HiddenState(
                hidden=self.initial_hidden.hidden.to(device),
                cell=self.initial_hidden.cell.to(device) if self.initial_hidden.cell is not None else None,
            )
        return self

    def reset(self):
        for tensor in (self.obs, self.actions, self.rewards, self.dones, self.behaviour_log_probs, self.values):
            tensor.zero_()
        self.initial_hidden = HiddenState(None, None)

    def set_initial_hidden(self, hidden: Optional[torch.Tensor], cell: Optional[torch.Tensor]):
        hidden_detached = hidden.detach().to(self.device) if hidden is not None else None
        cell_detached = cell.detach().to(self.device) if cell is not None else None
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

    def get_sequences(self):
        return {
            "obs": self.obs[:-1],
            "next_obs": self.obs[-1],
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "behaviour_log_probs": self.behaviour_log_probs,
            "values": self.values,
            "initial_hidden": self.initial_hidden,
        }
