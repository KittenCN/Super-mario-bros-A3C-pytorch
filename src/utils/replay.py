"""Prioritized experience replay buffer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class ReplaySample:
    observations: torch.Tensor
    actions: torch.Tensor
    target_values: torch.Tensor
    advantages: torch.Tensor
    weights: torch.Tensor
    indices: np.ndarray


class PrioritizedReplay:
    def __init__(
        self,
        capacity: int,
        alpha: float,
        beta_start: float,
        beta_final: float,
        beta_steps: int,
        device: torch.device,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_final = beta_final
        self.beta_steps = beta_steps
        self.device = device

        self.storage: list[dict[str, torch.Tensor]] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.step = 0

    def __len__(self) -> int:
        return len(self.storage)

    def beta(self) -> float:
        fraction = min(1.0, self.step / float(max(1, self.beta_steps)))
        return self.beta_start + fraction * (self.beta_final - self.beta_start)

    def push(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        target_values: torch.Tensor,
        advantages: torch.Tensor,
        priorities: Optional[torch.Tensor] = None,
    ) -> None:
        obs_np = observations.detach().cpu()
        act_np = actions.detach().cpu()
        tgt_np = target_values.detach().cpu()
        adv_np = advantages.detach().cpu()
        if priorities is None:
            priorities = adv_np.abs().sum(dim=tuple(range(1, adv_np.dim()))) if adv_np.dim() > 1 else adv_np.abs()
        prio_np = priorities.detach().cpu().numpy().flatten()

        for idx in range(obs_np.shape[0]):
            data = {
                "obs": obs_np[idx],
                "action": act_np[idx].view(-1),
                "target_value": tgt_np[idx].view(-1),
                "advantage": adv_np[idx].view(-1),
            }
            if len(self.storage) < self.capacity:
                self.storage.append(data)
            else:
                self.storage[self.pos] = data
            self.priorities[self.pos] = prio_np[idx] + 1e-5
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Optional[ReplaySample]:
        if len(self.storage) < batch_size:
            return None
        priorities = self.priorities[: len(self.storage)]
        scaled = priorities ** self.alpha
        probs = scaled / scaled.sum()
        indices = np.random.choice(len(self.storage), batch_size, p=probs)
        beta = self.beta()
        weights = (len(self.storage) * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        obs = torch.stack([self.storage[i]["obs"] for i in indices]).to(self.device)
        actions = torch.stack([self.storage[i]["action"] for i in indices]).to(self.device).squeeze(-1).long()
        target_values = torch.stack([self.storage[i]["target_value"] for i in indices]).to(self.device).squeeze(-1)
        advantages = torch.stack([self.storage[i]["advantage"] for i in indices]).to(self.device).squeeze(-1)
        weights_t = torch.tensor(weights, device=self.device, dtype=torch.float32)
        self.step += 1
        return ReplaySample(obs, actions, target_values, advantages, weights_t, indices)

    def update_priorities(self, indices: np.ndarray, priorities: torch.Tensor):
        self.priorities[indices] = priorities.detach().cpu().numpy() + 1e-5
