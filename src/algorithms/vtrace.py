"""V-trace target computation from IMPALA."""

from __future__ import annotations

from typing import Tuple

import torch


def vtrace_returns(
    behaviour_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    discounts: torch.Tensor,
    clip_rho_threshold: float = 1.0,
    clip_c_threshold: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute V-trace returns and advantages.

    Args:
        behaviour_log_probs: (T, B) log probs from behaviour policy.
        target_log_probs: (T, B) log probs from target policy.
        rewards: (T, B) rewards.
        values: (T, B) value estimates for steps.
        bootstrap_value: (B,) value estimate for final state.
        discounts: (T, B) discounts (gamma * (1 - done)).
    Returns:
        Tuple of (vs, advantages) each of shape (T, B).
    """

    log_rhos = target_log_probs - behaviour_log_probs
    rhos = log_rhos.exp()
    if clip_rho_threshold is not None:
        clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    else:
        clipped_rhos = rhos

    if clip_c_threshold is not None:
        cs = torch.clamp(rhos, max=clip_c_threshold)
    else:
        cs = rhos

    values_t_plus_1 = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)

    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    vs = torch.zeros_like(values)
    acc = torch.zeros_like(bootstrap_value)

    for t in reversed(range(values.shape[0])):
        acc = deltas[t] + discounts[t] * cs[t] * acc
        vs[t] = values[t] + acc

    advantages = clipped_rhos * (
        rewards
        + discounts * torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
        - values
    )
    return vs, advantages


__all__ = ["vtrace_returns"]
