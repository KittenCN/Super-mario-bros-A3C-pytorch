"""Learning rate schedules."""

from __future__ import annotations

import math


class CosineWithWarmup:
    def __init__(self, warmup_steps: int, total_steps: int):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        progress = (step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))
