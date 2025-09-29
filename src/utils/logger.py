"""Logging utilities for training."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

try:  # pragma: no cover - optional dependency
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore


def create_logger(log_dir: str, project: Optional[str] = None, resume: bool = False, enable_tb: bool = True):
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    run_name = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb_run = None
    if project and wandb is not None:
        wandb_run = wandb.init(project=project, name=run_name, dir=str(path), resume=resume)

    if not enable_tb:
        # Provide a minimal no-op writer with the same interface we use.
        class _NoopWriter:
            def add_scalar(self, *args, **kwargs):
                return

            def close(self):
                return

            @property
            def log_dir(self):
                return str(path / run_name)

        return _NoopWriter(), wandb_run

    writer = SummaryWriter(path / run_name)
    return writer, wandb_run

