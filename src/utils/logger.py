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


def create_logger(log_dir: str, project: Optional[str] = None, resume: bool = False):
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    run_name = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(path / run_name)

    wandb_run = None
    if project and wandb is not None:
        wandb_run = wandb.init(project=project, name=run_name, dir=str(path), resume=resume)

    return writer, wandb_run

