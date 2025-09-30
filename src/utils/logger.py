"""Logging utilities for training."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Optional, Tuple

from torch.utils.tensorboard import SummaryWriter

try:  # pragma: no cover - optional dependency
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore


def create_logger(
    log_dir: str,
    project: Optional[str] = None,
    resume: bool = False,
    enable_tb: bool = True,
) -> Tuple[Any, Optional["wandb.sdk.wandb_run.Run"], Path]:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    run_name = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_path = path / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if project and wandb is not None:
        wandb_run = wandb.init(project=project, name=run_name, dir=str(path), resume=resume)

    if not enable_tb:
        # Provide a minimal no-op writer with the same interface we use but backed by the run directory.
        class _NoopWriter:
            def __init__(self, base: Path):
                self._base = base

            def add_scalar(self, *args, **kwargs):
                return

            def close(self):
                return

            @property
            def log_dir(self):
                return str(self._base)

        return _NoopWriter(run_path), wandb_run, run_path

    writer = SummaryWriter(run_path)
    return writer, wandb_run, run_path

