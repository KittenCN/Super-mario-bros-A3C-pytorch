"""Helpers for exporting structured metrics artefacts."""

from __future__ import annotations

import gzip
import shutil
from datetime import UTC, datetime

from pathlib import Path
from typing import Any, Dict

try:  # Optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def write_latest_metrics(entry: Dict[str, Any], path: Path) -> bool:
    """Persist the latest metrics row to a Parquet file.

    Returns True when the export succeeds (and pandas/pyarrow are available),
    otherwise False. Failures are swallowed so the training loop can continue.
    """

    if pd is None:
        return False
    try:
        df = pd.DataFrame([entry])
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def rotate_metrics_file(path: Path, max_mb: float, retain: int = 5) -> bool:
    """Rotate and gzip metrics JSONL file when it grows beyond ``max_mb``.

    Returns True when a rotation happened. Failures are ignored so training
    can continue uninterrupted.
    """

    if max_mb <= 0:
        return False
    if not path.exists():
        return False
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb < max_mb:
            return False
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        archive = path.with_suffix(path.suffix + f".{timestamp}.gz")
        archive.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as src, gzip.open(archive, "wb") as dst:
            shutil.copyfileobj(src, dst)
        path.write_text("", encoding="utf-8")
        if retain >= 0:
            archives = sorted(
                archive.parent.glob(f"{path.name}.*.gz"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old in archives[retain:]:
                try:
                    old.unlink()
                except Exception:
                    pass
        return True
    except Exception:
        return False


__all__ = ["write_latest_metrics", "rotate_metrics_file"]
