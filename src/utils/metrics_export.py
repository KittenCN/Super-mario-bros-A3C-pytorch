"""Helpers for exporting structured metrics artefacts."""

from __future__ import annotations

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


__all__ = ["write_latest_metrics"]
