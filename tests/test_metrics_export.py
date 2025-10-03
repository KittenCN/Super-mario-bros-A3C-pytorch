from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from src.utils.metrics_export import write_latest_metrics


def test_write_latest_metrics(tmp_path):
    entry = {
        "timestamp": 123.456,
        "update": 17,
        "global_step": 1024,
        "loss_total": 0.5,
    }
    out_path = tmp_path / "latest.parquet"
    assert write_latest_metrics(entry, out_path) is True
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert df.loc[0, "update"] == entry["update"]
    assert abs(df.loc[0, "loss_total"] - entry["loss_total"]) < 1e-9
