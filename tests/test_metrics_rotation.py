from pathlib import Path

import gzip

from src.utils.metrics_export import rotate_metrics_file


def test_rotate_metrics_file(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    data = "x" * (1024 * 1024)  # 1 MB
    metrics_path.write_text(data, encoding="utf-8")

    rotated = rotate_metrics_file(metrics_path, max_mb=0.5, retain=1)
    assert rotated is True
    assert metrics_path.exists()
    assert metrics_path.read_text(encoding="utf-8") == ""
    archives = list(tmp_path.glob("metrics.jsonl.*.gz"))
    assert len(archives) == 1
    with gzip.open(archives[0], "rb") as fp:
        assert fp.read().decode("utf-8") == data
