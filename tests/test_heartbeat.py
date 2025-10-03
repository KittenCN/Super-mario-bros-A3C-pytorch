import json
import time

from pathlib import Path

from src.utils.heartbeat import HeartbeatReporter


def test_heartbeat_structured_log(tmp_path):
    log_path = tmp_path / "hb.jsonl"
    reporter = HeartbeatReporter(
        component="train", interval=1.0, stall_timeout=5.0, enabled=True
    )
    reporter.set_log_path(log_path)
    dump_dir = tmp_path / "stall"
    reporter.set_stall_dump_dir(dump_dir, cooldown=0.5)

    reporter.notify(update_idx=3, global_step=42, phase="init", message="starting")
    reporter.heartbeat_now()
    reporter._dump_stall_trace(time.time())  # force dump for test

    assert log_path.exists()
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert lines
    payload = json.loads(lines[-1])
    assert payload["component"] == "train"
    assert payload["update"] == 3
    assert payload["global_step"] == 42
    assert payload["phase"] == "init"
    assert "idle_seconds" in payload
    dumps = list(dump_dir.glob("stall_dump_*.log"))
    assert dumps
