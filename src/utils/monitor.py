"""Lightweight background resource monitor for training.

This module provides a simple thread-based monitor that periodically
collects GPU and system stats and writes them to a TensorBoard writer,
optionally to wandb, and appends to a metrics JSONL file.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import torch


def _get_resource_stats_once():
    stats = {}
    try:
        if torch.cuda.is_available():
            stats["gpu_count"] = torch.cuda.device_count()
            for gi in range(torch.cuda.device_count()):
                stats[f"gpu_{gi}_mem_alloc_bytes"] = float(
                    torch.cuda.memory_allocated(gi)
                )
                stats[f"gpu_{gi}_mem_reserved_bytes"] = float(
                    torch.cuda.memory_reserved(gi)
                )
    except Exception:
        pass

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if res.returncode == 0 and res.stdout.strip():
            lines = [line_var for line_var in res.stdout.strip().splitlines() if line_var.strip()]
            for idx, line in enumerate(lines):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    util, mem_used, mem_total = parts[:3]
                    stats[f"gpu_{idx}_util_pct"] = float(util)
                    stats[f"gpu_{idx}_mem_used_mb"] = float(mem_used)
                    stats[f"gpu_{idx}_mem_total_mb"] = float(mem_total)
    except Exception:
        pass

    try:
        import psutil

        p = psutil.Process()
        stats["proc_cpu_percent"] = float(p.cpu_percent(interval=None))
        stats["proc_mem_rss_bytes"] = float(p.memory_info().rss)
        stats["system_cpu_percent"] = float(psutil.cpu_percent(interval=None))
        stats["system_mem_percent"] = float(psutil.virtual_memory().percent)
    except Exception:
        try:
            load1, load5, load15 = os.getloadavg()
            stats["load1"] = float(load1)
        except Exception:
            pass

    return stats


class Monitor:
    def __init__(
        self,
        writer,
        wandb_run,
        metrics_path: Path,
        interval: float = 10.0,
        metrics_lock: Optional[threading.Lock] = None,
    ):
        self.writer = writer
        self.wandb_run = wandb_run
        self.metrics_path = metrics_path
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.metrics_lock = metrics_lock

    def _loop(self):
        while not self._stop_event.is_set():
            ts = time.time()
            stats = _get_resource_stats_once()
            entry = {"timestamp": ts, "monitor": stats}
            try:
                # write to tensorboard
                for k, v in stats.items():
                    try:
                        self.writer.add_scalar(f"Resource/{k}", float(v), int(ts))
                    except Exception:
                        pass
                # wandb
                if self.wandb_run is not None and stats:
                    try:
                        self.wandb_run.log({**stats, "timestamp": ts})
                    except Exception:
                        pass
                # append to metrics file
                try:
                    if self.metrics_lock is not None:
                        with self.metrics_lock:
                            with self.metrics_path.open("a", encoding="utf-8") as fp:
                                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    else:
                        with self.metrics_path.open("a", encoding="utf-8") as fp:
                            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            except Exception:
                pass
            # sleep with early exit
            for _ in range(int(self.interval * 10)):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


__all__ = ["Monitor", "_get_resource_stats_once"]
