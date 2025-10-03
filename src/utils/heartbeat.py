"""简单的状态心跳与卡死检测工具。"""

from __future__ import annotations

import faulthandler
import json
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Optional


@dataclass(frozen=True)
class _ProgressSample:
    timestamp: float
    update_idx: Optional[int]
    global_step: Optional[int]


class HeartbeatReporter:
    """后台线程周期性打印训练/评估心跳，并对长时间无进展进行告警。"""

    def __init__(
        self,
        component: str,
        interval: float = 30.0,
        stall_timeout: float = 180.0,
        print_fn: Optional[Callable[[str], None]] = None,
        enabled: bool = True,
        log_path: Optional[Path] = None,
    ) -> None:
        self._component = component
        self._interval = max(interval, 1.0)
        self._stall_timeout = max(stall_timeout, self._interval * 2)
        self._print = print_fn or print
        self._enabled = enabled

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._log_lock = threading.Lock()
        self._log_path: Optional[Path] = None
        if log_path is not None:
            self.set_log_path(log_path)
        self._stall_dump_dir: Optional[Path] = None
        self._stall_dump_cooldown = max(stall_timeout, interval * 4)
        self._last_stall_dump = 0.0

        self._history: Deque[_ProgressSample] = deque(maxlen=8)
        now = time.time()
        self._last_progress_ts = now
        self._last_phase = "init"
        self._last_message = ""
        self._last_update: Optional[int] = None
        self._last_step: Optional[int] = None

    def start(self) -> None:
        if not self._enabled:
            return
        if self._thread is not None:
            return
        self._stop_event.clear()
        thread = threading.Thread(
            target=self._run, name=f"{self._component}-heartbeat", daemon=True
        )
        self._thread = thread
        thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        if not self._enabled:
            return
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        self._thread = None

    def notify(
        self,
        *,
        update_idx: Optional[int] = None,
        global_step: Optional[int] = None,
        phase: Optional[str] = None,
        message: Optional[str] = None,
        progress: bool = True,
    ) -> None:
        if not self._enabled:
            return
        now = time.time()
        with self._lock:
            if update_idx is not None:
                self._last_update = int(update_idx)
            if global_step is not None:
                self._last_step = int(global_step)
            if phase:
                self._last_phase = phase
            if message:
                self._last_message = message
            if progress and (update_idx is not None or global_step is not None):
                self._last_progress_ts = now
                self._history.append(
                    _ProgressSample(
                        timestamp=now,
                        update_idx=self._last_update,
                        global_step=self._last_step,
                    )
                )

    def heartbeat_now(self) -> None:
        if not self._enabled:
            return
        self._emit(force=True)

    def set_log_path(self, path: Path) -> None:
        with self._log_lock:
            self._log_path = path
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    def set_stall_dump_dir(
        self, path: Path, cooldown: Optional[float] = None
    ) -> None:
        with self._log_lock:
            self._stall_dump_dir = path
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        if cooldown is not None and cooldown > 0:
            self._stall_dump_cooldown = max(cooldown, 1.0)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            self._emit(force=False)

    def _emit(self, *, force: bool) -> None:
        now = time.time()
        with self._lock:
            history = list(self._history)
            last_phase = self._last_phase
            last_message = self._last_message
            last_update = self._last_update
            last_step = self._last_step
            last_progress_ts = self._last_progress_ts

        since_progress = now - last_progress_ts
        since_progress_text = f"{since_progress:.1f}s"

        update_rate = 0.0
        step_rate = 0.0
        if len(history) >= 2:
            sample_start = history[0]
            sample_end = history[-1]
            dt = max(sample_end.timestamp - sample_start.timestamp, 1e-6)
            if (
                sample_start.update_idx is not None
                and sample_end.update_idx is not None
            ):
                update_rate = (sample_end.update_idx - sample_start.update_idx) / dt
            if (
                sample_start.global_step is not None
                and sample_end.global_step is not None
            ):
                step_rate = (sample_end.global_step - sample_start.global_step) / dt

        status_bits = [f"phase={last_phase}"]
        if last_update is not None:
            status_bits.append(f"update={last_update}")
        if last_step is not None:
            status_bits.append(f"step={last_step}")
        status_bits.append(f"updates/s={update_rate:.2f}")
        status_bits.append(f"steps/s={step_rate:.1f}")
        status_bits.append(f"idle={since_progress_text}")
        if last_message:
            status_bits.append(last_message)

        if force or history:
            self._print(f"[{self._component}][hb] " + " ".join(status_bits))

        self._append_log(
            {
                "timestamp": now,
                "component": self._component,
                "phase": last_phase,
                "update": last_update,
                "global_step": last_step,
                "updates_per_sec": update_rate,
                "steps_per_sec": step_rate,
                "idle_seconds": since_progress,
                "message": last_message,
            }
        )

        if since_progress >= self._stall_timeout:
            self._dump_stall_trace(now)
            self._print(
                f"[{self._component}][warn] 已连续 {since_progress_text} 无训练/评估进度，请检查环境是否卡死或线程阻塞。"
            )

    def _append_log(self, payload: dict) -> None:
        if not self._enabled:
            return
        path = None
        with self._log_lock:
            path = self._log_path
        if path is None:
            return
        try:
            with self._log_lock:
                if self._log_path is None:
                    return
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _dump_stall_trace(self, timestamp: float) -> None:
        path = None
        with self._log_lock:
            path = self._stall_dump_dir
        if path is None:
            return
        if timestamp - self._last_stall_dump < self._stall_dump_cooldown:
            return
        self._last_stall_dump = timestamp
        dump_path = path / f"stall_dump_{int(timestamp)}.log"
        try:
            path.mkdir(parents=True, exist_ok=True)
            with dump_path.open("w", encoding="utf-8") as fp:
                fp.write(
                    f"# Stall dump generated at {time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(timestamp))}Z\n"
                )
                try:
                    faulthandler.dump_traceback(file=fp)
                except Exception:
                    pass
                fp.write("\n# Thread stacks\n")
                for tid, frame in sys._current_frames().items():
                    fp.write(f"\n# Thread {tid}\n")
                    fp.write("".join(traceback.format_stack(frame)))
        except Exception:
            pass
