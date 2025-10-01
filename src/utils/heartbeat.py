"""简单的状态心跳与卡死检测工具。"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple


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
    ) -> None:
        self._component = component
        self._interval = max(interval, 1.0)
        self._stall_timeout = max(stall_timeout, self._interval * 2)
        self._print = print_fn or print
        self._enabled = enabled

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

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
        thread = threading.Thread(target=self._run, name=f"{self._component}-heartbeat", daemon=True)
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
                    _ProgressSample(timestamp=now, update_idx=self._last_update, global_step=self._last_step)
                )

    def heartbeat_now(self) -> None:
        if not self._enabled:
            return
        self._emit(force=True)

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
            if sample_start.update_idx is not None and sample_end.update_idx is not None:
                update_rate = (sample_end.update_idx - sample_start.update_idx) / dt
            if sample_start.global_step is not None and sample_end.global_step is not None:
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

        if since_progress >= self._stall_timeout:
            self._print(
                f"[{self._component}][warn] 已连续 {since_progress_text} 无训练/评估进度，请检查环境是否卡死或线程阻塞。"
            )
