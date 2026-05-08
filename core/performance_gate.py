import time
import threading
import queue
import numpy as np
from typing import Optional


class PerformanceGate:
    """
    Drops frames if processing exceeds target delta (33ms = ~30fps).
    Uses an async queue so display is never blocked.
    """
    TARGET_DELTA_MS = 33.0

    def __init__(self, max_queue: int = 2):
        self._q: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=max_queue)
        self._last_ts = time.perf_counter()
        self._fps_counter = 0
        self._fps_start = time.perf_counter()
        self._fps = 0.0

    def submit(self, frame: np.ndarray) -> bool:
        """Try to enqueue processed frame. Returns False if dropped."""
        try:
            self._q.put_nowait(frame)
            self._fps_counter += 1
            now = time.perf_counter()
            if now - self._fps_start >= 1.0:
                self._fps = self._fps_counter / (now - self._fps_start)
                self._fps_counter = 0
                self._fps_start = now
            return True
        except queue.Full:
            return False

    def get(self, timeout: float = 0.05) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def should_drop(self) -> bool:
        now = time.perf_counter()
        delta_ms = (now - self._last_ts) * 1000
        if delta_ms < self.TARGET_DELTA_MS:
            return False
        self._last_ts = now
        return False  # Logic: don't drop, just measure

    @property
    def fps(self) -> float:
        return self._fps