"""FPS tracker – running average with configurable window."""

from __future__ import annotations

import time
from collections import deque
from typing import Optional


class FPSTracker:
    """Lightweight real-time FPS counter.

    Keeps a sliding window of frame timestamps and reports the average
    frames-per-second over that window.
    """

    def __init__(self, window: int = 30) -> None:
        """
        Args:
            window: Number of recent frames to average over.
        """
        self._window = window
        self._timestamps: deque[float] = deque(maxlen=window)
        self._fps: float = 0.0

    # ── public API ──────────────────────────────────────────────────

    def tick(self) -> None:
        """Record a new frame timestamp."""
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self._fps = (len(self._timestamps) - 1) / elapsed

    @property
    def fps(self) -> float:
        """Current smoothed FPS."""
        return self._fps

    def reset(self) -> None:
        self._timestamps.clear()
        self._fps = 0.0

    def __repr__(self) -> str:
        return f"FPSTracker(fps={self._fps:.1f})"
