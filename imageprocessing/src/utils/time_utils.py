"""Time-related helpers – timestamps and formatting."""

from __future__ import annotations

import time
from datetime import datetime, timezone


def now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def now_stamp() -> str:
    """Filesystem-safe timestamp string, e.g. ``20260220_153045``."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """Human-readable ``mm:ss`` string from seconds."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def monotonic_seconds() -> float:
    """High-resolution monotonic clock (seconds)."""
    return time.monotonic()
