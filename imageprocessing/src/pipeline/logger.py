"""Session logger – writes CSV event log and JSON session summary."""

from __future__ import annotations

import atexit
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline.timer import BehaviorSegment, BehaviorTimer
from src.utils.time_utils import now_iso, now_stamp

logger = logging.getLogger(__name__)


class SessionLogger:
    """Writes logs to ``outputs/logs/`` with timestamped filenames.

    Two log files are produced:

    * **CSV** – one row per behaviour segment (track_id, behaviour,
      start, end, duration).
    * **JSON** – full session summary flushed at shutdown.

    Call :meth:`close` (or rely on atexit) to flush remaining data.
    """

    def __init__(self, output_dir: str = "outputs/logs") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        stamp = now_stamp()
        self._csv_path = self._output_dir / f"events_{stamp}.csv"
        self._json_path = self._output_dir / f"session_{stamp}.json"

        # Open CSV
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["track_id", "behavior", "start_time", "end_time", "duration_s"])

        self._session_start = now_iso()
        self._event_count = 0

        # Ensure we flush on exit
        atexit.register(self.close)
        logger.info("Logging to %s  /  %s", self._csv_path, self._json_path)

    # ── public API ──────────────────────────────────────────────────

    def log_segment(self, seg: BehaviorSegment) -> None:
        """Write a single finalized segment to the CSV."""
        self._csv_writer.writerow([
            seg.track_id,
            seg.behavior,
            seg.start_iso,
            seg.end_iso,
            round(seg.duration, 2),
        ])
        self._csv_file.flush()
        self._event_count += 1

    def log_segments(self, segments: List[BehaviorSegment]) -> None:
        for s in segments:
            self.log_segment(s)

    def write_json_summary(
        self,
        timer: BehaviorTimer,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write full session JSON (all segments + metadata)."""
        data: Dict[str, Any] = {
            "session_start": self._session_start,
            "session_end": now_iso(),
            "total_events": self._event_count,
            "segments": [s.to_dict() for s in timer.all_segments],
        }
        if extra:
            data.update(extra)
        with open(self._json_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("JSON session written → %s", self._json_path)

    def close(self) -> None:
        """Flush and close the CSV file."""
        if not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()
            logger.info("CSV log closed → %s (%d events)", self._csv_path, self._event_count)
