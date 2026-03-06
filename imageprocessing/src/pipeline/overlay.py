"""Real-time video overlay – draws all detections, skeletons, and stats."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.pipeline.behavior import BehaviorResult
from src.pipeline.timer import TrackState
from src.pipeline.tracker import Track
from src.utils.draw import color_for_id, draw_bbox, draw_skeleton, draw_text_block
from src.utils.time_utils import format_duration


class Overlay:
    """Composites all visual information onto the video frame.

    Draws:
    * Bounding box + track ID + behaviour label + duration
    * Pose skeleton
    * Global stats (FPS, person count)
    """

    def __init__(self, pose_conf: float = 0.3) -> None:
        self._pose_conf = pose_conf

    def draw(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        keypoints_map: Optional[Dict[int, np.ndarray]] = None,
        behavior_map: Optional[Dict[int, BehaviorResult]] = None,
        timer_states: Optional[Dict[int, TrackState]] = None,
        fps: float = 0.0,
        count_current: int = 0,
        count_total: int = 0,
    ) -> np.ndarray:
        """Draw overlays on *frame* in-place and return it."""
        keypoints_map = keypoints_map or {}
        behavior_map = behavior_map or {}
        timer_states = timer_states or {}

        for track in tracks:
            tid = track.track_id
            color = color_for_id(tid)
            x1, y1, x2, y2 = track.bbox.astype(int)

            # ── label text ──────────────────────────────────────────
            parts = [f"ID:{tid}"]
            beh = behavior_map.get(tid)
            if beh is not None:
                parts.append(beh.label)
            ts = timer_states.get(tid)
            if ts is not None:
                parts.append(ts.current_duration_str)
            label = " | ".join(parts)

            draw_bbox(frame, x1, y1, x2, y2, color=color, label=label)

            # ── skeleton ────────────────────────────────────────────
            kps = keypoints_map.get(tid)
            if kps is not None:
                draw_skeleton(frame, kps, conf_threshold=self._pose_conf, color=color)

        # ── global stats ────────────────────────────────────────────
        stats_lines = [
            f"FPS: {fps:.1f}",
            f"People: {count_current}  (total: {count_total})",
        ]
        draw_text_block(frame, stats_lines, origin=(10, 30))

        return frame
