"""Drawing primitives – kept separate so overlay.py stays high-level."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ── COCO 17-keypoint skeleton ─────────────────────────────────────
SKELETON_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                  # shoulders
    (5, 7), (7, 9),                          # left arm
    (6, 8), (8, 10),                         # right arm
    (5, 11), (6, 12),                        # torso
    (11, 12),                                # hips
    (11, 13), (13, 15),                      # left leg
    (12, 14), (14, 16),                      # right leg
]

# Color palette (BGR) for different track ids
_PALETTE = [
    (255, 76, 76), (76, 255, 76), (76, 76, 255),
    (255, 200, 76), (76, 255, 200), (200, 76, 255),
    (255, 128, 0), (0, 200, 255), (200, 0, 128),
    (128, 255, 0),
]


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Deterministic BGR colour for a track id."""
    return _PALETTE[track_id % len(_PALETTE)]


def draw_bbox(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: Optional[str] = None,
) -> None:
    """Draw a bounding box and optional label on *frame* in-place."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        font_scale = 0.55
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    conf_threshold: float = 0.3,
    color: Tuple[int, int, int] = (0, 255, 0),
    point_radius: int = 3,
    line_thickness: int = 2,
) -> None:
    """Draw COCO 17-keypoint skeleton on *frame*.

    Args:
        keypoints: shape (17, 3) – x, y, confidence per keypoint.
    """
    if keypoints is None or len(keypoints) == 0:
        return
    for idx, (x, y, c) in enumerate(keypoints):
        if c >= conf_threshold:
            cv2.circle(frame, (int(x), int(y)), point_radius, color, -1)
    for i, j in SKELETON_EDGES:
        if i < len(keypoints) and j < len(keypoints):
            ci, cj = keypoints[i][2], keypoints[j][2]
            if ci >= conf_threshold and cj >= conf_threshold:
                pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                cv2.line(frame, pt1, pt2, color, line_thickness, cv2.LINE_AA)


def draw_text_block(
    frame: np.ndarray,
    lines: Sequence[str],
    origin: Tuple[int, int] = (10, 30),
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    line_gap: int = 28,
) -> None:
    """Draw multiple lines of text starting from *origin*."""
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (x, y + i * line_gap),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA,
        )


def draw_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> None:
    """Overlay a binary *mask* on *frame* with transparency."""
    if mask is None:
        return
    overlay = frame.copy()
    overlay[mask > 0] = color
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
