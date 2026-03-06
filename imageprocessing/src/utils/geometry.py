"""Geometry helpers – angles, distances, velocities for keypoints."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def angle_between_points(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
) -> float:
    """Return the angle (degrees) at vertex *b* formed by segments ba and bc.

    Args:
        a: First point (x, y).
        b: Vertex point (x, y).
        c: Third point (x, y).

    Returns:
        Angle in degrees [0, 180].
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def euclidean_distance(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> float:
    """Euclidean distance between two 2-D points."""
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def midpoint(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> Tuple[float, float]:
    """Midpoint of two 2-D points."""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def keypoint_velocity(
    prev: Optional[Tuple[float, float]],
    curr: Optional[Tuple[float, float]],
    dt: float,
) -> float:
    """Pixel velocity (px / s) between two keypoint positions.

    Returns 0.0 when either position is missing or *dt* ≤ 0.
    """
    if prev is None or curr is None or dt <= 0:
        return 0.0
    return euclidean_distance(prev, curr) / dt


def torso_inclination(
    shoulder_mid: Tuple[float, float],
    hip_mid: Tuple[float, float],
) -> float:
    """Angle (degrees) of the torso w.r.t. the vertical axis.

    0° = perfectly upright, 90° = horizontal.
    """
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    # Vertical direction is +y downward in image coords
    angle_rad = math.atan2(abs(dx), abs(dy) + 1e-8)
    return math.degrees(angle_rad)
