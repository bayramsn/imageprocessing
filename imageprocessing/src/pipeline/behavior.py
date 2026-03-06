"""Rule-based behavior / posture classifier from COCO-17 keypoints."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.pipeline.pose import (
    KP_LEFT_ANKLE,
    KP_LEFT_HIP,
    KP_LEFT_KNEE,
    KP_LEFT_SHOULDER,
    KP_RIGHT_ANKLE,
    KP_RIGHT_HIP,
    KP_RIGHT_KNEE,
    KP_RIGHT_SHOULDER,
    get_keypoint,
)
from src.utils.geometry import angle_between_points, keypoint_velocity

logger = logging.getLogger(__name__)

# ── Behavior labels ─────────────────────────────────────────────────
LABEL_STANDING = "standing"
LABEL_SITTING = "sitting"
LABEL_WALKING = "walking"
LABEL_RUNNING = "running"
LABEL_UNKNOWN = "unknown"


@dataclass
class BehaviorResult:
    """Output of the classifier for one person."""
    label: str
    score: float          # 0..1 confidence-like heuristic score
    details: str = ""     # human-readable reasoning


class BehaviorClassifier:
    """Heuristic posture/action classifier.

    Rules (evaluated in order):
    1. If either knee angle < *sit_knee_angle* → **sitting**
    2. Compute ankle speed from previous frame:
       - > *run_speed* → **running**
       - > *walk_speed* → **walking**
    3. Otherwise → **standing**

    Falls back to *unknown* when keypoints are missing.
    """

    def __init__(
        self,
        sit_knee_angle: float = 120.0,
        run_speed_threshold: float = 60.0,
        walk_speed_threshold: float = 15.0,
        kp_conf: float = 0.3,
    ) -> None:
        self.sit_knee_angle = sit_knee_angle
        self.run_speed = run_speed_threshold
        self.walk_speed = walk_speed_threshold
        self.kp_conf = kp_conf

        # Previous-frame ankle positions per track_id
        self._prev_ankles: Dict[int, Tuple[Optional[Tuple[float, float]],
                                            Optional[Tuple[float, float]]]] = {}

    # ── public API ──────────────────────────────────────────────────

    def classify(
        self,
        track_id: int,
        keypoints: Optional[np.ndarray],
        dt: float,
    ) -> BehaviorResult:
        """Classify behaviour for one person.

        Args:
            track_id:  Unique track identifier.
            keypoints: Shape ``(17, 3)`` COCO keypoints or *None*.
            dt:        Seconds since last frame (for velocity).

        Returns:
            BehaviorResult with label and score.
        """
        if keypoints is None:
            return BehaviorResult(LABEL_UNKNOWN, 0.0, "no keypoints")

        # ── Extract key joints ──────────────────────────────────────
        l_hip = get_keypoint(keypoints, KP_LEFT_HIP, self.kp_conf)
        r_hip = get_keypoint(keypoints, KP_RIGHT_HIP, self.kp_conf)
        l_knee = get_keypoint(keypoints, KP_LEFT_KNEE, self.kp_conf)
        r_knee = get_keypoint(keypoints, KP_RIGHT_KNEE, self.kp_conf)
        l_ankle = get_keypoint(keypoints, KP_LEFT_ANKLE, self.kp_conf)
        r_ankle = get_keypoint(keypoints, KP_RIGHT_ANKLE, self.kp_conf)
        l_shoulder = get_keypoint(keypoints, KP_LEFT_SHOULDER, self.kp_conf)
        r_shoulder = get_keypoint(keypoints, KP_RIGHT_SHOULDER, self.kp_conf)

        # Need minimum keypoints for classification
        if not _at_least_one(l_hip, r_hip) or not _at_least_one(l_knee, r_knee):
            return BehaviorResult(LABEL_UNKNOWN, 0.0, "insufficient keypoints")

        # ── 1) Sitting check (knee angle) ──────────────────────────
        knee_angles = []
        if l_hip and l_knee and l_ankle:
            knee_angles.append(angle_between_points(l_hip, l_knee, l_ankle))
        if r_hip and r_knee and r_ankle:
            knee_angles.append(angle_between_points(r_hip, r_knee, r_ankle))

        if knee_angles and min(knee_angles) < self.sit_knee_angle:
            score = 1.0 - (min(knee_angles) / 180.0)
            return BehaviorResult(
                LABEL_SITTING, round(score, 2),
                f"knee_angle={min(knee_angles):.0f}°",
            )

        # ── 2) Walking / Running (ankle velocity) ──────────────────
        prev = self._prev_ankles.get(track_id, (None, None))
        prev_l_ankle, prev_r_ankle = prev

        # Update stored ankles
        self._prev_ankles[track_id] = (l_ankle, r_ankle)

        speed = 0.0
        count = 0
        if l_ankle and prev_l_ankle and dt > 0:
            speed += keypoint_velocity(prev_l_ankle, l_ankle, dt)
            count += 1
        if r_ankle and prev_r_ankle and dt > 0:
            speed += keypoint_velocity(prev_r_ankle, r_ankle, dt)
            count += 1
        if count:
            speed /= count

        if speed > self.run_speed:
            return BehaviorResult(LABEL_RUNNING, min(speed / (self.run_speed * 2), 1.0),
                                  f"speed={speed:.1f}px/s")
        if speed > self.walk_speed:
            return BehaviorResult(LABEL_WALKING, min(speed / (self.run_speed), 1.0),
                                  f"speed={speed:.1f}px/s")

        # ── 3) Default: standing ────────────────────────────────────
        return BehaviorResult(LABEL_STANDING, 0.8, "default posture")

    def remove_track(self, track_id: int) -> None:
        """Clean up state for a disappeared track."""
        self._prev_ankles.pop(track_id, None)


# ── helpers ─────────────────────────────────────────────────────────

def _at_least_one(*args: Optional[Tuple[float, float]]) -> bool:
    return any(a is not None for a in args)
