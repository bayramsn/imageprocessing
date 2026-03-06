"""Unit tests for behaviour classification heuristics."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline.behavior import (
    LABEL_RUNNING,
    LABEL_SITTING,
    LABEL_STANDING,
    LABEL_UNKNOWN,
    LABEL_WALKING,
    BehaviorClassifier,
)

# ── helpers ─────────────────────────────────────────────────────────

def _make_keypoints(**overrides: tuple[float, float, float]) -> np.ndarray:
    """Build a (17, 3) keypoint array with sensible defaults.

    All keypoints default to (100, 100, 0.9).  Pass index names as keyword
    args to override, e.g. ``_make_keypoints(**{11: (100, 80, 0.9)})``.
    For convenience we accept int keys via ``overrides`` dict built outside.
    """
    kps = np.full((17, 3), [100.0, 100.0, 0.9])
    for idx, val in overrides.items():
        kps[idx] = val
    return kps


def make_standing_keypoints() -> np.ndarray:
    """Person standing upright – knees almost straight (≈170°)."""
    kps = np.full((17, 3), 0.9)  # conf
    # Shoulders
    kps[5] = [90, 100, 0.9]    # left shoulder
    kps[6] = [110, 100, 0.9]   # right shoulder
    # Hips
    kps[11] = [90, 200, 0.9]   # left hip
    kps[12] = [110, 200, 0.9]  # right hip
    # Knees – almost directly below hips
    kps[13] = [90, 300, 0.9]   # left knee
    kps[14] = [110, 300, 0.9]  # right knee
    # Ankles – directly below knees
    kps[15] = [90, 400, 0.9]   # left ankle
    kps[16] = [110, 400, 0.9]  # right ankle
    return kps


def make_sitting_keypoints() -> np.ndarray:
    """Person sitting – knees bent at ≈90°."""
    kps = make_standing_keypoints().copy()
    # Bend knees forward (horizontal thigh)
    kps[13] = [150, 200, 0.9]   # left knee far forward at hip height
    kps[14] = [170, 200, 0.9]   # right knee far forward
    # Ankles below the forward knees
    kps[15] = [150, 300, 0.9]
    kps[16] = [170, 300, 0.9]
    return kps


# ═══════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════


class TestBehaviorClassifier:
    """Test suite for the rule-based BehaviorClassifier."""

    def setup_method(self) -> None:
        self.clf = BehaviorClassifier(
            sit_knee_angle=120.0,
            run_speed_threshold=60.0,
            walk_speed_threshold=15.0,
        )

    # ── standing ────────────────────────────────────────────────────

    def test_standing_posture(self) -> None:
        kps = make_standing_keypoints()
        result = self.clf.classify(track_id=1, keypoints=kps, dt=0.033)
        assert result.label == LABEL_STANDING

    # ── sitting ─────────────────────────────────────────────────────

    def test_sitting_posture(self) -> None:
        kps = make_sitting_keypoints()
        result = self.clf.classify(track_id=2, keypoints=kps, dt=0.033)
        assert result.label == LABEL_SITTING

    # ── walking ─────────────────────────────────────────────────────

    def test_walking_detected_via_ankle_speed(self) -> None:
        kps1 = make_standing_keypoints()
        kps2 = make_standing_keypoints().copy()
        # Move ankles by ~1.5 px/frame → at 30 FPS = 45 px/s  (above walk threshold)
        kps2[15][0] += 1.5
        kps2[16][0] += 1.5

        # First call stores previous ankles
        self.clf.classify(track_id=3, keypoints=kps1, dt=0.033)
        result = self.clf.classify(track_id=3, keypoints=kps2, dt=0.033)
        assert result.label == LABEL_WALKING

    # ── running ─────────────────────────────────────────────────────

    def test_running_detected_via_high_ankle_speed(self) -> None:
        kps1 = make_standing_keypoints()
        kps2 = make_standing_keypoints().copy()
        # Move ankles by ~6 px/frame → at 30 FPS = 180 px/s  (above run threshold)
        kps2[15][0] += 6.0
        kps2[16][0] += 6.0

        self.clf.classify(track_id=4, keypoints=kps1, dt=0.033)
        result = self.clf.classify(track_id=4, keypoints=kps2, dt=0.033)
        assert result.label == LABEL_RUNNING

    # ── unknown (missing keypoints) ─────────────────────────────────

    def test_unknown_when_no_keypoints(self) -> None:
        result = self.clf.classify(track_id=5, keypoints=None, dt=0.033)
        assert result.label == LABEL_UNKNOWN

    def test_unknown_when_low_confidence_keypoints(self) -> None:
        kps = make_standing_keypoints()
        kps[:, 2] = 0.05  # all confidences below threshold
        result = self.clf.classify(track_id=6, keypoints=kps, dt=0.033)
        assert result.label == LABEL_UNKNOWN

    # ── edge cases ──────────────────────────────────────────────────

    def test_zero_dt_does_not_crash(self) -> None:
        kps = make_standing_keypoints()
        result = self.clf.classify(track_id=7, keypoints=kps, dt=0.0)
        # Should still classify – just no walk/run from velocity
        assert result.label in {LABEL_STANDING, LABEL_SITTING, LABEL_UNKNOWN}

    def test_remove_track_cleans_state(self) -> None:
        kps = make_standing_keypoints()
        self.clf.classify(track_id=8, keypoints=kps, dt=0.033)
        self.clf.remove_track(8)
        # Re-classify same ID – should act like new track
        result = self.clf.classify(track_id=8, keypoints=kps, dt=0.033)
        assert result.label == LABEL_STANDING

    def test_score_in_valid_range(self) -> None:
        kps = make_standing_keypoints()
        result = self.clf.classify(track_id=9, keypoints=kps, dt=0.033)
        assert 0.0 <= result.score <= 1.0
