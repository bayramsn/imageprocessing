"""Unit tests for the BehaviorTimer state machine."""

from __future__ import annotations

import time

import pytest

from src.pipeline.timer import BehaviorTimer


class TestBehaviorTimer:
    """Test suite for per-track duration tracking and transitions."""

    def setup_method(self) -> None:
        self.timer = BehaviorTimer()

    # ── basic tracking ──────────────────────────────────────────────

    def test_new_track_creates_state(self) -> None:
        state = self.timer.update(track_id=1, behavior="standing")
        assert state.track_id == 1
        assert state.current_behavior == "standing"
        assert state.current_segment is not None

    def test_same_behavior_does_not_create_new_segment(self) -> None:
        self.timer.update(1, "standing")
        state = self.timer.update(1, "standing")
        assert len(state.segments) == 1

    def test_behavior_transition_creates_new_segment(self) -> None:
        self.timer.update(1, "standing")
        time.sleep(0.01)  # small delay for duration > 0
        state = self.timer.update(1, "sitting")
        assert state.current_behavior == "sitting"
        assert len(state.segments) == 2
        # First segment should be finalized
        assert state.segments[0].end_time is not None

    def test_multiple_transitions(self) -> None:
        self.timer.update(1, "standing")
        self.timer.update(1, "walking")
        state = self.timer.update(1, "running")
        assert len(state.segments) == 3
        assert state.current_behavior == "running"

    # ── duration tracking ───────────────────────────────────────────

    def test_current_duration_positive(self) -> None:
        self.timer.update(1, "standing")
        time.sleep(0.02)
        state = self.timer.update(1, "standing")
        assert state.current_duration > 0

    def test_cumulative_time_after_transition(self) -> None:
        self.timer.update(1, "standing")
        time.sleep(0.02)
        state = self.timer.update(1, "sitting")
        assert "standing" in state.cumulative
        assert state.cumulative["standing"] > 0

    # ── finalization ────────────────────────────────────────────────

    def test_finalize_track_closes_segment(self) -> None:
        self.timer.update(1, "standing")
        state = self.timer.finalize_track(1)
        assert state is not None
        assert all(s.end_time is not None for s in state.segments)

    def test_finalize_track_removes_from_active(self) -> None:
        self.timer.update(1, "standing")
        self.timer.finalize_track(1)
        assert self.timer.get_state(1) is None

    def test_finalize_nonexistent_track(self) -> None:
        result = self.timer.finalize_track(999)
        assert result is None

    def test_finalize_all(self) -> None:
        self.timer.update(1, "standing")
        self.timer.update(2, "walking")
        states = self.timer.finalize_all()
        assert len(states) == 2
        assert self.timer.active_track_ids == []

    # ── segment data ────────────────────────────────────────────────

    def test_segment_to_dict(self) -> None:
        self.timer.update(1, "standing")
        state = self.timer.finalize_track(1)
        assert state is not None
        d = state.segments[0].to_dict()
        assert d["track_id"] == 1
        assert d["behavior"] == "standing"
        assert "duration_s" in d

    def test_all_segments_includes_active_and_finalized(self) -> None:
        self.timer.update(1, "standing")
        self.timer.update(2, "walking")
        self.timer.finalize_track(1)
        # Track 1 finalized, track 2 still active
        segs = self.timer.all_segments
        assert len(segs) >= 2

    # ── edge cases ──────────────────────────────────────────────────

    def test_rapid_transitions(self) -> None:
        for i, beh in enumerate(["standing", "sitting", "walking", "running", "standing"]):
            self.timer.update(1, beh)
        state = self.timer.get_state(1)
        assert state is not None
        assert len(state.segments) == 5

    def test_multiple_tracks_independent(self) -> None:
        self.timer.update(1, "standing")
        self.timer.update(2, "sitting")
        s1 = self.timer.get_state(1)
        s2 = self.timer.get_state(2)
        assert s1 is not None and s1.current_behavior == "standing"
        assert s2 is not None and s2.current_behavior == "sitting"

    def test_duration_format_string(self) -> None:
        self.timer.update(1, "standing")
        state = self.timer.get_state(1)
        assert state is not None
        # Should be "00:00" or similar
        assert ":" in state.current_duration_str
