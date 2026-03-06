"""Per-track behaviour duration timer and state machine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.utils.time_utils import monotonic_seconds, format_duration, now_iso

logger = logging.getLogger(__name__)


@dataclass
class BehaviorSegment:
    """One continuous segment of a specific behaviour."""
    track_id: int
    behavior: str
    start_time: float            # monotonic seconds
    end_time: Optional[float] = None
    start_iso: str = ""
    end_iso: str = ""

    @property
    def duration(self) -> float:
        end = self.end_time if self.end_time is not None else monotonic_seconds()
        return max(0.0, end - self.start_time)

    def finalize(self) -> None:
        if self.end_time is None:
            self.end_time = monotonic_seconds()
            self.end_iso = now_iso()

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "behavior": self.behavior,
            "start_iso": self.start_iso,
            "end_iso": self.end_iso or now_iso(),
            "duration_s": round(self.duration, 2),
        }


@dataclass
class TrackState:
    """Full timing state for one tracked person."""
    track_id: int
    current_behavior: str = "unknown"
    current_segment: Optional[BehaviorSegment] = None
    cumulative: Dict[str, float] = field(default_factory=dict)
    segments: List[BehaviorSegment] = field(default_factory=list)
    last_seen: float = 0.0

    @property
    def current_duration(self) -> float:
        if self.current_segment is None:
            return 0.0
        return self.current_segment.duration

    @property
    def current_duration_str(self) -> str:
        return format_duration(self.current_duration)


class BehaviorTimer:
    """Manages per-track behavior durations and transitions.

    Call :meth:`update` every frame for each visible track.
    Call :meth:`finalize_track` when a track disappears.
    """

    def __init__(self) -> None:
        self._states: Dict[int, TrackState] = {}
        self._all_segments: List[BehaviorSegment] = []

    # ── public API ──────────────────────────────────────────────────

    def update(self, track_id: int, behavior: str) -> TrackState:
        """Update the timer for *track_id* with the current *behavior*.

        Creates a new segment on behaviour transitions.
        """
        now = monotonic_seconds()
        state = self._states.get(track_id)

        if state is None:
            # New track
            seg = BehaviorSegment(
                track_id=track_id,
                behavior=behavior,
                start_time=now,
                start_iso=now_iso(),
            )
            state = TrackState(
                track_id=track_id,
                current_behavior=behavior,
                current_segment=seg,
                last_seen=now,
            )
            state.segments.append(seg)
            self._states[track_id] = state
            return state

        state.last_seen = now

        if behavior != state.current_behavior:
            # Transition – close old segment
            self._close_segment(state, now)
            # Open new segment
            seg = BehaviorSegment(
                track_id=track_id,
                behavior=behavior,
                start_time=now,
                start_iso=now_iso(),
            )
            state.current_behavior = behavior
            state.current_segment = seg
            state.segments.append(seg)

        return state

    def finalize_track(self, track_id: int) -> Optional[TrackState]:
        """Close all open segments for a disappeared track."""
        state = self._states.pop(track_id, None)
        if state is None:
            return None
        now = monotonic_seconds()
        self._close_segment(state, now)
        self._all_segments.extend(state.segments)
        return state

    def finalize_all(self) -> List[TrackState]:
        """Finalize every active track (call at shutdown)."""
        states = []
        for tid in list(self._states.keys()):
            s = self.finalize_track(tid)
            if s:
                states.append(s)
        return states

    def get_state(self, track_id: int) -> Optional[TrackState]:
        return self._states.get(track_id)

    @property
    def all_segments(self) -> List[BehaviorSegment]:
        """Return all finalized + currently active segments."""
        active = []
        for s in self._states.values():
            active.extend(s.segments)
        return self._all_segments + active

    @property
    def active_track_ids(self) -> List[int]:
        return list(self._states.keys())

    # ── internals ───────────────────────────────────────────────────

    @staticmethod
    def _close_segment(state: TrackState, now: float) -> None:
        seg = state.current_segment
        if seg is not None and seg.end_time is None:
            seg.end_time = now
            seg.end_iso = now_iso()
            # accumulate
            state.cumulative[seg.behavior] = (
                state.cumulative.get(seg.behavior, 0.0) + seg.duration
            )
