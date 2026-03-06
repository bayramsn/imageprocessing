"""Multi-object tracker – wraps Ultralytics built-in ByteTrack / BotSORT."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Single tracked person."""
    track_id: int
    bbox: np.ndarray          # (x1, y1, x2, y2)
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class Tracker:
    """Person tracker using Ultralytics YOLO built-in tracking.

    Ultralytics YOLO provides ``model.track()`` which internally uses
    ByteTrack or BotSORT.  We delegate to that so we get robust
    tracking out-of-the-box without external dependencies.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        tracker_type: str = "bytetrack",
        confidence: float = 0.45,
        iou: float = 0.5,
        imgsz: int = 640,
        person_class_id: int = 0,
        max_detections: int = 50,
        tracker_config: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self._confidence = confidence
        self._iou = iou
        self._imgsz = imgsz
        self._person_cls = person_class_id
        self._max_detections = max_detections
        self._device = device or ""
        self._tracker_type = tracker_type
        self._tracker_config = tracker_config

        logger.info("Loading tracking model: %s (tracker=%s)", model_path, tracker_type)
        self._model = YOLO(model_path)

        # Bookkeeping for people counting
        self._seen_ids: Set[int] = set()

    # ── public API ──────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> List[Track]:
        """Run detection + tracking on *frame* and return active tracks.

        Returns:
            List of Track objects for persons currently visible.
        """
        tracker_yaml = self._tracker_config or f"{self._tracker_type}.yaml"
        results = self._model.track(
            source=frame,
            conf=self._confidence,
            iou=self._iou,
            imgsz=self._imgsz,
            classes=[self._person_cls],
            tracker=tracker_yaml,
            max_det=self._max_detections,
            persist=True,
            verbose=False,
            device=self._device if self._device else None,
        )

        tracks: List[Track] = []
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
            ids = r.boxes.id.cpu().numpy().astype(int)
            bboxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for tid, bbox, conf in zip(ids, bboxes, confs):
                tid = int(tid)
                self._seen_ids.add(tid)
                tracks.append(Track(track_id=tid, bbox=bbox.astype(float), confidence=float(conf)))
        return tracks

    # ── counting ────────────────────────────────────────────────────

    @property
    def count_total(self) -> int:
        """Total unique track IDs ever observed."""
        return len(self._seen_ids)

    def count_current(self, tracks: List[Track]) -> int:
        """Current number of people visible."""
        return len(tracks)

    def reset(self) -> None:
        self._seen_ids.clear()
