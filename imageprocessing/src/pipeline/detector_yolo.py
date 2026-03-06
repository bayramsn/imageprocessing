"""YOLOv8 person detector – wraps Ultralytics inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single person detection."""
    bbox: np.ndarray          # (x1, y1, x2, y2) absolute pixels
    confidence: float
    class_id: int = 0         # always 0 (person)

    @property
    def xyxy(self) -> np.ndarray:
        return self.bbox

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class YOLODetector:
    """Wraps Ultralytics YOLOv8 for *person-only* detection.

    Usage::

        det = YOLODetector("yolov8n.pt")
        results = det.detect(frame)
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.45,
        iou: float = 0.5,
        imgsz: int = 640,
        person_class_id: int = 0,
        max_detections: int = 50,
        device: Optional[str] = None,
    ) -> None:
        self._confidence = confidence
        self._iou = iou
        self._imgsz = imgsz
        self._person_cls = person_class_id
        self._max_detections = max_detections
        self._device = device or ""
        logger.info("Loading detection model: %s", model_path)
        self._model = YOLO(model_path)

    # ── public API ──────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference and return person-only detections."""
        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            iou=self._iou,
            imgsz=self._imgsz,
            classes=[self._person_cls],
            max_det=self._max_detections,
            verbose=False,
            device=self._device if self._device else None,
        )
        detections: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(float)
                conf = float(box.conf[0].cpu().numpy())
                detections.append(Detection(bbox=xyxy, confidence=conf))
        return detections
