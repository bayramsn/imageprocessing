"""Optional: YOLOv8-seg person segmentation with mask overlay."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from src.pipeline.tracker import Track
from src.utils.draw import draw_mask_overlay

logger = logging.getLogger(__name__)


class PersonSegmentor:
    """Runs YOLOv8-seg and associates person masks with tracked bboxes."""

    def __init__(
        self,
        model_path: str = "yolov8n-seg.pt",
        confidence: float = 0.45,
        imgsz: int = 640,
        person_class_id: int = 0,
        device: Optional[str] = None,
    ) -> None:
        logger.info("Loading segmentation model: %s", model_path)
        self._model = YOLO(model_path)
        self._confidence = confidence
        self._imgsz = imgsz
        self._person_cls = person_class_id
        self._device = device or ""

    def segment(
        self,
        frame: np.ndarray,
        tracks: List[Track],
    ) -> Dict[int, np.ndarray]:
        """Return ``{track_id: binary_mask}`` for each matched track.

        Masks are at the same resolution as *frame*.
        """
        if not tracks:
            return {}

        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            imgsz=self._imgsz,
            classes=[self._person_cls],
            verbose=False,
            device=self._device if self._device else None,
        )

        seg_bboxes: List[np.ndarray] = []
        seg_masks: List[np.ndarray] = []
        h, w = frame.shape[:2]

        for r in results:
            if r.boxes is None or r.masks is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            masks = r.masks.data.cpu().numpy()  # (N, mask_h, mask_w)
            for i in range(len(boxes)):
                seg_bboxes.append(boxes[i])
                # Resize mask to frame resolution
                m = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_NEAREST)
                seg_masks.append((m > 0.5).astype(np.uint8))

        # Match to tracks by IoU
        result: Dict[int, np.ndarray] = {}
        for track in tracks:
            best_iou = 0.0
            best_idx = -1
            for i, sb in enumerate(seg_bboxes):
                iou = _iou(track.bbox, sb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0 and best_iou > 0.3:
                result[track.track_id] = seg_masks[best_idx]

        return result

    def draw_masks(
        self,
        frame: np.ndarray,
        masks: Dict[int, np.ndarray],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Draw all person masks onto frame."""
        from src.utils.draw import color_for_id
        for tid, mask in masks.items():
            draw_mask_overlay(frame, mask, color=color_for_id(tid), alpha=alpha)
        return frame


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-8)
