"""Pose estimation – top-down with YOLOv8-pose."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from src.pipeline.tracker import Track

logger = logging.getLogger(__name__)

# COCO 17-keypoint indices
KP_NOSE = 0
KP_LEFT_EYE, KP_RIGHT_EYE = 1, 2
KP_LEFT_EAR, KP_RIGHT_EAR = 3, 4
KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER = 5, 6
KP_LEFT_ELBOW, KP_RIGHT_ELBOW = 7, 8
KP_LEFT_WRIST, KP_RIGHT_WRIST = 9, 10
KP_LEFT_HIP, KP_RIGHT_HIP = 11, 12
KP_LEFT_KNEE, KP_RIGHT_KNEE = 13, 14
KP_LEFT_ANKLE, KP_RIGHT_ANKLE = 15, 16


class PoseEstimator:
    """Run YOLOv8-pose on each tracked person bounding box (top-down).

    For each person crop we run the pose model and return COCO-17
    keypoints mapped back to the original frame coordinates.
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        confidence: float = 0.3,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        logger.info("Loading pose model: %s", model_path)
        self._model = YOLO(model_path)
        self._confidence = confidence
        self._imgsz = imgsz
        self._device = device or ""

    # ── public API ──────────────────────────────────────────────────

    def estimate(
        self,
        frame: np.ndarray,
        tracks: List[Track],
    ) -> Dict[int, np.ndarray]:
        """Run pose estimation on the full frame and match to tracks.

        Returns:
            Mapping ``{track_id: keypoints}`` where *keypoints* has shape
            ``(17, 3)`` – (x, y, confidence).
        """
        if not tracks:
            return {}

        # Run pose on the full frame – the model detects persons internally
        results = self._model.predict(
            source=frame,
            conf=self._confidence,
            imgsz=self._imgsz,
            verbose=False,
            device=self._device if self._device else None,
        )

        # Collect all pose detections with their bboxes
        pose_bboxes: List[np.ndarray] = []
        pose_keypoints: List[np.ndarray] = []
        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue
            kps = r.keypoints.data.cpu().numpy()   # (N, 17, 3)
            boxes = r.boxes.xyxy.cpu().numpy()      # (N, 4)
            for i in range(len(kps)):
                pose_bboxes.append(boxes[i])
                pose_keypoints.append(kps[i])

        # Match pose detections to tracks by IoU
        result: Dict[int, np.ndarray] = {}
        for track in tracks:
            best_iou = 0.0
            best_idx = -1
            for i, pb in enumerate(pose_bboxes):
                iou = _iou(track.bbox, pb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0 and best_iou > 0.3:
                result[track.track_id] = pose_keypoints[best_idx]

        return result


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / (union + 1e-8)


def get_keypoint(
    keypoints: np.ndarray,
    idx: int,
    conf_threshold: float = 0.3,
) -> Optional[Tuple[float, float]]:
    """Extract a single keypoint as ``(x, y)`` or ``None`` if below threshold."""
    if keypoints is None or idx >= len(keypoints):
        return None
    x, y, c = keypoints[idx]
    if c < conf_threshold:
        return None
    return (float(x), float(y))
