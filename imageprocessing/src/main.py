"""CLI entry-point – orchestrates the full pipeline.

Run examples::

    # Webcam with all features
    python -m src.main --source 0 --enable_tracking --enable_pose --enable_logging --save_video

    # Video file, no overlay window
    python -m src.main --source video.mp4 --no_show --save_video

    # RTSP stream, stride=2
    python -m src.main --source "rtsp://..." --stride 2 --enable_tracking
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import yaml

from src.pipeline.behavior import BehaviorClassifier, BehaviorResult
from src.pipeline.detector_yolo import YOLODetector
from src.pipeline.logger import SessionLogger
from src.pipeline.overlay import Overlay
from src.pipeline.pose import PoseEstimator
from src.pipeline.segmentation import PersonSegmentor
from src.pipeline.timer import BehaviorTimer
from src.pipeline.tracker import Track, Tracker
from src.pipeline.video_source import VideoSource
from src.utils.fps import FPSTracker
from src.utils.time_utils import monotonic_seconds, now_stamp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Argument parser
# ═══════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Real-time human behaviour analytics pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Source
    p.add_argument("--source", type=str, default="0",
                   help="Video source: webcam index, file path, or RTSP URL.")
    p.add_argument("--config", type=str, default="configs/default.yaml",
                   help="Path to YAML config file.")

    # Models
    p.add_argument("--model", type=str, default=None,
                   help="Detection model (e.g. yolov8n.pt). Overrides config.")
    p.add_argument("--pose_model", type=str, default=None,
                   help="Pose model (e.g. yolov8n-pose.pt). Overrides config.")
    p.add_argument("--seg_model", type=str, default=None,
                   help="Segmentation model. Overrides config.")

    # Toggles
    p.add_argument("--enable_tracking", action="store_true", default=None,
                   help="Enable multi-object tracking (ByteTrack).")
    p.add_argument("--disable_tracking", dest="enable_tracking", action="store_false",
                   help="Disable multi-object tracking.")
    p.add_argument("--enable_pose", action="store_true", default=None,
                   help="Enable pose estimation.")
    p.add_argument("--disable_pose", dest="enable_pose", action="store_false",
                   help="Disable pose estimation.")
    p.add_argument("--enable_logging", action="store_true", default=None,
                   help="Enable CSV/JSON logging.")
    p.add_argument("--disable_logging", dest="enable_logging", action="store_false",
                   help="Disable CSV/JSON logging.")
    p.add_argument("--enable_overlay", action="store_true", default=None,
                   help="Enable overlay drawing.")
    p.add_argument("--disable_overlay", dest="enable_overlay", action="store_false",
                   help="Disable overlay drawing.")
    p.add_argument("--enable_behavior", action="store_true", default=None,
                   help="Enable behavior classification from pose keypoints.")
    p.add_argument("--disable_behavior", dest="enable_behavior", action="store_false",
                   help="Disable behavior classification.")
    p.add_argument("--enable_segmentation", action="store_true", default=None,
                   help="Enable person segmentation masks.")
    p.add_argument("--disable_segmentation", dest="enable_segmentation", action="store_false",
                   help="Disable person segmentation masks.")

    # Performance
    p.add_argument("--stride", type=int, default=None,
                   help="Process every N-th frame.")
    p.add_argument("--imgsz", type=int, default=None,
                   help="Inference image size.")
    p.add_argument("--conf", type=float, default=None,
                   help="Detection confidence threshold.")
    p.add_argument("--device", type=str, default=None,
                   help="Inference device: 'cpu', 'cuda', 'cuda:0', 'mps'. Default: auto.")

    # Output
    p.add_argument("--save_video", action="store_true", default=False,
                   help="Save annotated video to outputs/videos/.")
    p.add_argument("--no_show", action="store_true", default=False,
                   help="Disable cv2 imshow window.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Base output directory.")
    return p


# ═══════════════════════════════════════════════════════════════════
#  Config loading
# ═══════════════════════════════════════════════════════════════════

def load_config(args: argparse.Namespace) -> dict:
    """Merge YAML config with CLI overrides."""
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        logger.warning("Config %s not found – using defaults.", cfg_path)
        cfg = {}

    # CLI overrides (only if explicitly provided)
    if args.source is not None:
        cfg["source"] = args.source
    if args.model is not None:
        cfg["detection_model"] = args.model
    if args.pose_model is not None:
        cfg["pose_model"] = args.pose_model
    if args.seg_model is not None:
        cfg["segmentation_model"] = args.seg_model
    if args.stride is not None:
        cfg["stride"] = args.stride
    if args.imgsz is not None:
        cfg["imgsz"] = args.imgsz
    if args.conf is not None:
        cfg["confidence_threshold"] = args.conf
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.device is not None:
        cfg["device"] = args.device

    # Boolean toggles – CLI flags override config
    if args.enable_tracking is not None:
        cfg["enable_tracking"] = args.enable_tracking
    if args.enable_pose is not None:
        cfg["enable_pose"] = args.enable_pose
    if args.enable_behavior is not None:
        cfg["enable_behavior"] = args.enable_behavior
    if args.enable_logging is not None:
        cfg["enable_logging"] = args.enable_logging
    if args.enable_overlay is not None:
        cfg["enable_overlay"] = args.enable_overlay
    if args.enable_segmentation is not None:
        cfg["enable_segmentation"] = args.enable_segmentation

    cfg.setdefault("show_window", not args.no_show)
    if args.no_show:
        cfg["show_window"] = False
    cfg["save_video"] = args.save_video or cfg.get("save_video", False)

    # Defaults
    cfg.setdefault("source", "0")
    cfg.setdefault("detection_model", "yolov8n.pt")
    cfg.setdefault("pose_model", "yolov8n-pose.pt")
    cfg.setdefault("segmentation_model", "yolov8n-seg.pt")
    cfg.setdefault("confidence_threshold", 0.45)
    cfg.setdefault("iou_threshold", 0.5)
    cfg.setdefault("person_class_id", 0)
    cfg.setdefault("max_detections", 50)
    cfg.setdefault("stride", 1)
    cfg.setdefault("imgsz", 640)
    cfg.setdefault("enable_tracking", True)
    cfg.setdefault("enable_pose", True)
    cfg.setdefault("enable_behavior", True)
    cfg.setdefault("enable_logging", True)
    cfg.setdefault("enable_overlay", True)
    cfg.setdefault("enable_segmentation", False)
    cfg.setdefault("tracker", "bytetrack")
    cfg.setdefault("tracker_config", None)
    cfg.setdefault("output_dir", "outputs")
    cfg.setdefault("device", "")
    cfg.setdefault("pose_confidence", 0.3)
    cfg.setdefault("behavior", {})
    cfg["behavior"].setdefault("sit_knee_angle", 120)
    cfg["behavior"].setdefault("run_speed_threshold", 60)
    cfg["behavior"].setdefault("walk_speed_threshold", 15)

    return cfg


# ═══════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════

def run(cfg: dict) -> None:
    """Run the analytics pipeline with the given configuration."""

    # ── Build components ────────────────────────────────────────────
    source_id = cfg["source"]
    # Convert numeric strings
    if isinstance(source_id, str) and source_id.isdigit():
        source_id = int(source_id)

    video = VideoSource(source=source_id, stride=cfg["stride"])
    device = cfg.get("device", "")

    tracker: Optional[Tracker] = None
    detector: Optional[YOLODetector] = None
    pose_estimator: Optional[PoseEstimator] = None
    behavior_clf: Optional[BehaviorClassifier] = None
    segmentor: Optional[PersonSegmentor] = None
    session_logger: Optional[SessionLogger] = None
    overlay: Optional[Overlay] = None
    timer = BehaviorTimer()
    fps_tracker = FPSTracker()

    if cfg["enable_tracking"]:
        tracker = Tracker(
            model_path=cfg["detection_model"],
            tracker_type=cfg["tracker"],
            confidence=cfg["confidence_threshold"],
            iou=cfg["iou_threshold"],
            imgsz=cfg["imgsz"],
            person_class_id=cfg["person_class_id"],
            max_detections=cfg["max_detections"],
            tracker_config=cfg["tracker_config"],
            device=device,
        )
    else:
        detector = YOLODetector(
            model_path=cfg["detection_model"],
            confidence=cfg["confidence_threshold"],
            iou=cfg["iou_threshold"],
            imgsz=cfg["imgsz"],
            person_class_id=cfg["person_class_id"],
            max_detections=cfg["max_detections"],
            device=device,
        )

    if cfg["enable_pose"]:
        pose_estimator = PoseEstimator(
            model_path=cfg["pose_model"],
            confidence=cfg["pose_confidence"],
            imgsz=cfg["imgsz"],
            device=device,
        )

    if cfg["enable_pose"] and cfg["enable_behavior"]:
        beh_cfg = cfg["behavior"]
        behavior_clf = BehaviorClassifier(
            sit_knee_angle=beh_cfg["sit_knee_angle"],
            run_speed_threshold=beh_cfg["run_speed_threshold"],
            walk_speed_threshold=beh_cfg["walk_speed_threshold"],
        )

    if cfg["enable_segmentation"]:
        segmentor = PersonSegmentor(
            model_path=cfg["segmentation_model"],
            confidence=cfg["confidence_threshold"],
            imgsz=cfg["imgsz"],
            device=device,
        )

    if cfg["enable_logging"]:
        log_dir = str(Path(cfg["output_dir"]) / "logs")
        session_logger = SessionLogger(output_dir=log_dir)

    if cfg["enable_overlay"]:
        overlay = Overlay(pose_conf=cfg["pose_confidence"])

    # ── Video writer ────────────────────────────────────────────────
    writer: Optional[cv2.VideoWriter] = None

    # ── Run loop ────────────────────────────────────────────────────
    prev_time = monotonic_seconds()

    with video:
        _fps = video.fps or 30.0
        w, h = video.frame_size

        if cfg["save_video"]:
            vid_dir = Path(cfg["output_dir"]) / "videos"
            vid_dir.mkdir(parents=True, exist_ok=True)
            vid_path = vid_dir / f"output_{now_stamp()}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(vid_path), fourcc, _fps, (w, h))
            logger.info("Video writer → %s", vid_path)

        logger.info("Pipeline running. Press 'q' to stop.")

        active_ids_last_frame: set = set()

        for frame_idx, frame in video:
            now = monotonic_seconds()
            dt = now - prev_time
            prev_time = now

            fps_tracker.tick()

            # ── 1) Detection / Tracking ─────────────────────────────
            tracks: list[Track] = []
            if tracker is not None:
                tracks = tracker.update(frame)
            elif detector is not None:
                detections = detector.detect(frame)
                # Without tracking we create ephemeral "tracks"
                for i, det in enumerate(detections):
                    tracks.append(Track(track_id=i, bbox=det.bbox, confidence=det.confidence))

            current_ids = {t.track_id for t in tracks}

            # ── 2) Finalize disappeared tracks ──────────────────────
            disappeared = active_ids_last_frame - current_ids
            for tid in disappeared:
                state = timer.finalize_track(tid)
                if state and session_logger:
                    session_logger.log_segments(state.segments)
                if behavior_clf:
                    behavior_clf.remove_track(tid)
            active_ids_last_frame = current_ids

            # ── 3) Pose estimation ──────────────────────────────────
            keypoints_map: Dict[int, np.ndarray] = {}
            if pose_estimator is not None and tracks:
                keypoints_map = pose_estimator.estimate(frame, tracks)

            # ── 4) Behavior classification ──────────────────────────
            behavior_map: Dict[int, BehaviorResult] = {}
            if behavior_clf is not None:
                for track in tracks:
                    kps = keypoints_map.get(track.track_id)
                    result = behavior_clf.classify(track.track_id, kps, dt)
                    behavior_map[track.track_id] = result

            # ── 5) Timer update ─────────────────────────────────────
            timer_states: Dict[int, object] = {}
            for track in tracks:
                beh = behavior_map.get(track.track_id)
                label = beh.label if beh else "unknown"
                state = timer.update(track.track_id, label)
                timer_states[track.track_id] = state

            # ── 6) Segmentation (optional) ──────────────────────────
            if segmentor is not None:
                masks = segmentor.segment(frame, tracks)
                segmentor.draw_masks(frame, masks)

            # ── 7) Overlay ──────────────────────────────────────────
            count_current = tracker.count_current(tracks) if tracker else len(tracks)
            count_total = tracker.count_total if tracker else len(tracks)

            if overlay is not None:
                frame = overlay.draw(
                    frame,
                    tracks=tracks,
                    keypoints_map=keypoints_map,
                    behavior_map=behavior_map,
                    timer_states=timer_states,
                    fps=fps_tracker.fps,
                    count_current=count_current,
                    count_total=count_total,
                )

            # ── 8) Output ──────────────────────────────────────────
            if writer is not None:
                writer.write(frame)

            if cfg.get("show_window", True):
                cv2.imshow("Human Behavior Analytics", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User pressed 'q' – stopping.")
                    break

    # ── Cleanup ─────────────────────────────────────────────────────
    logger.info("Shutting down pipeline …")
    remaining = timer.finalize_all()
    if session_logger:
        for state in remaining:
            session_logger.log_segments(state.segments)
        session_logger.write_json_summary(timer)
        session_logger.close()

    if writer is not None:
        writer.release()
        logger.info("Video saved.")

    cv2.destroyAllWindows()
    logger.info("Done.")


# ═══════════════════════════════════════════════════════════════════
#  Entry-point
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args)
    logger.info("Config: %s", {k: v for k, v in cfg.items() if k != "behavior"})
    run(cfg)


if __name__ == "__main__":
    main()
