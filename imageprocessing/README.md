# cv-human-behavior-analytics

> Real-time **multi-person behaviour analytics** pipeline:
> YOLO Detection → ByteTrack Tracking → Pose Estimation → Behavior Classification → Duration Tracking → Logging → Video Overlay

Built with **Ultralytics YOLOv8** and **OpenCV** — designed as an internship portfolio project demonstrating end-to-end computer vision engineering.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VIDEO SOURCE                                     │
│              (Webcam · Video File · RTSP Stream)                        │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │ frame
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — PERSON DETECTION + TRACKING                                  │
│                                                                         │
│  ┌─────────────────┐         ┌──────────────────────┐                   │
│  │ YOLOv8 Detector │────────►│ ByteTrack Tracker    │                   │
│  │ (person only)   │  dets   │ (stable track_ids)   │                   │
│  └─────────────────┘         └──────────┬───────────┘                   │
│                                         │ tracks[]                      │
└─────────────────────────────────────────┼───────────────────────────────┘
                                          │
                           ┌──────────────┼──────────────┐
                           ▼              ▼              ▼
                   ┌──────────────┐ ┌──────────┐ ┌──────────────┐
 STAGE 2           │ YOLOv8-Pose  │ │ Segment  │ │  People      │
                   │ (keypoints)  │ │ (masks)  │ │  Counter     │
                   └──────┬───────┘ └────┬─────┘ └──────────────┘
                          │              │
                          ▼              │
                   ┌──────────────┐      │
 STAGE 3           │  Behavior    │      │
                   │  Classifier  │      │
                   │  (rules)     │      │
                   └──────┬───────┘      │
                          │              │
                          ▼              │
                   ┌──────────────┐      │
 STAGE 4           │  Duration    │      │
                   │  Timer       │      │
                   │  (per track) │      │
                   └──────┬───────┘      │
                          │              │
              ┌───────────┼──────────────┘
              ▼           ▼
       ┌─────────────┐  ┌─────────────────────────────────────┐
       │   Logger     │  │           Overlay                    │
       │  CSV + JSON  │  │  bbox · ID · skeleton · label · FPS │
       └─────────────┘  └──────────────┬──────────────────────┘
                                       │
                           ┌───────────┼───────────┐
                           ▼                       ▼
                    ┌─────────────┐         ┌─────────────┐
                    │  Display    │         │  Save Video  │
                    │  (imshow)   │         │  (.mp4)      │
                    └─────────────┘         └─────────────┘
```

### Feature Matrix

| Feature | Description | Toggle |
|---------|-------------|--------|
| Person Detection | YOLOv8n/s, COCO person class only | always on |
| Multi-Object Tracking | ByteTrack via Ultralytics — stable IDs, no double counting | `--enable_tracking` |
| People Counter | Current + total unique person count | automatic with tracking |
| Pose Estimation | YOLOv8-pose, COCO 17-keypoint format | `--enable_pose` |
| Behavior Classification | Rule-based: standing / sitting / walking / running | `--enable_behavior` |
| Duration Tracking | Per-person behaviour timers with transition history | automatic |
| Logging | CSV event log + JSON session summary | `--enable_logging` |
| Video Overlay | Bboxes, skeletons, labels, durations, FPS, counts | `--enable_overlay` |
| Segmentation | Optional YOLOv8-seg person masks | `--enable_segmentation` |

---

## Installation

### Prerequisites
- Python ≥ 3.10
- (Optional) NVIDIA GPU + CUDA for faster inference

### Setup

```bash
git clone <repo-url>
cd cv-human-behavior-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

Or install as a package with a console command:

```bash
pip install -e .
cv-human-behavior-analytics --source 0
```

### Model Weights

Ultralytics **auto-downloads** weights on first run — no manual steps needed:
- `yolov8n.pt` — detection (~6 MB)
- `yolov8n-pose.pt` — pose estimation (~6 MB)
- `yolov8n-seg.pt` — segmentation (~7 MB)

For higher accuracy (at the cost of speed), use `yolov8s.pt` / `yolov8s-pose.pt` / `yolov8s-seg.pt`.

---

## Usage

### Webcam — all features on

```bash
python -m src.main \
  --source 0 \
  --enable_tracking \
  --enable_pose \
  --enable_behavior \
  --enable_logging \
  --enable_overlay
```

### Video file

```bash
python -m src.main \
  --source path/to/video.mp4 \
  --enable_tracking \
  --enable_pose \
  --enable_behavior \
  --enable_logging \
  --enable_overlay
```

### RTSP stream with frame skipping

```bash
python -m src.main \
  --source "rtsp://user:pass@192.168.1.10:554/stream" \
  --stride 2 \
  --enable_tracking \
  --enable_overlay
```

### Save annotated video (no display window)

```bash
python -m src.main \
  --source video.mp4 \
  --save_video \
  --no_show \
  --enable_tracking \
  --enable_pose \
  --enable_behavior \
  --enable_overlay
```

### Detection only (no tracking, no pose)

```bash
python -m src.main --source 0 --enable_overlay
```

### With segmentation masks

```bash
python -m src.main \
  --source 0 \
  --enable_tracking \
  --enable_pose \
  --enable_behavior \
  --enable_segmentation \
  --enable_overlay
```

### GPU inference

```bash
python -m src.main \
  --source 0 \
  --device cuda \
  --enable_tracking \
  --enable_pose \
  --enable_behavior \
  --enable_overlay
```

### Use a larger model for higher accuracy

```bash
python -m src.main \
  --source 0 \
  --model yolov8s.pt \
  --pose_model yolov8s-pose.pt \
  --enable_tracking \
  --enable_pose \
  --enable_behavior \
  --enable_overlay
```

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | Webcam index, video path, or RTSP URL |
| `--config` | `configs/default.yaml` | YAML config file |
| `--model` | `yolov8n.pt` | Detection model |
| `--pose_model` | `yolov8n-pose.pt` | Pose model |
| `--seg_model` | `yolov8n-seg.pt` | Segmentation model |
| `--device` | auto | `cpu`, `cuda`, `cuda:0`, `mps` |
| `--enable_tracking` / `--disable_tracking` | config | Enable or disable ByteTrack multi-object tracking |
| `--enable_pose` / `--disable_pose` | config | Enable or disable pose estimation |
| `--enable_behavior` / `--disable_behavior` | config | Enable or disable behavior classification |
| `--enable_logging` / `--disable_logging` | config | Enable or disable CSV/JSON logging |
| `--enable_overlay` / `--disable_overlay` | config | Enable or disable overlay drawing |
| `--enable_segmentation` / `--disable_segmentation` | config | Enable or disable segmentation masks |
| `--stride N` | `1` | Process every N-th frame |
| `--imgsz` | `640` | Inference resolution |
| `--conf` | `0.45` | Confidence threshold |
| `--save_video` | off | Save annotated output video |
| `--no_show` | off | Disable cv2.imshow window |
| `--output_dir` | `outputs` | Base output directory |

> **Tip:** All parameters can also be set in `configs/default.yaml` to avoid long CLI commands.

> **Note:** The default `configs/default.yaml` enables tracking, pose, behavior, logging, and overlay. CLI flags only override those config values when passed explicitly.

---

## Output Artifacts

### Logs — `outputs/logs/`

**CSV event log** (`events_YYYYMMDD_HHMMSS.csv`):
```csv
track_id,behavior,start_time,end_time,duration_s
1,standing,2026-02-20T15:30:45.123Z,2026-02-20T15:31:02.456Z,17.33
1,walking,2026-02-20T15:31:02.456Z,2026-02-20T15:31:15.789Z,13.33
2,sitting,2026-02-20T15:30:50.000Z,2026-02-20T15:31:20.000Z,30.0
```

**JSON session summary** (`session_YYYYMMDD_HHMMSS.json`):
```json
{
  "session_start": "2026-02-20T15:30:45.000Z",
  "session_end": "2026-02-20T15:35:00.000Z",
  "total_events": 12,
  "segments": [
    {"track_id": 1, "behavior": "standing", "start_iso": "...", "end_iso": "...", "duration_s": 17.33}
  ]
}
```

### Videos — `outputs/videos/`

Annotated `.mp4` files when `--save_video` is passed. Filename: `output_YYYYMMDD_HHMMSS.mp4`.

---

## Behavior Classification

The classifier uses **keypoint geometry heuristics** on COCO 17-keypoint poses:

| Behavior | Rule | Keypoints Used |
|----------|------|----------------|
| **sitting** | Either knee angle (hip→knee→ankle) < 120° | hips, knees, ankles |
| **running** | Avg ankle velocity > 60 px/s between frames | ankles (frame t vs t-1) |
| **walking** | Avg ankle velocity > 15 px/s | ankles (frame t vs t-1) |
| **standing** | Default when no other rule triggers | — |
| **unknown** | Keypoints missing or below confidence | — |

Rules are evaluated in priority order: sitting → running → walking → standing.

Thresholds are configurable in `configs/default.yaml`:
```yaml
behavior:
  sit_knee_angle: 120       # degrees
  run_speed_threshold: 60   # px/s
  walk_speed_threshold: 15  # px/s
```

---

## Project Structure

```
cv-human-behavior-analytics/
├── README.md                            ← You are here
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   └── default.yaml                     # All tuneable parameters
├── src/
│   ├── __init__.py
│   ├── main.py                          # CLI entry-point + pipeline orchestration
│   ├── pipeline/
│   │   ├── video_source.py              # Unified frame generator (cam/file/RTSP)
│   │   ├── detector_yolo.py             # YOLOv8 person detection wrapper
│   │   ├── tracker.py                   # ByteTrack multi-object tracker
│   │   ├── pose.py                      # YOLOv8-pose top-down estimator
│   │   ├── behavior.py                  # Rule-based behaviour classifier
│   │   ├── timer.py                     # Per-track duration state machine
│   │   ├── overlay.py                   # Video overlay compositor
│   │   ├── logger.py                    # CSV + JSON session logger
│   │   └── segmentation.py             # Optional YOLOv8-seg person masks
│   └── utils/
│       ├── fps.py                       # FPS tracker (sliding window)
│       ├── geometry.py                  # Angles, distances, velocities
│       ├── draw.py                      # Drawing primitives (bbox, skeleton)
│       └── time_utils.py               # Timestamp helpers
├── outputs/
│   ├── logs/                            # CSV/JSON logs (gitignored)
│   └── videos/                          # Annotated videos (gitignored)
├── docs/
│   ├── Internship_Report_Template.md    # Full report outline
│   ├── YOLO_v3_to_v8_Differences.md     # YOLO evolution (Turkish)
│   ├── Benchmark_Template.md            # FPS/latency benchmark guide
│   ├── YOLO_Theory.md
│   ├── Pose_Theory.md
│   ├── Segmentation_Theory.md
│   ├── Research_OpenCLIP.md
│   └── Research_DINOv2.md
└── tests/
    ├── test_behavior_rules.py           # 9 behaviour classifier tests
    └── test_timer.py                    # 15 timer state-machine tests
```

---

## Performance Tips

### Model selection

| Model | Params | Speed (A100) | Accuracy (mAP) | Best for |
|-------|--------|-------------|-----------------|----------|
| yolov8n | 3.2M | ~6 ms | 37.3 | Real-time on CPU / laptop |
| yolov8s | 11.2M | ~10 ms | 44.9 | Balanced (with GPU) |
| yolov8m | 25.9M | ~20 ms | 50.2 | High accuracy (GPU required) |

### Speed Optimizations

1. **Use nano models** (`yolov8n.pt`) for real-time on CPU.
2. **Stride > 1** (`--stride 2` or `--stride 3`) skips frames — useful for RTSP streams.
3. **GPU acceleration** (`--device cuda`) provides 5–10× speedup.
4. **Disable unused modules**: don't pass `--enable_pose` if you only need detection + tracking.
5. **Lower resolution** (`--imgsz 320`) trades accuracy for speed.
6. **Export to ONNX/TensorRT**: Ultralytics supports `model.export()` for deployment.

### CPU vs GPU Guidance

| Setup | Expected FPS (640px) | Notes |
|-------|---------------------|-------|
| CPU (i7) det only | 15–25 | yolov8n |
| CPU (i7) det + track + pose | 5–10 | Two model forward passes |
| GPU (RTX 3060) full pipeline | 30–60 | Easily real-time |
| Apple M1/M2 (`--device mps`) | 20–40 | Metal acceleration |

---

## Running Tests

```bash
pytest tests/ -v
```

24 unit tests covering:
- Behaviour classifier rules (standing, sitting, walking, running, unknown, edge cases)
- Timer state machine (transitions, finalization, cumulative durations)

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'ultralytics'`
```bash
pip install ultralytics
```

### `cv2.error: ... Can't open video source`
- Check that the webcam index is correct (`0`, `1`, etc.).
- For video files, verify the path exists and the codec is supported.
- For RTSP, confirm network connectivity and URL format.

### `CUDA out of memory`
- Use a smaller model: `--model yolov8n.pt`.
- Lower resolution: `--imgsz 320`.
- Increase stride: `--stride 2`.
- Fallback to CPU: `--device cpu`.

### Display window not appearing
- On headless Linux servers, `imshow` needs X11. Use `--no_show --save_video`.
- On macOS, the pip `opencv-python` package includes GUI support.

### FPS is very low on CPU
- Use `yolov8n.pt` (nano).
- Disable pose if not needed.
- Increase stride: `--stride 3`.

### Tracking IDs keep resetting
- Ensure `--enable_tracking` is passed.
- ByteTrack needs consecutive frames — avoid very large stride values (> 5).
- On RTSP with high latency, dropped frames may cause ID resets.

### Log files are empty
- Ensure `--enable_logging` is passed.
- Logs flush on exit — press `q` to quit cleanly (don't `Ctrl+C` or kill).
- Check `outputs/logs/` for timestamped files.

### `ImportError` on Apple Silicon (M1/M2/M3)
```bash
pip install --upgrade ultralytics opencv-python-headless
```
Use `--device mps` for Metal acceleration.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Internship_Report_Template.md](docs/Internship_Report_Template.md) | Full report outline with all section headings |
| [YOLO_v3_to_v8_Differences.md](docs/YOLO_v3_to_v8_Differences.md) | YOLO evolution (Turkish, copy-paste ready) |
| [Benchmark_Template.md](docs/Benchmark_Template.md) | FPS/latency benchmarking methodology |
| [YOLO_Theory.md](docs/YOLO_Theory.md) | YOLO architecture deep-dive |
| [Pose_Theory.md](docs/Pose_Theory.md) | Pose estimation paradigms |
| [Segmentation_Theory.md](docs/Segmentation_Theory.md) | Segmentation theory |
| [Research_OpenCLIP.md](docs/Research_OpenCLIP.md) | OpenCLIP research notes |
| [Research_DINOv2.md](docs/Research_DINOv2.md) | DINOv2 research notes |

---

## License

MIT
