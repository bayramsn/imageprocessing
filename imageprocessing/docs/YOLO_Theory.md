# YOLO Theory – From v1 to v8

## 1. What is YOLO?

**YOLO** (You Only Look Once) is a family of single-stage object detection models
that frame detection as a **regression problem** – predicting bounding boxes and
class probabilities directly from the full image in one forward pass. This makes
YOLO extremely fast compared to two-stage approaches like Faster R-CNN.

## 2. Core Idea – Grid-Based Detection

The original YOLO (v1) divides the input image into an **S × S grid**. Each grid
cell is responsible for predicting:

| Output              | Description                                    |
|---------------------|------------------------------------------------|
| B bounding boxes    | (x, y, w, h) relative to the grid cell         |
| Confidence scores   | P(object) × IoU(pred, truth)                   |
| C class probs       | P(class_i \| object)                           |

This is a **single-shot** architecture: one forward pass → all detections.

## 3. Evolution from v1 to v8

### YOLOv1 (2016)
- Grid + fully-connected head.
- Struggled with small objects and objects near grid boundaries.

### YOLOv2 / YOLO9000 (2017)
- Introduced **anchor boxes** (predefined aspect ratios).
- Batch normalization, multi-scale training.
- Darknet-19 backbone.

### YOLOv3 (2018)
- **Feature Pyramid Network (FPN)** – predictions at 3 scales.
- Darknet-53 backbone (residual connections).
- Better at detecting small objects.

### YOLOv4 (2020)
- CSPDarknet53 backbone + SPP + PAN neck.
- Bag of freebies (data augmentation tricks) + bag of specials (architectural
  improvements).

### YOLOv5 (2020 – Ultralytics)
- PyTorch implementation, easy CLI, auto-augment.
- Focus layer, C3 modules.
- Model zoo: n / s / m / l / x sizes.

### YOLOv7 (2022)
- E-ELAN (Extended Efficient Layer Aggregation Network).
- Auxiliary head training, re-parameterized convolutions.

### YOLOv8 (2023 – Ultralytics)
- **Anchor-free** detection head (decoupled head).
- C2f (Cross Stage Partial with two convolutions) module.
- Unified framework for **detection, segmentation, pose, classification**.
- State-of-the-art speed / accuracy trade-off.

## 4. Anchor-Based vs Anchor-Free

| Feature            | Anchor-Based (v2-v5)               | Anchor-Free (v8)                  |
|--------------------|-------------------------------------|-----------------------------------|
| Priors             | Predefined aspect-ratio anchors     | None – predicts offsets directly  |
| Training           | Anchor–GT matching complicated      | Simpler label assignment (TAL)    |
| NMS sensitivity    | Higher                             | Lower                             |
| Small objects      | Depends on anchor design            | Generally better                  |

YOLOv8 uses **Task-Aligned Assignment (TAL)** for positive sample selection
during training, replacing the IoU-based matching of earlier versions.

## 5. YOLOv8 Architecture (Detection)

```
Input (640×640)
  │
  ▼
Backbone: CSPDarknet (C2f modules)
  │  ┌────────────┐
  ├──► P3 (80×80) │
  │  └────────────┘
  │  ┌────────────┐
  ├──► P4 (40×40) │
  │  └────────────┘
  │  ┌────────────┐
  └──► P5 (20×20) │
     └────────────┘
  │
  ▼
Neck: PAN-FPN (feature fusion)
  │
  ▼
Head: Decoupled anchor-free
  ├── Classification branch → class logits
  └── Regression branch   → bbox (ltrb) + objectness
```

## 6. Model Sizes

| Model     | Params (M) | mAP@50-95 | Speed (ms) |
|-----------|-----------|-----------|------------|
| yolov8n   | 3.2       | 37.3      | ~6         |
| yolov8s   | 11.2      | 44.9      | ~10        |
| yolov8m   | 25.9      | 50.2      | ~20        |
| yolov8l   | 43.7      | 52.9      | ~35        |
| yolov8x   | 68.2      | 53.9      | ~55        |

*(Benchmarks on COCO val2017, NVIDIA A100, FP16)*

## 7. Loss Functions in YOLOv8

- **CIoU loss** – bounding box regression (considers overlap, center distance,
  aspect ratio).
- **BCE loss** – classification.
- **DFL (Distribution Focal Loss)** – regression quality estimation.

## 8. Why YOLOv8 for This Project?

1. **Real-time** – yolov8n runs >100 FPS on modern GPUs.
2. **Unified** – same Ultralytics API for det / seg / pose / classify.
3. **Built-in tracking** – ByteTrack and BotSORT integrated.
4. **Community** – active development, extensive documentation.
5. **Easy deployment** – export to ONNX, TensorRT, CoreML, etc.

## References

- Redmon et al., "You Only Look Once" (2016)
- Ultralytics YOLOv8 docs: https://docs.ultralytics.com
- Jocher et al., Ultralytics repository: https://github.com/ultralytics/ultralytics
