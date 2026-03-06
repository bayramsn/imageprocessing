# Image Segmentation Theory

## 1. What is Image Segmentation?

Image segmentation assigns a label to **every pixel** in an image, producing
fine-grained spatial understanding beyond bounding boxes.

## 2. Three Types of Segmentation

### Semantic Segmentation
- Every pixel gets a class label (e.g., "person", "road", "sky").
- **No distinction between instances** of the same class.
- Output: H × W class map.

### Instance Segmentation
- Detects individual objects **and** produces a pixel mask per instance.
- Distinguishes "person #1" from "person #2".
- Output: Set of (class, confidence, mask) per instance.

### Panoptic Segmentation
- Combines semantic (stuff) and instance (things) segmentation.
- Every pixel is assigned to either a stuff class or a specific thing instance.

| Type       | Classes | Instances | Stuff + Things |
|------------|---------|-----------|----------------|
| Semantic   | ✓       | ✗         | ✗              |
| Instance   | ✓       | ✓         | ✗              |
| Panoptic   | ✓       | ✓         | ✓              |

## 3. Common Architectures

### FCN (Fully Convolutional Network)
- First end-to-end semantic segmentation model.
- Replaces FC layers with convolutional layers for dense prediction.

### U-Net
- Encoder-decoder with skip connections.
- Originally designed for biomedical image segmentation.
- Still widely used for binary segmentation tasks.

### Mask R-CNN
- Extends Faster R-CNN with a mask branch.
- Two-stage: detect objects → predict per-instance masks.
- Standard benchmark model for instance segmentation.

### DeepLab (v1–v3+)
- Atrous (dilated) convolutions for multi-scale context.
- ASPP (Atrous Spatial Pyramid Pooling).
- CRF post-processing (v1/v2).

### YOLO-Seg (YOLOv8-seg)
- Single-stage instance segmentation.
- Shares the YOLOv8 backbone + neck.
- Adds a **proto-mask** head that predicts mask coefficients.

## 4. YOLOv8-Seg Architecture

```
Input Image
    │
    ▼
Backbone (CSPDarknet)
    │
    ▼
Neck (PAN-FPN)
    │
    ▼
┌─────────────────────────────────┐
│ Detection Head                  │
│  ├── bbox regression            │
│  └── classification             │
├─────────────────────────────────┤
│ Segmentation Head               │
│  ├── Proto-net → k proto-masks  │  (32 prototypes, H/4 × W/4)
│  └── Mask coefficients (per box)│  (32 coefficients per detection)
└─────────────────────────────────┘

Final mask = sigmoid(coefficients · proto-masks)  → resized to bbox crop
```

### Key Concepts
- **Proto-masks**: A set of learned basis masks shared across all instances.
- **Mask coefficients**: Per-detection linear combination weights.
- **Crop + threshold**: Final masks are cropped to bounding boxes and thresholded.

## 5. Metrics

### IoU (Intersection over Union)
$$IoU = \frac{|A \cap B|}{|A \cup B|}$$

### mAP (mask)
Same as detection mAP but IoU is computed on **pixel masks** instead of boxes.

### Dice Coefficient
$$Dice = \frac{2|A \cap B|}{|A| + |B|}$$
Common in medical imaging; equivalent to F1 at the pixel level.

## 6. Typical Outputs

For this project, the segmentation module produces:
1. A **binary mask** per detected person (same resolution as input frame).
2. An **overlay** visualization blending the mask colour onto the frame.

```python
# Conceptual output structure
{
    track_id_1: np.ndarray(shape=(H, W), dtype=uint8),  # 0 or 1
    track_id_2: np.ndarray(shape=(H, W), dtype=uint8),
}
```

## 7. Use Cases in This Project

- **Person silhouette** overlay for qualitative visualization.
- **Occupancy heatmaps** – accumulate masks over time.
- **Precise body area** measurement (pixel count inside mask).
- Could be extended with **semantic parts** segmentation for finer behaviour cues.

## 8. References

- Long et al., "Fully Convolutional Networks for Semantic Segmentation" (2015)
- He et al., "Mask R-CNN" (2017)
- Chen et al., "DeepLab" (2017)
- Bolya et al., "YOLACT" (2019)
- Ultralytics YOLOv8-Seg: https://docs.ultralytics.com/tasks/segment/
