# Pose Estimation Theory

## 1. What is Pose Estimation?

**Human pose estimation** predicts the spatial locations of body joints (keypoints)
from images or video. Each keypoint represents an anatomical landmark:

```
       0: Nose
     1   2: Eyes (L/R)
   3       4: Ears (L/R)
     5   6: Shoulders (L/R)
     7   8: Elbows (L/R)
     9  10: Wrists (L/R)
    11  12: Hips (L/R)
    13  14: Knees (L/R)
    15  16: Ankles (L/R)
```

The **COCO keypoint format** defines 17 keypoints per person, each with
`(x, y, visibility/confidence)`.

## 2. Two Paradigms

### Top-Down

1. **Detect** all persons with a bounding box detector.
2. **Crop** each person and run a single-person pose model on each crop.
3. **Map** keypoints back to original image coordinates.

**Pros**: Higher accuracy per person.
**Cons**: Scales O(N) with number of persons – slower for crowds.

### Bottom-Up

1. Detect **all keypoints** in the entire image at once.
2. **Group** keypoints into individuals using associative embeddings or
   part affinity fields.

**Pros**: Constant time regardless of person count.
**Cons**: Lower per-person accuracy, harder to implement.

| Method      | Examples                  | Speed       | Accuracy |
|-------------|---------------------------|-------------|----------|
| Top-Down    | HRNet, SimpleBaseline     | O(N·k)     | Higher   |
| Bottom-Up   | OpenPose, HigherHRNet     | O(1)       | Lower    |

## 3. The COCO Keypoint Format

Each annotation contains:
```json
{
  "keypoints": [x1, y1, v1, x2, y2, v2, ...],
  "num_keypoints": 17
}
```

Visibility flags:
- `0` = not labelled
- `1` = labelled but occluded
- `2` = labelled and visible

## 4. YOLOv8-Pose

YOLOv8-Pose is a **single-stage, top-down** pose model that predicts bounding
boxes **and** keypoints simultaneously in one forward pass:

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
Detection Head (anchor-free)
  ├── bbox regression → (x1, y1, x2, y2)
  ├── classification → person confidence
  └── keypoint head  → 17 × (x, y, conf)
```

This is neither purely top-down nor bottom-up – the detector and pose estimator
share the same backbone and are jointly trained.

### Advantages
- **Fast**: One forward pass for detection + pose.
- **Unified API**: Same Ultralytics interface as detection.
- **Accurate**: Competitive with dedicated top-down models at small scale.

## 5. MoveNet (Alternative)

Google's **MoveNet** is a lightweight bottom-up model:
- **Lightning**: Runs on mobile devices at 50+ FPS.
- **Thunder**: Higher accuracy, slightly slower.
- Available via TensorFlow Hub / TFLite.

MoveNet predicts 17 COCO keypoints for a single person (or multiple with
multi-pose variant).

## 6. Metrics

### OKS (Object Keypoint Similarity)

The standard metric for pose evaluation:

$$OKS = \frac{\sum_i \exp\left(-d_i^2 / (2 s^2 \kappa_i^2)\right) \cdot \delta(v_i > 0)}{\sum_i \delta(v_i > 0)}$$

Where:
- $d_i$ = Euclidean distance between predicted and GT keypoint $i$
- $s$ = object scale (sqrt of segment area)
- $\kappa_i$ = per-keypoint constant (controls falloff – e.g., eyes have smaller tolerance than hips)

**AP@OKS** works like mAP for detection, using OKS instead of IoU.

## 7. Keypoint-Based Behaviour Analysis

In this project we use keypoints for heuristic posture classification:

| Feature                | Formula                                           | Used for          |
|------------------------|--------------------------------------------------|--------------------|
| Knee angle             | angle(hip, knee, ankle)                          | Sitting detection  |
| Torso inclination      | angle between shoulder-hip line and vertical      | Bending / leaning  |
| Ankle inter-frame speed| ‖ankle_t − ankle_{t-1}‖ / Δt                    | Walk vs run        |

## 8. References

- Cao et al., "OpenPose" (2019)
- Sun et al., "HRNet" (2019)
- Ronchi & Perona, "Benchmarking and Error Diagnosis in Multi-Instance Pose Estimation" (2017)
- Ultralytics YOLOv8-Pose docs: https://docs.ultralytics.com/tasks/pose/
- MoveNet: https://www.tensorflow.org/hub/tutorials/movenet
