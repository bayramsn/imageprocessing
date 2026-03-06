# Research: DINOv2 – Self-Supervised Vision Foundation Model

## 1. What is DINOv2?

**DINOv2** (Meta AI, 2023) is a family of self-supervised Vision Transformers (ViT)
that produce **general-purpose visual features** without any text supervision.

Unlike CLIP which requires paired image-text data, DINOv2 learns from **images
alone** using a combination of:
- **Self-distillation** (student-teacher with EMA)
- **iBOT** masked image modelling
- **Sinkhorn-Knopp centering** to avoid collapse

## 2. Architecture

```
Image (224×224 or 518×518)
    │
    ▼
Patch Embedding (14×14 or 16×16 patches)
    │
    ▼
ViT Encoder (L layers of self-attention)
    │
    ├─► [CLS] token → global feature (d-dim)
    └─► patch tokens → dense feature map (N_patches × d)
```

### Model Sizes

| Model        | Params | Embedding dim | Patch | ImageNet kNN |
|-------------|--------|---------------|-------|--------------|
| ViT-S/14    | 21M    | 384           | 14    | 81.1%        |
| ViT-B/14    | 86M    | 768           | 14    | 82.1%        |
| ViT-L/14    | 300M   | 1024          | 14    | 83.5%        |
| ViT-g/14    | 1.1B   | 1536          | 14    | 83.5%        |

## 3. Training Recipe

1. **Curated data**: LVD-142M dataset (curated from uncurated web images).
2. **Self-distillation**: Student sees augmented crops; teacher (EMA of student)
   provides targets.
3. **iBOT head**: Masked patch prediction for local feature learning.
4. **No labels needed**: Entirely self-supervised.

### Loss Function (simplified)

$$\mathcal{L} = \mathcal{L}_{\text{DINO}} + \lambda \cdot \mathcal{L}_{\text{iBOT}}$$

- $\mathcal{L}_{\text{DINO}}$: Cross-entropy between student [CLS] and teacher [CLS].
- $\mathcal{L}_{\text{iBOT}}$: Cross-entropy between student masked patches and teacher patches.

## 4. Key Properties of DINOv2 Features

| Property                 | Description                                        |
|--------------------------|----------------------------------------------------|
| **Semantic richness**    | Features capture object parts, materials, shapes   |
| **Spatial locality**     | Patch tokens preserve spatial information           |
| **Transfer quality**     | SoTA linear-probe on many benchmarks               |
| **No text needed**       | Works purely on visual data                        |
| **Dense prediction**     | Patch tokens useful for segmentation, depth, etc.  |

## 5. DINOv2 vs CLIP

| Aspect              | DINOv2                        | CLIP / OpenCLIP               |
|---------------------|-------------------------------|-------------------------------|
| Supervision         | Self-supervised (images only) | Contrastive (image-text)      |
| Text understanding  | ✗ No text encoder             | ✓ Aligned with language       |
| Zero-shot classify  | ✗ (needs probe/kNN)           | ✓ (via text prompts)          |
| Feature quality     | Excellent local + global      | Strong global, weaker local   |
| Dense tasks         | ✓ Segmentation, depth         | Weaker                        |
| Data requirement    | Images only                   | Image-text pairs              |

**Summary**: Use CLIP when you need language grounding; use DINOv2 when you need
high-quality visual features for downstream tasks.

## 6. Potential Integration with This Project

### A) Person Appearance Embedding for Re-ID

DINOv2 features are very discriminative for person appearance:

```python
import torch
from torchvision import transforms

# Load DINOv2
dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dinov2.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Extract feature for a person crop
crop_tensor = transform(pil_crop).unsqueeze(0)
with torch.no_grad():
    features = dinov2(crop_tensor)  # (1, 384)

# Compare across frames for re-identification
similarity = torch.cosine_similarity(feat_frame1, feat_frame2)
```

### B) Dense Part Segmentation

DINOv2 patch tokens can segment person body parts without any supervision:

```python
with torch.no_grad():
    output = dinov2.forward_features(crop_tensor)
    patch_tokens = output["x_norm_patchtokens"]  # (1, N, D)

# PCA or clustering on patch tokens → body part segments
from sklearn.cluster import KMeans
tokens_np = patch_tokens[0].cpu().numpy()
clusters = KMeans(n_clusters=5).fit_predict(tokens_np)
# Reshape to spatial grid → visualize as coloured map
```

### C) Anomaly Detection via Feature Distance

Compute a "normal" feature distribution from training frames, then flag
outlier crops at inference time:

```python
# Offline: collect features from normal behaviour
normal_features = [dinov2(crop) for crop in normal_crops]
mean_feat = torch.stack(normal_features).mean(0)

# Online: compare new crops
dist = torch.cdist(new_feat.unsqueeze(0), mean_feat.unsqueeze(0))
if dist > threshold:
    flag_anomaly()
```

### D) Fine-Grained Posture Embedding

Instead of heuristic rules, train a small MLP on top of frozen DINOv2 features
to classify sitting/standing/walking/running:

```python
class PostureHead(nn.Module):
    def __init__(self, in_dim=384, n_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )
    
    def forward(self, x):
        return self.fc(x)

# Freeze DINOv2, only train the head
head = PostureHead()
for crop, label in dataloader:
    with torch.no_grad():
        feat = dinov2(crop)
    logits = head(feat)
    loss = F.cross_entropy(logits, label)
```

## 7. Practical Considerations

| Concern          | Notes                                                        |
|------------------|--------------------------------------------------------------|
| **Speed**        | ViT-S/14 ≈ 4ms (GPU), ViT-B/14 ≈ 8ms – feasible per-person |
| **Memory**       | ViT-g needs >8 GB VRAM; S/B fit on consumer GPUs            |
| **No text**      | Cannot do zero-shot text prompting like CLIP                 |
| **Best use**     | Re-ID, dense features, fine-tuned classifiers                |
| **Integration**  | Run on crops every N frames; cache embeddings per track_id   |

## 8. References

- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision" (2023)
- Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (DINO, 2021)
- Repository: https://github.com/facebookresearch/dinov2
