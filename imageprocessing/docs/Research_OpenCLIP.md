# Research: OpenCLIP – Open-Source CLIP

## 1. What is CLIP?

**CLIP** (Contrastive Language-Image Pre-training) by OpenAI (2021) learns to
align **images** and **text** in a shared embedding space using contrastive learning
on 400 M image-text pairs scraped from the internet.

### Core Idea

```
Image  ──► Image Encoder (ViT or ResNet) ──► image embedding (d-dim)
                                                        ↕  cosine similarity
Text   ──► Text Encoder  (Transformer)   ──► text embedding  (d-dim)
```

During training, matching (image, text) pairs are pulled together while
non-matching pairs are pushed apart via a symmetric **InfoNCE loss**:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(I_i, T_j)/\tau)}$$

Where $\text{sim}$ is cosine similarity and $\tau$ is a learned temperature.

## 2. What is OpenCLIP?

**OpenCLIP** is an open-source reproduction and extension of CLIP by LAION:

- Trained on **LAION-2B** (2 billion image-text pairs) and LAION-400M.
- Supports multiple backbone architectures (ViT-B/32, ViT-L/14, ViT-H/14, ViT-G/14).
- Achieves **similar or better** zero-shot performance than original CLIP.
- Apache-2.0 licensed.

Repository: https://github.com/mlfoundations/open_clip

## 3. Key Capabilities

| Capability               | Description                                              |
|--------------------------|----------------------------------------------------------|
| Zero-shot classification | Classify images using free-form text prompts             |
| Image-text retrieval     | Find images matching a text query (or vice versa)        |
| Feature extraction       | Dense visual features for downstream tasks               |
| Linear probing           | Train a simple classifier on frozen CLIP features        |

## 4. Architecture Details

### Image Encoder
- **ViT** (Vision Transformer) variants: patch embedding → self-attention layers → [CLS] token.
- Common sizes: ViT-B/32 (patch=32), ViT-B/16, ViT-L/14, ViT-H/14.

### Text Encoder
- Standard Transformer with causal masking.
- Tokenization: BPE (Byte Pair Encoding), max 77 tokens.
- [EOS] token embedding used as the text representation.

## 5. Differences: CLIP vs OpenCLIP

| Aspect          | CLIP (OpenAI)                 | OpenCLIP (LAION)                  |
|-----------------|-------------------------------|-----------------------------------|
| Data            | WIT (400M, private)           | LAION-2B / LAION-400M (public)    |
| License         | MIT (model only)              | Apache 2.0                        |
| Reproducibility | Data not released             | Fully reproducible                |
| Model sizes     | Up to ViT-L/14               | Up to ViT-G/14, ViT-bigG/14      |
| Fine-tuning     | Limited community tooling     | Full training scripts provided    |

## 6. Potential Integration with This Project

### A) Person Re-identification
Use CLIP embeddings to match the **same person** across different camera views:

```python
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Crop person from bbox → embed
person_crop = preprocess(Image.fromarray(crop)).unsqueeze(0)
with torch.no_grad():
    person_embedding = model.encode_image(person_crop)

# Compare across cameras/frames
similarity = F.cosine_similarity(emb_cam1, emb_cam2)
```

### B) Behaviour Description with Natural Language
Instead of heuristic rules, use CLIP to score text prompts against person crops:

```python
# Define behaviour prompts
texts = tokenizer([
    "a person standing upright",
    "a person sitting on a chair",
    "a person walking",
    "a person running",
])

# Score each prompt against the person crop
with torch.no_grad():
    text_features = model.encode_text(texts)
    image_features = model.encode_image(person_crop)
    # Normalize + similarity
    similarities = (image_features @ text_features.T).softmax(dim=-1)
    # → [0.1, 0.7, 0.15, 0.05]  → "sitting"
```

### C) Anomaly / Novelty Detection
Compute similarity to "normal" prompt templates; low similarity → anomaly.

## 7. Practical Considerations

| Concern          | Notes                                                    |
|------------------|----------------------------------------------------------|
| **Speed**        | ViT-B/32 ≈ 5 ms/image (GPU); too slow for real-time per-person |
| **Accuracy**     | Zero-shot behaviour classification is noisy              |
| **Best use**     | Offline analysis, re-ID, anomaly tagging                 |
| **Integration**  | Run asynchronously; batch person crops every N frames     |

## 8. References

- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021)
- Ilharco et al., "OpenCLIP" (2021) – https://github.com/mlfoundations/open_clip
- Schuhmann et al., "LAION-5B" (2022)
