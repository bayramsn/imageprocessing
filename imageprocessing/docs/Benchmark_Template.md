# FPS / Gecikme Benchmark Şablonu

Bu belge, pipeline'daki her aşamanın performansını sistematik olarak ölçmek için
kullanılacak yöntem, tablo şablonları ve örnek Python kodu içerir.

---

## 1. Ne Ölçülmeli?

| Metrik | Açıklama | Birim |
|--------|----------|-------|
| **FPS** | Saniyede işlenen kare sayısı | frame/s |
| **Kare başı gecikme** (per-frame latency) | Bir karenin tüm pipeline'dan geçiş süresi | ms |
| **Aşama gecikmesi** | Tek bir aşamanın (tespit, takip, poz, segmentasyon, çizim) süresi | ms |
| **Bellek kullanımı** | GPU VRAM veya CPU RAM tüketimi | MB |
| **Model yükleme süresi** | Model ilk yüklenirken geçen süre | s |

---

## 2. Deney Protokolü

### 2.1 Sabit Girdi

Tekrarlanabilirlik için **sabit bir video dosyası** kullanılmalıdır:
- Çözünürlük: 1280×720 veya 1920×1080
- Süre: en az 30 saniye (minimum ~900 kare @30fps)
- İçerik: 2-5 kişi, hareketli sahne

### 2.2 Isınma (Warm-up)

İlk N kare, model ve GPU'nun ısınması için atlanmalıdır:

```
WARMUP_FRAMES = 30
```

- İlk kareler genellikle daha yavaştır (model derleme, CUDA çekirdek önbellekleme)
- Bu kareler ölçüm dışı bırakılır

### 2.3 Ölçüm Döngüsü

```
MEASURE_FRAMES = 300
```

- Isınma sonrası ardışık 300 kare ölçülür
- Her karenin aşama süreleri kaydedilir
- Ortanca (median), ortalama (mean) ve p95 değerleri raporlanır

---

## 3. Python Zamanlama Kodu

```python
import time
import statistics
from dataclasses import dataclass, field


@dataclass
class StageTimer:
    """Her pipeline aşaması için süre kaydedici."""
    detection: list[float] = field(default_factory=list)
    tracking: list[float] = field(default_factory=list)
    pose: list[float] = field(default_factory=list)
    segmentation: list[float] = field(default_factory=list)
    overlay: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)


def measure_stage(func, *args, **kwargs):
    """Bir fonksiyonun çalışma süresini ölçer (ms cinsinden)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms


# ── Kullanım örneği ──────────────────────────────────────────────────
timer = StageTimer()

for frame_idx, frame in enumerate(video_source):
    if frame_idx < WARMUP_FRAMES:
        # Isınma – süre kaydetme
        _ = detector.detect(frame)
        continue

    if frame_idx >= WARMUP_FRAMES + MEASURE_FRAMES:
        break

    # Her aşamayı ölç
    t_total_start = time.perf_counter()

    detections, dt_det = measure_stage(detector.detect, frame)
    timer.detection.append(dt_det)

    tracks, dt_trk = measure_stage(tracker.update, detections, frame)
    timer.tracking.append(dt_trk)

    poses, dt_pose = measure_stage(pose_estimator.estimate, frame)
    timer.pose.append(dt_pose)

    masks, dt_seg = measure_stage(segmentor.segment, frame)
    timer.segmentation.append(dt_seg)

    _, dt_draw = measure_stage(overlay.draw, frame, tracks, poses, masks)
    timer.overlay.append(dt_draw)

    total_ms = (time.perf_counter() - t_total_start) * 1000.0
    timer.total.append(total_ms)


# ── Sonuçları raporla ────────────────────────────────────────────────
def report(name: str, values: list[float]) -> dict:
    return {
        "stage": name,
        "mean_ms": round(statistics.mean(values), 2),
        "median_ms": round(statistics.median(values), 2),
        "p95_ms": round(sorted(values)[int(len(values) * 0.95)], 2),
        "min_ms": round(min(values), 2),
        "max_ms": round(max(values), 2),
    }


stages = [
    ("detection", timer.detection),
    ("tracking", timer.tracking),
    ("pose", timer.pose),
    ("segmentation", timer.segmentation),
    ("overlay", timer.overlay),
    ("TOTAL", timer.total),
]

print(f"{'Stage':<15} {'Mean':>8} {'Median':>8} {'P95':>8} {'Min':>8} {'Max':>8}")
print("-" * 63)
for name, vals in stages:
    r = report(name, vals)
    print(f"{r['stage']:<15} {r['mean_ms']:>7.2f}  {r['median_ms']:>7.2f}  "
          f"{r['p95_ms']:>7.2f}  {r['min_ms']:>7.2f}  {r['max_ms']:>7.2f}")

fps = 1000.0 / statistics.mean(timer.total)
print(f"\nOrtalama FPS: {fps:.1f}")
```

---

## 4. Deney Planı

Aşağıdaki konfigürasyonların her biri için ayrı ölçüm yapılmalıdır:

### Deney 1: Pipeline Bileşen Etkisi

Amaç: Her bileşenin eklenmesiyle artan gecikmeyi ölçmek.

| # | Konfigürasyon | Komut |
|---|---------------|-------|
| 1 | Yalnızca tespit | `--enable_pose false --enable_seg false --enable_behavior false` |
| 2 | Tespit + takip | (varsayılan, davranış kapalı) |
| 3 | Tespit + takip + poz | `--enable_pose true` |
| 4 | Tespit + takip + poz + segmentasyon | `--enable_pose true --enable_seg true` |
| 5 | Tam pipeline | `--enable_pose true --enable_seg true --enable_behavior true` |

### Deney 2: Stride Etkisi

Amaç: Kare atlama ile FPS kazancını ölçmek.

| # | Stride | Beklenen Etki |
|---|--------|---------------|
| 1 | 1 | Tüm kareler işlenir (referans) |
| 2 | 2 | Her 2. kare → ~2× FPS artışı |
| 3 | 3 | Her 3. kare → ~3× FPS artışı |

### Deney 3: Model Boyutu Karşılaştırması

| # | Model | Parametre (M) |
|---|-------|--------------|
| 1 | yolov8n | 3.2 |
| 2 | yolov8s | 11.2 |
| 3 | yolov8m | 25.9 |

### Deney 4: Cihaz Karşılaştırması

| # | Cihaz | Flag |
|---|-------|------|
| 1 | CPU | `--device cpu` |
| 2 | GPU (CUDA) | `--device cuda:0` |
| 3 | Apple MPS | `--device mps` |

---

## 5. Sonuç Tablosu Şablonu

Aşağıdaki tabloyu benchmark sonuçlarıyla doldurun:

### Tablo A: Bileşen Bazlı Gecikme

| Konfigürasyon | Detection (ms) | Tracking (ms) | Pose (ms) | Seg (ms) | Overlay (ms) | **Toplam (ms)** | **FPS** |
|---------------|---------------|--------------|-----------|----------|-------------|----------------|---------|
| Yalnızca tespit | | | — | — | | | |
| + Takip | | | — | — | | | |
| + Poz | | | | — | | | |
| + Segmentasyon | | | | | | | |
| Tam pipeline | | | | | | | |

### Tablo B: Stride Etkisi

| Stride | FPS | Toplam Gecikme (ms) | Takip Kalitesi Notu |
|--------|-----|---------------------|---------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |

### Tablo C: Model Boyutu

| Model | FPS (CPU) | FPS (GPU) | Gecikme (ms) | VRAM (MB) |
|-------|-----------|-----------|-------------|-----------|
| yolov8n | | | | |
| yolov8s | | | | |
| yolov8m | | | | |

### Tablo D: Cihaz Karşılaştırması

| Cihaz | Model | FPS | Gecikme (ms) | Bellek (MB) |
|-------|-------|-----|-------------|-------------|
| MacBook CPU (M1/M2) | yolov8n | | | |
| MacBook MPS (M1/M2) | yolov8n | | | |
| NVIDIA GPU (model) | yolov8n | | | |

---

## 6. Bellek Ölçümü

### GPU (CUDA) VRAM

```python
import torch

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    # ... pipeline çalıştır ...
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Peak GPU VRAM: {peak_mb:.1f} MB")
```

### CPU RSS (Resident Set Size)

```python
import psutil
import os

process = psutil.Process(os.getpid())
rss_mb = process.memory_info().rss / (1024 ** 2)
print(f"CPU RSS: {rss_mb:.1f} MB")
```

> Not: `psutil` ek olarak kurulmalıdır: `pip install psutil`

---

## 7. Sonuçların Yorumlanması

### Darboğaz Tespiti

1. **Detection süresi > %60 toplam** → Model büyük, daha küçük model veya TensorRT dışa aktarma deneyin
2. **Tracking süresi yüksek** → Çok fazla iz var, `conf` eşiğini yükseltmeyi deneyin
3. **Pose / Segmentation yüksek** → Bu bileşenler isteğe bağlıdır; gerçek zamanlı ihtiyacınız yoksa kapatın
4. **Overlay süresi yüksek** → Çizim işlemlerini sadeleştirin, `imshow` yerine video yazıcı kullanın

### Optimizasyon Önerileri

| Yöntem | Beklenen Kazanç |
|--------|----------------|
| Model küçültme (m → s → n) | 2-3× FPS artışı |
| Stride artırma (1 → 2) | ~2× FPS artışı |
| TensorRT dışa aktarma | 1.5-3× FPS artışı |
| GPU kullanma (CPU → CUDA) | 3-10× FPS artışı |
| Çözünürlük düşürme (1080p → 720p) | 1.5-2× FPS artışı |
| half precision (FP16) | 1.3-2× FPS artışı (GPU) |
| Poz/Seg kapatma | Aşama başına ~5-15 ms tasarruf |

---

## 8. Örnek Çalıştırma Komutu

```bash
# 1) Tam benchmark: 30 kare ısınma + 300 kare ölçüm
python -m src.main --source benchmark_video.mp4 \
    --model yolov8n.pt \
    --device cpu \
    --stride 1 \
    --conf 0.5

# 2) GPU ile karşılaştırma
python -m src.main --source benchmark_video.mp4 \
    --model yolov8n.pt \
    --device cuda:0 \
    --stride 1 \
    --conf 0.5
```

> Bu komutlar pipeline'ı çalıştırır ve overlay üzerinde anlık FPS gösterir.
> Detaylı aşama bazlı ölçüm için yukarıdaki Python kodunu pipeline'a entegre edin.
