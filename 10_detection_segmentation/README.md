# 🎯 Detection - Classification - Segmentation

## 🎯 Amaç
Üç temel CV görevini tek görüntü üzerinde karşılaştırarak, problem türlerini ve doğru model seçimini kavramak.

## 📦 Gereksinimler
```bash
pip install torch torchvision opencv-python numpy matplotlib
```

## 🚀 Kullanım
```bash
python compare_tasks.py resim.jpg
python compare_tasks.py resim.jpg --save sonuc.png
python compare_tasks.py resim.jpg --all
```

## 🧠 Üç Temel Görev

### 1️⃣ Classification (Sınıflandırma)
**Soru: "Bu ne?"**

```
Giriş: Görüntü
Çıkış: Tek etiket + güven skoru

Örnek: "kedi" (%95)
```

**Kullanım Alanları:**
- Spam/spam değil
- Hastalık teşhisi (X-ray)
- Ürün kategorileme

### 2️⃣ Detection (Nesne Tespiti)
**Soru: "Nerede ne var?"**

```
Giriş: Görüntü
Çıkış: Bounding box'lar + etiketler + skorlar

Örnek: [(x1,y1,x2,y2), "kedi", %92]
        [(x1,y1,x2,y2), "köpek", %88]
```

**Kullanım Alanları:**
- Otonom araçlar (yaya, araç tespiti)
- Güvenlik kameraları
- Raf sayımı

### 3️⃣ Segmentation (Bölütleme)
**Soru: "Hangi piksel neye ait?"**

```
Giriş: Görüntü (HxWx3)
Çıkış: Maske (HxWx1 veya HxWxN)

Örnek: Her piksel için sınıf ID'si
```

**Türleri:**
- **Semantic**: Aynı sınıf → aynı renk (kediler ayrılmaz)
- **Instance**: Her nesne ayrı (kedi1, kedi2 farklı)
- **Panoptic**: Semantic + Instance birleşik

**Kullanım Alanları:**
- Tıbbi görüntüleme (tümör sınırları)
- Arka plan kaldırma
- Harita oluşturma (uydu görüntüleri)

## 📊 Karşılaştırma Tablosu

| Özellik | Classification | Detection | Segmentation |
|---------|---------------|-----------|--------------|
| Çıkış | Tek etiket | Kutu + etiket | Piksel bazlı maske |
| Konum bilgisi | ❌ Yok | ✅ Kutu | ✅ Piksel düzeyinde |
| Çoklu nesne | ❌ (multi-label hariç) | ✅ | ✅ |
| Hesaplama | Hafif | Orta | Ağır |
| Popüler Model | ResNet, EfficientNet | YOLO, Faster R-CNN | U-Net, Mask R-CNN |

## 🔧 Popüler Modeller

### Classification
```python
from torchvision import models

# ResNet, EfficientNet, MobileNet
model = models.resnet50(pretrained=True)
# Çıkış: (batch, 1000) - 1000 sınıf olasılığı
```

### Detection
```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=True)
# Çıkış: boxes, labels, scores
```

### Segmentation
```python
from torchvision.models.segmentation import deeplabv3_resnet50

model = deeplabv3_resnet50(pretrained=True)
# Çıkış: (batch, 21, H, W) - 21 sınıf için maske
```

## ⚠️ Yanlış Model Seçmenin Bedeli

| Görev | Yanlış Model | Sonuç |
|-------|--------------|-------|
| Nesne sayma | Classification | ❌ Sayı bilgisi yok |
| Arka plan kaldırma | Detection | ❌ Kaba kenarlar |
| Hızlı sınıflandırma | Segmentation | ❌ Gereksiz yavaş |

## 💡 Hangi Görevi Seçmeliyim?

```
Tek nesne mi?
├── Evet → Classification
└── Hayır → Konumu bilmem lazım mı?
            ├── Hayır → Multi-label Classification
            └── Evet → Kesin sınırlar lazım mı?
                        ├── Hayır → Detection
                        └── Evet → Segmentation
```

## 🌍 Gerçek Dünya Örnekleri

### Otonom Araç
- Classification: Trafik işareti türü ("dur", "yavaşla")
- Detection: Yaya ve araç konumları
- Segmentation: Yol / kaldırım / araç bölgeleri

### Tıbbi Görüntüleme
- Classification: Hastalık var/yok
- Detection: Lezyon konumları
- Segmentation: Tümör sınırları

### E-ticaret
- Classification: Ürün kategorisi
- Detection: Birden fazla ürün tespiti
- Segmentation: Ürünü arka plandan ayırma

## 🔗 Önceki Konular
← [08_cnn_intro](../08_cnn_intro/) - CNN temelleri
