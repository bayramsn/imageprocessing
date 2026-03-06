# YOLO v3'ten v8'e: Mimari Farklar ve Evrim

> Bu bölüm, bir staj raporuna doğrudan kopyalanarak yapıştırılabilecek şekilde
> hazırlanmıştır. Teknik Türkçe ile yazılmıştır.

---

## 1. YOLO'nun Temel Fikri

YOLO (You Only Look Once), nesne tespitini **tek bir sinir ağı geçişinde** (single forward pass) gerçekleştiren bir aile modeldir. Geleneksel iki aşamalı (two-stage) detektörlerin aksine (örn. Faster R-CNN), YOLO:

1. Girdi görüntüsünü **S × S boyutlu bir ızgaraya** (grid) böler.
2. Her ızgara hücresi, o hücreye düşen nesneler için **sınırlayıcı kutu** (bounding box) ve **sınıf olasılıklarını** doğrudan tahmin eder.
3. Tüm bu tahminler **tek bir ileri geçişte** yapılır → gerçek zamanlı hız.

Bu yaklaşım, tespiti bir **regresyon problemi** olarak çerçeveler: girdiden doğrudan kutu koordinatlarına ve sınıf olasılıklarına eşleme yapılır.

### Izgara ve Hücre Atama Mantığı

Bir nesnenin merkez noktası hangi ızgara hücresine düşüyorsa, **o hücre o nesneyi tahmin etmekten sorumludur**. Her hücre birden fazla kutu önerebilir, ancak her kutunun sorumluluğu çapa (anchor) mekanizması veya doğrudan regresyon ile belirlenir.

```
┌───┬───┬───┬───┐
│   │   │ ● │   │  ← Nesnenin merkezi (●) bu hücreye düşer
├───┼───┼───┼───┤     → Bu hücre kutu + sınıf tahmini yapar
│   │   │   │   │
├───┼───┼───┼───┤
│   │   │   │   │
├───┼───┼───┼───┤
│   │   │   │   │
└───┴───┴───┴───┘
      S × S ızgara
```

---

## 2. YOLOv3 Mimarisi (2018)

### Omurga (Backbone): Darknet-53
- 53 katmanlı konvolüsyonel ağ
- Artık bağlantılar (residual connections) kullanır
- ImageNet üzerinde ön eğitimli

### Boyun (Neck): Özellik Piramidi Ağı (FPN)
- Üç farklı ölçekte tahmin: **büyük** (13×13), **orta** (26×26), **küçük** (52×52) nesneler için
- Alt katmanlardaki ince ayrıntılar + üst katmanlardaki semantik bilgi birleştirilir

### Kafa (Head): Çapa Tabanlı (Anchor-Based)
- Her ölçek seviyesinde **3 önceden tanımlanmış çapa kutusu** kullanılır (toplamda 9)
- Tahmin: `(tx, ty, tw, th, objectness, class_probs)` — çapa kutusuna göre ofsetler
- Eğitim sırasında her gerçek kutu (ground truth), en yüksek IoU'ya sahip çapa ile eşleştirilir

### K-Means ile Çapa Belirleme
- Çapa boyutları, eğitim veri setindeki kutu boyutlarına **k-means** kümeleme uygulanarak belirlenir
- COCO veri seti için tipik çapalar: `(10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198), (373,326)`

---

## 3. YOLOv4 ve YOLOv5 Notları (2020)

### YOLOv4 (Bochkovskiy et al.)
- **CSPDarknet53** omurga: Cross Stage Partial bağlantıları ile daha verimli gradyan akışı
- **SPP** (Spatial Pyramid Pooling): farklı ölçeklerde özellik havuzlama
- **PAN** (Path Aggregation Network): yukarıdan aşağı + aşağıdan yukarı özellik birleştirme
- "Bag of Freebies" (eğitim hileleri): CutMix, Mosaic augmentation, DropBlock
- "Bag of Specials" (mimari iyileştirmeler): Mish aktivasyon, SAM dikkat modülü

### YOLOv5 (Ultralytics – Jocher)
- **PyTorch** implementasyonu (orijinal YOLO C/Darknet tabanlıydı)
- Kolay CLI arayüzü, otomatik veri artırma
- Focus katmanı, C3 (CSP Bottleneck with 3 convolutions) modülleri
- Model çeşitliliği: **n (nano)** / **s (small)** / **m (medium)** / **l (large)** / **x (extra-large)**
- Pratik üretim kullanımı için optimize edilmiş: model dışa aktarma (ONNX, TensorRT, CoreML)

---

## 4. YOLOv7 Öne Çıkan Özellikler (2022)

- **E-ELAN** (Extended Efficient Layer Aggregation Network): katman birleştirmede verimliliği artırır
- **Yeniden parametrelendirme** (re-parameterization): eğitim sırasında karmaşık yapı, çıkarım sırasında basit yapı → hız kazancı
- Ek kafa (auxiliary head) ile eğitim: ana kafanın performansını artırır, çıkarımda ek maliyet yok
- Etiket atama (label assignment) stratejisi: dinamik etiket atama ile eğitim kalitesi artırılmış

---

## 5. YOLOv8 Mimarisi (2023 – Ultralytics)

YOLOv8, YOLO ailesinde birçok önemli mimari değişiklik getirmiştir:

### 5.1 Çapa-Sız (Anchor-Free) Tespit Kafası

YOLOv8, çapa kutularını **tamamen kaldırmıştır**. Bunun yerine:
- Her ızgara hücresi, kutu merkezine olan **doğrudan uzaklıkları** (sol, üst, sağ, alt → LTRB) tahmin eder
- **Görev Hizalı Atama** (Task-Aligned Assignment, TAL) ile pozitif örnekler belirlenir
- Bu yaklaşım, çapa tasarımı ve eşleştirme karmaşıklığını ortadan kaldırır

### 5.2 Ayrıştırılmış Kafa (Decoupled Head)

- Sınıflandırma ve regresyon görevleri **ayrı dallara** ayrılmıştır
- Her dal kendi konvolüsyon katmanlarına sahiptir
- Bu ayrım, çakışan gradyanları önler ve her görevin bağımsız optimize edilmesini sağlar

```
Neck çıktısı
     │
     ├──► Sınıflandırma dalı → Conv → Conv → sınıf logitleri
     │
     └──► Regresyon dalı     → Conv → Conv → bbox (LTRB) + kalite
```

### 5.3 C2f Modülü

- **C2f** (Cross Stage Partial with 2 convolutions): YOLOv5'teki C3 modülünün geliştirilmiş hali
- Daha fazla gradyan akış yolu → daha zengin özellik öğrenimi
- Hesaplama maliyeti benzer, doğruluk daha yüksek

### 5.4 Birleşik Ultralytics API

YOLOv8, tek bir API üzerinden dört farklı görevi destekler:
- **Tespit** (detection): `yolov8n.pt`
- **Segmentasyon** (segmentation): `yolov8n-seg.pt`
- **Poz tahmini** (pose estimation): `yolov8n-pose.pt`
- **Sınıflandırma** (classification): `yolov8n-cls.pt`

Ayrıca yerleşik takip desteği (ByteTrack, BotSORT) bulunur.

### 5.5 Kayıp Fonksiyonları

| Kayıp | Açıklama |
|-------|----------|
| CIoU Loss | Kutu regresyonu – örtüşme, merkez mesafesi ve en-boy oranını dikkate alır |
| BCE Loss | Sınıflandırma – ikili çapraz entropi |
| DFL (Distribution Focal Loss) | Regresyon kalite tahmini – kutu kenar dağılımını öğrenir |

---

## 6. Çapa Tabanlı vs Çapa-Sız Karşılaştırma

### Çapa (Anchor) Nedir?

Çapa kutuları, modelin tahminlerine başlangıç noktası olarak kullanılan **önceden tanımlanmış referans kutulardır**. Model, doğrudan kutu koordinatlarını değil, çapaya göre **ofsetleri** tahmin eder:
- `bx = σ(tx) + cx` (merkez x)
- `by = σ(ty) + cy` (merkez y)
- `bw = pw · e^tw` (genişlik, pw = çapa genişliği)
- `bh = ph · e^th` (yükseklik, ph = çapa yüksekliği)

### Neden Kullanıldılar?

- Regresyon görevini kolaylaştırır: model küçük ofsetler öğrenir, sıfırdan kutu boyutu tahmin etmez
- Farklı en-boy oranlarını (uzun, kısa, kare) daha iyi yakalar
- Eğitim yakınsamasını hızlandırır

### Karşılaştırma Tablosu

| Özellik | Çapa Tabanlı (v3-v5) | Çapa-Sız (v8) |
|---------|----------------------|---------------|
| **Ön tanımlı kutular** | Evet – k-means ile belirlenir | Yok |
| **Eşleştirme** | IoU tabanlı, karmaşık | TAL ile basitleştirilmiş |
| **Hiperparametre** | Çapa boyutları ayrı tasarlanmalı | Daha az hiperparametre |
| **NMS hassasiyeti** | Yüksek | Düşük |
| **Küçük nesne başarısı** | Çapa tasarımına bağlı | Genel olarak daha iyi |
| **Eğitim karmaşıklığı** | Daha karmaşık | Daha basit |
| **Doğruluk** | Rekabetçi | Hafif üstün |

---

## 7. Versiyon Karşılaştırma Özet Tablosu

| Özellik | YOLOv3 | YOLOv4 | YOLOv5 | YOLOv7 | YOLOv8 |
|---------|--------|--------|--------|--------|--------|
| **Yıl** | 2018 | 2020 | 2020 | 2022 | 2023 |
| **Framework** | Darknet (C) | Darknet (C) | PyTorch | PyTorch | PyTorch |
| **Omurga** | Darknet-53 | CSPDarknet53 | CSPDarknet | E-ELAN | CSPDarknet (C2f) |
| **Boyun** | FPN | SPP + PAN | PAN-FPN | E-ELAN | PAN-FPN |
| **Çapa** | ✓ (9 çapa) | ✓ (9 çapa) | ✓ (9 çapa) | ✓ | ✗ (anchor-free) |
| **Kafa** | Birleşik | Birleşik | Birleşik | Birleşik | Ayrıştırılmış |
| **Görev** | Tespit | Tespit | Tespit | Tespit | Tespit+Seg+Poz+Cls |
| **mAP (COCO)** | ~33 | ~43 | ~37-50 | ~51 | ~37-54 |
| **Hız (nano)** | Orta | Orta | Hızlı | Hızlı | Çok hızlı |

> *mAP değerleri model boyutuna (n/s/m/l/x) göre değişir. Tablodaki değerler aralık olarak verilmiştir.*

---

## 8. Pratik Değerlendirmeler

### Hız vs Doğruluk Dengesi

| Model | Parametre (M) | mAP@50-95 | Gecikme (ms, A100) |
|-------|--------------|-----------|---------------------|
| yolov8n | 3.2 | 37.3 | ~6 |
| yolov8s | 11.2 | 44.9 | ~10 |
| yolov8m | 25.9 | 50.2 | ~20 |
| yolov8l | 43.7 | 52.9 | ~35 |
| yolov8x | 68.2 | 53.9 | ~55 |

### Dağıtım (Deployment) Hususları

- **Gerçek zamanlı uygulamalar** (güvenlik kameraları, IoT): yolov8n + TensorRT dışa aktarma
- **Kenar cihazlar** (Jetson Nano, Raspberry Pi): yolov8n + ONNX veya TFLite
- **Sunucu tarafı** (bulut işleme): yolov8m/l + GPU batch inference
- **Mobil uygulamalar**: yolov8n + CoreML (iOS) veya TFLite (Android)

### Bu Projede Kullanılan Seçim

Bu projede **yolov8n** (nano) modeli tercih edilmiştir çünkü:
1. Bir dizüstü bilgisayar CPU'sunda bile gerçek zamanlı çalışabilir (~15-25 FPS)
2. Model dosya boyutu küçük (~6 MB): hızlı indirme ve yükleme
3. Staj projesi kapsamında doğruluk yeterli (mAP 37.3)
4. Birleşik Ultralytics API sayesinde aynı altyapıyla tespit + poz + segmentasyon

---

## 9. Madde Özeti

- **YOLOv3**: İlk pratik kullanılabilir çok ölçekli YOLO. Darknet-53 omurga, çapa tabanlı, FPN ile 3 ölçek.
- **YOLOv4/v5**: Eğitim hilelerinin ve mimari iyileştirmelerin sistematik uygulanması. PyTorch geçişi.
- **YOLOv7**: E-ELAN ile verimli özellik birleştirme, yeniden parametrelendirme ile hız artışı.
- **YOLOv8**: Çapa-sız, ayrıştırılmış kafa. Dört görevi (tespit, segmentasyon, poz, sınıflandırma) tek API altında birleştirir.
- **Çapa-sız yaklaşım**: Daha az hiperparametre, daha basit eğitim, benzer veya daha iyi doğruluk.
- **Pratik seçim**: Gerçek zamanlı ihtiyaçlar için nano modeller; doğruluk gerektiğinde medium/large modellerle GPU üzerinde çalışma.

---

## Kaynaklar

1. Redmon, J., et al. "You Only Look Once" (2016)
2. Redmon, J., & Farhadi, A. "YOLOv3: An Incremental Improvement" (2018)
3. Bochkovskiy, A., et al. "YOLOv4" (2020)
4. Jocher, G., et al. Ultralytics YOLOv5 / YOLOv8. https://github.com/ultralytics/ultralytics
5. Wang, C.-Y., et al. "YOLOv7: Trainable bag-of-freebies" (2022)
