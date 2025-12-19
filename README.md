# 🖼️ Bilgisayarlı Görü ve Derin Öğrenme Projeleri

Bu depo, **klasik görüntü işlemeden derin öğrenmeye** kadar adım adım ilerleyen mini projeleri içerir. Her proje bağımsız çalışabilir ve **Türkçe yorumlarla** açıklanmıştır.

> 🎯 **Amaç:** OpenCV ve PyTorch kullanarak görüntü işleme temellerini öğrenmek

---

## 📦 Kurulum

```bash
# 1. Sanal ortam oluştur
python -m venv .venv

# 2. Aktive et (Windows PowerShell)
.venv\Scripts\activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
```

**Gereksinimler:** `numpy`, `opencv-python`, `matplotlib`, `torch`, `torchvision`, `tensorflow/keras` (CNN eğitimi için)

---

## 🗂️ Proje Yapısı ve Öğrenme Yolu

```
imageprocessing/
│
├── 📁 03_opencv_giris/          # ADIM 1: OpenCV'ye giriş
├── 📁 04_gaussian_blur_opencv/  # ADIM 2: Bulanıklaştırma temelleri
├── 📁 05_gaussian_blur_manual/  # ADIM 3: Filtrelerin matematiği
├── 📁 06_traditional_image_processing/  # ADIM 4: Geleneksel yöntemler
├── 📁 07_keypoints_features/    # ADIM 5: Özellik çıkarımı
├── 📁 08_cnn_intro/             # ADIM 6: CNN'e giriş
│
├── 📄 utils.py                  # ADIM 7: Ortak yardımcılar
├── 📄 project_1_similarity.py   # ADIM 8: Keypoint benzerliği
├── 📄 project_2_edges.py        # ADIM 9: Kural tabanlı sınıflandırma
├── 📄 project_3_cnn_ready.py    # ADIM 10: Hazır CNN çıkarımı
├── 📄 project_4_compare.py      # ADIM 11: Sınıflandırma vs Tespit
│
├── 📄 requirements.txt          # Bağımlılıklar
├── 📄 PROJE_ANLATIMI.md         # Detaylı anlatım (1. şahıs)
├── 📄 SPECIAL_USAGE_README.md   # Özel kullanımlar sözlüğü
└── 📄 README.md                 # Bu dosya
```

---

## 📚 ADIM ADIM PROJELER

### 🔹 ADIM 1: OpenCV'ye Giriş
**Klasör:** `03_opencv_giris/`

| Dosya | Ne Öğrendim | Kullandığım Fonksiyonlar |
|-------|-------------|-------------------------|
| `webcam_filter.py` | Kameradan canlı görüntü alma, tuşla filtre değiştirme | `cv2.VideoCapture`, `cv2.cvtColor`, `cv2.GaussianBlur`, `cv2.resize`, `cv2.waitKey` |

**Çalıştırma:**
```bash
python 03_opencv_giris/webcam_filter.py
# Tuşlar: c=normal, g=gri, b=blur, r=yarı çözünürlük, q=çık
```

---

### 🔹 ADIM 2: Gaussian Blur (OpenCV)
**Klasör:** `04_gaussian_blur_opencv/`

| Dosya | Ne Öğrendim | Kullandığım Fonksiyonlar |
|-------|-------------|-------------------------|
| `gaussian_blur_app.py` | Trackbar ile canlı parametre ayarlama | `cv2.createTrackbar`, `cv2.getTrackbarPos`, `cv2.GaussianBlur`, `np.hstack` |

**Çalıştırma:**
```bash
python 04_gaussian_blur_opencv/gaussian_blur_app.py --image foto.jpg
```

---

### 🔹 ADIM 3: Manuel Konvolüsyon
**Klasör:** `05_gaussian_blur_manual/`

| Dosya | Ne Öğrendim | Kullandığım Fonksiyonlar |
|-------|-------------|-------------------------|
| `gaussian_blur_manual.py` | Elle 2D Gaussian kernel oluşturma, konvolüsyon | `np.meshgrid`, `np.exp`, `np.pad`, manuel döngü |
| `gaussian_blur_scipy.py` | SciPy ile konvolüsyon | `scipy.signal.convolve2d` |
| `all_filters_demo.py` | Farklı filtreleri karşılaştırma | `cv2.GaussianBlur`, `cv2.medianBlur`, `cv2.bilateralFilter` |
| `filters_from_blog.py` | Sobel, Laplacian, keskinleştirme | `cv2.boxFilter`, `cv2.Sobel`, `cv2.Laplacian`, `cv2.addWeighted` |

**Çalıştırma:**
```bash
python 05_gaussian_blur_manual/gaussian_blur_manual.py --image foto.jpg --ksize 5 --sigma 1.0
python 05_gaussian_blur_manual/all_filters_demo.py --image foto.jpg
```

---

### 🔹 ADIM 4: Geleneksel Görüntü İşleme
**Klasör:** `06_traditional_image_processing/`

| Dosya | Ne Öğrendim | Kullandığım Fonksiyonlar |
|-------|-------------|-------------------------|
| `coin_counter.py` | CLAHE, eşikleme, morfoloji, watershed, kontur analizi | `cv2.createCLAHE`, `cv2.adaptiveThreshold`, `cv2.morphologyEx`, `cv2.distanceTransform`, `cv2.watershed`, `cv2.findContours`, `cv2.Canny` |

**Çalıştırma:**
```bash
python 06_traditional_image_processing/coin_counter.py --image coins.jpg --watershed --show
```

**Örnek çıktı:** Para sayısını tespit edip kutular içinde gösterir.

---

### 🔹 ADIM 5: Keypoint ve Özellik Eşleştirme
**Klasör:** `07_keypoints_features/`

| Dosya | Ne Öğrendim | Kullandığım Fonksiyonlar |
|-------|-------------|-------------------------|
| `logo_match.py` | ORB/SIFT keypoint, Lowe oran testi, homografi | `cv2.ORB_create`, `cv2.SIFT_create`, `cv2.BFMatcher`, `cv2.findHomography`, `cv2.perspectiveTransform`, `cv2.drawMatches` |

**Çalıştırma:**
```bash
python 07_keypoints_features/logo_match.py --template logo.png --scene sahne.jpg --feature orb
```

---

### 🔹 ADIM 6: CNN'e Giriş (Eğitim)
**Klasör:** `08_cnn_intro/`

| Dosya | Ne Öğrendim | Kullandığım Fonksiyonlar |
|-------|-------------|-------------------------|
| `mnist_cnn.py` | CNN mimarisi, eğitim döngüsü, doğrulama | `keras.Sequential`, `Conv2D`, `MaxPool2D`, `Dense`, `model.fit`, `model.evaluate` |

**Çalıştırma:**
```bash
python 08_cnn_intro/mnist_cnn.py --epochs 20 --batch-size 128
```

**Eğitim sonucu:** ~%99 doğruluk, kayıp/doğruluk grafikleri

---

### 🔹 ADIM 7: Ortak Yardımcı Fonksiyonlar
**Dosya:** `utils.py`

| Fonksiyon | Açıklama |
|-----------|----------|
| `load_image_bgr(path)` | Unicode yollarda bile çalışan güvenli görüntü yükleme (`cv2.imdecode` fallback) |
| `bgr_to_rgb(img)` | BGR → RGB dönüşümü |
| `to_torch_tensor(img)` | NumPy → PyTorch tensör (HWC→CHW, /255) |
| `normalize_tensor(tensor, mean, std)` | ImageNet normalizasyonu |
| `resize_rgb(img, size)` | Boyut değiştirme |
| `draw_boxes(img, boxes, labels, scores)` | Detection kutuları çizme |
| `show_image(title, img)` | Yeniden boyutlanabilir pencerede gösterme |

---

### 🔹 ADIM 8: Görüntü Benzerliği (ORB)
**Dosya:** `project_1_similarity.py`

**Ne yapıyor:**
1. İki görüntüyü yükler
2. Gri tona çevirir
3. ORB ile keypoint ve tanımlayıcı çıkarır
4. BFMatcher ile eşleştirir
5. Lowe oran testi uygular
6. İyi eşleşme sayısına göre BENZER/BENZEMİYOR der

**Çalıştırma:**
```bash
python project_1_similarity.py resim1.jpg resim2.jpg --show --ratio 0.75 --min-matches 20
```

**Öğrenilen kavramlar:** Keypoint, Descriptor, Lowe ratio test

---

### 🔹 ADIM 9: Kural Tabanlı Sınıflandırma
**Dosya:** `project_2_edges.py`

**Ne yapıyor:**
1. Görüntüyü yükler ve gri yapar
2. Gaussian blur uygular
3. Canny kenar tespiti yapar
4. Kenar piksellerini sayar
5. Eşiğe göre EMPTY/NOT EMPTY der

**Çalıştırma:**
```bash
python project_2_edges.py raf.jpg --show --edge-thresh 500
```

**Öğrenilen kavramlar:** Elle özellik tanımlama, kural tabanlı sistemlerin sınırları, CNN motivasyonu

---

### 🔹 ADIM 10: Hazır CNN ile Sınıflandırma
**Dosya:** `project_3_cnn_ready.py`

**Ne yapıyor:**
1. MobileNet veya ResNet yükler (ön eğitimli)
2. Görüntüyü 224×224'e boyutlandırır
3. ImageNet normalizasyonu uygular
4. Çıkarım yapar (eğitim YOK)
5. Top-K tahminleri yazdırır

**Çalıştırma:**
```bash
python project_3_cnn_ready.py kopek.jpg --model mobilenet --topk 5
```

**Öğrenilen kavramlar:** Transfer öğrenme, çıkarım vs eğitim, softmax olasılıkları

---

### 🔹 ADIM 11: Sınıflandırma vs Nesne Tespiti
**Dosya:** `project_4_compare.py`

**Ne yapıyor:**
1. **Sınıflandırma:** ResNet ile "bu ne?" sorusuna cevap
2. **Tespit:** Faster R-CNN ile "nerede ne var?" sorusuna cevap + kutular

**Çalıştırma:**
```bash
python project_4_compare.py sokak.jpg --score 0.5 --show
```

**Öğrenilen kavramlar:**
| Sınıflandırma | Nesne Tespiti |
|---------------|---------------|
| Tek etiket | Birden fazla kutu + etiket |
| Konum yok | Bounding box koordinatları |
| Hafif | Ağır (FPN, RPN, NMS) |

---

## 📄 Dokümantasyon Dosyaları

| Dosya | İçerik |
|-------|--------|
| `PROJE_ANLATIMI.md` | 4 yeni projenin 1. şahıs ağzından adım adım anlatımı |
| `SPECIAL_USAGE_README.md` | Tüm dosyalardaki özel OpenCV/PyTorch kullanımlarının sözlüğü |

---

## 🎯 Öğrenme Akışı Özeti

```
TEMEL OpenCV                    GELENEKSEL CV                    DERİN ÖĞRENME
     │                               │                                │
     ▼                               ▼                                ▼
┌─────────────┐              ┌─────────────┐              ┌─────────────────┐
│ webcam_filter│              │coin_counter │              │   mnist_cnn     │
│ gaussian_blur│     ───►    │ logo_match  │     ───►    │ project_3_cnn   │
│ manuel filtre│              │project_1,2  │              │ project_4_compare│
└─────────────┘              └─────────────┘              └─────────────────┘

   cv2.resize                   cv2.findContours               torch.no_grad()
   cv2.cvtColor                 cv2.ORB_create                  model.eval()
   cv2.GaussianBlur             cv2.BFMatcher                   softmax
```

---

## 🚀 Hızlı Başlangıç Komutları

```bash
# Webcam filtresi
python 03_opencv_giris/webcam_filter.py

# İki resim benzerliği
python project_1_similarity.py resim1.jpg resim2.jpg --show

# Kenar tabanlı sınıflandırma
python project_2_edges.py foto.jpg --show

# CNN ile nesne tanıma
python project_3_cnn_ready.py kopek.jpg --model resnet

# Sınıflandırma vs Detection karşılaştırması
python project_4_compare.py sokak.jpg --show
```

---

## 📝 Sonraki Adımlar

- [ ] YOLO ile gerçek zamanlı nesne tespiti
- [ ] Kendi veri setiyle fine-tuning
- [ ] Segmentasyon (piksel bazlı maske)
- [ ] Video analizi ve nesne takibi

---

*Bu proje, bilgisayarlı görü öğrenme yolculuğumun bir kaydıdır. Her dosya Türkçe yorumlarla açıklanmıştır.*
