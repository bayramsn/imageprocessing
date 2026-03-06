# 🖼️ Bilgisayarlı Görü ve Derin Öğrenme Projeleri

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Bu depo, **klasik görüntü işlemeden derin öğrenmeye** kadar adım adım ilerleyen mini projeleri içerir. Her proje bağımsız çalışabilir ve **Türkçe yorumlarla** açıklanmıştır.

> 🎯 **Amaç:** OpenCV ve PyTorch kullanarak görüntü işleme temellerini öğrenmek

> 📁 **Not:** Bu repo iki katmandan oluşur: kökte eğitim/demolar bulunur, [imageprocessing/README.md](imageprocessing/README.md) altında ise gerçek zamanlı insan davranışı analitiği için ayrı paketlenmiş bir uygulama yer alır.

---

## 🚀 Hızlı Başlangıç

### 🖥️ Grafik Arayüz (Önerilen)

Tüm projeleri tek bir yerden yönetmek için GUI başlatıcıyı kullanabilirsiniz:

```bash
python app_launcher.py
```

### ⌨️ Terminal Kullanımı

Manuel kurulum ve çalıştırma için:

```bash
# 1. Repoyu klonla
git clone https://github.com/bayramsn/imageprocessing.git
cd imageprocessing

# 2. Sanal ortam oluştur
python -m venv .venv

# 3. Aktive et (Windows PowerShell)
.venv\Scripts\activate

# 4. Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## 📦 Gereksinimler

| Paket           | Versiyon | Kullanım         |
| --------------- | -------- | ---------------- |
| `numpy`         | ≥1.21    | Matris işlemleri |
| `opencv-python` | ≥4.5     | Görüntü işleme   |
| `matplotlib`    | ≥3.4     | Görselleştirme   |
| `torch`         | ≥2.0     | Derin öğrenme    |
| `torchvision`   | ≥0.15    | Hazır modeller   |
| `tensorflow`    | ≥2.10    | CNN eğitimi      |
| `scipy`         | ≥1.7     | Sinyal işleme    |
| `Pillow`        | ≥10.0    | Görsel yükleme   |
| `PyYAML`        | ≥6.0     | Konfigürasyon    |
| `ultralytics`   | ≥8.0     | YOLOv8 pipeline  |
| `pytest`        | ≥7.0     | Testler          |

---

## 🗂️ Proje Yapısı

```
imageprocessing/
│
├── 📁 03_opencv_giris/              # OpenCV temelleri
│   ├── webcam_filter.py
│   └── webcam_paint.py              # 🆕 Sanal çizim tahtası
│
├── 📁 04_gaussian_blur_opencv/      # Blur türleri karşılaştırması
│   ├── blur_comparison.py
│   └── tilt_shift_effect.py         # 🆕 Minyatür şehir efekti
│
├── 📁 05_gaussian_blur_manual/      # Kernel ve convolution
│   ├── custom_gaussian.py
│   └── kernel_playground.py         # 🆕 Filtre bahçesi (Sharpen/Emboss)
│
├── 📁 06_traditional_image_processing/
│   ├── preprocessing_tool.py
│   └── shape_detector.py            # 🆕 Geometrik şekil tespiti
│
├── 📁 07_keypoints_features/        # Feature matching
│   ├── feature_matcher.py
│   └── panorama_maker.py            # 🆕 Panorama oluşturucu
│
├── 📁 08_cnn_intro/                 # CNN eğitimi
│   ├── mnist_cnn.py
│   └── data_augmentation_demo.py    # 🆕 Veri çoğaltma demosu
│
├── 📁 09_numpy_matplotlib/          # Matris analizi
│   ├── image_analyzer.py
│   └── color_distribution_3d.py     # 🆕 3D renk analizi
│
├── 📁 10_detection_segmentation/
│   ├── compare_tasks.py
│   └── face_eye_detector.py         # 🆕 Yüz ve göz tespiti
│
├── 📄 app_launcher.py               # 🚀 GUI Başlatıcı (Tüm projeler için)
├── 📄 YENI_ORNEKLER.md              # 📚 Yeni örneklerin detaylı anlatımı
├── 📄 PROJE_ANLATIMI.md             # Orijinal projelerin hikayesi
├── 📄 requirements.txt
├── 📄 SPECIAL_USAGE_README.md
└── 📄 README.md
```

---

## 📚 Öğrenme Yolu

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ÖĞRENME HARİTASI                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  🟢 BAŞLANGIÇ          🟡 ORTA SEVİYE           🔴 İLERİ SEVİYE            │
│                                                                             │
│  03_opencv_giris       05_gaussian_manual       08_cnn_intro               │
│       ↓                      ↓                       ↓                     │
│  04_gaussian_blur      06_traditional           09_numpy_matplotlib        │
│       ↓                      ↓                       ↓                     │
│  project_2_edges       07_keypoints             10_detection_segmentation  │
│                              ↓                       ↓                     │
│                        project_1_similarity     project_3_cnn              │
│                                                      ↓                     │
│                                                 project_4_compare          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📖 Proje Detayları

### 🟢 Başlangıç Seviyesi

<details>
<summary><b>03 - OpenCV Giriş (Webcam Laboratuvarı)</b></summary>

**Amaç:** OpenCV'nin temel yapı taşlarını öğrenmek

| Dosya              | Açıklama                                                    |
| ------------------ | ----------------------------------------------------------- |
| `webcam_filter.py` | Tuşla filtre değiştirme (normal/gri/blur/resize)            |
| `webcam_fps.py`    | FPS gösterimi + ekstra filtreler (cartoon, sepia, negative) |
| `webcam_paint.py`  | **(Yeni)** Sanal çizim tahtası (Webcam ile çizim)           |

```bash
python 03_opencv_giris/webcam_filter.py
# Tuşlar: c=normal, g=gri, b=blur, r=yarı çözünürlük, q=çık
```

**Öğrenilen:** `cv2.VideoCapture`, `cv2.imshow`, `cv2.setMouseCallback`

</details>

<details>
<summary><b>04 - Gaussian Blur (Blur Karşılaştırma)</b></summary>

**Amaç:** Farklı blur türlerini anlamak ve karşılaştırmak

| Dosya                  | Açıklama                                      |
| ---------------------- | --------------------------------------------- |
| `gaussian_blur_app.py` | Trackbar ile canlı parametre ayarlama         |
| `blur_comparison.py`   | Gaussian/Median/Bilateral/Box karşılaştırması |
| `tilt_shift_effect.py` | **(Yeni)** Minyatür şehir efekti oluşturma    |

```bash
python 04_gaussian_blur_opencv/blur_comparison.py resim.jpg --interactive
```

**Öğrenilen:** Blur teknikleri, maskeleme, doygunluk artırma

</details>

---

### 🟡 Orta Seviye

<details>
<summary><b>05 - Manuel Gaussian (Convolution Matematiği)</b></summary>

**Amaç:** CNN'in temelini oluşturan convolution'ı sıfırdan yazmak

| Dosya                     | Açıklama                                   |
| ------------------------- | ------------------------------------------ |
| `gaussian_blur_manual.py` | Elle kernel oluşturma                      |
| `custom_gaussian.py`      | Benchmark + OpenCV karşılaştırması         |
| `kernel_playground.py`    | **(Yeni)** Özel filtreler (Sharpen/Emboss) |

```bash
python 05_gaussian_blur_manual/custom_gaussian.py resim.jpg --benchmark
```

**Öğrenilen:** Kernel nedir, convolution, filtre matrisleri

</details>

<details>
<summary><b>06 - Geleneksel Görüntü İşleme</b></summary>

**Amaç:** AI'sız klasik yöntemlerle sonuç almak

| Dosya                   | Açıklama                                        |
| ----------------------- | ----------------------------------------------- |
| `coin_counter.py`       | Para sayma (watershed)                          |
| `preprocessing_tool.py` | Threshold/Canny/Morphology karşılaştırması      |
| `shape_detector.py`     | **(Yeni)** Geometrik şekil tespiti (Kare/Daire) |

```bash
python 06_traditional_image_processing/preprocessing_tool.py resim.jpg --mode all
```

**Öğrenilen:** Threshold, Canny Edge, Contours, ApproxPolyDP

</details>

<details>
<summary><b>07 - Keypoint ve Özellik Çıkarımı</b></summary>

**Amaç:** Görüntüden ayırt edici noktalar çıkarmak

| Dosya                | Açıklama                                   |
| -------------------- | ------------------------------------------ |
| `logo_match.py`      | Logo eşleştirme                            |
| `feature_matcher.py` | ORB/SIFT/AKAZE karşılaştırması             |
| `panorama_maker.py`  | **(Yeni)** Panorama oluşturucu (Stitching) |

```bash
python 07_keypoints_features/feature_matcher.py resim1.jpg resim2.jpg --method all
```

**Öğrenilen:** Feature matching, Homography, Image Stitching

</details>

---

### 🔴 İleri Seviye

<details>
<summary><b>08 - CNN Giriş (Feature Map Görselleştirme)</b></summary>

**Amaç:** CNN'in içini "kara kutu" olmaktan çıkarmak

| Dosya                       | Açıklama                             |
| --------------------------- | ------------------------------------ |
| `mnist_cnn.py`              | MNIST üzerinde CNN eğitimi           |
| `cnn_visualizer.py`         | Feature map ve kernel görselleştirme |
| `data_augmentation_demo.py` | **(Yeni)** Veri çoğaltma teknikleri  |

```bash
python 08_cnn_intro/mnist_cnn.py --epochs 10
```

**Öğrenilen:** Conv2D, Torchvision Transforms, Augmentation

</details>

<details>
<summary><b>09 - NumPy & Matplotlib (Görüntü Analizi)</b></summary>

**Amaç:** Matris mantığını ve görselleştirmeyi öğrenmek

| Dosya                      | Açıklama                                            |
| -------------------------- | --------------------------------------------------- |
| `image_analyzer.py`        | Histogram, istatistikler, threshold karşılaştırması |
| `color_distribution_3d.py` | **(Yeni)** 3D RGB renk uzayı analizi                |

```bash
python 09_numpy_matplotlib/image_analyzer.py resim.jpg --demo
```

**Öğrenilen:** NumPy slicing, 3D Plotting, RGB uzayı

</details>

<details>
<summary><b>10 - Detection vs Segmentation</b></summary>

**Amaç:** Üç temel CV görevini karşılaştırmak

| Dosya                  | Açıklama                                       |
| ---------------------- | ---------------------------------------------- |
| `compare_tasks.py`     | Classification/Detection/Segmentation yan yana |
| `face_eye_detector.py` | **(Yeni)** Haar Cascade ile yüz/göz tespiti    |

```bash
python 10_detection_segmentation/compare_tasks.py resim.jpg --save sonuc.png
```

**Öğrenilen:** Haar Cascades, Object Detection, ROI

</details>

---

## 🎯 Bağımsız Projeler

| Proje                     | Açıklama                      | Komut                                                 |
| ------------------------- | ----------------------------- | ----------------------------------------------------- |
| `project_1_similarity.py` | ORB ile görüntü benzerliği    | `python project_1_similarity.py r1.jpg r2.jpg --show` |
| `project_2_edges.py`      | Kenar tabanlı EMPTY/NOT EMPTY | `python project_2_edges.py raf.jpg --show`            |
| `project_3_cnn_ready.py`  | Hazır CNN ile sınıflandırma   | `python project_3_cnn_ready.py kopek.jpg`             |
| `project_4_compare.py`    | Classification vs Detection   | `python project_4_compare.py sokak.jpg --show`        |

---

## 📊 Teknoloji Karşılaştırması

| Görev            | Klasik Yöntem          | Derin Öğrenme      |
| ---------------- | ---------------------- | ------------------ |
| Kenar tespiti    | `cv2.Canny`            | Conv2D katmanı     |
| Özellik çıkarımı | ORB, SIFT              | CNN feature maps   |
| Sınıflandırma    | Kural tabanlı          | ResNet, MobileNet  |
| Nesne tespiti    | Kontur analizi         | YOLO, Faster R-CNN |
| Segmentasyon     | Threshold + Morphology | U-Net, DeepLab     |

---

## 📄 Dokümantasyon

| Dosya                                                | İçerik                                                       |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| [`SPECIAL_USAGE_README.md`](SPECIAL_USAGE_README.md) | Tüm dosyalardaki özel OpenCV/PyTorch kullanımlarının sözlüğü |
| [`PROJE_ANLATIMI.md`](PROJE_ANLATIMI.md)             | Detaylı proje anlatımı (1. şahıs)                            |
| [`YENI_ORNEKLER.md`](YENI_ORNEKLER.md)               | **Yeni eklenen** uygulama örneklerinin detaylı açıklamaları  |
| Her klasördeki `README.md`                           | Proje bazlı detaylı dokümantasyon                            |

---

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Commit yapın (`git commit -m 'Yeni özellik eklendi'`)
4. Push yapın (`git push origin feature/yeni-ozellik`)
5. Pull Request açın

---

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

---

## 📬 İletişim

**GitHub:** [@bayramsn](https://github.com/bayramsn)

---

<p align="center">
  <i>Bu proje, bilgisayarlı görü öğrenme yolculuğumun bir kaydıdır. Her dosya Türkçe yorumlarla açıklanmıştır.</i>
</p>
