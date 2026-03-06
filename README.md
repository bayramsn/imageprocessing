# 🖼️ Bilgisayarlı Görü, Derin Öğrenme ve OCR Projeleri

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Tesseract](https://img.shields.io/badge/Tesseract-5.x-blueviolet.svg)](https://github.com/tesseract-ocr/tesseract)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Bu depo, **klasik görüntü işlemeden derin öğrenmeye** ve **OCR'a** kadar adım adım ilerleyen projeleri içerir. Her modül bağımsız çalışabilir ve **Türkçe yorumlarla** açıklanmıştır.

> 🎯 **Amaç:** OpenCV, PyTorch ve Tesseract kullanarak görüntü işleme, bilgisayarlı görü ve metin tanıma temellerini öğrenmek  
> 🗂️ **Kapsam:** 3 bağımsız alt proje — CV öğrenme modülleri · davranış analitiği pipeline · OCR masaüstü uygulaması

---

## 📂 Repo Haritası

```
imageproccesing/
│
├── 📁 03_opencv_giris/              # OpenCV temelleri & webcam lab
├── 📁 04_gaussian_blur_opencv/      # Blur türleri & tilt-shift efekti
├── 📁 05_gaussian_blur_manual/      # Convolution matematiği sıfırdan
├── 📁 06_traditional_image_processing/  # AI'sız klasik yöntemler
├── 📁 07_keypoints_features/        # Feature matching & panorama
├── 📁 08_cnn_intro/                 # CNN eğitimi & görselleştirme
├── 📁 09_numpy_matplotlib/          # Matris analizi & 3D renk uzayı
├── 📁 10_detection_segmentation/    # Yüz tespiti & görev karşılaştırması
│
├── 📁 cv_projects/                  # Bağımsız CV projeleri (ORB, Kenar, CNN)
│
├── 📁 imageprocessing/              # 🔬 Gerçek zamanlı insan davranışı analitiği
│   ├── src/pipeline/               #    YOLOv8 · ByteTrack · COCO-17 pose
│   ├── configs/                    #    YAML konfigürasyon
│   └── tests/                      #    pytest test paketi (24 test)
│
├── 📁 teserract/                    # 📝 OCR masaüstü uygulaması
│   ├── src/                        #    Basit→ileri OCR modülleri
│   ├── streamlit_app.py            #    Web arayüzü (Streamlit)
│   ├── desktop_app.py              #    Tkinter masaüstü GUI
│   └── docs/                       #    Kapsamlı dokümantasyon
│
├── 🚀 app_launcher.py              # Tkinter GUI – tüm modülleri başlatır
├── 📄 requirements.txt
└── 📄 README.md
```

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
| `numpy`         | ≥1.24    | Matris işlemleri                  |
| `opencv-python` | ≥4.8     | Görüntü işleme & kamera           |
| `matplotlib`    | ≥3.7     | Görselleştirme                    |
| `torch`         | ≥2.0     | Derin öğrenme                     |
| `torchvision`   | ≥0.15    | Hazır modeller & veri artırma     |
| `tensorflow`    | ≥2.10    | CNN eğitimi (MNIST)               |
| `scipy`         | ≥1.10    | Sinyal/görüntü işleme             |
| `Pillow`        | ≥10.0    | Görüntü yükleme/kaydetme          |
| `PyYAML`        | ≥6.0     | Pipeline konfigürasyonu           |
| `ultralytics`   | ≥8.0     | YOLOv8 tespit & segmentasyon      |
| `pytesseract`   | ≥0.3     | Tesseract OCR bağlayıcısı         |
| `streamlit`     | ≥1.30    | OCR web arayüzü                   |
| `pytest`        | ≥7.0     | Otomatik test                     |

---

## � Öğrenme Modülleri (03–10)

---

## �️ Öğrenme Yolu

```
🟢 BAŞLANGIÇ             🟡 ORTA SEVİYE              🔴 İLERİ SEVİYE
─────────────────────    ─────────────────────────    ──────────────────────────────
03_opencv_giris          05_gaussian_manual           08_cnn_intro
       ↓                        ↓                            ↓
04_gaussian_blur         06_traditional               09_numpy_matplotlib
       ↓                        ↓                            ↓
project_2_edges          07_keypoints_features        10_detection_segmentation
                                ↓                            ↓
                         project_1_similarity         project_3_cnn_ready
                                                             ↓
                                                     imageprocessing/ pipeline
                                                             ↓
                                                     teserract/ OCR studio
```

---

### 🟢 Başlangıç Seviyesi

<details>
<summary><b>03 — OpenCV Giriş · Webcam Laboratuvarı</b></summary>

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
<summary><b>04 — Gaussian Blur · Blur Türleri Karşılaştırması</b></summary>

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
<summary><b>05 — Manuel Gaussian · Convolution Matematiği</b></summary>

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
<summary><b>06 — Geleneksel Görüntü İşleme · AI'sız Klasik Yöntemler</b></summary>

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
<summary><b>07 — Keypoint & Özellik Çıkarımı · Feature Matching</b></summary>

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
<summary><b>08 — CNN Giriş · Feature Map Görselleştirme</b></summary>

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
<summary><b>09 — NumPy & Matplotlib · Görüntü Analizi</b></summary>

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
<summary><b>10 — Detection & Segmentation · Görev Karşılaştırması</b></summary>

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

## 🎯 Bağımsız Projeler (`cv_projects/`)

| Proje                     | Açıklama                                   | Komut                                                         |
| ------------------------- | ------------------------------------------ | ------------------------------------------------------------- |
| `project_1_similarity.py` | ORB ile iki görüntü benzerlik skoru        | `python cv_projects/project_1_similarity.py r1.jpg r2.jpg`    |
| `project_2_edges.py`      | Kenar yoğunluğuyla RAF DOLU/BOŞ tespiti   | `python cv_projects/project_2_edges.py raf.jpg --show`        |
| `project_3_cnn_ready.py`  | Hazır ResNet ile görüntü sınıflandırma     | `python cv_projects/project_3_cnn_ready.py kopek.jpg`         |
| `project_4_compare.py`    | Klasik vs derin öğrenme karşılaştırması    | `python cv_projects/project_4_compare.py sokak.jpg`           |

---

## 🔬 Alt Proje — Gerçek Zamanlı İnsan Davranışı Analitiği

> 📁 [`imageprocessing/`](imageprocessing/README.md) · Ayrı paketlenmiş uygulama

YOLOv8 + ByteTrack + COCO-17 iskelet noktaları kullanarak video akışında kişileri takip eder, duruş ve hareketleri (duruyor / oturuyor / yürüyor / koşuyor) sınıflandırır.

```
imageprocessing/
├── src/
│   ├── pipeline/
│   │   ├── detector_yolo.py   # YOLOv8 kişi tespiti (max_detections destekli)
│   │   ├── tracker.py         # ByteTrack çok-kişi takibi
│   │   ├── pose_estimator.py  # COCO-17 keypoint çıkarımı
│   │   └── behavior.py        # Kural tabanlı duruş sınıflandırıcı
│   └── main.py                # CLI giriş noktası
├── configs/default.yaml       # Pipeline konfigürasyonu
└── tests/                     # 24 pytest testi
```

**Kurulum ve Çalıştırma:**

```bash
cd imageprocessing
pip install -e .                         # Konsol betiği kaydeder
cv-human-behavior-analytics --source 0  # Webcam'den canlı analiz

# Ya da:
python src/main.py --source video.mp4 --disable_segmentation
pytest tests/ -q                         # → 24 passed
```

**CLI Seçenekleri:**

| Bayrak | Karşıtı | Açıklama |
|--------|---------|----------|
| `--enable_tracking` | `--disable_tracking` | ByteTrack takip |
| `--enable_pose` | `--disable_pose` | Keypoint tahmini |
| `--enable_behavior` | `--disable_behavior` | Duruş sınıflandırma |
| `--enable_overlay` | `--disable_overlay` | Ekran bindirme |
| `--enable_logging` | `--disable_logging` | CSV kayıt |
| `--source` | — | Video dosyası veya kamera indeksi |

---

## 📝 Alt Proje — OCR Masaüstü Uygulaması

> 📁 [`teserract/`](teserract/README.md) · Tesseract tabanlı OCR stüdyosu

Görüntüden, PDF'den veya kameradan metin tanıma işlemlerini gerçekleştiren kapsamlı bir OCR uygulaması. Hem web arayüzü (Streamlit) hem masaüstü GUI (Tkinter) içerir.

```
teserract/
├── src/
│   ├── 01_basic_ocr.py          # Basit görüntü → metin
│   ├── 02_preprocess_ocr.py     # Gürültü temizleme + OCR
│   ├── 03_table_ocr.py          # Tablo yapısı çıkarımı
│   ├── 04_pdf_ocr.py            # PDF sayfa işleme
│   └── ...                      # İleri seviye modüller
├── streamlit_app.py             # Sürükle-bırak web OCR arayüzü
├── desktop_app.py               # Tkinter GUI (offline çalışır)
├── build_desktop_exe.bat        # PyInstaller ile .exe paketleme
├── tessdata/                    # Türkçe + İngilizce dil paketleri
└── docs/                        # Kapsamlı rehberler ve soru bankası
```

**Çalıştırma:**

```bash
cd teserract
pip install -r requirements.txt

# Web arayüzü
streamlit run streamlit_app.py

# Masaüstü GUI
python desktop_app.py

# Komut satırı
python src/01_basic_ocr.py goruntu.png

# EXE derle (Windows)
build_desktop_exe.bat
```

**OCR İşlem Akışı:**

```
Görüntü/PDF → Grileştirme → Gürültü Azaltma → Otsu Threshold
    → Tesseract Engine → Ham Metin → Regex/NLP → Yapılandırılmış Çıktı
```

---

## 📊 Teknoloji Karşılaştırması

| Görev               | Klasik Yöntem               | Derin Öğrenme              |
| ------------------- | --------------------------- | -------------------------- |
| Kenar tespiti       | `cv2.Canny`                 | Conv2D gradyan katmanı     |
| Özellik çıkarımı    | ORB, SIFT, AKAZE            | CNN feature map            |
| Nesne sınıflandırma | Kural + renk histogramı     | ResNet, MobileNet          |
| Nesne tespiti       | Haar Cascade, Kontur analizi| YOLOv8, Faster R-CNN       |
| Segmentasyon        | Threshold + Morphology      | U-Net, SAM                 |
| Poz tahmini         | —                           | COCO-17 keypoint modeli    |
| Metin tanıma        | Tesseract (template)        | TrOCR, EasyOCR             |

---

## 📄 Dokümantasyon

| Dosya | İçerik |
| ----- | ------ |
| [`SPECIAL_USAGE_README.md`](SPECIAL_USAGE_README.md) | Tüm dosyalardaki özel OpenCV/PyTorch kullanımı sözlüğü |
| [`PROJE_ANLATIMI.md`](PROJE_ANLATIMI.md)             | Detaylı 1. şahıs proje hikayesi |
| [`YENI_ORNEKLER.md`](YENI_ORNEKLER.md)               | Yeni eklenen örneklerin açıklamaları |
| [`imageprocessing/README.md`](imageprocessing/README.md) | Davranış analitiği pipeline dokümantasyonu |
| [`teserract/README.md`](teserract/README.md)         | OCR projesi rehberi |
| [`teserract/docs/`](teserract/docs/)                 | Soru bankası, kod analizi, EXE rehberi |

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
  <i>Bu depo, bilgisayarlı görü ve derin öğrenme yolculuğumun canlı kaydıdır.<br>Her dosya Türkçe yorumlarla açıklanmıştır.</i>
</p>
