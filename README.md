# 🖼️ Bilgisayarlı Görü, Derin Öğrenme ve OCR Laboratuvarı

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF)](https://github.com/ultralytics/ultralytics)
[![Tesseract](https://img.shields.io/badge/Tesseract-5.x-blueviolet)](https://github.com/tesseract-ocr/tesseract)
[![pytest](https://img.shields.io/badge/tests-24%20passed-brightgreen?logo=pytest)](imageprocessing/tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Klasik görüntü işlemeden derin öğrenmeye, özellik çıkarımından nesne tespitine, OCR uygulamalarından gerçek zamanlı insan davranışı analizine** kadar uzanan, **50+ Python dosyası** içeren kapsamlı bir bilgisayarlı görü laboratuvarı.

Her modül bağımsız çalışabilir ve tüm kod **Türkçe yorumlarla** açıklanmıştır.

---

## ✨ Öne Çıkan Özellikler

| Özellik | Detay |
|---------|-------|
| 🎓 **8 Eğitim Modülü** | OpenCV temellerinden CNN'e adım adım ilerleyen dersler |
| 🔬 **4 Bağımsız Proje** | ORB benzerlik, kenar tespiti, ResNet sınıflandırma, görev karşılaştırması |
| 🏃 **Davranış Analitiği** | YOLOv8 + ByteTrack + COCO-17 iskelet → duruyor/oturuyor/yürüyor/koşuyor |
| 📝 **OCR Stüdyosu** | 15 Tesseract modülü + Streamlit web + Tkinter masaüstü + EXE paketleme |
| 🚀 **GUI Başlatıcı** | `app_launcher.py` ile tüm modüllere tek tıkla erişim |
| ✅ **24 Otomatik Test** | `pytest` ile doğrulanmış pipeline |
| 📚 **Türkçe Dokümantasyon** | Her klasörde README + 3 detaylı anlatım dosyası |

---

## 📂 Tam Repo Yapısı

```
📦 imageprocessing/
│
│  ─────────────── EĞİTİM MODÜLLERİ ───────────────
│
├── 📁 03_opencv_giris/                   # 🟢 Başlangıç — OpenCV temelleri
│   ├── webcam_filter.py                  #    Tuşla filtre geçişi (gri/blur/resize)
│   ├── webcam_fps.py                     #    FPS + cartoon/sepia/negative efekt
│   ├── webcam_paint.py                   #    Sanal çizim tahtası (fare ile)
│   └── README.md
│
├── 📁 04_gaussian_blur_opencv/           # 🟢 Başlangıç — Blur algoritmaları
│   ├── gaussian_blur_app.py              #    Trackbar ile canlı kernel ayarı
│   ├── blur_comparison.py                #    Gaussian / Median / Bilateral / Box
│   ├── tilt_shift_effect.py              #    Minyatür şehir (tilt-shift) efekti
│   └── README.md
│
├── 📁 05_gaussian_blur_manual/           # 🟡 Orta — Convolution sıfırdan
│   ├── gaussian_blur_manual.py           #    Gaussian kernel formülü elle
│   ├── gaussian_blur_scipy.py            #    SciPy ndimage.convolve yaklaşımı
│   ├── custom_gaussian.py                #    Benchmark — elle vs OpenCV hız testi
│   ├── all_filters_demo.py               #    Sharpen/Emboss/Laplacian/Sobel
│   ├── filters_from_blog.py              #    Blog tabanlı filtre deneyleri
│   ├── kernel_playground.py              #    İnteraktif filtre bahçesi
│   └── README.md
│
├── 📁 06_traditional_image_processing/   # 🟡 Orta — AI'sız klasik CV
│   ├── coin_counter.py                   #    Watershed ile para sayma
│   ├── preprocessing_tool.py             #    Threshold / Canny / Morphology
│   ├── shape_detector.py                 #    Geometrik şekil tanıma (approxPolyDP)
│   └── README.md
│
├── 📁 07_keypoints_features/             # 🟡 Orta — Feature matching
│   ├── logo_match.py                     #    ORB ile logo eşleştirme
│   ├── feature_matcher.py                #    ORB / SIFT / AKAZE karşılaştırması
│   ├── panorama_maker.py                 #    Çoklu görüntü dikişi (stitching)
│   └── README.md
│
├── 📁 08_cnn_intro/                      # 🔴 İleri — CNN eğitimi
│   ├── mnist_cnn.py                      #    TensorFlow/Keras MNIST eğitimi
│   ├── cnn_visualizer.py                 #    Feature map + kernel görselleştirme
│   ├── data_augmentation_demo.py         #    Flip/rotate/crop/jitter teknikleri
│   └── README.md
│
├── 📁 09_numpy_matplotlib/               # 🔴 İleri — Matris analizi
│   ├── image_analyzer.py                 #    Histogram + istatistik + threshold
│   ├── color_distribution_3d.py          #    3D RGB renk uzayı scatter plot
│   └── README.md
│
├── 📁 10_detection_segmentation/         # 🔴 İleri — Tespit & segmentasyon
│   ├── compare_tasks.py                  #    Classification/Detection/Segmentation
│   ├── face_eye_detector.py              #    Haar Cascade yüz + göz tespiti
│   └── README.md
│
│  ─────────────── BAĞIMSIZ PROJELER ───────────────
│
├── 📁 cv_projects/
│   ├── project_1_similarity.py           #    ORB ile 2 görüntü benzerlik skoru
│   ├── project_2_edges.py                #    Kenar yoğunluğu → raf dolu/boş
│   ├── project_3_cnn_ready.py            #    Hazır ResNet-50 sınıflandırma
│   ├── project_4_compare.py              #    Klasik vs DL karşılaştırması
│   └── utils.py                          #    Paylaşımlı yardımcı fonksiyonlar
│
│  ─────────────── DAVRANIŞ ANALİTİĞİ PİPELINE ───────────────
│
├── 📁 imageprocessing/
│   ├── src/
│   │   ├── main.py                       #    CLI giriş noktası
│   │   ├── pipeline/
│   │   │   ├── detector_yolo.py          #    YOLOv8 kişi tespiti
│   │   │   ├── tracker.py                #    ByteTrack çok-kişi takibi
│   │   │   ├── pose.py                   #    COCO-17 keypoint çıkarımı
│   │   │   ├── behavior.py               #    Duruş sınıflandırıcı
│   │   │   ├── segmentation.py           #    Semantik segmentasyon
│   │   │   ├── overlay.py                #    Video bindirme katmanı
│   │   │   ├── logger.py                 #    CSV kayıt modülü
│   │   │   ├── timer.py                  #    Performans zamanlayıcı
│   │   │   └── video_source.py           #    Kamera/dosya video kaynağı
│   │   └── utils/
│   │       ├── geometry.py               #    Açı/mesafe/hız hesaplama
│   │       ├── draw.py                   #    Çizim yardımcıları
│   │       ├── fps.py                    #    FPS sayacı
│   │       └── time_utils.py             #    Zaman damgası araçları
│   ├── configs/default.yaml              #    Pipeline konfigürasyonu
│   ├── gui_runner.py                     #    Tkinter GUI çalıştırıcı
│   ├── fix_camera_permission.py          #    Kamera izin düzeltici
│   ├── pyproject.toml                    #    Modern build-meta + konsol betiği
│   ├── requirements.txt
│   ├── tests/
│   │   ├── conftest.py                   #    sys.path enjeksiyonu
│   │   ├── test_behavior_rules.py        #    Duruş kuralları testleri
│   │   ├── test_timer.py                 #    Zamanlayıcı testleri
│   │   └── run_all_manual.py             #    Manuel test çalıştırıcı
│   ├── docs/
│   │   ├── YOLO_Theory.md                #    YOLO algoritma teorisi
│   │   ├── YOLO_v3_to_v8_Differences.md  #    YOLOv3→v8 evrim karşılaştırması
│   │   ├── Pose_Theory.md                #    İskelet nokta tahmini teorisi
│   │   ├── Segmentation_Theory.md        #    Segmentasyon teorisi
│   │   ├── Research_DINOv2.md            #    DINOv2 araştırma notu
│   │   ├── Research_OpenCLIP.md          #    OpenCLIP araştırma notu
│   │   ├── Benchmark_Template.md         #    Performans kıyaslama şablonu
│   │   └── Internship_Report_Template.md #    Staj raporu şablonu
│   └── README.md
│
│  ─────────────── OCR MASAÜSTÜ UYGULAMASI ───────────────
│
├── 📁 teserract/
│   ├── src/
│   │   ├── 01_basic_ocr.py              #    Temel görüntü → metin
│   │   ├── 02_preprocess_ocr.py         #    Gri + blur + Otsu → temiz OCR
│   │   ├── 03_document_regions_ocr.py   #    Belge bölge tespiti + OCR
│   │   ├── 04_bounding_boxes.py         #    Kelime bazlı kutu çizimi
│   │   ├── 05_turkish_ocr.py            #    Türkçe karakter desteği
│   │   ├── 06_clean_ocr_text.py         #    Regex ile metin temizleme
│   │   ├── 07_postprocess_correction.py #    Sözlük tabanlı hata düzeltme
│   │   ├── 08_field_extraction.py       #    Yapılandırılmış alan çıkarımı
│   │   ├── 09_ocr_nlp_insights.py       #    NLP ile anlamlı bilgi çıkarma
│   │   ├── 10_form_digitization.py      #    Form dijitalleştirme
│   │   ├── 11_table_ocr.py             #    Tablo yapısı çıkarımı
│   │   ├── 12_template_runner.py        #    Şablon tabanlı OCR çalıştırıcı
│   │   ├── 13_live_webcam_ocr.py        #    Canlı kamera OCR
│   │   ├── 14_pdf_ocr.py               #    PDF sayfa işleme
│   │   ├── 15_batch_folder_ocr.py       #    Toplu klasör OCR
│   │   ├── common.py                    #    Paylaşımlı yardımcı fonksiyonlar
│   │   └── pipelines.py                 #    OCR pipeline yapılandırmaları
│   ├── streamlit_app.py                 #    Sürükle-bırak Streamlit web arayüzü
│   ├── desktop_app.py                   #    Tkinter masaüstü GUI
│   ├── build_desktop_exe.bat            #    PyInstaller ile .exe paketleme
│   ├── tessdata/                        #    eng + tur + osd dil paketleri
│   ├── templates/                       #    Fatura ve kimlik JSON şablonları
│   ├── docs/
│   │   ├── INDEX.md                     #    Dokümantasyon ana sayfası
│   │   ├── PROJECT_NARRATIVE_GUIDE.md   #    Proje anlatım rehberi
│   │   ├── CODE_WALKTHROUGH.md          #    Kod analiz dokümanı
│   │   ├── QUESTION_BANK.md             #    Soru bankası
│   │   ├── GITHUB_PAGE_CONTENT.md       #    GitHub Pages içeriği
│   │   └── DESKTOP_EXE_GUIDE.md         #    EXE paketleme rehberi
│   ├── tests/test_quality_improvements.py
│   ├── tools/
│   │   ├── generate_sample_assets.py    #    Örnek veri üretici
│   │   └── generate_training_spreadsheet.py
│   ├── requirements.txt
│   └── README.md
│
│  ─────────────── KÖK DOSYALAR ───────────────
│
├── 🚀 app_launcher.py                   # Tkinter GUI başlatıcı
├── 📄 requirements.txt                  # Ana bağımlılık listesi
├── 📄 SPECIAL_USAGE_README.md           # OpenCV/PyTorch kullanım sözlüğü
├── 📄 PROJE_ANLATIMI.md                 # Detaylı 1. şahıs proje anlatımı
├── 📄 YENI_ORNEKLER.md                  # Yeni örneklerin açıklamaları
├── 📄 LICENSE                           # MIT Lisansı
└── 📄 README.md                         # ← Bu dosya
```

---

## 🚀 Hızlı Başlangıç

```bash
# 1. Repoyu klonla
git clone https://github.com/bayramsn/imageprocessing.git
cd imageprocessing

# 2. Sanal ortam oluştur ve aktive et
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # Linux / macOS

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. GUI başlatıcıyı aç (opsiyonel)
python app_launcher.py
```

### Alt Projelerin Kurulumu

```bash
# Davranış Analitiği Pipeline
cd imageprocessing
pip install -e .
cv-human-behavior-analytics --source 0    # Webcam'den canlı analiz

# OCR Stüdyosu
cd ../teserract
pip install -r requirements.txt
streamlit run streamlit_app.py            # Web arayüzü
python desktop_app.py                      # Masaüstü GUI
```

---

## 📦 Bağımlılıklar

| Paket | Versiyon | Kullanım Alanı |
|-------|----------|----------------|
| `numpy` | ≥1.24 | Matris işlemleri, piksel dizileri |
| `opencv-python` | ≥4.8 | Görüntü işleme, kamera, Haar Cascade |
| `matplotlib` | ≥3.7 | Histogram, 3D plot, görselleştirme |
| `torch` | ≥2.0 | Feature extraction, model inference |
| `torchvision` | ≥0.15 | Hazır modeller (ResNet, MobileNet) |
| `tensorflow` | ≥2.10 | MNIST CNN eğitimi |
| `scipy` | ≥1.10 | ndimage.convolve, sinyal işleme |
| `Pillow` | ≥10.0 | Görüntü yükleme / format dönüşümü |
| `PyYAML` | ≥6.0 | Pipeline YAML konfigürasyonu |
| `ultralytics` | ≥8.0 | YOLOv8 tespit, segmentasyon, pose |
| `pytesseract` | ≥0.3 | Tesseract OCR Python bağlayıcısı |
| `streamlit` | ≥1.30 | OCR web arayüzü |
| `pytest` | ≥7.0 | Otomatik test altyapısı |

---

## 🗺️ Öğrenme Yolu

```
🟢 BAŞLANGIÇ               🟡 ORTA SEVİYE                 🔴 İLERİ SEVİYE
───────────────────        ───────────────────────         ─────────────────────────────
 03 OpenCV Giriş            05 Manuel Convolution           08 CNN Eğitimi
        ↓                          ↓                              ↓
 04 Gaussian Blur            06 Klasik CV Yöntemleri         09 NumPy & Matplotlib
        ↓                          ↓                              ↓
 cv: project_2               07 Feature Matching             10 Detection & Segmentation
    (kenar tespiti)                ↓                              ↓
                              cv: project_1                  cv: project_3 & _4
                                 (benzerlik)                      ↓
                                                             imageprocessing/
                                                                (davranış pipeline)
                                                                   ↓
                                                             teserract/
                                                                (OCR stüdyo)
```

---

## 📖 Modül Detayları

### 🟢 Modül 03 — OpenCV Giriş · Webcam Laboratuvarı

**Ne öğrenilir:** Kamerayı açma, kare okuma, filtre uygulama, fare olaylarıyla çizim

| Dosya | Açıklama | Temel API |
|-------|----------|-----------|
| `webcam_filter.py` | Tuşla filtre geçişi: normal → gri → blur → resize | `cv2.VideoCapture`, `cv2.cvtColor` |
| `webcam_fps.py` | FPS overlay + cartoon, sepia, negative efektler | `cv2.putText`, `cv2.getTickCount` |
| `webcam_paint.py` | Fare ile sanal çizim tahtası | `cv2.setMouseCallback`, `cv2.line` |

```bash
python 03_opencv_giris/webcam_filter.py
# Tuşlar: c=normal  g=gri  b=blur  r=½çözünürlük  q=çık
```

---

### 🟢 Modül 04 — Gaussian Blur · Blur Türleri

**Ne öğrenilir:** Farklı blur algoritmalarının parametresel karşılaştırması

| Dosya | Açıklama | Temel API |
|-------|----------|-----------|
| `gaussian_blur_app.py` | Trackbar ile canlı kernel boyutu ayarlama | `cv2.createTrackbar` |
| `blur_comparison.py` | Gaussian / Median / Bilateral / Box yan yana | `cv2.GaussianBlur`, `cv2.bilateralFilter` |
| `tilt_shift_effect.py` | Minyatür şehir efekti (odaklanma maskesi) | Gradient mask + `cv2.addWeighted` |

```bash
python 04_gaussian_blur_opencv/blur_comparison.py resim.jpg --interactive
```

---

### 🟡 Modül 05 — Manuel Gaussian · Convolution Matematiği

**Ne öğrenilir:** CNN'in temel yapı taşı olan convolution'ı formülden koda dönüştürmek

| Dosya | Açıklama | Temel API |
|-------|----------|-----------|
| `gaussian_blur_manual.py` | Gaussian kernel formülü ile elle hesaplama | `math.exp`, NumPy |
| `gaussian_blur_scipy.py` | SciPy tabanlı convolution | `scipy.ndimage.convolve` |
| `custom_gaussian.py` | Performans kıyaslama: elle vs OpenCV | `time.perf_counter` |
| `all_filters_demo.py` | Sharpen, Emboss, Laplacian, Sobel karşılaştırması | Kernel matrisleri |
| `filters_from_blog.py` | Blog tabanlı filtre deneyleri | Özel kernel dizileri |
| `kernel_playground.py` | İnteraktif filtre test ortamı | `cv2.filter2D` |

```bash
python 05_gaussian_blur_manual/custom_gaussian.py resim.jpg --benchmark
```

---

### 🟡 Modül 06 — Geleneksel Görüntü İşleme

**Ne öğrenilir:** AI olmadan, klasik algoritmalarla nesne sayma ve şekil tanıma

| Dosya | Açıklama | Temel API |
|-------|----------|-----------|
| `coin_counter.py` | Watershed ile para sayma | `cv2.watershed`, `cv2.connectedComponents` |
| `preprocessing_tool.py` | Threshold / Canny / Morphology karşılaştırması | `cv2.threshold`, `cv2.Canny` |
| `shape_detector.py` | Geometrik şekil tanıma (kare, daire, üçgen) | `cv2.approxPolyDP`, `cv2.minEnclosingCircle` |

```bash
python 06_traditional_image_processing/shape_detector.py resim.jpg --show
```

---

### 🟡 Modül 07 — Keypoint & Feature Matching

**Ne öğrenilir:** Görüntüdeki ayırt edici noktaları bulma, eşleştirme ve panorama dikişi

| Dosya | Açıklama | Temel API |
|-------|----------|-----------|
| `logo_match.py` | ORB ile logo eşleştirme | `cv2.ORB_create`, `cv2.BFMatcher` |
| `feature_matcher.py` | ORB / SIFT / AKAZE karşılaştırması | `cv2.SIFT_create`, `cv2.AKAZE_create` |
| `panorama_maker.py` | Çoklu görüntüden panorama oluşturma | `cv2.Stitcher`, Homography |

```bash
python 07_keypoints_features/feature_matcher.py r1.jpg r2.jpg --method all
```

---

### 🔴 Modül 08 — CNN Eğitimi & Görselleştirme

**Ne öğrenilir:** Sinir ağını sıfırdan eğitmek, her katmanın ne gördüğünü anlamak

| Dosya | Açıklama | Temel API |
|-------|----------|-----------|
| `mnist_cnn.py` | TensorFlow/Keras ile MNIST el yazısı tanıma | `tf.keras.Sequential`, `Conv2D` |
| `cnn_visualizer.py` | Her katmanın feature map ve kernel görseli | `torchvision.models`, hook'lar |
| `data_augmentation_demo.py` | Flip, rotate, crop, color jitter | `torchvision.transforms` |

```bash
python 08_cnn_intro/mnist_cnn.py --epochs 10
```

---

### 🔴 Modül 09 — NumPy & Matplotlib Analizi

**Ne öğrenilir:** Görüntüyü matris olarak düşünmek, 3D renk uzayını görselleştirmek

| Dosya | Açıklama | Temel API |
|-------|----------|-----------|
| `image_analyzer.py` | Histogram, istatistikler, çoklu threshold | `np.histogram`, `cv2.calcHist` |
| `color_distribution_3d.py` | 3D RGB scatter plot ile piksel dağılımı | `mpl_toolkits.mplot3d` |

```bash
python 09_numpy_matplotlib/image_analyzer.py resim.jpg --demo
```

---

### 🔴 Modül 10 — Detection & Segmentation

**Ne öğrenilir:** Sınıflandırma / tespit / segmentasyon farkları, Haar Cascade

| Dosya | Açıklama | Temel API |
|-------|----------|-----------|
| `compare_tasks.py` | 3 temel CV görevi yan yana | ResNet + YOLO + segmentasyon |
| `face_eye_detector.py` | Haar Cascade ile yüz ve göz tespiti | `cv2.CascadeClassifier`, `detectMultiScale` |

```bash
python 10_detection_segmentation/face_eye_detector.py foto.jpg --output sonuc.png
```

---

## 🎯 Bağımsız Projeler (`cv_projects/`)

| # | Proje | Ne Yapar? | Komut |
|---|-------|-----------|-------|
| 1 | `project_1_similarity.py` | ORB keypoint'leri ile 2 görüntünün benzerlik skorunu hesaplar | `python cv_projects/project_1_similarity.py a.jpg b.jpg --show` |
| 2 | `project_2_edges.py` | Canny kenar yoğunluğu analizi → raf dolu mu boş mu? | `python cv_projects/project_2_edges.py raf.jpg --show` |
| 3 | `project_3_cnn_ready.py` | Hazır ResNet-50 ile görüntü sınıflandırma (ImageNet 1000 sınıf) | `python cv_projects/project_3_cnn_ready.py kopek.jpg` |
| 4 | `project_4_compare.py` | Klasik (Canny+Contour) vs DL (YOLO) karşılaştırması | `python cv_projects/project_4_compare.py sokak.jpg` |

**Paylaşımlı Modül:** `utils.py` — `load_image_bgr()`, `to_torch_tensor()`, `normalize_tensor()`, `draw_boxes()` gibi ortak fonksiyonlar. Torch lazy import ile yüklenir (torch olmadan da çalışır).

---

## 🔬 Davranış Analitiği Pipeline (`imageprocessing/`)

Gerçek zamanlı video akışında kişi tespiti, takibi ve hareket sınıflandırması yapan uçtan uca bir uygulama.

### Mimari

```
Video Kaynağı (kamera / dosya)
       │
       ▼
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐
│  YOLOv8 Dedektör│────▶│  ByteTrack   │────▶│  COCO-17     │
│  (detector_yolo) │     │  (tracker)   │     │  Pose Est.   │
│  max_detections  │     │  tracker_cfg │     │  (pose.py)   │
└─────────────────┘     └──────────────┘     └──────────────┘
                                                     │
                                                     ▼
                              ┌──────────────────────────────┐
                              │  Davranış Sınıflandırıcı      │
                              │  (behavior.py)                │
                              │  Duruyor / Oturuyor /         │
                              │  Yürüyor / Koşuyor            │
                              │  → keypoint_velocity() tabanlı│
                              └──────────────────────────────┘
                                        │
                         ┌──────────────┼──────────────┐
                         ▼              ▼              ▼
                    overlay.py     logger.py     segmentation.py
                    (video üstü)   (CSV kayıt)   (semantik seg.)
```

### Pipeline Bileşenleri

| Dosya | Rol | Açıklama |
|-------|-----|----------|
| `detector_yolo.py` | Tespit | YOLOv8 ile yalnız "person" sınıfı tespiti, `max_detections` limiti |
| `tracker.py` | Takip | ByteTrack ile çoklu kişi ID ataması, `tracker_config` destekli |
| `pose.py` | Poz | COCO-17 keypoint (burun, omuz, dirsek, bilek, kalça, diz, ayak bileği) |
| `behavior.py` | Davranış | Kalça-diz açısı + keypoint hızına dayalı kural motoru |
| `segmentation.py` | Segmentasyon | İsteğe bağlı semantik segmentasyon katmanı |
| `overlay.py` | Görselleştirme | Tespit kutuları + iskelet + etiket bindirme |
| `logger.py` | Kayıt | Frame bazlı CSV log dosyası üretimi |
| `timer.py` | Performans | Pipeline adımlarının süre ölçümü |
| `video_source.py` | Kaynak | Kamera veya video dosyası sarmalayıcısı |

### Yardımcı Fonksiyonlar (`src/utils/`)

| Dosya | İçerik |
|-------|--------|
| `geometry.py` | `angle_between_points()`, `keypoint_velocity()`, `euclidean_distance()` |
| `draw.py` | İskelet ve kutu çizim fonksiyonları |
| `fps.py` | Kayan ortalama FPS sayacı |
| `time_utils.py` | Zaman damgası formatlama |

### CLI Kullanımı

```bash
cd imageprocessing
pip install -e .

# Webcam'den canlı analiz
cv-human-behavior-analytics --source 0

# Video dosyasından
python src/main.py --source video.mp4

# Seçici modüller
python src/main.py --source 0 --disable_segmentation --disable_logging
```

| Bayrak | Karşıtı | Varsayılan | Açıklama |
|--------|---------|-----------|----------|
| `--enable_tracking` | `--disable_tracking` | config | ByteTrack çoklu kişi takibi |
| `--enable_pose` | `--disable_pose` | config | COCO-17 iskelet noktaları |
| `--enable_behavior` | `--disable_behavior` | config | Duruş/hareket sınıflandırma |
| `--enable_overlay` | `--disable_overlay` | config | Video üstü görselleştirme |
| `--enable_logging` | `--disable_logging` | config | CSV kayıt |
| `--enable_segmentation` | `--disable_segmentation` | config | Semantik segmentasyon |
| `--source` | — | `0` | Video yolu veya kamera indeksi |
| `--config` | — | `configs/default.yaml` | YAML konfigürasyon dosyası |

### Test Paketi

```bash
cd imageprocessing
pytest tests/ -q          # → 24 passed
```

### Araştırma Dokümanları (`docs/`)

| Doküman | Konu |
|---------|------|
| `YOLO_Theory.md` | YOLO algoritma ailesi teorisi |
| `YOLO_v3_to_v8_Differences.md` | v3 → v5 → v8 evrim karşılaştırması |
| `Pose_Theory.md` | İskelet noktası tahmin yöntemleri |
| `Segmentation_Theory.md` | Semantik / instance / panoptic segmentasyon |
| `Research_DINOv2.md` | DINOv2 self-supervised learning araştırması |
| `Research_OpenCLIP.md` | OpenCLIP metin-görüntü eşleştirme araştırması |
| `Benchmark_Template.md` | Performans kıyaslama rapor şablonu |
| `Internship_Report_Template.md` | Staj raporu şablonu |

---

## 📝 OCR Stüdyosu (`teserract/`)

Tesseract OCR motoru üzerine kurulu, basit metin çıkarmadan PDF işlemeye, canlı kamera OCR'dan form dijitalleştirmeye kadar 15 adımlı bir öğrenme serisi.

### OCR Modülleri (Sıralı)

| # | Dosya | Ne Yapar? |
|---|-------|-----------|
| 01 | `basic_ocr.py` | Görüntüden düz metin çıkarma — en temel örnek |
| 02 | `preprocess_ocr.py` | Grileştirme → Gaussian blur → Otsu threshold → OCR |
| 03 | `document_regions_ocr.py` | Belgedeki farklı bölgeleri (başlık/gövde/tablo) ayırt etme |
| 04 | `bounding_boxes.py` | Her kelime/satır etrafına dikdörtgen çizme |
| 05 | `turkish_ocr.py` | Türkçe karakter desteği (`tur.traineddata`) |
| 06 | `clean_ocr_text.py` | Regex ile OCR çıktısından gürültü temizleme |
| 07 | `postprocess_correction.py` | Sözlük tabanlı yazım hatası düzeltme |
| 08 | `field_extraction.py` | Ad, TC No, tarih gibi yapılandırılmış alan çıkarma |
| 09 | `ocr_nlp_insights.py` | NLP ile anlamlı bilgi madenciliği |
| 10 | `form_digitization.py` | Form alanlarını dijital veri yapısına dönüştürme |
| 11 | `table_ocr.py` | Tablo yapısını satır-sütun olarak çıkarma |
| 12 | `template_runner.py` | JSON şablon ile alan bazlı OCR (fatura, kimlik) |
| 13 | `live_webcam_ocr.py` | Canlı kamera akışında gerçek zamanlı OCR |
| 14 | `pdf_ocr.py` | Çok sayfalı PDF işleme |
| 15 | `batch_folder_ocr.py` | Klasördeki tüm dosyaları toplu OCR |

### Arayüzler

| Arayüz | Dosya | Açıklama |
|--------|-------|----------|
| 🌐 **Web** | `streamlit_app.py` | Tarayıcıda sürükle-bırak OCR, önizleme, indirme |
| 🖥️ **Masaüstü** | `desktop_app.py` | Tkinter GUI — offline çalışır, internet gerekmez |
| ⚡ **EXE** | `build_desktop_exe.bat` | PyInstaller ile tek dosya .exe paketleme |

### OCR İşlem Akışı

```
Görüntü/PDF/Kamera
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Grileştirme  │────▶│ Gürültü      │────▶│ Otsu         │
│ (grayscale)  │     │ Azaltma      │     │ Threshold    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                     ┌──────────────┐     ┌──────────────┐
                     │ Tesseract    │────▶│ Post-process │
                     │ Engine       │     │ (regex/NLP)  │
                     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                              Yapılandırılmış çıktı (JSON/CSV/metin)
```

### Dil Paketleri (`tessdata/`)

| Dosya | Dil |
|-------|-----|
| `eng.traineddata` | İngilizce |
| `tur.traineddata` | Türkçe |
| `osd.traineddata` | Yön ve betik tespiti |

### Şablonlar (`templates/`)

| Dosya | Kullanım |
|-------|----------|
| `fatura_template.json` | Fatura alanları (tarih, tutar, firma) |
| `kimlik_template.json` | Kimlik kartı alanları (TC, ad, soyad) |

---

## 📊 Teknoloji Matrisi

| Görev | Klasik Yöntem | Derin Öğrenme | Bu Repoda |
|-------|---------------|---------------|-----------|
| Kenar tespiti | `cv2.Canny` | Conv2D gradyan | Modül 05, 06, cv_project_2 |
| Özellik çıkarımı | ORB, SIFT, AKAZE | CNN feature map | Modül 07, cv_project_1 |
| Nesne sınıflandırma | Renk histogramı | ResNet, MobileNet | Modül 08, cv_project_3 |
| Nesne tespiti | Haar Cascade | YOLOv8 | Modül 10, imageprocessing/ |
| Segmentasyon | Threshold + Morphology | U-Net, SAM | Modül 06, imageprocessing/ |
| Poz tahmini | — | COCO-17 keypoint | imageprocessing/pipeline |
| Davranış analizi | — | Keypoint hızı + açı | imageprocessing/behavior |
| Metin tanıma | Tesseract | TrOCR, EasyOCR | teserract/ |

---

## 📄 Dokümantasyon Haritası

| Dosya | İçerik |
|-------|--------|
| [`SPECIAL_USAGE_README.md`](SPECIAL_USAGE_README.md) | Tüm Python dosyalarındaki OpenCV/PyTorch API kullanım sözlüğü |
| [`PROJE_ANLATIMI.md`](PROJE_ANLATIMI.md) | 1. şahıs anlatımla projenin hikayesi |
| [`YENI_ORNEKLER.md`](YENI_ORNEKLER.md) | Yeni eklenen 8 örneğin detaylı açıklamaları |
| [`imageprocessing/README.md`](imageprocessing/README.md) | Davranış analitiği pipeline kurulum & kullanım |
| [`imageprocessing/docs/`](imageprocessing/docs/) | YOLO teorisi, DINOv2, OpenCLIP, Pose, Segmentasyon |
| [`teserract/README.md`](teserract/README.md) | OCR projesi kurulum & kullanım rehberi |
| [`teserract/docs/`](teserract/docs/) | Kod analizi, soru bankası, EXE paketleme rehberi |
| Her modül klasöründe `README.md` | Modül bazlı detaylı açıklama |

---

## 🤝 Katkıda Bulunma

1. **Fork** yapın
2. Feature branch oluşturun: `git checkout -b feature/yeni-ozellik`
3. Değişiklikleri commit edin: `git commit -m 'feat: Yeni özellik eklendi'`
4. Push edin: `git push origin feature/yeni-ozellik`
5. **Pull Request** açın

---

## 📝 Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

---

## 📬 İletişim

**GitHub:** [@bayramsn](https://github.com/bayramsn)

---

<p align="center">
  <b>50+ Python dosyası · 8 eğitim modülü · 4 bağımsız proje · 1 gerçek zamanlı pipeline · 15 OCR modülü</b><br>
  <i>Tüm kodlar Türkçe yorumlarla açıklanmıştır.</i>
</p>
