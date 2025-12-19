# Bilgisayarlı Görü ve Derin Öğrenme Projeleri

Bu depo, klasik görüntü işlemeden derin öğrenmeye kadar adım adım ilerleyen mini projeleri içerir. Her proje bağımsız çalışabilir ve Türkçe yorumlarla açıklanmıştır.

## Kurulum
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows PowerShell
pip install -r requirements.txt
```

## Proje Yapısı

### 📁 03_opencv_giris/
| Dosya | Açıklama |
|-------|----------|
| `webcam_filter.py` | Webcam'den canlı görüntü alıp filtreler uygular. **c**: normal, **g**: gri ton, **b**: blur, **r**: yarı çözünürlük. `cv2.VideoCapture`, `cv2.cvtColor`, `cv2.GaussianBlur` kullanımı. |

### 📁 04_gaussian_blur_opencv/
| Dosya | Açıklama |
|-------|----------|
| `gaussian_blur_app.py` | Trackbar ile canlı kernel ve sigma ayarlama. Orijinal ve blur görüntüyü yan yana gösterir. `cv2.createTrackbar`, `cv2.getTrackbarPos` örneği. |

### 📁 05_gaussian_blur_manual/
| Dosya | Açıklama |
|-------|----------|
| `gaussian_blur_manual.py` | Elle yazılmış 2D Gaussian kernel ve konvolüsyon. OpenCV sonucuyla karşılaştırma yapar. |
| `gaussian_blur_scipy.py` | SciPy `convolve2d` ile aynı işlem; kanal bazlı konvolüsyon örneği. |
| `all_filters_demo.py` | Gaussian, median, bilateral filtreleri tek ekranda karşılaştırır. |
| `filters_from_blog.py` | Box, Sobel, Laplacian, sharpen gibi filtreleri ızgara halinde gösterir. |

### 📁 06_traditional_image_processing/
| Dosya | Açıklama |
|-------|----------|
| `coin_counter.py` | Geleneksel yöntemlerle para sayma: CLAHE, adaptive threshold, morphology, watershed, Canny, contour analizi. CNN kullanmadan nesne sayımı örneği. |

### 📁 07_keypoints_features/
| Dosya | Açıklama |
|-------|----------|
| `logo_match.py` | ORB/SIFT ile logo tespiti. Lowe oran testi, homografi hesaplama, perspektif dönüşümü. Şablonu sahnede bulup kutu çizer. |

### 📁 08_cnn_intro/
| Dosya | Açıklama |
|-------|----------|
| `mnist_cnn.py` | Keras ile basit CNN eğitimi. MNIST veri seti, Conv2D+MaxPool katmanları, eğitim/doğrulama grafikleri. |
| `README.md` | CNN giriş notları. |

### 📁 Kök Dizin (Yeni Mini Projeler)
| Dosya | Açıklama |
|-------|----------|
| `utils.py` | Ortak yardımcı fonksiyonlar: görüntü yükleme (`cv2.imdecode` fallback), BGR↔RGB, tensör dönüşümü, normalizasyon, kutu çizme. |
| `project_1_similarity.py` | **ORB ile Görüntü Benzerliği**: İki fotoğrafı karşılaştırır, keypoint eşleştirir, Lowe oran testi uygular, benzerlik skoru verir. |
| `project_2_edges.py` | **Kural Tabanlı Sınıflandırma**: Canny kenar sayısına göre EMPTY/NOT EMPTY kararı. CNN öncesi yaklaşımın sınırlarını gösterir. |
| `project_3_cnn_ready.py` | **Hazır CNN ile Sınıflandırma**: MobileNet veya ResNet ile yalnızca çıkarım. Eğitim yok, ImageNet ağırlıkları kullanılır. |
| `project_4_compare.py` | **Sınıflandırma vs Tespit**: Aynı görüntüde ResNet (sınıflandırma) ve Faster R-CNN (detection) karşılaştırması. |
| `requirements.txt` | Gerekli Python paketleri: numpy, opencv-python, matplotlib, torch, torchvision. |

### 📄 Dokümantasyon
| Dosya | Açıklama |
|-------|----------|
| `PROJE_ANLATIMI.md` | 4 yeni projenin 1. şahıs ağzından adım adım anlatımı. |
| `SPECIAL_USAGE_README.md` | Tüm dosyalardaki özel OpenCV/PyTorch kullanımlarının sözlüğü. |

## Çalıştırma Örnekleri
```
python project_1_similarity.py img_a.jpg img_b.jpg --show
python project_2_edges.py shelf.jpg --show --edge-thresh 800
python project_3_cnn_ready.py dog.jpg --model resnet
python project_4_compare.py street.jpg --score 0.6 --show
```

## Öğrenme Notları
- Proje 1: Anahtar nokta, tanımlayıcı, Lowe oran testi; derin öğrenme değildir.
- Proje 2: Kural tabanlı yaklaşımın sınırlamaları; neden CNN’e ihtiyaç var.
- Proje 3: Ön eğitimli modelle yalnız çıkarım; eğitim yok, yalnızca ileri geçiş.
- Proje 4: Sınıflandırma (var/yok) ile tespit (kutu çiz) farkı; tespitin daha maliyetli oluşu.
