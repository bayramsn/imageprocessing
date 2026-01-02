# 🌟 Yeni Eklenen Uygulamalı Örnekler

Bu doküman, projeye sonradan eklenen ve her konsepti pekiştirmeyi amaçlayan pratik uygulamaları anlatır. Her biri belirli bir görüntü işleme tekniğine odaklanır.

---

## 🎨 1. Sanal Çizim Tahtası (Webcam Paint)

**Konum:** `03_opencv_giris/webcam_paint.py`

**Amaç:** Webcam ve temel çizim komutlarını interaktif bir uygulamaya dönüştürmek.

**Nasıl Çalışır?**

- `cv2.setMouseCallback` ile fare hareketlerini dinler.
- Çizimleri siyah bir "canvas" (maske) katmanına yapar.
- Görüntü birleştirme (Blending) ile canvas'ı webcam görüntüsünün üzerine bindirir.
- Renk değiştirmek için klavye kısayollarını (r, g, b, c) kullanır.

---

## 🏙️ 2. Tilt-Shift Efekti (Minyatür Şehir)

**Konum:** `04_gaussian_blur_opencv/tilt_shift_effect.py`

**Amaç:** Sıradan bir şehir fotoğrafını maket/oyuncak gibi göstermek.

**Teknik:**

- **Bulanıklaştırma:** Görüntünün üst ve alt kısımlarına Gaussian Blur uygular.
- **Maskeleme:** Odak noktasından kenarlara doğru artan bir maske kullanır.
- **Doygunluk (Saturation):** HSV renk uzayına geçip renkleri canlandırır (oyuncak etkisi için).

---

## 🧪 3. Kernel Bahçesi (Custom Filters)

**Konum:** `05_gaussian_blur_manual/kernel_playground.py`

**Amaç:** Konvolüsyon matrislerinin (kernel) görüntü üzerindeki etkisini doğrudan gözlemlemek.

**Kullanılan Filtreler:**

- **Sharpen:** Kenarları belirginleştirir.
- **Emboss:** Kabartma efekti verir (3D gibi görünür).
- **Edge Detect:** Sadece kenar çizgilerini bırakır.
- **Motion Blur:** Hareketsiz görüntüye hız efekti verir.

---

## 📐 4. Geometrik Şekil Tespiti

**Konum:** `06_traditional_image_processing/shape_detector.py`

**Amaç:** Görüntüdeki temel şekilleri (Kare, Üçgen, Daire) sınıflandırmak.

**Adımlar:**

1. **Ön İşleme:** Griye çevir + Blur + Threshold (Inverse).
2. **Kontur Bulma:** `cv2.findContours` ile şekil sınırlarını çıkarır.
3. **Köşe Sayma:** `cv2.approxPolyDP` ile şekli çokgene yaklaştırır ve köşe sayısına göre isimlendirir (3=Üçgen, 4=Kare vb.).

---

## 🏞️ 5. Panorama Oluşturucu (Image Stitching)

**Konum:** `07_keypoints_features/panorama_maker.py`

**Amaç:** Yan yana çekilmiş fotoğrafları birleştirip geniş açılı tek bir fotoğraf yapmak.

**Teknik:**

- `cv2.Stitcher` sınıfını kullanır.
- Arka planda: Özellik noktalarını (Keypoints) bulur -> Eşleştirir -> Homografi matrisini hesaplar -> Görüntüleri büker (Warp) ve birleştirir.

---

## 🔄 6. Veri Çoğaltma (Data Augmentation)

**Konum:** `08_cnn_intro/data_augmentation_demo.py`

**Amaç:** Derin öğrenme için veri setini yapay olarak zenginleştirmek.

**Yöntemler:**

- Rastgele Döndürme (Rotation)
- Aynalama (Flip)
- Renk Oynamaları (Jitter)
- Kesme (Crop)
  PyTorch `torchvision.transforms` kütüphanesi kullanılmıştır.

---

## 🌈 7. 3D Renk Uzayı Analizi

**Konum:** `09_numpy_matplotlib/color_distribution_3d.py`

**Amaç:** Bir resmin renk paletini 3 boyutlu uzayda analiz etmek.

**Teknik:**

- Görüntüyü piksellere ayırır.
- Kırmızı, Yeşil ve Mavi değerlerini X, Y, Z eksenlerine oturtur.
- Matplotlib kullanarak 3D "Scatter Plot" çizer.

---

## 👤 8. Yüz ve Göz Tespiti

**Konum:** `10_detection_segmentation/face_eye_detector.py`

**Amaç:** Fotoğraftaki insan yüzlerini bulmak.

**Teknik:**

- **Haar Cascades:** OpenCV'nin klasik, hızlı nesne tespit yöntemi.
- XML dosyalarındaki eğitilmiş özellikler kullanılarak tarama yapılır.
- Önce yüz bulunur, sonra yüzün içinde göz aranır (ROI - Region of Interest).

---

## 🚀 Proje Başlatıcı (GUI)

**Dosya:** `app_launcher.py`

Tüm bu projeleri tek bir pencereden yönetmek için geliştirdiğimiz arayüz.

- Dosya seçme işlemlerini otomatikleştirir.
- Proje açıklamalarını gösterir.
- Hata yönetimini sağlar.
