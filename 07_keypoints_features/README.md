# 🔑 Keypoint ve Özellik Çıkarımı

## 🎯 Amaç
Görüntüden "ayırt edici noktalar" çıkarmak. AR, panorama ve nesne takibi temellerini öğrenmek.

## 📦 Gereksinimler
```bash
pip install opencv-python opencv-contrib-python numpy matplotlib
```

## 🚀 Kullanım
```bash
python feature_matcher.py resim1.jpg resim2.jpg
python feature_matcher.py resim1.jpg resim2.jpg --method sift
python feature_matcher.py resim1.jpg resim2.jpg --show-keypoints
```

## 🧠 Öğrenecekleriniz

### Keypoint Nedir?

Keypoint = Görüntüdeki ayırt edici nokta
- Köşeler, kenarlar, dokular
- Döndürme ve ölçeklemeye dayanıklı
- Her keypoint'in koordinatı ve yönü var

### Descriptor Nedir?

Descriptor = Keypoint'in "parmak izi"
- Keypoint etrafındaki bölgeyi tanımlar
- 128-512 boyutlu vektör
- İki görüntüdeki aynı noktayı bulmak için kullanılır

## 🔍 Popüler Algoritmalar

| Algoritma | Hız | Doğruluk | Lisans |
|-----------|-----|----------|--------|
| ORB | ⚡⚡⚡ | ⭐⭐ | Ücretsiz |
| SIFT | ⚡ | ⭐⭐⭐ | Ücretsiz (OpenCV 4.4+) |
| SURF | ⚡⚡ | ⭐⭐⭐ | Patentli |
| AKAZE | ⚡⚡ | ⭐⭐⭐ | Ücretsiz |
| BRISK | ⚡⚡⚡ | ⭐⭐ | Ücretsiz |

### ORB (Oriented FAST and Rotated BRIEF)
```python
orb = cv2.ORB_create(nfeatures=2000)
keypoints, descriptors = orb.detectAndCompute(gray, None)
```
- En hızlı
- Binary descriptor (Hamming mesafesi)
- Gerçek zamanlı uygulamalar için ideal

### SIFT (Scale-Invariant Feature Transform)
```python
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
```
- En doğru
- Float descriptor (L2 mesafesi)
- Yavaş ama güçlü

### AKAZE
```python
akaze = cv2.AKAZE_create()
keypoints, descriptors = akaze.detectAndCompute(gray, None)
```
- SIFT ve ORB arası denge
- Binary descriptor
- Ücretsiz ve modern

## 🔗 Eşleştirme (Matching)

### Brute-Force Matcher
```python
# ORB için (binary descriptor)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc1, desc2)

# SIFT için (float descriptor)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(desc1, desc2)
```

### FLANN Matcher (Daha hızlı)
```python
# SIFT için
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc1, desc2, k=2)
```

### Lowe's Ratio Test
```python
# İyi eşleşmeleri filtrele
good_matches = []
for m, n in knn_matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```
- En yakın 2 eşleşmeyi al
- Birinci çok iyi değilse (ratio > 0.75), reddet
- Yanlış eşleşmeleri azaltır

## 📊 Homography (Perspektif Dönüşümü)

```python
# En az 4 iyi eşleşme gerekli
if len(good_matches) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # H = 3x3 dönüşüm matrisi
```

## 🌍 Gerçek Kullanım

### Panorama Stitching
```
Resim1 + Resim2 → Keypoint bul → Eşleştir → Homography → Birleştir
```

### Nesne Takibi
```
Referans görüntü → Keypoint çıkar → Her frame'de eşleştir → Konum bul
```

### AR (Artırılmış Gerçeklik)
```
Marker tanı → Keypoint eşleştir → Kamera pozisyonu bul → 3D nesne yerleştir
```

### Logo/Marka Tanıma
```
Logo veritabanı → Her görüntüde logo ara → Benzerlik skoru hesapla
```

## ⚠️ İpuçları

1. **ORB yeterliyse SIFT kullanma** - Gereksiz yere yavaşlama
2. **Lowe ratio'yu ayarla** - 0.7-0.8 arası iyi başlangıç
3. **RANSAC kullan** - Yanlış eşleşmeleri temizler
4. **Minimum keypoint sayısı** - Güvenilir sonuç için 10+ iyi eşleşme

## 🔗 Sonraki Adım
→ [08_cnn_intro](../08_cnn_intro/) - CNN'e giriş
