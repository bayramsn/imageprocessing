# 🌫️ OpenCV ve Gaussian Blur - Blur Karşılaştırma Aracı

## 🎯 Amaç
Gaussian Blur'un ne işe yaradığını gerçekten anlamak ve farklı blur türlerini karşılaştırmak.

## 📦 Gereksinimler
```bash
pip install opencv-python numpy matplotlib
```

## 🚀 Kullanım
```bash
python blur_comparison.py resim.jpg
python blur_comparison.py resim.jpg --kernel 7
python blur_comparison.py resim.jpg --interactive
```

## 🧠 Öğrenecekleriniz

### Blur Türleri

| Blur Türü | Ne Zaman Kullanılır | Kenar Koruma |
|-----------|---------------------|--------------|
| **Gaussian** | Genel amaçlı bulanıklaştırma | ❌ Düşük |
| **Median** | Tuz-biber gürültüsü | ✅ Orta |
| **Bilateral** | Yüz güzelleştirme | ✅ Yüksek |
| **Box (Average)** | Hızlı blur gerektiğinde | ❌ Düşük |

### 1. Gaussian Blur
```python
# Kernel boyutu TEK SAYI olmalı (3, 5, 7, 9...)
# Sigma: 0 = otomatik hesapla
blurred = cv2.GaussianBlur(img, (5, 5), 0)
```
- Normal dağılım (çan eğrisi) kullanır
- Merkezdeki piksele daha fazla ağırlık verir
- Doğal görünümlü bulanıklık

### 2. Median Blur
```python
# Tuz-biber gürültüsü için ideal
blurred = cv2.medianBlur(img, 5)
```
- Medyan değeri alır (orta değer)
- Aykırı değerleri (outlier) etkisiz kılar
- Kenarları daha iyi korur

### 3. Bilateral Filter
```python
# d: komşu çapı, sigmaColor: renk hassasiyeti, sigmaSpace: mesafe hassasiyeti
blurred = cv2.bilateralFilter(img, 9, 75, 75)
```
- Kenarları koruyarak blur uygular
- Yüz güzelleştirme için mükemmel
- Hesaplama maliyeti yüksek

### 4. Box Filter (Average)
```python
blurred = cv2.blur(img, (5, 5))
```
- En basit blur
- Tüm piksellere eşit ağırlık
- Çok hızlı ama kalitesiz

## 📊 Kernel Size Etkisi

```
Kernel 3x3  → Hafif blur, detay korunur
Kernel 7x7  → Orta blur
Kernel 15x15 → Güçlü blur, detay kaybolur
Kernel 31x31 → Çok güçlü blur
```

## 🔧 Sigma Parametresi

Gaussian Blur'da sigma değeri:
- **Düşük sigma** → Keskin blur, dar çan eğrisi
- **Yüksek sigma** → Yumuşak blur, geniş çan eğrisi
- **0** → Kernel boyutundan otomatik hesapla

## 🎨 Gürültü (Noise) Türleri

| Gürültü | Özellik | En İyi Çözüm |
|---------|---------|--------------|
| Gaussian | Her yerde hafif | Gaussian Blur |
| Tuz-Biber | Siyah/beyaz noktalar | Median Blur |
| Speckle | Benekli | Bilateral |

## 🌍 Gerçek Kullanım

### Preprocessing (Ön İşleme)
```python
# CNN'e vermeden önce
img = cv2.GaussianBlur(img, (5, 5), 0)  # Gürültüyü azalt
# Bu adım olmadan model başarısı düşer!
```

### Depth of Field Efekti
```python
# Arka planı bulanıklaştırma
mask = get_foreground_mask(img)
blurred_bg = cv2.GaussianBlur(img, (21, 21), 0)
result = np.where(mask, img, blurred_bg)
```

## ⚠️ Önemli Notlar

1. **Kernel boyutu TEK SAYI olmalı**: 3, 5, 7, 9...
2. **Büyük kernel = Daha yavaş işlem**
3. **Blur, geri dönüşü olmayan bir işlemdir** - orijinal detaylar kaybolur
4. **Edge detection öncesi blur gerekli** - gürültüyü kenar olarak algılamamak için

## 🔗 Sonraki Adım
→ [05_gaussian_blur_manual](../05_gaussian_blur_manual/) - Blur'u kendin yaz!
