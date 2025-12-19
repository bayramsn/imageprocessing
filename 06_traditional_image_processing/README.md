# 🔧 Geleneksel Görüntü İşleme Yöntemleri

## 🎯 Amaç
Klasik yöntemlerle "AI'sız" sonuç almak. OCR, plaka tanıma, belge tarama gibi uygulamaların temeli.

## 📦 Gereksinimler
```bash
pip install opencv-python numpy matplotlib
```

## 🚀 Kullanım
```bash
python preprocessing_tool.py resim.jpg
python preprocessing_tool.py resim.jpg --mode threshold
python preprocessing_tool.py resim.jpg --mode morphology
python preprocessing_tool.py resim.jpg --interactive
```

## 🧠 Öğrenecekleriniz

### 1. Threshold (Eşikleme)

Görüntüyü siyah-beyaza çevirme:

```python
# Basit threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Otsu (otomatik eşik bulma)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptif (bölgesel eşik)
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
```

| Tür | Ne Zaman | Avantaj |
|-----|----------|---------|
| Binary | Tek renk arka plan | Basit, hızlı |
| Otsu | Bimodal histogram | Otomatik eşik |
| Adaptive | Değişken aydınlatma | Gölgelere dayanıklı |

### 2. Canny Edge Detection

```python
# Canny kenar tespiti
edges = cv2.Canny(gray, 50, 150)
# 50 = alt eşik, 150 = üst eşik
```

**Canny Adımları:**
1. Gaussian Blur (gürültü azaltma)
2. Gradient hesaplama (Sobel)
3. Non-maximum suppression (inceltme)
4. Hysteresis thresholding (bağlama)

### 3. Morphological İşlemler

```python
kernel = np.ones((5, 5), np.uint8)

# Erosion - Aşındırma (beyazı küçültür)
eroded = cv2.erode(binary, kernel, iterations=1)

# Dilation - Genişletme (beyazı büyütür)
dilated = cv2.dilate(binary, kernel, iterations=1)

# Opening - Erosion + Dilation (gürültü temizler)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Closing - Dilation + Erosion (delikleri kapatır)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

| İşlem | Etki | Kullanım |
|-------|------|----------|
| Erode | Küçültür | İnce çizgileri siler |
| Dilate | Büyütür | Kopuk parçaları birleştirir |
| Open | Gürültü temizler | Küçük noktaları siler |
| Close | Delik kapatır | İç boşlukları doldurur |

## 📊 İşlem Sırası (Pipeline)

Tipik OCR/belge tarama pipeline'ı:

```
1. Gri seviye dönüşümü
   ↓
2. Gaussian Blur (gürültü azaltma)
   ↓
3. Adaptive Threshold (siyah-beyaz)
   ↓
4. Morphological Open (gürültü temizle)
   ↓
5. Morphological Close (boşlukları doldur)
   ↓
6. Contour bulma (şekilleri tespit)
```

## 🎯 Kernel Şekilleri

```python
# Dikdörtgen (varsayılan)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Elips
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Çapraz
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
```

## 🌍 Gerçek Kullanım Örnekleri

### Plaka Tanıma
```python
# 1. Gri + Blur
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(gray, 11, 17, 17)

# 2. Kenar tespiti
edges = cv2.Canny(blur, 30, 200)

# 3. Kontur bul
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 4. Dikdörtgen konturları filtrele
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.018 * cv2.arcLength(cnt, True), True)
    if len(approx) == 4:  # 4 köşe = dikdörtgen
        # Plaka bulundu!
```

### Belge Tarama
```python
# Adaptif threshold (gölgelere dayanıklı)
binary = cv2.adaptiveThreshold(gray, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

# Gürültü temizle
clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
```

## ⚠️ Klasik Yöntemlerin Sınırları

| Sorun | Klasik Yöntem | Çözüm |
|-------|---------------|-------|
| Değişken aydınlatma | Adaptif threshold | ✅ |
| Karmaşık arka plan | Başarısız | CNN gerekli |
| Dönük/eğik nesneler | Zor | Derin öğrenme |
| Çoklu nesne türü | Parametre ayarı zor | YOLO/SSD |

## 💡 İpuçları

1. **Blur önce**: Threshold'dan önce mutlaka blur uygula
2. **Otsu dene**: Eşik değerini elle ayarlamak yerine
3. **Kernel boyutu**: İşlenecek nesne boyutuna göre seç
4. **Iteration sayısı**: Morphology'de 1-2 genelde yeterli

## 🔗 Sonraki Adım
→ [07_keypoints_features](../07_keypoints_features/) - Özellik çıkarımı
