# 🎥 OpenCV Giriş - Webcam Görüntü Deney Laboratuvarı

## 🎯 Amaç
OpenCV'nin temel yapı taşlarını öğrenmek ve gerçek zamanlı görüntü işlemeye giriş yapmak.

## 📦 Gereksinimler
```bash
pip install opencv-python numpy
```

## 🚀 Kullanım
```bash
python webcam_filter.py
```

## ⌨️ Tuş Kontrolleri

| Tuş | Mod | Açıklama |
|-----|-----|----------|
| `c` | Normal | Orijinal renkli görüntü |
| `g` | Gri | Gri tonlama (grayscale) |
| `b` | Blur | Gaussian bulanıklaştırma |
| `r` | Resize | Yarı çözünürlük |
| `q` | Çıkış | Programı kapat |

## 🧠 Öğrenecekleriniz

### 1. Video Yakalama
```python
cap = cv2.VideoCapture(0)  # 0 = varsayılan kamera
ret, frame = cap.read()     # Her kare için döngüde çağrılır
```

### 2. Renk Dönüşümü
```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# OpenCV varsayılan olarak BGR kullanır (RGB değil!)
```

### 3. Boyut Değiştirme
```python
resized = cv2.resize(frame, (width // 2, height // 2))
# Performans için küçültme sık kullanılır
```

### 4. Görüntü Gösterme
```python
cv2.imshow('Pencere Adı', frame)
key = cv2.waitKey(1)  # 1ms bekle, tuş oku
```

### 5. Kaynak Temizliği
```python
cap.release()           # Kamerayı serbest bırak
cv2.destroyAllWindows() # Pencereleri kapat
```

## 📊 FPS Hesaplama

Performans ölçümü için FPS ekleyebilirsiniz:
```python
import time

prev_time = time.time()
while True:
    ret, frame = cap.read()
    
    # FPS hesapla
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```

## 🌍 Gerçek Hayat Kullanımları

- **Güvenlik kameraları**: 7/24 video akışı
- **Video konferans**: Zoom, Teams arka plan efektleri
- **Oyun**: Hareket yakalama, gesture control
- **Sürücüsüz araçlar**: Kamera tabanlı algılama

## ⚠️ Sık Yapılan Hatalar

1. **Kamera açılmıyor**: `cap.isOpened()` ile kontrol edin
2. **Siyah ekran**: Başka bir program kamerayı kullanıyor olabilir
3. **Yavaş performans**: Çözünürlüğü düşürün veya işlemleri optimize edin

## 🔗 Sonraki Adım
→ [04_gaussian_blur_opencv](../04_gaussian_blur_opencv/) - Blur filtreleri
