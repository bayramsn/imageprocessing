# 🔬 Gaussian Blur'u Kendin Yaz - Custom Gaussian Filter

## 🎯 Amaç
"Hazır fonksiyon kullanmadan" filtre mantığını öğrenmek. Bu kısmı anlayan CNN'i de anlar!

## 📦 Gereksinimler
```bash
pip install opencv-python numpy scipy matplotlib
```

## 🚀 Kullanım
```bash
python custom_gaussian.py resim.jpg
python custom_gaussian.py resim.jpg --kernel 7 --sigma 2.0
python custom_gaussian.py resim.jpg --benchmark
```

## 🧠 Öğrenecekleriniz

### Kernel Nedir?

Kernel (çekirdek), görüntü üzerinde gezdirilen küçük bir matristir.

```
Gaussian Kernel 3x3 örneği:
┌─────────────────────┐
│ 0.075  0.124  0.075 │
│ 0.124  0.204  0.124 │
│ 0.075  0.124  0.075 │
└─────────────────────┘
Toplam = 1.0 (normalize)
```

### Gaussian Formülü

$$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

Python'da:
```python
def gaussian_kernel(size, sigma):
    x = np.arange(size) - size // 2
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum()  # Normalize
```

### Convolution (Evrişim) İşlemi

```
Her piksel için:
1. Kernel'i pikselin üzerine yerleştir
2. Karşılıklı değerleri çarp
3. Topla
4. Sonuç = yeni piksel değeri

Görüntü:          Kernel:         Sonuç:
┌───┬───┬───┐    ┌───┬───┬───┐
│ 10│ 20│ 30│    │0.1│0.2│0.1│
├───┼───┼───┤ ⊛  ├───┼───┼───┤  = 25
│ 40│ 50│ 60│    │0.2│0.4│0.2│
├───┼───┼───┤    ├───┼───┼───┤
│ 70│ 80│ 90│    │0.1│0.2│0.1│
└───┴───┴───┘    └───┴───┴───┘
```

### Elle Convolution Kodu

```python
def convolve2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Padding ekle (kenarlar için)
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    
    output = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            # Bölgeyi al ve kernel ile çarp
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    
    return output
```

## ⚡ Performans Karşılaştırması

| Yöntem | 512x512 Görüntü | Hız |
|--------|-----------------|-----|
| Elle (Python loop) | ~5-10 saniye | 🐢 |
| NumPy vectorized | ~0.5 saniye | 🐇 |
| SciPy convolve2d | ~0.1 saniye | 🚀 |
| OpenCV GaussianBlur | ~0.01 saniye | ⚡ |

## 🌉 CNN Bağlantısı

**Convolution = CNN'in temeli!**

```
CNN'de:
- Kernel = Öğrenilebilir filtre
- Birden fazla kernel = Farklı özellikler
- Edge kernel → Kenar bulur
- Blur kernel → Yumuşatır
- Özel kernel → Model öğrenir
```

### CNN Katmanı vs Gaussian Blur

| Gaussian Blur | CNN Conv Layer |
|---------------|----------------|
| Sabit kernel | Öğrenilen kernel |
| Tek kernel | Birçok kernel |
| Blur amaçlı | Özellik çıkarma |
| Elle tasarlanır | Eğitimle bulunur |

## 🎯 Kernel Örnekleri

```python
# Sharpen (Keskinleştirme)
sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# Sobel X (Yatay kenar)
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Emboss (Kabartma)
emboss = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
])
```

## 📊 Sigma ve Kernel Boyutu İlişkisi

```
Sigma büyürse → Kernel de büyümeli
Kural: kernel_size ≈ 6 * sigma + 1

Sigma=0.5 → Kernel 5x5
Sigma=1.0 → Kernel 7x7
Sigma=2.0 → Kernel 13x13
```

## ⚠️ Sık Yapılan Hatalar

1. **Kernel normalize edilmemiş** → Görüntü karanlık/aydınlık olur
2. **Padding unutulmuş** → Kenarlar siyah kalır
3. **Yanlış veri tipi** → overflow/underflow

## 🔗 Sonraki Adım
→ [06_traditional_image_processing](../06_traditional_image_processing/) - Geleneksel CV yöntemleri
