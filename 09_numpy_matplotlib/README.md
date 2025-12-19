# NumPy & Matplotlib ile Görüntü Analiz Aracı

Bu proje, görüntü işlemenin temelini oluşturan **matris mantığını** ve **veri görselleştirmeyi** öğretir.

## 🎯 Öğrenecekleriniz

| Konu | Açıklama |
|------|----------|
| NumPy Slicing | `img[100:200, 150:250]` gibi bölge seçimi |
| Reshape | 2D/3D matrisleri düzleştirme ve yeniden şekillendirme |
| Boolean Indexing | `img[img > 200]` gibi koşullu seçim |
| Matris = Görüntü | Her piksel bir sayı, görüntü bir matris |
| Matplotlib Subplot | Çoklu grafik düzeni |
| Histogram | Işık dağılımı analizi |

## 📦 Gereksinimler

```bash
pip install numpy matplotlib opencv-python
```

## 🚀 Kullanım

### Temel Kullanım
```bash
python image_analyzer.py resim.jpg
```

### Analizi Kaydet
```bash
python image_analyzer.py resim.jpg --save analiz.png
```

### NumPy Demo Modu
```bash
python image_analyzer.py resim.jpg --demo
```

### Sadece İstatistik (Grafik Yok)
```bash
python image_analyzer.py resim.jpg --no-plot
```

## 📊 Çıktı Örneği

Program çalıştırıldığında 3x3 subplot gösterir:

```
┌─────────────────┬─────────────────┬─────────────────┐
│  Orijinal RGB   │   Gri Seviye    │ Yoğunluk Haritası│
├─────────────────┼─────────────────┼─────────────────┤
│ Gri Histogram   │  RGB Histogram  │Kümülatif Histogram│
├─────────────────┼─────────────────┼─────────────────┤
│Binary Threshold │Adaptif Threshold│   İstatistikler  │
└─────────────────┴─────────────────┴─────────────────┘
```

## 🧠 NumPy Kavramları

### Shape ve Boyutlar
```python
img.shape  # (480, 640, 3) -> (yükseklik, genişlik, kanal)
img.ndim   # 3 -> 3 boyutlu
img.size   # 921600 -> toplam eleman sayısı
```

### Slicing
```python
img[0, 0]           # İlk piksel (RGB değerleri)
img[0, 0, 0]        # İlk pikselin Red değeri
img[:, :, 0]        # Sadece Red kanalı
img[100:200, 50:150] # Belirli bir bölge
```

### Boolean Indexing
```python
bright = img[img > 200]  # 200'den büyük tüm pikseller
img[img < 50] = 0        # 50'den küçükleri siyah yap
```

### Matematiksel İşlemler
```python
negative = 255 - img     # Negatif görüntü
brighter = img + 50      # Parlaklık artır
contrast = img * 1.5     # Kontrast artır
```

## 📈 Histogram Nedir?

Histogram, görüntüdeki piksel değerlerinin dağılımını gösterir:

- **Sol tarafta yoğunluk** → Koyu görüntü
- **Sağ tarafta yoğunluk** → Parlak görüntü
- **Yayılmış histogram** → İyi kontrast
- **Dar histogram** → Düşük kontrast

## 🎨 Threshold Türleri

| Tür | Açıklama |
|-----|----------|
| Binary | Sabit eşik değeri (127) ile siyah/beyaz |
| Adaptif | Her bölge için farklı eşik, gölgeli görüntüler için ideal |

## 📝 Örnek Çıktı

```
GÖRÜNTÜ ANALİZİ
==================================================
Boyut: 640x480 piksel
Kanal sayısı: 3
Veri tipi: uint8

Gri Seviye İstatistikleri:
  Min: 12, Max: 255
  Ortalama: 128.45
  Standart Sapma: 52.31
  Medyan: 130.0
```

## 💡 İpuçları

1. **Histogram Eşitleme**: Düşük kontrastlı görüntüleri iyileştirmek için `cv2.equalizeHist()` kullanın
2. **Renk Uzayları**: RGB yerine HSV veya LAB kullanmak bazen daha iyi sonuç verir
3. **Threshold Seçimi**: Otsu yöntemi (`cv2.THRESH_OTSU`) optimal eşik değerini otomatik bulur
