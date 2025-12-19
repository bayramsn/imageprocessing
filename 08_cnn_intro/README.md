# 🧠 CNN Nedir? - Mini CNN Görselleştirici

Bu klasör, MNIST veri seti üzerinde basit bir evrişimli sinir ağı (CNN) eğiten,
eğitim sürecini görselleştiren ve feature map'leri analiz eden araçlar içerir.

## 🎯 Amaç
CNN'in içini "kara kutu" olmaktan çıkarmak. Her katmandan çıkan feature map'leri görselleştirmek.

## 📦 Gereksinimler
```bash
pip install tensorflow numpy matplotlib opencv-python
```

## 🚀 Hızlı Başlangıç

### Model Eğitimi
```powershell
& "C:\opencv yakalayıcı\.venv\Scripts\python.exe" "C:/opencv yakalayıcı/08_cnn_intro/mnist_cnn.py" ^
  --epochs 5 --batch-size 128 ^
  --model-out "mnist_cnn.h5" ^
  --plot-out "training_curve.png"
```

### Feature Map Görselleştirme
```bash
python cnn_visualizer.py mnist_cnn.h5
python cnn_visualizer.py mnist_cnn.h5 --kernels
```

- Eğitim sonunda test doğruluğu yazdırılır, model `mnist_cnn.h5` olarak kaydedilir, kayıp/doğruluk grafiği `training_curve.png` olarak çıkartılır.

## Özel Rakam Tahmini
1. 28x28 boyutlu, gri tonlamalı bir PNG hazırlayın (açık zemin, koyu rakam). Gerekirse Paint’te çizip yeniden boyutlandırın.
2. Çalıştırın:
```powershell
& "C:\opencv yakalayıcı\.venv\Scripts\python.exe" "C:/opencv yakalayıcı/08_cnn_intro/mnist_cnn.py" ^
  --predict "C:/opencv yakalayıcı/08_cnn_intro/my_digit.png"
```
- Model önceden eğitilmediyse bu komut önce eğitir, sonra tahmin yapar. Eğitili modelin varsa `--model-out` ile aynı ada kaydetmiş olmanız yeterli, yeniden eğitmek istemiyorsanız kodda yükleme adımı ekleyip çağırabilirsiniz.

## Parametreler
- `--epochs`: Eğitim epoch sayısı (varsayılan 5)
- `--batch-size`: Batch boyutu (varsayılan 128)
- `--model-out`: Kayıt yolu (varsayılan `mnist_cnn.h5`)
- `--plot-out`: Grafik çıktısı (varsayılan `training_curve.png`)
- `--predict`: Özel 28x28 gri görüntü yolu (boşsa sadece eğitim/test yapılır)

## İpuçları
- Daha yüksek doğruluk için epoch sayısını artırın; overfitting görürseniz `validation_split=0.1` zaten etkili, gerekirse erken durdurma ekleyebilirsiniz.
- GPU varsa TensorFlow otomatik kullanır; yoksa CPU’da çalışır (daha yavaş). CUDA kurulu değilse uyarı alabilirsiniz; sorun değil.
- Tahminlerde rakamı siyah, zemini beyaz yapın; tersse koddaki `1.0 - (arr / 255.0)` satırını kaldırın.
