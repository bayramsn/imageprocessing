# Beginner CV + DL Mini Projeleri

Bu depo dört küçük projeyi ve ortak yardımcıları içerir. Amaç: klasik bilgisayarlı görü ve hazır derin öğrenme modelleriyle çıkarımı adım adım göstermek.

## Kurulum
```
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell için
pip install -r requirements.txt
```

## Yapı
- utils.py — Ortak yardımcılar (görüntü yükleme, tensöre çevirme, kutu çizme).
- project_1_similarity.py — ORB ile iki görüntüde benzerlik; iyi eşleşme sayısına göre skor.
- project_2_edges.py — Kenar sayısına dayalı EMPTY / NOT EMPTY kural tabanlı sınıflandırma.
- project_3_cnn_ready.py — Hazır (ön eğitimli) Mobilenet veya ResNet ile sınıflandırma, yalnızca çıkarım.
- project_4_compare.py — Aynı görüntüde sınıflandırma ve nesne tespitini yan yana gösterir; Faster R-CNN kullanır.
- requirements.txt — Gerekli paketler.

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
