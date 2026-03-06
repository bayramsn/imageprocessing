# Staj Raporu Şablonu – Gerçek Zamanlı Çok Kişili Davranış Analitiği

> Bu belge, staj süresince geliştirilen **cv-human-behavior-analytics** projesi için
> hazırlanacak raporun bölüm başlıklarını ve her bölümde yazılması gereken içeriklerin
> kısa açıklamalarını sunar. Her bölümü doldurup kendi deney sonuçlarınız ve
> gözlemlerinizle zenginleştirmeniz beklenir.

---

## Kapak Sayfası

- Üniversite / Fakülte / Bölüm logosu ve adı
- Rapor başlığı: *"Gerçek Zamanlı Çok Kişili İnsan Davranış Analitiği: YOLOv8, ByteTrack ve Poz Tahmini Tabanlı Uçtan Uca Bir Bilgisayarla Görü Sistemi"*
- Öğrencinin adı, numarası
- Staj yeri (şirket / kurum adı)
- Danışman adı
- Tarih

---

## Özet (Abstract)

> **Burada ne yazılmalı:**
> - Projenin amacı (1-2 cümle)
> - Kullanılan ana yöntemler (YOLOv8, ByteTrack, poz tahmini, kural tabanlı davranış sınıflandırma)
> - Elde edilen ana sonuçlar (FPS, tespit doğruluğu, davranış tanıma başarısı)
> - Sonuç cümlesi
> - 150-250 kelime arasında

**Anahtar Kelimeler:** Nesne Tespiti, YOLO, Çoklu Nesne Takibi, ByteTrack, Poz Tahmini, Davranış Tanıma, Bilgisayarla Görü, Gerçek Zamanlı Sistem

---

## 1. Giriş

### 1.1 Problem Tanımı
- Güvenlik kameraları ve akıllı bina sistemlerinde insan davranışlarının gerçek zamanlı izlenmesi ihtiyacı
- Mevcut sistemlerin sınırlamaları (yalnızca tespit, kimlik tutarlılığı yok, davranış analizi yok)
- Bu projenin çözmeyi hedeflediği somut problem

### 1.2 Motivasyon
- Endüstriyel/ticari alandaki uygulama alanları (güvenlik, perakende, sağlık)
- Staj bağlamında öğrenme hedefleri
- Neden uçtan uca (end-to-end) bir boru hattı gerekli?

### 1.3 Projenin Kapsamı
- Neleri kapsıyor (tespit, takip, poz, davranış, loglama, görselleştirme)
- Neleri kapsamıyor (yüz tanıma, aktivite tahmini, bulut dağıtımı vb.)

### 1.4 Raporun Organizasyonu
- Bölümlerin kısa tanıtımı

---

## 2. Kuramsal Arka Plan (Literature / Background)

### 2.1 Nesne Tespiti (Object Detection)
- İki aşamalı (two-stage) vs tek aşamalı (single-stage) detektörler
- R-CNN ailesi (R-CNN → Fast R-CNN → Faster R-CNN) kısa özet
- SSD, RetinaNet gibi tek aşamalı yaklaşımlar
- Değerlendirme metrikleri: mAP, IoU, precision, recall

### 2.2 YOLO Ailesi (v1 → v8)
- YOLOv1 temel fikir: ızgara tabanlı, tek geçişli (single-pass) tespit
- YOLOv3: Darknet-53, çok ölçekli tahmin, çapa kutuları (anchor boxes)
- YOLOv5: PyTorch geçişi, pratik iyileştirmeler
- YOLOv8: çapa-sız (anchor-free) kafa, C2f modülleri, birleşik API
- Çapa tabanlı vs çapa-sız karşılaştırma
- *(Detaylı karşılaştırma: bkz. `docs/YOLO_v3_to_v8_Differences.md`)*

### 2.3 Çoklu Nesne Takibi (Multi-Object Tracking)
- Tracking-by-detection paradigması
- Veri ilişkilendirme (data association) problemi
- ByteTrack: düşük güvenlikli tespitlerin dahil edilmesi
- Track ID tutarlılığı, yeniden tanımlama (re-ID) kavramı
- Metrikler: MOTA, MOTP, IDF1

### 2.4 Poz Tahmini (Pose Estimation)
- COCO 17-keypoint formatı
- Yukarıdan aşağı (top-down) vs aşağıdan yukarı (bottom-up) yaklaşımlar
- YOLOv8-Pose: tespit + poz tek geçişte
- Metrik: OKS (Object Keypoint Similarity)

### 2.5 Davranış Tanıma (Behavior Recognition)
- Eylem tanıma (action recognition) alanına genel bakış
- Keypoint tabanlı basit sınıflandırma (bu projede kullanılan yaklaşım)
- Kural tabanlı (rule-based) vs öğrenme tabanlı (learning-based) yöntemler
- Açı hesaplama, hız hesaplama, postür kuralları

### 2.6 Görüntü Segmentasyonu (Image Segmentation)
- Semantik vs örnek (instance) segmentasyon
- YOLO-Seg mimarisi: proto-mask + katsayılar
- Bu projede opsiyonel kullanımı

### 2.7 Araştırma: OpenCLIP ve DINOv2
- CLIP: kontrastif dil-görüntü ön eğitimi, sıfır atış (zero-shot) sınıflandırma
- OpenCLIP: açık kaynak CLIP, LAION veri seti
- DINOv2: öz-denetimli (self-supervised) görsel öznitelikler
- Bu projeyle potansiyel entegrasyon senaryoları

---

## 3. Sistem Tasarımı ve Mimari

### 3.1 Genel Boru Hattı (Pipeline) Diyagramı
- ASCII veya çizim olarak uçtan uca akış
- Video kaynağı → Tespit → Takip → Poz → Davranış → Zamanlayıcı → Log → Görselleştirme

### 3.2 Modüler Yapı
- Her modülün sorumluluğu ve arayüzü (interface)
- Bağımlılık ilişkileri

### 3.3 Yapılandırma Yönetimi
- YAML config dosyası
- CLI argümanları ve config birleştirme mantığı

### 3.4 Kullanılan Teknolojiler
- Python 3.10+, Ultralytics, OpenCV, NumPy
- ByteTrack (Ultralytics entegrasyonu)
- YAML, CSV, JSON

---

## 4. Uygulama Detayları (Implementation)

### 4.1 `video_source.py` — Video Kaynağı
- Webcam, video dosyası, RTSP desteği
- Stride (kare atlama) mekanizması
- Context manager (`with` bloğu) kullanımı

### 4.2 `detector_yolo.py` — Kişi Tespiti
- YOLOv8 model yükleme
- Yalnızca "person" sınıfı filtreleme
- Detection veri sınıfı: bbox, confidence

### 4.3 `tracker.py` — Çoklu Nesne Takibi
- Ultralytics `model.track()` API kullanımı
- ByteTrack entegrasyonu
- Track ID yönetimi, kişi sayımı (current / total)
- "Aynı kişiyi iki kez saymama" mantığı

### 4.4 `pose.py` — Poz Tahmini
- YOLOv8-Pose model kullanımı
- Track-poz eşleştirme (IoU tabanlı)
- 17 anahtar nokta çıktısı

### 4.5 `behavior.py` — Davranış Sınıflandırma
- Kural tabanlı sınıflandırıcı
- Diz açısı → oturma tespiti
- Ayak bileği hızı → yürüme / koşma ayrımı
- Varsayılan → ayakta durma
- Güven skoru hesaplama

### 4.6 `timer.py` — Süre Takibi
- Track başına durum makinesi (state machine)
- Davranış geçişlerinde segment kapatma / açma
- Kümülatif süre hesaplama
- Track kaybolduğunda segment sonlandırma

### 4.7 `logger.py` — Veri Kayıt
- CSV olay kaydı (event log)
- JSON oturum özeti (session summary)
- Dosya adlandırması (zaman damgalı)
- Exit handler ile güvenli kapatma

### 4.8 `overlay.py` — Görselleştirme
- Bbox + ID + davranış etiketi + süre çizimi
- İskelet (keypoint) çizimi
- Global istatistikler (FPS, kişi sayısı)

### 4.9 `segmentation.py` — Segmentasyon (Opsiyonel)
- YOLOv8-Seg ile kişi maskeleri
- Track-maske eşleştirme
- Yarı saydam renk katmanı

### 4.10 `main.py` — CLI ve Orkestrasyon
- Argparse yapılandırması
- Config-CLI birleştirme
- Ana döngü mantığı
- Temiz kapatma (cleanup)

---

## 5. Deneyler ve Sonuçlar

### 5.1 Deney Ortamı
- Donanım: CPU modeli, GPU modeli (varsa), RAM
- İşletim sistemi
- Python ve kütüphane sürümleri
- Test videoları: kaynak, çözünürlük, süre, kişi sayısı

### 5.2 FPS ve Gecikme Ölçümleri

| Yapılandırma | Model | Cihaz | Stride | Ort. FPS | Ort. Gecikme (ms) |
|-------------|-------|-------|--------|----------|-------------------|
| Yalnızca tespit | yolov8n | CPU | 1 | — | — |
| Tespit + takip | yolov8n | CPU | 1 | — | — |
| Tespit + takip + poz | yolov8n | CPU | 1 | — | — |
| Tam boru hattı | yolov8n | CPU | 1 | — | — |
| Tam boru hattı | yolov8n | CPU | 2 | — | — |
| Tam boru hattı | yolov8n | CUDA | 1 | — | — |

> *(Tabloyu kendi ölçümlerinizle doldurun. Bkz. `docs/Benchmark_Template.md`)*

### 5.3 Ablasyon Çalışması (Ablation Study)

#### 5.3.1 Modül Etkisi
- Yalnızca tespit vs tespit+takip vs tam boru hattı
- Her modülün FPS üzerindeki etkisi

#### 5.3.2 Stride Etkisi
- stride=1 vs 2 vs 3 karşılaştırması
- FPS kazancı vs takip kalitesi kaybı

#### 5.3.3 Model Boyutu Etkisi
- yolov8n vs yolov8s: hız ve doğruluk farkı

### 5.4 Davranış Sınıflandırma Doğruluğu
- Gözleme dayalı nitel değerlendirme
- Hangi durumlarda doğru, hangi durumlarda hatalı?
- Yaygın hata senaryoları (oturma tespit başarısızlığı, yürüme/koşma karışıklığı)

### 5.5 Takip Kalitesi
- ID tutarlılığı gözlemleri
- ID sıçramaları (ID switches) hangi durumlarda oluyor?
- Kışı sayımı doğruluğu

### 5.6 Örnek Çıktılar
- Ekran görüntüleri (overlay ile)
- Örnek CSV log çıktısı
- Örnek JSON session çıktısı

---

## 6. Tartışma (Discussion)

### 6.1 Başarılar
- Gerçek zamanlı çalışma performansı
- Modüler ve genişletilebilir mimari
- Uçtan uca entegrasyon

### 6.2 Sınırlamalar
- Kural tabanlı davranış sınıflandırmanın kısıtları (4 sınıf, basit kurallar)
- Poz tahmini hataları (ışık, örtüşme, kısmi görünürlük)
- ByteTrack uzun süreli tıkanıklıklarda (occlusion) ID kaybı
- Tek kameralı operasyon – çoklu kamera desteği yok

### 6.3 Gelecekte İyileştirme Önerileri
- Öğrenme tabanlı davranış sınıflandırma (LSTM, Transformer)
- DINOv2 / OpenCLIP tabanlı kişi yeniden tanıma (re-ID)
- TensorRT / ONNX ile dağıtım optimizasyonu
- Anomali tespiti eklenmesi
- Çoklu kamera füzyonu

---

## 7. Etik, Gizlilik ve KVKK Değerlendirmeleri

### 7.1 Kişisel Verilerin Korunması
- KVKK (Kişisel Verilerin Korunması Kanunu) bağlamında yükümlülükler
- Video izleme sistemlerinin hukuki çerçevesi
- Veri saklama, erişim ve silme politikaları

### 7.2 Gizlilik İlkeleri
- Yalnızca tespit ve takip – yüz tanıma yapılmıyor
- Kişi kimliklendirme: track_id ≠ gerçek kimlik
- Verilerin anonimleştirilmesi

### 7.3 Etik Kullanım
- Gözetim vs güvenlik ayrımı
- Rıza bildirimi (kamera uyarı tabelaları)
- Ayrımcılık riski (model önyargıları)

---

## 8. Sonuç

- Projenin özet değerlendirmesi
- Elde edilen temel çıktılar
- Öğrenilen dersler
- Gelecek çalışmalar

---

## 9. Kaynakça (References)

> **Not:** Aşağıdaki referansları kullanılan kaynaklara göre düzenleyin ve ekleyin.

1. Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." CVPR 2016.
2. Redmon, J., & Farhadi, A. "YOLOv3: An Incremental Improvement." arXiv:1804.02767, 2018.
3. Bochkovskiy, A., et al. "YOLOv4: Optimal Speed and Accuracy of Object Detection." arXiv:2004.10934, 2020.
4. Jocher, G., et al. "Ultralytics YOLOv8." GitHub, 2023. https://github.com/ultralytics/ultralytics
5. Zhang, Y., et al. "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." ECCV 2022.
6. Cao, Z., et al. "OpenPose: Realtime Multi-Person 2D Pose Estimation." IEEE TPAMI 2019.
7. Sun, K., et al. "Deep High-Resolution Representation Learning for Visual Recognition." CVPR 2019.
8. He, K., et al. "Mask R-CNN." ICCV 2017.
9. Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.
10. Oquab, M., et al. "DINOv2: Learning Robust Visual Features without Supervision." arXiv:2304.07193, 2023.
11. Ilharco, G., et al. "OpenCLIP." GitHub, 2021. https://github.com/mlfoundations/open_clip

---

## Ekler (Appendices)

### Ek A: Kurulum ve Çalıştırma Rehberi
*(README.md'den alınabilir)*

### Ek B: Yapılandırma Dosyası (`configs/default.yaml`)
*(Tam içerik)*

### Ek C: Benchmark Sonuçları
*(Detaylı tablolar)*

### Ek D: Örnek Log Dosyaları
*(CSV ve JSON örnekleri)*
