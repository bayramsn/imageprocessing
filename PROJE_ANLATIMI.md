# 4 Mini Projeyi Adım Adım Anlattım

Bu dokümanda yaptığım dört mini projeyi kendi ağzımdan, adım adım anlatıyorum. Amacım klasik bilgisayarlı görü ile derin öğrenme çıkarımını öğrenmekti.

---

## 🔹 Proje 1: Görüntü Benzerliği (ORB Keypoints)

**Dosya:** `project_1_similarity.py`

**Ne yapmak istedim?**
İki fotoğrafın aynı nesneyi gösterip göstermediğini anlamak istedim. Mesela aynı binanın iki farklı açıdan çekilmiş fotoğrafı mı, yoksa tamamen farklı iki bina mı?

### Adımlarım:

**1. Önce iki görüntüyü yükledim:**
```python
img1 = load_image_bgr(args.image1)
img2 = load_image_bgr(args.image2)
```
`load_image_bgr` fonksiyonumu kullandım. Bu fonksiyon Unicode karakterli yollarda bile çalışıyor çünkü `cv2.imdecode` fallback'i var.

**2. Görüntüleri gri tona çevirdim:**
```python
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
```
ORB algoritması tek kanallı görüntü istiyor. Renk bilgisine ihtiyacım yok, sadece yoğunluk değerleri yeterli.

**3. ORB dedektörünü oluşturdum:**
```python
orb = cv2.ORB_create(nfeatures=2000)
```
ORB'u seçtim çünkü:
- Hızlı çalışıyor
- Patentsiz (SIFT gibi değil)
- `nfeatures=2000` ile en fazla 2000 anahtar nokta bulmasını söyledim

**4. Her görüntüde keypoint ve tanımlayıcıları çıkardım:**
```python
kp1, desc1 = orb.detectAndCompute(gray1, None)
```
Burada iki şey elde ettim:
- `kp1`: Anahtar noktaların koordinatları, açıları, boyutları
- `desc1`: Her keypoint için 32 baytlık ikili tanımlayıcı (parmak izi gibi düşün)

**5. Tanımlayıcıları eşleştirdim:**
```python
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
knn_matches = matcher.knnMatch(desc1, desc2, k=2)
```
Neden Hamming mesafesi? Çünkü ORB ikili tanımlayıcı üretiyor, Hamming bit farklarını sayıyor.
Neden `k=2`? Lowe oran testi için en yakın iki eşleşmeyi almam gerekiyordu.

**6. Lowe oran testini uyguladım:**
```python
for m, n in knn_matches:
    if m.distance < args.ratio * n.distance:
        good.append(m)
```
Bu testin mantığı şu: Eğer en iyi eşleşme (m) ikinci en iyiden (n) belirgin şekilde daha iyiyse, bu güvenilir bir eşleşmedir. Yoksa muhtemelen gürültüdür.

**7. Sonucu değerlendirdim:**
```python
similarity_score = len(good)
if similarity_score >= args.min_matches:
    print("Sonuç: BENZER")
```
İyi eşleşme sayısı benim benzerlik skorum oldu. 20'nin üstündeyse "BENZER" dedim.

### Bu projeden ne öğrendim?
- Keypoint: Görüntüdeki ayırt edici noktalardır (köşeler, bloblar)
- Descriptor: O noktanın sayısal parmak izi
- Bu yöntem derin öğrenme **değil**; elle tasarlanmış özellikler kullanıyor
- Işık ve açı değişiminde zorlanıyor — bu normal

---

## 🔹 Proje 2: Kural Tabanlı Sınıflandırma (Kenar Sayımı)

**Dosya:** `project_2_edges.py`

**Ne yapmak istedim?**
Bir rafın fotoğrafına bakıp "boş mu dolu mu?" sorusuna cevap vermek istedim. Ama CNN kullanmadan, basit bir kuralla.

### Adımlarım:

**1. Görüntüyü yükleyip gri yaptım:**
```python
img = load_image_bgr(args.image)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

**2. Gaussian blur uyguladım:**
```python
ksize = max(3, args.blur | 1)
blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
```
Neden blur? Gürültüyü azaltmak için. Yoksa Canny her yerde sahte kenarlar buluyor.
`| 1` hilesini kullandım: bu bit işlemi çift sayıyı tek yapıyor (OpenCV tek kernel istiyor).

**3. Canny kenar tespiti yaptım:**
```python
edges = cv2.Canny(blurred, args.canny_low, args.canny_high)
```
Canny'nin iki eşiği var:
- Alt eşik (50): Bunun altındaki gradyanlar kenar değil
- Üst eşik (150): Bunun üstündekiler kesinlikle kenar
- Aradakiler: Güçlü kenarlara bağlıysa kenar sayılır

**4. Kenar piksellerini saydım:**
```python
edge_pixels = int(np.count_nonzero(edges))
```
Canny çıktısı siyah-beyaz. Beyaz pikseller kenar. `count_nonzero` ile saydım.

**5. Basit bir kuralla karar verdim:**
```python
label = "NOT EMPTY" if edge_pixels > args.edge_thresh else "EMPTY"
```
Eğer kenar sayısı 500'ün üstündeyse rafta bir şeyler var demektir.

### Bu projeden ne öğrendim?
- CNN öncesi dönemde insanlar özellikleri elle tanımlıyordu
- Bu yöntem ölçeklenmiyor: farklı ışık, açı, nesne türlerinde çuvalladı
- **İşte bu yüzden CNN'lere ihtiyaç var** — motivasyonu anladım

---

## 🔹 Proje 3: Hazır CNN ile Sınıflandırma

**Dosya:** `project_3_cnn_ready.py`

**Ne yapmak istedim?**
Ön eğitimli bir CNN kullanarak görüntüdeki nesneyi tanımak istedim. Ama model eğitmedim, sadece çıkarım yaptım.

### Adımlarım:

**1. Cihazı seçtim:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
GPU varsa kullan dedim, yoksa CPU'da çalışsın.

**2. Ön eğitimli modeli yükledim:**
```python
weights = models.MobileNet_V3_Large_Weights.DEFAULT
model = models.mobilenet_v3_large(weights=weights)
model.eval()
```
- `weights=...`: ImageNet üzerinde eğitilmiş ağırlıkları otomatik indirdi
- `model.eval()`: Dropout ve BatchNorm'u çıkarım moduna aldım (önemli!)

**3. Görüntüyü modelin beklediği formata getirdim:**
```python
img_resized = resize_rgb(img_rgb, target_hw[::-1])  # 224x224 yaptım
tensor = to_torch_tensor(img_resized)               # HWC → CHW, /255 ile normalize ettim
tensor = normalize_tensor(tensor, mean, std)        # ImageNet ortalaması/std ile normalize ettim
batch = tensor.unsqueeze(0)                         # Batch boyutu ekledim
```
Bu adımlar zorunlu. Model tam olarak bu formatı bekliyor, yoksa saçma sonuçlar veriyor.

**4. Çıkarım yaptım:**
```python
with torch.no_grad():
    outputs = model(batch)
    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
```
- `torch.no_grad()`: Gradyan hesaplamadım — bellek ve hız kazandım
- `softmax`: Ham skorları olasılıklara çevirdim (toplamı 1 oldu)

**5. En yüksek olasılıklı sınıfları yazdırdım:**
```python
scores, indices = torch.topk(probs, topk)
label = categories[idx]
print(f"{rank}. {label}: {score:.3f}")
```

### Bu projeden ne öğrendim?
- Sınıflandırma: "Bu görüntüde ne var?" sorusuna tek cevap verir
- Eğitim vs Çıkarım: Ben sadece ileri geçiş yaptım, ağırlıklar güncellenmedi
- Transfer öğrenme: Başkasının (ImageNet'te) eğittiği modeli kullandım
- Ön işleme kritik: Yanlış normalize edersem model çöp üretiyor

---

## 🔹 Proje 4: Sınıflandırma vs Nesne Tespiti

**Dosya:** `project_4_compare.py`

**Ne yapmak istedim?**
Aynı görüntüde hem sınıflandırma hem nesne tespiti yapıp farkı görmek istedim.

### BÖLÜM A — Sınıflandırma Yaptım:

```python
cls_label, cls_score, exists = run_classification(img_rgb)
```
ResNet18 kullandım. Tek etiket ve güven skoru döndü. Basit bir eşikle "var/yok" dedim.

### BÖLÜM B — Nesne Tespiti Yaptım:

**1. Detection modelini yükledim:**
```python
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
```
Faster R-CNN kullandım. COCO veri setinde 80 sınıf üzerinde eğitilmiş.

**2. Çıkarım yaptım:**
```python
batch = [tensor.to(device)]  # Liste olarak verdim!
outputs = model(batch)
```
Önemli bir fark: Detection modelleri giriş olarak **liste** bekliyor. Çünkü farklı boyutlu görüntüler olabilir.

**3. Çıktıyı çözümledim:**
```python
boxes = out["boxes"]   # Her nesne için [x1, y1, x2, y2] koordinatları
scores = out["scores"] # Her kutu için güven skoru
labels = out["labels"] # Her kutu için sınıf indeksi
```
Sınıflandırmadan farklı olarak burada **birden fazla nesne** ve **konum bilgisi** var.

**4. Kutuları çizdim:**
```python
drawn = draw_boxes(img, boxes, labels, scores, score_thresh=0.5)
```
Skor eşiğinin üstündeki tespitler için dikdörtgen ve etiket çizdim.

### İki yaklaşımı karşılaştırdım:

| Özellik | Sınıflandırma | Nesne Tespiti |
|---------|---------------|---------------|
| Çıktı | Tek etiket | Birden fazla kutu + etiket |
| Konum bilgisi | Yok | Bounding box koordinatları var |
| Soru | "Bu ne?" | "Nerede ne var?" |
| Hesaplama maliyeti | Daha hafif | Çok daha ağır |

### Bu projeden ne öğrendim?
- Sınıflandırma sadece "var/yok" diyor, nerede olduğunu söylemiyor
- Detection hem sınıfı hem konumu veriyor
- Detection çok daha fazla hesaplama gerektiriyor (FPN, RPN, NMS aşamaları var)
- Kullanım senaryosuna göre doğru olanı seçmeliyim

---

## 🎯 Genel Özet

```
Görüntü
   │
   ├─► Proje 1: Keypoint çıkardım → Eşleştirdim → Benzerlik skoru aldım
   │
   ├─► Proje 2: Blur → Canny → Piksel saydım → Kural ile karar verdim
   │
   ├─► Proje 3: Resize → Normalize → CNN → Softmax → Top-K etiket aldım
   │
   └─► Proje 4: ┬─ Sınıflandırma → "Var/Yok" dedim
                └─ Detection → Kutular çizdim + Etiketler yazdım
```

## Sonraki Adımlarım

1. **YOLO denemeliyim** — Faster R-CNN'den daha hızlı
2. **Kendi veri setimle fine-tuning yapmalıyım** — Transfer öğrenmeyi pratiğe dökmeliyim
3. **Segmentasyon öğrenmeliyim** — Kutu yerine piksel bazlı maske

---


