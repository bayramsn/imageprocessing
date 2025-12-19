# Özel Kullanım Sözlüğü

Her dosyadaki dikkat çekici / alışılmışın dışındaki çağrılar ve yanındaki kısa açıklamalar.

## Genel Kalıplar
- `cv2.imdecode`: Unicode/bozuk yollar için dosyayı `open(..., "rb")` ile okuyup OpenCV'ye bayt dizisi vererek güvenli yükleme.
- `cv2.namedWindow(..., cv2.WINDOW_NORMAL)`: Pencereyi serbestçe yeniden boyutlandırmak için.
- `cv2.waitKey`: Döngü içinde tuş okuma ve çıkış kontrolü; çoğu yerde `& 0xFF` ile maskeleme yapılıyor.
- `np.pad(..., mode="reflect")`: Manuel konvolüsyonda sınır tekrarı için kullanılan yansımalı pad.

## Dosya Bazlı Liste
- [03_opencv_giris/webcam_filter.py](03_opencv_giris/webcam_filter.py)
  - `cv2.VideoCapture(0)`: Varsayılan kamerayı açar; `isOpened()` ile doğrulama yapılıyor.
  - `cv2.resize(..., (w//2, h//2))`: Çözünürlüğü hızlıca yarıya indirmek için integer bölme kullanımı.
  - `cv2.putText`: Gri/BGR durumuna göre dinamik yazı rengi seçimi.
  - `cv2.waitKey(1)`: Canlı akışta mod değiştirip çıkmak için düşük gecikmeli tuş okuma.

- [04_gaussian_blur_opencv/gaussian_blur_app.py](04_gaussian_blur_opencv/gaussian_blur_app.py)
  - `cv2.createTrackbar` + `cv2.getTrackbarPos`: Kernel ve sigma'yı canlı slider ile ayarlama.
  - `kernel_size = max(1, slider) * 2 + 1`: Trackbar değerini her zaman tek sayıya zorlayan formül.
  - `np.hstack((original, blurred))`: Yan yana gösterim için basit mosaik.
  - `cv2.imdecode`: Unicode yol fallback'i olarak manuel decode.

- [05_gaussian_blur_manual/all_filters_demo.py](05_gaussian_blur_manual/all_filters_demo.py)
  - `make_gaussian_kernel`: Elle normalleştirilmiş 2D Gaussian çekirdek üretimi.
  - `convolve2d_manual`: Renkli görüntüde kanal kanal kayan pencere konvolüsyonu; `padding` seçilebilir (`reflect`/`zero`).
  - `convolve2d_scipy`: SciPy varsa kanal bazlı `convolve2d` kullanımı; yoksa ImportError uyarısı.
  - `cv2.bilateralFilter`: Kenar koruyan bulanıklaştırma örneği.
  - `np.hstack` / `np.vstack`: Farklı filtre sonuçlarını ızgara halinde birleştirme.

- [05_gaussian_blur_manual/filters_from_blog.py](05_gaussian_blur_manual/filters_from_blog.py)
  - `cv2.boxFilter(..., normalize=True)` ve `cv2.blur`: Ortalama alma filtreleri arasındaki farkı görmek için.
  - `cv2.Sobel` + `cv2.magnitude`: X/Y gradyanlardan kenar büyüklüğü türetme.
  - `cv2.addWeighted`: Unsharp masking tarzı keskinleştirme (`1.5*orijinal - 0.5*gauss`).
  - Izgara oluştururken eksik hücreleri siyah pad ile doldurma (genişlik eşitleme).

- [05_gaussian_blur_manual/gaussian_blur_manual.py](05_gaussian_blur_manual/gaussian_blur_manual.py)
  - `make_gaussian_kernel`: Formülle üretilmiş ve 1'e normalleştirilmiş çekirdek.
  - `convolve2d`: Yansımalı (`reflect`) veya sıfır dolgulu manuel konvolüsyon.
  - `cv2.GaussianBlur(..., borderType=cv2.BORDER_REFLECT)`: OpenCV ile karşılaştırma için aynı sınır koşulunu seçme.
  - `np.abs(...).mean()`: Manuel ve OpenCV sonuçlarının ortalama mutlak farkını ölçme.

- [05_gaussian_blur_manual/gaussian_blur_scipy.py](05_gaussian_blur_manual/gaussian_blur_scipy.py)
  - `scipy.signal.convolve2d`: SciPy ile kanal bazlı Gaussian konvolüsyonu.
  - `cv2.getWindowProperty(..., cv2.WND_PROP_VISIBLE)`: Pencere kapandığında döngüyü otomatik bitirmek için görünürlük kontrolü.

- [06_traditional_image_processing/coin_counter.py](06_traditional_image_processing/coin_counter.py)
  - `cv2.createCLAHE`: Yerel kontrast artırımı için uyarlanabilir histogram eşitleme.
  - `ksize | 1`: Kernel değerini tek sayıya zorlamak için bitwise OR hilesi.
  - `cv2.adaptiveThreshold` ve `cv2.threshold(... | cv2.THRESH_OTSU)`: Uyarlamalı eşik + Otsu kombinasyonunu OR ile birleştirme seçeneği.
  - `cv2.morphologyEx` (open/close/erode/dilate) ve isteğe bağlı `close->open` sıralı temizlik.
  - `cv2.distanceTransform` + `cv2.watershed`: Dokunan paraları ayırmak için tohum tabanlı bölütleme.
  - `cv2.connectedComponents`: Watershed tohum etiketleme.
  - `cv2.Canny`: Kenar maskesi üretip sonuç tablosuna ekleme.
  - Dinamik min-alan: Yüzdelik (percentile) ve sabit taban değerin maksimumunu alarak kontur eleme.
  - `cv2.getWindowProperty` ile pencere kapanma algısı; `cv2.rectangle` ile bulunan coin'leri kutulama.

- [07_keypoints_features/logo_match.py](07_keypoints_features/logo_match.py)
  - `cv2.SIFT_create` / `cv2.ORB_create`: Özellik dedektörünü seçme; SIFT yoksa ORB fallback.
  - `cv2.BFMatcher` + `knnMatch(k=2)` ve Lowe oran testi (`m.distance < ratio * n.distance`).
  - `cv2.findHomography(..., cv2.RANSAC, 5.0)`: Eşleşmelerden sağlam homografi kestirimi.
  - `cv2.perspectiveTransform` + `cv2.polylines`: Şablon köşelerini sahneye projeksiyonla çizme.
  - `cv2.drawMatches`: İyi eşleşmeleri tek görselle birleştirme.

- [08_cnn_intro/mnist_cnn.py](08_cnn_intro/mnist_cnn.py)
  - `keras.datasets.mnist.load_data`: MNIST'i indirip (28x28) gri ton veri olarak alma.
  - Normalizasyon ve `np.expand_dims`: Veri biçimini `(N,28,28,1)` yapma.
  - `keras.Sequential` CNN: 2x Conv+MaxPool, ardından `Flatten` ve Dense katmanları.
  - `model.fit(..., validation_split=0.1)`: Egitim sırasında otomatik doğrulama ayırma.
  - `model.save`: Eğitilmiş modeli `.h5`/`.keras` olarak kaydetme.
  - `keras.preprocessing.image.load_img(..., color_mode="grayscale", target_size=(28, 28))`: Harici rakam görselini ölçekleyip tersine çevirerek (`1.0 - x`) tahmin etme.
  - `matplotlib` ile eğitim/val kayıp-dogruluk grafiği kaydı (`plot_history`).

- [project_1_similarity.py](project_1_similarity.py)
  - `cv2.ORB_create(nfeatures=2000)`: Hafif keypoint dedektörü ve tanımlayıcı.
  - `cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(k=2)`: Hamming mesafesiyle ikili en yakın komşu.
  - Lowe oran testi (`m.distance < ratio * n.distance`): Gürültülü eşleşmeleri elemek.
  - `cv2.drawMatches(..., flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)`: Eşleşmeleri tek karede göstermek.

- [project_2_edges.py](project_2_edges.py)
  - `args.blur | 1`: Blur çekirdeğini tek sayıya zorlamak için bit işlemi.
  - `cv2.Canny` + `np.count_nonzero`: Kenar piksel sayısıyla kural tabanlı sınıflandırma.
  - `cv2.putText` ile kenar sayısını görselleştirme.

- [project_3_cnn_ready.py](project_3_cnn_ready.py)
  - `torchvision.models.mobilenet_v3_large` / `resnet18` ve `.Weights.DEFAULT`: Ön eğitimli ağırlıkları otomatik indirip yükleme.
  - `weights.meta['mean'/'std'/'categories']`: Manuel normalizasyon ve sınıf isimleri.
  - NumPy→tensor dönüşümü `permute(2,0,1)` + ölçekleme; PIL'siz iş akışı.

- [project_4_compare.py](project_4_compare.py)
  - Sınıflandırma: `ResNet18_Weights.DEFAULT` ile var/yok kararı (basit eşik).
  - Detection: `fasterrcnn_resnet50_fpn` + `FasterRCNN_ResNet50_FPN_Weights.DEFAULT` ile COCO kutuları.
  - Detection modeline giriş: Liste halinde tensör, normalize edilmiş RGB.
  - `draw_boxes` yardımcı fonksiyonu: Skor eşiğiyle kutu ve etiket çizer.
