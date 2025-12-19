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

---

## 🆕 Yeni Eklenen Projeler

- [03_opencv_giris/webcam_fps.py](03_opencv_giris/webcam_fps.py)
  - FPS hesaplama: `frame_count / (curr_time - prev_time)` ile gerçek zamanlı performans ölçümü.
  - `apply_cartoon_filter`: `cv2.adaptiveThreshold` + `cv2.bilateralFilter` + `cv2.bitwise_and` kombinasyonu.
  - `apply_sepia_filter`: `cv2.transform` ile özel renk dönüşüm matrisi uygulama.
  - `cv2.imwrite`: Tuşla ekran görüntüsü kaydetme (`p` tuşu).
  - Dinamik filtre sistemi: Dictionary tabanlı mod seçimi ve `chr(key)` ile tuş eşleştirme.

- [04_gaussian_blur_opencv/blur_comparison.py](04_gaussian_blur_opencv/blur_comparison.py)
  - `add_noise`: Gaussian ve tuz-biber gürültüsü ekleme için NumPy random fonksiyonları.
  - `compare_blur_types`: Gaussian, Median, Bilateral, Box filtrelerini tek fonksiyonda karşılaştırma.
  - `compare_kernel_sizes`: Farklı kernel boyutlarının etkisini görselleştirme.
  - `compare_sigma_values`: Sigma parametresinin blur üzerindeki etkisini analiz.
  - `interactive_blur`: `cv2.createTrackbar` ile canlı blur türü ve kernel seçimi.
  - `kernel | 1` yerine `k if k % 2 == 1 else k + 1`: Kernel'i tek sayıya zorlama.

- [05_gaussian_blur_manual/custom_gaussian.py](05_gaussian_blur_manual/custom_gaussian.py)
  - `create_gaussian_kernel`: Formülden elle 2D Gaussian kernel üretimi ve normalize etme.
  - `create_gaussian_kernel_fast`: `np.outer` ile vektörize kernel üretimi (1D→2D).
  - `convolve2d_manual`: Nested loop ile piksel piksel konvolüsyon (eğitim amaçlı).
  - `convolve2d_vectorized`: `np.lib.stride_tricks.sliding_window_view` ile hızlı konvolüsyon.
  - `gaussian_blur_separable`: 2D konvolüsyonu 2×1D'ye ayırarak O(n²)→O(2n) optimizasyonu.
  - `benchmark`: Farklı yöntemlerin hız karşılaştırması (`time.time()` ile).
  - `visualize_kernel`: `matplotlib 3D surface plot` ile kernel görselleştirme.

- [06_traditional_image_processing/preprocessing_tool.py](06_traditional_image_processing/preprocessing_tool.py)
  - `threshold_comparison`: Binary, Otsu, Adaptive Mean/Gaussian eşik yöntemlerini karşılaştırma.
  - `edge_detection_comparison`: Sobel X/Y, Laplacian, Canny kenar tespit yöntemleri.
  - `morphology_comparison`: Erosion, Dilation, Opening, Closing, Gradient, Top/Black Hat.
  - `document_preprocessing`: Belge tarama için adım adım pipeline (Gri→Blur→Adaptive→Morph).
  - `plate_detection_preprocessing`: Plaka tanıma için `cv2.bilateralFilter` + Canny + kontur analizi.
  - `cv2.approxPolyDP`: Konturları basitleştirip dikdörtgen (4 köşe) bulma.
  - `interactive_preprocessing`: Trackbar ile canlı threshold/canny/morph parametreleri.

- [07_keypoints_features/feature_matcher.py](07_keypoints_features/feature_matcher.py)
  - `detect_features`: ORB, SIFT, AKAZE, BRISK algoritmalarını tek fonksiyonda destekleme.
  - `match_features`: Descriptor tipine göre `NORM_HAMMING` veya `NORM_L2` otomatik seçimi.
  - `draw_matches_custom`: Rastgele renkli çizgilerle özel eşleşme görselleştirme.
  - `find_homography`: `cv2.findHomography(..., cv2.RANSAC)` ile sağlam homografi kestirimi.
  - `compare_methods`: Tüm algoritmaları aynı görüntü çiftinde karşılaştırma ve benchmark.
  - `visualize_keypoints`: `cv2.drawKeypoints(..., DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)` ile detaylı keypoint çizimi.

- [08_cnn_intro/cnn_visualizer.py](08_cnn_intro/cnn_visualizer.py)
  - `get_feature_maps`: Intermediate model oluşturarak Conv katman çıkışlarını alma.
  - `visualize_feature_maps`: Her katmandaki filtreleri grid halinde gösterme.
  - `visualize_kernels`: `layer.get_weights()` ile öğrenilen kernel ağırlıklarını görselleştirme.
  - `visualize_activations_grid`: Tüm Conv/Pool katmanlarını ayrı figürlerde gösterme.
  - `model_summary_visual`: Layer başına parametre sayısını manuel formatlama.

- [09_numpy_matplotlib/image_analyzer.py](09_numpy_matplotlib/image_analyzer.py)
  - `analyze_pixels`: NumPy ile min/max/mean/std/median istatistikleri.
  - `apply_threshold` / `apply_adaptive_threshold`: OpenCV eşikleme fonksiyonları.
  - `plot_analysis`: 3×3 subplot ile kapsamlı görüntü analizi (histogram, threshold, heatmap).
  - `demonstrate_numpy_operations`: Slicing, reshape, boolean indexing eğitim demonstrasyonu.
  - `gray.ravel()`: 2D→1D dönüşümü histogram için.
  - `np.cumsum`: Kümülatif histogram hesaplama.
  - RGB kanal analizi: `img[:, :, i]` ile kanal bazlı istatistikler.

- [10_detection_segmentation/compare_tasks.py](10_detection_segmentation/compare_tasks.py)
  - `run_classification`: `ResNet50_Weights.DEFAULT` ile ImageNet sınıflandırma.
  - `run_detection`: `FasterRCNN_ResNet50_FPN_Weights` ile COCO nesne tespiti.
  - `run_segmentation`: `DeepLabV3_ResNet50_Weights` ile semantic segmentation.
  - `outputs["out"].argmax(1)`: Segmentasyon çıkışından sınıf maskesi üretme.
  - `colors[mask]`: NumPy fancy indexing ile renkli maske oluşturma.
  - `cv2.rectangle` ile detection kutularını çizme.
  - Üç görevin aynı görüntü üzerinde yan yana karşılaştırması.
