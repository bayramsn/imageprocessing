# Soru - Cevap Bankası

Bu bölümde projeden çıkabilecek soruları ve kısa ama doğru cevapları topladım.

## Başlangıç Soruları

### 1. OCR nedir?
OCR, bir görüntü veya taranmış belge üzerindeki yazıları makine tarafından okunabilir metne çeviren teknolojidir.

### 2. Tesseract nedir?
Tesseract açık kaynaklı bir OCR motorudur. Python tarafında bunu `pytesseract` ile çağırıyorum.

### 3. OpenCV neden kullanıldı?
Görüntüyü OCR öncesi hazırlamak, çizgi bulmak, kutu çizmek, belge kenarı tespit etmek ve tablo yapısını çıkarmak için.

### 4. `pytesseract` ile Tesseract aynı şey mi?
Hayır. `pytesseract` Python arabirimi, Tesseract ise asıl OCR motorudur.

### 5. OCR neden bazen kötü sonuç verir?
Düşük çözünürlük, eğik belge, kötü aydınlatma, bulanıklık, yanlış dil seçimi ve kötü ön işleme sonucu düşürür.

## Orta Seviye Sorular

### 6. Ön işleme neden önemli?
Çünkü OCR motoru temiz, kontrastı yüksek ve net görüntülerde daha doğru çalışır.

### 7. Bu projede hangi ön işleme teknikleri var?
Grileştirme, Gaussian blur, Otsu threshold ve morfolojik temizlik.

### 8. `--psm` nedir?
Tesseract’ın sayfayı nasıl yorumlayacağını belirleyen page segmentation mode parametresidir.

### 9. Türkçe OCR için ne yaptın?
Yerel `tessdata` klasörü oluşturdum, `tur.traineddata` ekledim ve `lang="tur"` veya `lang="tur+eng"` kullandım.

### 10. OCR sonrası neden temizlik yaptın?
Çünkü ham OCR çıktısında bozuk karakter, birleşik satır, gereksiz boşluk ve yanlış sembol çok sık görülür.

### 11. RapidFuzz burada ne işe yarıyor?
OCR sonrası kelime benzerliği ile daha doğru kelime tahmini yapıyor.

### 12. Bounding box görselleştirme neden önemli?
OCR motorunun hangi bölgeleri okuduğunu görselleştirerek hata analizi yapmayı kolaylaştırır.

## İleri Seviye Sorular

### 13. Belge türüne göre özel pipeline neden kurdun?
Çünkü kimlik, fatura, form ve tablo gibi belgelerde çıkarılmak istenen veri yapısı farklıdır.

### 14. `run_specialized_pipeline()` ne yapıyor?
Belgenin türünü belirliyor, sonra türe göre uygun işleme stratejisini uyguluyor.

### 15. Kimlik belgesinde hangi alanları çıkardın?
TC kimlik no, isim, soyisim, doğum tarihi ve adres adayları.

### 16. Fatura için hangi bilgiler önemli?
Belge numarası, tarih, tutar, tutar listesi ve varsa tablo satırları.

### 17. Form otomasyonunda ana mantık ne?
OCR ile satırları okuyup `anahtar: değer` yapısına çevirmek.

### 18. Tablo tanıma mantığı ne?
Önce yatay ve dikey çizgileri buluyorum, sonra hücre konturlarını çıkarıp her hücreye ayrı OCR uyguluyorum.

### 19. PDF OCR nasıl çalışıyor?
Önce PDF sayfalarını görüntüye çeviriyorum, sonra her sayfayı OCR pipeline’ından geçiriyorum.

### 20. Webcam OCR’da belge kenarı neden bulunuyor?
Perspektif hatasını azaltmak için belgeyi düzleştirip OCR’ye daha temiz bir giriş vermek için.

## Veritabanı Soruları

### 21. Neden SQLite seçtin?
Kurulumsuz, hafif, taşınabilir ve eğitim/demolar için ideal olduğu için.

### 22. Neden tek tablo yerine birden çok tablo kullandın?
Belge türleri farklı alanlar taşıdığı için normalize bir yapı kurmak istedim.

### 23. Ana tablo ile özel tabloların farkı ne?
Ana tablo tüm belgelerin ortak alanlarını tutar, özel tablolar ise belge türüne özgü detayları tutar.

### 24. `ocr_documents` tablosu ne içeriyor?
Dosya adı, türü, belge tipi, pipeline adı, metin, JSON alanları ve zaman bilgisi.

### 25. `identity_documents` tablosu ne içeriyor?
TC kimlik no, isim, soyisim, doğum tarihi ve adres.

### 26. `invoice_documents` tablosu ne içeriyor?
Belge no, toplam tutar, tarih, tutarlar JSON ve tablo satırları JSON.

### 27. `form_documents` tablosu ne içeriyor?
Alan sayısı, alanlar JSON ve satır sayısı.

### 28. `table_documents` tablosu ne içeriyor?
Satır sayısı ve tablo JSON içeriği.

### 29. `generic_documents` tablosu ne içeriyor?
Metin uzunluğu ve kurum adayları gibi genel belge bilgileri.

## Streamlit Soruları

### 30. Streamlit neden eklendi?
Projeyi daha anlaşılır, gösterilebilir ve interaktif hale getirmek için.

### 31. Streamlit’te hangi kaynaklar var?
Örnek görsel, görsel yükleme, PDF yükleme, kamera çekimi, toplu klasör ve veritabanı ekranı.

### 32. Otomatik mod ne yapıyor?
Belge tipini tahmin edip uygun işleme akışını seçiyor.

### 33. Veritabanı ekranı ne sağlıyor?
Arama, filtreleme, kayıt detayı ve özel tablo şemalarını görüntüleme.

## Kod Tasarımı Soruları

### 34. Neden [../src/common.py](../src/common.py) var?
Ortak işlevleri tek yerde toplamak için.

### 35. Neden [../src/pipelines.py](../src/pipelines.py) var?
İş mantığını merkezi hale getirip tekrar kullanılabilir yapmak için.

### 36. Neden her konu için ayrı script yazdın?
Öğrenmeyi modüler hale getirmek, demo kolaylığı sağlamak ve kullanım senaryolarını net ayırmak için.

### 37. Hangi dosya proje omurgası sayılır?
[../src/pipelines.py](../src/pipelines.py)

### 38. Hangi dosya ortam ve yardımcı omurga sayılır?
[../src/common.py](../src/common.py)

## Sunumda Kullanılabilecek Sorular

### 39. Bu proje ile neyi kanıtlıyorsun?
Sadece OCR çağırmayı değil; veri kaynağı, ön işleme, OCR, post-processing, extraction, raporlama, veritabanı ve arayüz katmanlarını birleştirebildiğimi.

### 40. Bu projeyi ürünleştirmek istesen ne eklersin?
Kullanıcı yönetimi, dosya kuyruklama, loglama, API katmanı, bulut depolama ve model bazlı doğrulama eklerim.

### 41. OCR yerine derin öğrenme kullanır mıydın?
Belge tipi çok karmaşıksa evet. Ama eğitim ve modüler başlangıç için Tesseract daha hızlı ve erişilebilir bir çözüm.

### 42. Tablo tanıma neden zor?
Çünkü hücre yapısı, çizgi kalitesi, birleşik hücreler ve OCR hataları işleri zorlaştırır.

### 43. Form tanıma neden zor?
Bazı formlarda `:` karakteri kaybolabilir, alanlar kayabilir veya OCR satırları yanlış ayırabilir.

### 44. PDF OCR neden doğrudan yapılmadı?
Çünkü OCR motoru raster görüntü üstünde çalışır. PDF önce görüntüye çevrilmelidir.

### 45. `tur+eng` kullanmak neden iyi olabilir?
Türkçe karakterleri korurken İngilizce kısaltmaları ve karma belgeleri daha iyi okumaya yardımcı olur.

## Mini Teknik Sözlük

### 46. Threshold nedir?
Piksel değerlerini siyah-beyaz ayırma işlemidir.

### 47. Contour nedir?
Görüntüde bir şeklin sınırını temsil eden nokta dizisidir.

### 48. ROI nedir?
Region of Interest. Yani ilgilenilen görüntü bölgesi.

### 49. Template matching nedir?
Küçük bir şablon görseli, büyük görsel içinde arama yöntemidir.

### 50. Perspective transform nedir?
Eğik çekilmiş belgeyi düzleştirmek için kullanılan geometrik dönüşümdür.
