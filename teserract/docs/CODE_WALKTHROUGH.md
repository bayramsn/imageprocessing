# Kod Yürüyüşü ve Satır Mantığı

Bu dosyada kodları, hiç bilmeyen birinin anlayacağı dilde anlatıyorum. Amacım sadece "bu satır çalışıyor" demek değil; neden o satırı yazdığımı da açıklamak.

## 1. [../src/common.py](../src/common.py)

Bu dosya projenin yardımcı araç kutusu.

### Mantık
- en üstte importlar var,
- sonra proje klasör yollarını tanımlıyorum,
- sonra Tesseract yapılandırmasını yapıyorum,
- sonra görsel okuma ve çıktı kaydetme fonksiyonları geliyor,
- en sonda OCR temizlik ve satır birleştirme yardımcıları var.

### Satır mantığı
- İlk blokta `Path`, `cv2`, `numpy`, `pytesseract` gibi temel araçları içeri alıyorum.
- `PROJECT_ROOT`, `IMAGES_DIR`, `OUTPUTS_DIR`, `TESSDATA_DIR` ile projenin klasör omurgasını kuruyorum.
- `configure_tesseract()` içinde önce kullanıcı özel yol vermiş mi diye bakıyorum.
- Sonra ortam değişkenine bakıyorum.
- Sonra yaygın Windows kurulum yollarını deniyorum.
- Son olarak yerel [../tessdata](../tessdata) klasörü varsa `TESSDATA_PREFIX` ayarlıyorum.
- `read_image()` sadece görsel okumuyor; aynı zamanda eksik dosya durumunda açıklayıcı hata veriyor.
- `save_output()`, `save_text()` ve `save_json()` proje çıktılarının standart şekilde yazılmasını sağlıyor.
- `preprocess_for_ocr()` OCR’den önce görüntüyü temizliyor.
- `clean_ocr_text()` OCR sonrasında bozuk karakterleri düzeltiyor.
- `extract_text_lines()` `image_to_data()` çıktısını satır bazında toparlıyor.

## 2. [../src/01_basic_ocr.py](../src/01_basic_ocr.py)

Bu dosya projedeki en sade başlangıç noktası.

### Mantık
- komut satırından görsel adı al,
- Tesseract’ı hazırla,
- görseli oku,
- OCR yap,
- sonucu yazdır ve dosyaya kaydet.

Bu dosyayı özellikle basit bıraktım çünkü yeni başlayan biri önce en kısa akışı görmeli.

## 3. [../src/02_preprocess_ocr.py](../src/02_preprocess_ocr.py)

Bu dosyada OCR’den önce temizliğin etkisini gösteriyorum.

### Mantık
- görseli oku,
- `preprocess_for_ocr()` uygula,
- hem işlenmiş görseli hem OCR metnini kaydet,
- istersem `--show` ile pencere aç.

Burada vermek istediğim mesaj şu:
> OCR kalitesi yalnızca OCR motorundan ibaret değildir. Görüntü hazırlığı en az OCR kadar önemlidir.

## 4. [../src/03_document_regions_ocr.py](../src/03_document_regions_ocr.py)

Bu dosyayı iyileştirdim.

### Önceki problem
Sabit bölgeler yanlış konumlandığı için örnek kimlik belgesinde yanlış alanlar okunuyordu.

### Yeni mantık
- `Region` veri sınıfı ile alanları anlamlı isimlerle tanımlıyorum.
- Bölgeleri örnek belgeye göre yeniden ayarladım.
- Her alanı tek tek kırpıyorum.
- Her kırpılmış alanı ayrıca preprocess ediyorum.
- `--psm 7` ile tek satır gibi okumasını sağlıyorum.
- Sonucu ekrana ve JSON dosyasına yazıyorum.

### Ben bu dosyada ne anlatıyorum?
Sabit şablonlu belgelerde her şeyi tüm sayfadan çıkarmak zorunda değilim. İlgili alanları doğrudan okuyabilirim.

## 5. [../src/04_bounding_boxes.py](../src/04_bounding_boxes.py)

Bu dosya OCR motorunun kelimeleri nerede gördüğünü gösteriyor.

### Mantık
- `image_to_data()` ile sadece metin değil, koordinat da alıyorum.
- güven skoru düşük kelimeleri elemek için `conf > 40` kullanıyorum.
- kutu çiziyorum,
- kelimeyi üstüne yazıyorum,
- sonucu kaydediyorum.

Bu dosya özellikle hata ayıklama için çok faydalı.

## 6. [../src/05_turkish_ocr.py](../src/05_turkish_ocr.py)

Burada aynı OCR mantığını Türkçe dil modeli ile çalıştırıyorum.

### Ana fikir
Tesseract’a doğru dil dosyasını vermediğimde Türkçe karakterlerde hata oranı artar.
Bu yüzden `lang="tur"` kullanıyorum.

## 7. [../src/06_clean_ocr_text.py](../src/06_clean_ocr_text.py)

Bu dosyayı önemli ölçüde iyileştirdim.

### Önceki problem
Bazı satırlar birleşiyor, cümleler birbirine yapışıyordu.

### Yeni mantık
- önce `image_to_data()` ile satır bilgisi topluyorum,
- sonra `extract_text_lines()` ile metni satır bazında kuruyorum,
- her satıra tek tek `clean_ocr_text()` uyguluyorum,
- ham çıktı, temiz çıktı ve satır listesi olarak JSON üretiyorum.

### Sonuç
Temiz metin artık daha okunabilir ve satır düzeni daha tutarlı.

## 8. [../src/07_postprocess_correction.py](../src/07_postprocess_correction.py)

Bu dosya OCR sonrası düzeltme katmanı.

### Mantık
- OCR metnini alıyorum,
- token’lara bölüyorum,
- `RapidFuzz` ile sözlüğe göre en yakın terimi arıyorum,
- eşik üzerindeyse kelimeyi düzeltiyorum,
- düzeltme raporu da oluşturuyorum.

### Neden gerekli?
OCR çoğu zaman harfleri karıştırır. Mesela `TARIHI` yerine `TAR1H1` benzeri bir sonuç gelebilir.
Bu katman bu tip hataları azaltır.

## 9. [../src/08_field_extraction.py](../src/08_field_extraction.py)

Bu dosya da iyileştirildi.

### Yaklaşım 1: regex tabanlı çıkarım
Metnin tamamını tarayıp desen eşleşmesi ile alan buluyorum.

### Yaklaşım 2: satır tabanlı çıkarım
`ADI`, `SOYADI`, `TC KIMLIK NO` gibi etiketleri satırlarda arıyorum.

### Yaklaşım 3: anchor-template çıkarımı
Belgedeki küçük bir şablonu bularak geri kalan alanları göreli koordinatla çıkartıyorum.

### Neden üç yaklaşım birden var?
Çünkü gerçek hayatta tek bir yöntem her belgede en iyi sonucu vermez.

## 10. [../src/09_ocr_nlp_insights.py](../src/09_ocr_nlp_insights.py)

Burada OCR metninden anlam çıkarıyorum.

### Mantık
- regex ile adres adayları arıyorum,
- kurum isimleri arıyorum,
- tutar, tarih, telefon tespit ediyorum,
- çıkan verilere göre olası belge türü tahmini yapıyorum.

Bu, klasik OCR’den bilgi çıkarım aşamasına geçiştir.

## 11. [../src/10_form_digitization.py](../src/10_form_digitization.py)

Bu dosyada form otomasyonu yaptım ve sonradan iyileştirdim.

### Önceki problem
Sadece preprocess edilmiş görüntü ile çalışınca form alanları yeterince çıkmıyordu.

### Yeni mantık
- hem preprocess edilmiş görüntüden hem orijinal görüntüden OCR alıyorum,
- `--psm 11` ile sparse text okuma kullanıyorum,
- satırları birleştiriyorum,
- `Anahtar: Değer` biçimini ve kolonlu satırları yorumluyorum,
- alanları JSON’a dönüştürüyorum.

### Sonuç
Form alanları artık daha iyi yakalanıyor.

## 12. [../src/11_table_ocr.py](../src/11_table_ocr.py)

Bu dosyada tablo hücrelerini ayrı ayrı okuyorum.

### Mantık
- adaptif threshold ile tablo çizgilerini belirginleştiriyorum,
- yatay ve dikey çizgileri ayrı çıkarıyorum,
- ikisini birleştirip ızgara elde ediyorum,
- konturları hücre adayına çeviriyorum,
- boyut filtreleme ve deduplication uyguluyorum,
- her hücreye OCR yapıyorum,
- CSV yazıyorum.

### İyileştirme
- medyan boyuta göre filtre ekledim,
- yakın kutuları tekrar saymıyorum,
- hücre değerlerini normalize ediyorum,
- çok boş satırları atıyorum.

## 13. [../src/12_template_runner.py](../src/12_template_runner.py)

Bu dosyada JSON tabanlı belge şablonu yürütüyorum.

### Mantık
- template dosyasını oku,
- varsa ROI kullan,
- yoksa regex kullan,
- alanları sırayla doldur,
- sonucu JSON yaz.

Bu yapı, ürünleştirme için çok değerlidir.

## 14. [../src/13_live_webcam_ocr.py](../src/13_live_webcam_ocr.py)

Bu dosyada canlı görüntü üstünde OCR çalıştırıyorum.

### Mantık
- kamerayı aç,
- belli kare aralığında OCR çalıştır,
- belge kenarı bul,
- varsa belgeyi düzleştir,
- OCR kutularını çiz,
- özet satırları ekranda göster,
- `s` ile anlık kayıt al,
- `q` ile çık.

Bu dosya projeyi oldukça canlı hale getiriyor.

## 15. [../src/14_pdf_ocr.py](../src/14_pdf_ocr.py)

Bu dosya PDF OCR kapısı.

### Mantık
- PDF yolunu çöz,
- PDF sayfalarını görsele çevir,
- her sayfada özel pipeline çalıştır,
- sayfa sayısını, belge tipini ve tablo tespit sayısını raporla,
- JSON, TXT, XLSX üret.

## 16. [../src/15_batch_folder_ocr.py](../src/15_batch_folder_ocr.py)

Bu dosyayı da iyileştirdim.

### Yeni özellikler
- `--recursive`
- `--db-path`
- `--skip-db`
- dosya bazlı özet çıktı

### Mantık
- klasörü tara,
- görselleri ve PDF’leri topla,
- her biri için özel pipeline çalıştır,
- raporu JSON ve Excel’e kaydet,
- istenirse SQLite’a yaz,
- hangi dosya hangi pipeline’dan geçti ekrana yazdır.

## 17. [../src/pipelines.py](../src/pipelines.py)

Bu dosya proje beynidir.

### Bu dosyada neler var?
- OCR alma
- OCR veri alma
- metin düzeltme
- alan çıkarımı
- içgörü çıkarımı
- belge tipi sınıflandırma
- özel pipeline çalıştırma
- belge kenarı bulma
- tablo tanıma
- PDF sayfalarını işleme
- toplu klasör işleme
- Excel üretme
- SQLite veritabanı yönetimi
- veritabanı sorgulama ve filtreleme

Kısacası bütün akışlar burada birleşiyor.

## 18. [../streamlit_app.py](../streamlit_app.py)

Bu dosya kullanıcıya görünen katman.

### Burada ne yaptım?
- kaynak seçimi ekledim,
- OCR mod seçimi ekledim,
- otomatik mod ekledim,
- PDF, kamera ve toplu klasör akışı ekledim,
- veritabanı ekranı ekledim,
- arama ve filtreleme paneli ekledim,
- kayıt detayı ve özel tablo görüntüleme ekledim.

Bu dosya sayesinde proje sadece geliştirici aracı olmaktan çıkıp sunulabilir bir mini uygulamaya dönüştü.

---

## Kendi ağzımdan kısa proje savunması

> Bu projede kodları parça parça ama ortak omurga üstüne kurdum. `common.py` ile tekrar eden işleri topladım, `pipelines.py` ile asıl iş mantığını merkezileştirdim, her senaryoyu ayrı script haline getirdim ve en sonunda Streamlit ile hepsini tek arayüzde birleştirdim. Böylece proje hem öğretici hem de demo yapılabilir bir yapı kazandı.
