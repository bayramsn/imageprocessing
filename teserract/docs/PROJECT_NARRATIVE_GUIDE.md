# Proje Anlatım Rehberi

## Bu projeyi neden yaptım?

Bu projeyi OCR dünyasını sadece yüzeysel göstermek için değil, gerçek bir öğrenme hattı kurmak için hazırladım. Amacım tek bir dosyada sadece metin okutmak değildi. Ben burada şunu göstermek istedim:

1. ham görüntüden metin nasıl çıkarılır,
2. görüntü temizliği neden gerekir,
3. belge türüne göre neden farklı yaklaşım gerekir,
4. OCR sonucunu neden sonradan düzeltmek gerekir,
5. neden sadece metin almak yetmez, anlam da çıkarmak gerekir,
6. neden bu verileri JSON, Excel ve veritabanına taşımak gerekir.

Kısacası ben bu projeyi, "OCR yaptım" demekten daha ileri taşıyıp, "OCR tabanlı mini belge işleme platformu geliştirdim" diyebilmek için kurguladım.

---

## Projenin genel resmi

Bu proje şu katmanlardan oluşur:

### 1. Girdi katmanı
Ben projeye farklı veri kaynakları koydum:
- tekil görsel
- PDF
- webcam
- toplu klasör
- Streamlit arayüzünden dosya yükleme

### 2. Ön işleme katmanı
Bu katmanda görüntü OCR için hazırlanır:
- grileştirme
- bulanıklaştırma
- eşikleme
- morfolojik temizlik
- bazı senaryolarda belge kenarı bulma ve perspektif düzeltme

### 3. OCR katmanı
Burada Tesseract çalışır.
Kullandığım ana kütüphane:
- `pytesseract`

### 4. Son işleme katmanı
Burada OCR çıktısı temizlenir:
- bozuk karakterler düzeltilir
- boşluklar normalize edilir
- satırlar bir araya getirilir
- bazı kelimeler sözlükle düzeltilir

### 5. Anlamlandırma katmanı
Burada sadece ham metin değil, anlamlı veri çıkarılır:
- isim
- soyisim
- TC kimlik no
- tarih
- adres
- kurum adı
- telefon
- tutar
- form alanları
- tablo hücreleri

### 6. Çıktı katmanı
Sonuçlar farklı formatlarda kaydedilir:
- TXT
- JSON
- CSV
- XLSX
- SQLite veritabanı

---

## Projede kullandığım teknolojiler

### Python
Bu projenin ana dili Python. Çünkü görüntü işleme, OCR ve veri analizi için zengin kütüphane desteği sunuyor.

### OpenCV
Ben OpenCV'yi görüntüyle çalışmak için kullandım.
Başlıca görevleri:
- görsel okuma
- grileştirme
- threshold
- contour bulma
- belge kenarı yakalama
- tablo çizgileri tespiti
- bounding box çizme

### pytesseract
Bu katman Python ile Tesseract arasında köprü görevi görüyor.
Yani OCR motorunu Python'dan çağırmamı sağlıyor.

### Tesseract OCR
Asıl karakter tanıma motoru bu.
Benim kodum OCR mantığını organize ediyor ama karakteri gerçekten tanıyan sistem Tesseract.

### RapidFuzz
OCR sonrası kelime düzeltme için kullandım.
Benzer kelime eşleştirme yapıyor.
Bu sayede `KIMLIK`, `TOPLAM`, `TARIHI` gibi alan adlarını daha temiz hale getirebiliyorum.

### pandas
Excel üretmek ve tablo halinde veri hazırlamak için kullandım.

### openpyxl
`xlsx` dosyası yazmak için kullandım.

### pypdfium2
PDF dosyasını görüntülere çevirmek için kullandım.
Çünkü Tesseract doğrudan PDF değil, görüntü üstünde çalışıyor.

### Streamlit
Projeyi sadece terminal aracı olmaktan çıkarıp demo yapılabilir hale getirmek için Streamlit ekledim.
Böylece:
- yükle,
- seç,
- incele,
- filtrele,
- indir

mantığında çalışan bir arayüz oluştu.

### SQLite
Toplu sonuçları saklamak için hafif ve taşınabilir bir veritabanı seçtim.
Kurulum gerektirmemesi öğrenme projesi için büyük avantaj sağladı.

---

## Klasör yapısını neden böyle kurdum?

- [../src](../src)
- [../images](../images)
- [../pdfs](../pdfs)
- [../templates](../templates)
- [../outputs](../outputs)
- [../docs](../docs)
- [../tools](../tools)

Bu yapı sayesinde proje karmaşık görünse de modüler kalıyor.

---

## Öğrenme mantığı

Bu projeyi doğrusal değil, katmanlı tasarladım.

### Başlangıç seviyesi
- [../src/01_basic_ocr.py](../src/01_basic_ocr.py)
- [../src/02_preprocess_ocr.py](../src/02_preprocess_ocr.py)
- [../src/05_turkish_ocr.py](../src/05_turkish_ocr.py)

### Orta seviye
- [../src/03_document_regions_ocr.py](../src/03_document_regions_ocr.py)
- [../src/04_bounding_boxes.py](../src/04_bounding_boxes.py)
- [../src/06_clean_ocr_text.py](../src/06_clean_ocr_text.py)
- [../src/07_postprocess_correction.py](../src/07_postprocess_correction.py)

### İleri seviye
- [../src/08_field_extraction.py](../src/08_field_extraction.py)
- [../src/09_ocr_nlp_insights.py](../src/09_ocr_nlp_insights.py)
- [../src/10_form_digitization.py](../src/10_form_digitization.py)
- [../src/11_table_ocr.py](../src/11_table_ocr.py)
- [../src/12_template_runner.py](../src/12_template_runner.py)
- [../src/14_pdf_ocr.py](../src/14_pdf_ocr.py)
- [../src/15_batch_folder_ocr.py](../src/15_batch_folder_ocr.py)

### Uygulama ve sunum seviyesi
- [../src/13_live_webcam_ocr.py](../src/13_live_webcam_ocr.py)
- [../streamlit_app.py](../streamlit_app.py)

---

## Belge türüne göre özel pipeline neden ekledim?

Her belge aynı değildir.
Ben burada kritik bir tasarım kararı aldım:

- kimlik için alan tabanlı veri önemli,
- fatura için tutar ve tablo önemli,
- form için anahtar-değer ilişkisi önemli,
- tablo için hücre yapısı önemli,
- genel belge için ise serbest metin ve kurum/anahtar bilgi önemli.

Bu yüzden [../src/pipelines.py](../src/pipelines.py) içinde `run_specialized_pipeline()` ekledim.
Bu fonksiyon aynı OCR motorunu kullanıyor ama belge türüne göre farklı odak seçiyor.

---

## Veritabanı tasarımını neden ayırdım?

Sadece tek tablo kullansaydım bütün belge türleri aynı kolonlara sıkışacaktı.
Bu hem anlamsız hem de raporlama açısından zayıf olurdu.

Bu yüzden iki seviyeli tasarım kullandım:

### Ana tablo
- `ocr_documents`

Bu tablo tüm kayıtların ortak üst tablosu.

### Özel tablolar
- `identity_documents`
- `invoice_documents`
- `form_documents`
- `table_documents`
- `generic_documents`

Bu yaklaşımın avantajı:
- veri daha temiz tutulur,
- sorgular daha anlamlı olur,
- belge türüne özel rapor hazırlamak kolaylaşır.

---

## Bu projeyi anlatırken nasıl konuşabilirim?

Aşağıdaki anlatım dili, sanki projeyi ben yazmışım gibi kullanılabilir:

> Bu projede amacım yalnızca OCR yapmak değildi. Ben görüntü, PDF, webcam ve klasör bazlı veri kaynaklarından metin çıkaran; bu metni temizleyen; belge tipini tahmin eden; belgeye göre özel işleme uygulayan; çıktıyı JSON, Excel ve SQLite veritabanına kaydeden modüler bir OCR platformu geliştirdim. Görüntü işleme için OpenCV, karakter tanıma için Tesseract, arayüz için Streamlit, raporlama için pandas/openpyxl ve veri saklama için SQLite kullandım.

---

## Bu projede en önemli mühendislik kararlarım

1. Ortak yardımcıları [../src/common.py](../src/common.py) içine koydum.
2. Ana iş mantığını [../src/pipelines.py](../src/pipelines.py) içine topladım.
3. Her kullanım senaryosu için ayrı script yazdım.
4. Demo ve sunum tarafı için [../streamlit_app.py](../streamlit_app.py) ekledim.
5. Test kolaylığı için [../images](../images) ve [../pdfs](../pdfs) altına örnek veriler koydum.
6. Sonuçları sadece ekranda göstermekle bırakmayıp dosya ve veritabanına yazdım.

---

## Bu projeyi geliştirirken öğrendiğim en önemli şeyler şunlar oldu:

- OCR başarısı sadece OCR motoruna bağlı değildir.
- Görüntü kalitesi sonucu doğrudan etkiler.
- Son işleme, ham OCR kadar önemlidir.
- Belge türü değişince extraction stratejisi de değişmelidir.
- Demo arayüzü eklemek projeyi daha görünür hale getirir.
- Veritabanı eklemek projeyi sadece demo olmaktan çıkarıp ürün mantığına yaklaştırır.
