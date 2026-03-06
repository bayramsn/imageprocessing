# OCR ve Tesseract Çalışma Projeleri

Bu klasör, OCR konularını başlangıçtan ileri seviyeye kadar adım adım çalışabilmeniz için hazırlanmış örnek bir Python projesidir.

## Yeni dokümantasyon paketi

Projeyi daha rahat anlatabilmek ve GitHub üzerinde güçlü gösterebilmek için ayrıntılı dokümanlar ekledim:

- [docs/INDEX.md](docs/INDEX.md)
- [docs/PROJECT_NARRATIVE_GUIDE.md](docs/PROJECT_NARRATIVE_GUIDE.md)
- [docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md)
- [docs/QUESTION_BANK.md](docs/QUESTION_BANK.md)
- [docs/GITHUB_PAGE_CONTENT.md](docs/GITHUB_PAGE_CONTENT.md)
- [docs/DESKTOP_EXE_GUIDE.md](docs/DESKTOP_EXE_GUIDE.md)
- [docs/OCR_egitim_paketi.xlsx](docs/OCR_egitim_paketi.xlsx)

## Masaüstü EXE desteği

Projeye GUI tabanlı masaüstü uygulama da eklendi:

- [desktop_app.py](desktop_app.py)
- [build_desktop_exe.bat](build_desktop_exe.bat)

Detaylı paketleme rehberi:
- [docs/DESKTOP_EXE_GUIDE.md](docs/DESKTOP_EXE_GUIDE.md)

## OCR Nedir?

OCR (Optical Character Recognition), bir görüntü veya taranmış belge içindeki yazıları makine tarafından okunabilir metne dönüştüren teknolojidir.

### Kullanım alanları
- Nüfus cüzdanı veya kimlik kartı okuma
- Fatura ve fiş tarama
- PDF veya taranmış dokümanlardan metin çıkarma
- Plaka, form, etiket ve arşiv dokümanları işleme

### Genel çalışma prensibi
1. Görüntü alınır.
2. Görüntü temizlenir: grileştirme, eşikleme, gürültü azaltma.
3. Metin bölgeleri tespit edilir.
4. OCR motoru karakterleri tanır.
5. Sonuç metin, JSON veya tablo olarak kaydedilir.
6. Gerekirse regex, sözlük veya NLP ile anlamlı bilgi çıkarılır.

---

## Bu projede bulunan örnekler

### 1. Basit OCR
Dosya: [src/01_basic_ocr.py](src/01_basic_ocr.py)

Amaç: Görüntüden doğrudan metin çıkarmak.

### 2. OCR Öncesi Görüntü Temizleme
Dosya: [src/02_preprocess_ocr.py](src/02_preprocess_ocr.py)

Uygulanan işlemler:
- Grileştirme
- Gaussian blur
- Otsu threshold
- Morphological open

### 3. Belge Üzerinde Bölgesel Okuma
Dosya: [src/03_document_regions_ocr.py](src/03_document_regions_ocr.py)

Amaç: Kimlik, fatura, fiş gibi belgelerde belirli alanları ayrı ayrı okumak.

### 4. Bounding Box Görselleştirme
Dosya: [src/04_bounding_boxes.py](src/04_bounding_boxes.py)

Amaç: OCR ile bulunan kelimeleri kutu içine alıp görüntü üzerine yazmak.

### 5. Türkçe OCR
Dosya: [src/05_turkish_ocr.py](src/05_turkish_ocr.py)

Amaç: Türkçe karakterleri daha doğru okumak için `lang="tur"` ve `--psm 6` kullanmak.

### 6. OCR Sonuçlarını Temizleme
Dosya: [src/06_clean_ocr_text.py](src/06_clean_ocr_text.py)

Amaç: OCR çıktısındaki bozuk karakterleri ve gereksiz boşlukları temizlemek.

### 7. OCR Post-Processing ve Düzeltme
Dosya: [src/07_postprocess_correction.py](src/07_postprocess_correction.py)

Amaç: Levenshtein benzeri benzerlik ile doğru kelime tahmini yapmak.

### 8. OCR ile Alan Bazlı Veri Çıkarımı
Dosya: [src/08_field_extraction.py](src/08_field_extraction.py)

Amaç: TC, isim, soyisim ve tarih gibi alanları otomatik çıkarmak.

### 9. OCR + NLP ile Anlam Çıkarımı
Dosya: [src/09_ocr_nlp_insights.py](src/09_ocr_nlp_insights.py)

Amaç: OCR metninden adres, kurum adı, tarih, telefon ve tutar gibi anlamlı bilgi çıkarmak.

### 10. OCR ile Form Otomasyonu
Dosya: [src/10_form_digitization.py](src/10_form_digitization.py)

Amaç: Formu satırlara ve anahtar-değer çiftlerine dönüştürmek.

### 11. OCR + Tablo Tanıma
Dosya: [src/11_table_ocr.py](src/11_table_ocr.py)

Amaç: Tablo çizgilerini tespit edip hücreleri ayrı OCR ile CSV dosyasına dönüştürmek.

### 12. Template Runner
Dosya: [src/12_template_runner.py](src/12_template_runner.py)

Amaç: Hazır JSON template dosyaları ile alan bazlı OCR çalıştırmak.

### 13. Streamlit Arayüzü
Dosya: [streamlit_app.py](streamlit_app.py)

Amaç: Tüm OCR demolarını tek ekranda görsel arayüz ile denemek.

### 14. Webcam ile Canlı OCR
Dosya: [src/13_live_webcam_ocr.py](src/13_live_webcam_ocr.py)

Amaç: Kameradan alınan görüntü üzerinde gerçek zamanlı OCR, belge tipi tahmini ve belge kenarı algılama yapmak.

### 15. PDF OCR + Excel
Dosya: [src/14_pdf_ocr.py](src/14_pdf_ocr.py)

Amaç: PDF sayfalarını OCR ile okuyup tablo ayrıştırma dahil JSON, TXT ve Excel çıktısına dönüştürmek.

### 16. Toplu Klasör OCR
Dosya: [src/15_batch_folder_ocr.py](src/15_batch_folder_ocr.py)

Amaç: Bir klasör içindeki görsel ve PDF dosyalarını toplu işleyip özet rapor üretmek.

### 17. Belge Türüne Göre Özel Pipeline
Dosya: [src/pipelines.py](src/pipelines.py)

Amaç: `kimlik`, `fatura_veya_fis`, `form`, `tablo` ve `genel_belge` türleri için farklı işleme akışları uygulamak.

### 18. Veritabanına Toplu Kayıt
Dosya: [src/15_batch_folder_ocr.py](src/15_batch_folder_ocr.py), [src/pipelines.py](src/pipelines.py)

Amaç: Toplu OCR sonuçlarını SQLite veritabanına kaydetmek.

### 19. Veritabanı Sorgulama Ekranı
Dosya: [streamlit_app.py](streamlit_app.py)

Amaç: Kaydedilen OCR sonuçlarını aramak, filtrelemek ve özel tablo şemalarını görüntülemek.

---

## Kurulum

### 1) Python sanal ortamı oluşturun
Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Tesseract OCR kurun
Windows için güvenilir başlangıç noktası:
- Tesseract GitHub: https://github.com/tesseract-ocr/tesseract

Kurulumdan sonra `tesseract.exe` yolu PATH içinde değilse [src/common.py](src/common.py) içindeki `TESSERACT_CMD` değişkenine tam yolu yazın.

Örnek:

```python
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### 3) Türkçe dil dosyası
Türkçe OCR için `tur.traineddata` dosyası gerekir.
Genelde şu klasörde bulunur:
- `C:\Program Files\Tesseract-OCR\tessdata`

Yoksa Tesseract dil dosyalarını indirip bu klasöre ekleyin.

### 4) Örnek görselleri yeniden üretme
Örnek test görsellerini tekrar oluşturmak isterseniz:

```powershell
python tools/generate_sample_assets.py
```

Bu komut ayrıca örnek bir PDF dosyası da üretir:
- [pdfs/ocr_demo_pack.pdf](pdfs/ocr_demo_pack.pdf)

---

## Görselleri nereye koymalıyım?

Denemek istediğiniz görselleri [images](images) klasörüne ekleyin.

Önerilen dosya adları:
- `sample.png`
- `document_sample.png`
- `turkish_sample.png`
- `form_sample.png`
- `table_sample.png`
- `tc_anchor.png`

---

## Çalıştırma örnekleri

### Basit OCR
```powershell
python src/01_basic_ocr.py sample.png
```

### Ön işleme + OCR
```powershell
python src/02_preprocess_ocr.py sample.png
```

### Bölgesel belge okuma
```powershell
python src/03_document_regions_ocr.py
```

> Not: Bölgesel okuma için [src/03_document_regions_ocr.py](src/03_document_regions_ocr.py) içindeki `REGIONS` koordinatlarını belgeye göre güncelleyin.

### Bounding box çizimi
```powershell
python src/04_bounding_boxes.py
```

### Türkçe OCR
```powershell
python src/05_turkish_ocr.py turkish_sample.png
```

### OCR çıktısını temizleme
```powershell
python src/06_clean_ocr_text.py sample.png
```

### Post-processing ve düzeltme
```powershell
python src/07_postprocess_correction.py sample.png
```

### Alan bazlı veri çıkarımı
```powershell
python src/08_field_extraction.py document_sample.png
```

Anchor-template ile örnek:

```powershell
python src/08_field_extraction.py document_sample.png --template tc_anchor.png
```

### OCR + NLP
```powershell
python src/09_ocr_nlp_insights.py document_sample.png
```

### Form otomasyonu
```powershell
python src/10_form_digitization.py form_sample.png
```

### Tablo tanıma
```powershell
python src/11_table_ocr.py table_sample.png
```

### Template ile OCR
```powershell
python src/12_template_runner.py kimlik_template.json
python src/12_template_runner.py fatura_template.json
```

### Streamlit arayüzü
```powershell
python -m streamlit run streamlit_app.py
```

Arayüzde:
- örnek görselleri seçebilirsiniz,
- kendi dosyanızı yükleyebilirsiniz,
- PDF yükleyebilirsiniz,
- kameradan tek kare alabilirsiniz,
- toplu klasör seçeneğiyle çok sayıda dosyayı birlikte işleyebilirsiniz,
- veritabanı ekranında kayıtları sorgulayabilirsiniz,
- temel OCR, temizleme, alan çıkarımı, form ve tablo modları arasında geçiş yapabilirsiniz.
- `Otomatik` modunda belge tipi tahmin edilip uygun akış önerilir.
- belge türüne göre özel pipeline özeti ekranda gösterilir.
- sonuçları Excel olarak indirebilirsiniz.

Veritabanı ekranında:
- serbest metin araması,
- belge tipi filtresi,
- dosya tipi filtresi,
- batch filtresi,
- sonuç limiti,
- kayıt detayı,
- belge türüne göre ayrı tablo şemaları

görüntülenebilir.

### Webcam ile canlı OCR
```powershell
python src/13_live_webcam_ocr.py
```

İpuçları:
- `q` ile çıkılır.
- `s` ile o anki görüntü, metin ve Excel raporu [outputs](outputs) klasörüne kaydedilir.
- belge kenarı bulunursa görüntü otomatik düzeltilmiş haliyle OCR yapılır.

### PDF OCR
```powershell
python src/14_pdf_ocr.py ocr_demo_pack.pdf
```

Bu komut:
- PDF sayfalarını görüntüye çevirir,
- OCR yapar,
- belge tipini tahmin eder,
- tablo bulunan sayfalarda satırları ayrı raporlar,
- sonuçları JSON, TXT ve XLSX olarak kaydeder.

### Toplu klasör işleme
```powershell
python src/15_batch_folder_ocr.py images
python src/15_batch_folder_ocr.py pdfs --pdfs-only
python src/15_batch_folder_ocr.py images --db-path outputs/ocr_results.db
```

Bu komut:
- klasördeki görselleri ve PDF'leri tarar,
- her dosya için belge tipini tahmin eder,
- belge türüne göre özel pipeline uygular,
- PDF'lerde sayfa ve tablo bilgilerini çıkarır,
- toplu JSON ve XLSX raporu oluşturur,
- sonuçları SQLite veritabanına kaydedebilir.

### SQLite veritabanı

Toplu kayıt varsayılan olarak şu dosyaya yapılır:
- [outputs/ocr_results.db](outputs/ocr_results.db)

İstemiyorsanız:

```powershell
python src/15_batch_folder_ocr.py images --skip-db
```

### Belge türüne göre ayrı tablo şemaları

Ana tablo:
- `ocr_documents`

Özel tablolar:
- `identity_documents`
- `invoice_documents`
- `form_documents`
- `table_documents`
- `generic_documents`

Bu yapı sayesinde her belge tipi için farklı alanlar ayrı tabloda tutulur.

---

## Hazır örnek görseller

Oluşturulan örnek görseller:
- [images/sample.png](images/sample.png)
- [images/turkish_sample.png](images/turkish_sample.png)
- [images/document_sample.png](images/document_sample.png)
- [images/form_sample.png](images/form_sample.png)
- [images/table_sample.png](images/table_sample.png)
- [images/tc_anchor.png](images/tc_anchor.png)

Hazır PDF:
- [pdfs/ocr_demo_pack.pdf](pdfs/ocr_demo_pack.pdf)

Bu PDF artık çok sayfalıdır ve tablo örneği de içerir.

Bu görseller [tools/generate_sample_assets.py](tools/generate_sample_assets.py) ile otomatik üretildi.

## Hazır template dosyaları

- [templates/kimlik_template.json](templates/kimlik_template.json)
- [templates/fatura_template.json](templates/fatura_template.json)

Bu template dosyaları şunları içerir:
- kullanılacak örnek belge,
- OCR dil ve `psm` bilgisi,
- alan ROI koordinatları,
- regex doğrulama kuralları.

---

## Dosya yapısı

- [README.md](README.md)
- [requirements.txt](requirements.txt)
- [src/common.py](src/common.py)
- [src/01_basic_ocr.py](src/01_basic_ocr.py)
- [src/02_preprocess_ocr.py](src/02_preprocess_ocr.py)
- [src/03_document_regions_ocr.py](src/03_document_regions_ocr.py)
- [src/04_bounding_boxes.py](src/04_bounding_boxes.py)
- [src/05_turkish_ocr.py](src/05_turkish_ocr.py)
- [src/06_clean_ocr_text.py](src/06_clean_ocr_text.py)
- [src/07_postprocess_correction.py](src/07_postprocess_correction.py)
- [src/08_field_extraction.py](src/08_field_extraction.py)
- [src/09_ocr_nlp_insights.py](src/09_ocr_nlp_insights.py)
- [src/10_form_digitization.py](src/10_form_digitization.py)
- [src/11_table_ocr.py](src/11_table_ocr.py)
- [src/12_template_runner.py](src/12_template_runner.py)
- [src/13_live_webcam_ocr.py](src/13_live_webcam_ocr.py)
- [src/14_pdf_ocr.py](src/14_pdf_ocr.py)
- [src/15_batch_folder_ocr.py](src/15_batch_folder_ocr.py)
- [src/pipelines.py](src/pipelines.py)
- [streamlit_app.py](streamlit_app.py)
- [tools/generate_sample_assets.py](tools/generate_sample_assets.py)
- [templates](templates)
- [images](images)
- [pdfs](pdfs)
- [outputs](outputs)

---

## Öğrenme sırası önerisi

1. Önce [src/01_basic_ocr.py](src/01_basic_ocr.py) ile düz metin çıkarın.
2. Sonra [src/02_preprocess_ocr.py](src/02_preprocess_ocr.py) ile görüntü temizleme etkisini görün.
3. Ardından [src/04_bounding_boxes.py](src/04_bounding_boxes.py) ile OCR'ın hangi bölgeden okuduğunu görün.
4. Sonra [src/05_turkish_ocr.py](src/05_turkish_ocr.py) ile Türkçe kaliteyi artırın.
5. Daha sonra [src/06_clean_ocr_text.py](src/06_clean_ocr_text.py) ile ham sonucu temizleyin.
6. Sonra [src/07_postprocess_correction.py](src/07_postprocess_correction.py) ile düzeltme yapın.
7. Ardından [src/08_field_extraction.py](src/08_field_extraction.py) ile alan çıkarımı uygulayın.
8. Sonra [src/09_ocr_nlp_insights.py](src/09_ocr_nlp_insights.py) ile anlamlı bilgi çıkarın.
9. Ardından [src/10_form_digitization.py](src/10_form_digitization.py) ile form otomasyonuna geçin.
10. Sonra [src/11_table_ocr.py](src/11_table_ocr.py) ile tablo OCR çalışın.
11. Ardından [src/12_template_runner.py](src/12_template_runner.py) ile hazır template mantığını deneyin.
12. Sonra [src/14_pdf_ocr.py](src/14_pdf_ocr.py) ile PDF OCR çalışın.
13. Ardından [src/13_live_webcam_ocr.py](src/13_live_webcam_ocr.py) ile canlı OCR deneyin.
14. Sonra [src/15_batch_folder_ocr.py](src/15_batch_folder_ocr.py) ile toplu klasör işleme yapın.
15. Ardından [src/pipelines.py](src/pipelines.py) içindeki belge türüne göre özel pipeline mantığını inceleyin.
16. En sonda [streamlit_app.py](streamlit_app.py) ile tüm akışları tek arayüzden test edin.

---

## İleri seviye notlar
- [src/07_postprocess_correction.py](src/07_postprocess_correction.py) içindeki `DEFAULT_TERMS` sözlüğünü kendi belge tipinize göre genişletin.
- [src/08_field_extraction.py](src/08_field_extraction.py) içindeki anchor-template yaklaşımı sabit tasarımlı kimlik ve form belgelerinde etkilidir.
- [src/09_ocr_nlp_insights.py](src/09_ocr_nlp_insights.py) regex tabanlıdır. İstenirse spaCy ile kişi, kurum ve adres varlık tanıma eklenebilir.
- [src/11_table_ocr.py](src/11_table_ocr.py) çizgili tablolar için uygundur. PDF tablolarında `camelot-py`, `tabula` veya `layoutparser` tercih edilebilir.
- [templates/kimlik_template.json](templates/kimlik_template.json) ve [templates/fatura_template.json](templates/fatura_template.json) dosyalarını çoğaltarak kendi belge şablonlarınızı oluşturabilirsiniz.
- [streamlit_app.py](streamlit_app.py) içinde belge tipi tahmini anahtar kelime ve yapı tabanlıdır; kendi belge tiplerinize göre geliştirebilirsiniz.
- [src/14_pdf_ocr.py](src/14_pdf_ocr.py) Excel çıktısını çok sayfalı rapor olarak üretir.
- [streamlit_app.py](streamlit_app.py) içindeki `Toplu klasör` kaynağı ile tek seferde çok sayıda görsel ve PDF işlenebilir.
- [src/13_live_webcam_ocr.py](src/13_live_webcam_ocr.py) belge sınırını bulursa perspektif düzeltmesi uygular.
- [src/pipelines.py](src/pipelines.py) içindeki `run_specialized_pipeline()` belge türüne göre farklı alan, tablo ve form çıkarımı çalıştırır.
- [src/pipelines.py](src/pipelines.py) içindeki `save_batch_to_database()` toplu sonuçları SQLite veritabanına yazar.
- [streamlit_app.py](streamlit_app.py) içindeki `Veritabanı` kaynağı arama/filtreleme paneli ve tablo şema görünümü sağlar.

## Sonraki geliştirme fikirleri
- Video dosyası üzerinden OCR
- PDF içinde tablo ve form alanlarını ayrı işleme
- Verileri veritabanına kaydetme
- Otomatik belge arşivleme ve klasörleme