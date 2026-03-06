# GitHub Sayfası İçeriği

Bu dosya, projeyi GitHub’da güçlü göstermek için kullanabileceğim anlatım metnidir.

## Kısa Tanım

Bu proje; görüntü, PDF, webcam ve toplu klasör kaynaklarından veri alıp OCR yapan; belge türünü tahmin eden; belgeye göre özel pipeline uygulayan; sonuçları JSON, CSV, Excel ve SQLite veritabanına kaydeden modüler bir Python OCR platformudur.

## Öne Çıkan Özellikler

- Tesseract tabanlı OCR
- OpenCV ile ön işleme
- Türkçe OCR desteği
- OCR sonrası metin temizleme
- Kelime düzeltme
- Alan bazlı veri çıkarımı
- Form otomasyonu
- Tablo tanıma
- PDF OCR
- Webcam ile canlı OCR
- Streamlit arayüzü
- Toplu klasör işleme
- SQLite veritabanı kaydı
- Veritabanı sorgulama ekranı

## Teknik Yığın

- Python
- OpenCV
- pytesseract
- Tesseract OCR
- RapidFuzz
- pandas
- openpyxl
- pypdfium2
- Streamlit
- SQLite

## Mimarinin özeti

Ben projeyi şu mantıkla kurdum:
- ortak fonksiyonlar ayrı,
- iş akışları ayrı,
- senaryo scriptleri ayrı,
- arayüz ayrı,
- örnek veri ayrı,
- çıktı klasörü ayrı,
- eğitim dokümanı ayrı.

Bu sayede proje hem öğrenme projesi hem de sunulabilir demo projesi oldu.

## Demo Senaryoları

### 1. Basit OCR
[../src/01_basic_ocr.py](../src/01_basic_ocr.py)

### 2. Türkçe OCR
[../src/05_turkish_ocr.py](../src/05_turkish_ocr.py)

### 3. PDF OCR
[../src/14_pdf_ocr.py](../src/14_pdf_ocr.py)

### 4. Webcam OCR
[../src/13_live_webcam_ocr.py](../src/13_live_webcam_ocr.py)

### 5. Streamlit Demo
[../streamlit_app.py](../streamlit_app.py)

## Sunumda Kullanabileceğim Cümleler

> Bu projede yalnızca OCR yaptım demiyorum. Ben, OCR etrafında çalışan mini bir belge işleme ekosistemi kurdum.

> Aynı sistem içinde görüntü işleme, OCR, belge sınıflandırma, alan çıkarımı, tablo tanıma, form digitization, PDF parsing, Excel raporlama ve veritabanı saklama katmanlarını birleştirdim.

> Projeyi modüler tasarladım; bu yüzden her parça bağımsız test edilebilir ve sonra tek arayüzde birleşebilir hale geldi.

## GitHub README içine eklenebilecek başlıklar

- Proje özeti
- Kurulum
- Kullanılan teknolojiler
- Modüller
- Örnek kullanım komutları
- Çıktı örnekleri
- Streamlit ekran görüntüleri
- Veritabanı şeması
- Gelecek geliştirmeler

## GitHub için önerilen ekran görüntüleri

- bounding box çıktısı
- form OCR JSON çıktısı
- tablo CSV çıktısı
- PDF OCR Excel çıktısı
- Streamlit ana ekranı
- veritabanı sorgulama ekranı
