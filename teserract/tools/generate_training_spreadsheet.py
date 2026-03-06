from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_PATH = DOCS_DIR / "OCR_egitim_paketi.xlsx"


def build_sheets() -> dict[str, pd.DataFrame]:
    proje_haritasi = pd.DataFrame(
        [
            ["src/common.py", "Ortak yardımcı fonksiyonlar", "Tesseract ayarı, preprocess, temizleme, satır çıkarımı"],
            ["src/pipelines.py", "Ana iş mantığı", "Belge sınıflandırma, özel pipeline, PDF, batch, DB"],
            ["streamlit_app.py", "Arayüz", "Tüm demoların tek ekranda kullanılması"],
            ["src/03_document_regions_ocr.py", "Bölgesel OCR", "Kimlik veya sabit şablonlu belgelerde alan odaklı okuma"],
            ["src/06_clean_ocr_text.py", "Metin temizleme", "Ham OCR çıktısını okunabilir hale getirme"],
            ["src/08_field_extraction.py", "Alan çıkarımı", "TC, isim, soyisim, tarih gibi alanları bulma"],
            ["src/10_form_digitization.py", "Form digitization", "Anahtar-değer alanlarını çıkarma"],
            ["src/11_table_ocr.py", "Tablo OCR", "Hücre bazlı tablo ayrıştırma"],
            ["src/14_pdf_ocr.py", "PDF OCR", "PDF sayfalarını OCR ile işleme"],
            ["src/15_batch_folder_ocr.py", "Toplu OCR", "Klasördeki tüm belgeleri sırayla işleme"],
        ],
        columns=["Dosya", "Rol", "Kısa Açıklama"],
    )

    teknoloji_sozlugu = pd.DataFrame(
        [
            ["Python", "Ana programlama dili", "OCR, veri işleme ve otomasyon kodları"],
            ["OpenCV", "Görüntü işleme", "Threshold, contour, çizgi ve belge kenarı tespiti"],
            ["pytesseract", "Python köprüsü", "Tesseract motorunu Python içinden çağırma"],
            ["Tesseract", "OCR motoru", "Görüntü üzerindeki karakterleri tanıma"],
            ["RapidFuzz", "Fuzzy matching", "OCR sonrası kelime düzeltme"],
            ["pandas", "Tablo ve raporlama", "Excel ve DataFrame üretimi"],
            ["openpyxl", "Excel yazımı", "xlsx çıktı dosyaları"],
            ["pypdfium2", "PDF render", "PDF sayfalarını görsele dönüştürme"],
            ["Streamlit", "Web arayüzü", "OCR demolarını görsel arayüzde toplama"],
            ["SQLite", "Veritabanı", "OCR sonuçlarını kalıcı saklama"],
        ],
        columns=["Teknoloji", "Kategori", "Projede Kullanım Amacı"],
    )

    dosya_ozet = pd.DataFrame(
        [
            ["01_basic_ocr", "Başlangıç", "En basit OCR akışı"],
            ["02_preprocess_ocr", "Başlangıç", "Ön işleme etkisi"],
            ["03_document_regions_ocr", "Orta", "ROI bazlı alan okuma"],
            ["04_bounding_boxes", "Orta", "OCR kutu görselleştirme"],
            ["05_turkish_ocr", "Başlangıç", "Türkçe OCR"],
            ["06_clean_ocr_text", "Orta", "Temizleme ve satır normalizasyonu"],
            ["07_postprocess_correction", "Orta", "Fuzzy correction"],
            ["08_field_extraction", "İleri", "Regex, line ve anchor extraction"],
            ["09_ocr_nlp_insights", "İleri", "Anlam çıkarımı"],
            ["10_form_digitization", "İleri", "Form alanlarını çıkarma"],
            ["11_table_ocr", "İleri", "Tablo hücre OCR"],
            ["12_template_runner", "İleri", "Template ile alan yürütme"],
            ["13_live_webcam_ocr", "İleri", "Canlı OCR ve belge düzeltme"],
            ["14_pdf_ocr", "İleri", "PDF OCR ve raporlama"],
            ["15_batch_folder_ocr", "İleri", "Batch işlem ve DB kaydı"],
        ],
        columns=["Modül", "Seviye", "Öğrenme Kazanımı"],
    )

    soru_cevap = pd.DataFrame(
        [
            ["OCR nedir?", "Görüntüdeki yazıyı makine tarafından okunabilir metne çeviren teknolojidir."],
            ["Neden preprocess kullanıyoruz?", "OCR motoruna daha temiz giriş vermek için."],
            ["Neden belge tipi sınıflandırması ekledik?", "Farklı belge türlerinde farklı veri çıkarımı gerektiği için."],
            ["Neden Streamlit kullandık?", "Projeyi interaktif ve sunulabilir hale getirmek için."],
            ["Neden SQLite kullandık?", "Kurulumsuz ve hafif bir veri saklama çözümü olduğu için."],
            ["Tablo OCR neden zor?", "Çizgiler, birleşik hücreler ve OCR gürültüsü nedeniyle."],
            ["Form OCR neden zor?", "Alan hizaları bozulabilir ve ':' karakteri kaybolabilir."],
            ["PDF OCR nasıl çalışıyor?", "PDF sayfaları görsele çevrilip sonra OCR uygulanıyor."],
        ],
        columns=["Soru", "Cevap"],
    )

    calistirma_komutlari = pd.DataFrame(
        [
            ["Basit OCR", "python src/01_basic_ocr.py sample.png"],
            ["Türkçe OCR", "python src/05_turkish_ocr.py turkish_sample.png"],
            ["Bölgesel OCR", "python src/03_document_regions_ocr.py"],
            ["Form OCR", "python src/10_form_digitization.py form_sample.png"],
            ["Tablo OCR", "python src/11_table_ocr.py table_sample.png"],
            ["PDF OCR", "python src/14_pdf_ocr.py pdfs/ocr_demo_pack.pdf"],
            ["Batch OCR", "python src/15_batch_folder_ocr.py images --recursive"],
            ["Streamlit", "streamlit run streamlit_app.py"],
        ],
        columns=["Senaryo", "Komut"],
    )

    iyilestirmeler = pd.DataFrame(
        [
            ["03_document_regions_ocr", "ROI koordinatları yenilendi, preprocess eklendi, JSON çıktı eklendi"],
            ["06_clean_ocr_text", "Satır bazlı toplama ve JSON rapor eklendi"],
            ["08_field_extraction", "Regex, line ve anchor çıkarımı güçlendirildi"],
            ["10_form_digitization", "Sparse text ve çok kaynaklı satır birleşimi eklendi"],
            ["11_table_ocr", "Median filtre, kutu tekrar temizliği ve hücre normalizasyonu eklendi"],
            ["15_batch_folder_ocr", "Recursive tarama, daha güçlü özet ve DB kontrolü eklendi"],
        ],
        columns=["Dosya", "Yapılan İyileştirme"],
    )

    github_icerik = pd.DataFrame(
        [
            ["Kısa Tanım", "Modüler OCR platformu"],
            ["Vurgu 1", "Görüntü, PDF, webcam ve batch desteği"],
            ["Vurgu 2", "Belge tipi tahmini ve özel pipeline"],
            ["Vurgu 3", "JSON, Excel ve SQLite kayıtları"],
            ["Vurgu 4", "Streamlit arayüzü"],
        ],
        columns=["Alan", "İçerik"],
    )

    return {
        "proje_haritasi": proje_haritasi,
        "teknoloji_sozlugu": teknoloji_sozlugu,
        "dosya_ozet": dosya_ozet,
        "soru_cevap": soru_cevap,
        "calistirma_komutlari": calistirma_komutlari,
        "iyilestirmeler": iyilestirmeler,
        "github_icerik": github_icerik,
    }


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    sheets = build_sheets()
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        for sheet_name, dataframe in sheets.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Spreadsheet oluşturuldu: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
