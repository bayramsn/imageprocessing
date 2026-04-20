from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import IMAGES_DIR, OUTPUTS_DIR, save_output, save_text  # noqa: E402
from pipelines import (  # noqa: E402
    analyze_pdf_pages,
    annotate_ocr_boxes,
    batch_process_folder,
    configure,
    detect_table,
    draw_boxes,
    excel_bytes_from_report,
    extract_form_fields,
    get_database_overview,
    get_specialized_records,
    pdf_bytes_to_images,
    query_database,
    run_specialized_pipeline,
    save_batch_to_database,
    save_excel_report,
    save_table_csv,
)

st.set_page_config(page_title="OCR Demo Studio", layout="wide")
st.title("OCR Demo Studio")
st.caption("Tesseract tabanlı OCR, PDF, tablo, form, kenar algılama ve toplu klasör işleme")

configure()
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resolve_auto_mode(document_type: str) -> str:
    mapping = {
        "kimlik": "Alan Çıkarımı",
        "fatura_veya_fis": "NLP Çıkarımı",
        "form": "Form Otomasyonu",
        "tablo": "Tablo Tanıma",
        "cmr": "CMR Çıkarımı",
        "genel_belge": "OCR Temizleme",
    }
    return mapping.get(document_type, "OCR Temizleme")


def image_from_upload(upload) -> np.ndarray | None:
    if upload is None:
        return None
    np_buffer = np.frombuffer(upload.read(), dtype=np.uint8)
    return cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)


sample_files = sorted([path.name for path in IMAGES_DIR.glob("*.png")])
source_type = st.sidebar.radio(
    "Kaynak",
    ["Örnek görsel seç", "Görsel yükle", "PDF yükle", "Kamera çekimi", "Toplu klasör", "Veritabanı"],
)
lang = st.sidebar.selectbox("Dil", ["tur", "eng"], index=0)
mode = st.sidebar.selectbox(
    "Mod",
    [
        "Otomatik",
        "Temel OCR",
        "OCR Temizleme",
        "Post-Processing",
        "Alan Çıkarımı",
        "NLP Çıkarımı",
        "Form Otomasyonu",
        "Tablo Tanıma",
        "CMR Çıkarımı",
        "PDF OCR",
    ],
)

if source_type == "Veritabanı":
    db_default = str(OUTPUTS_DIR / "ocr_results.db")
    db_path = st.sidebar.text_input("Veritabanı yolu", value=db_default)
    overview = get_database_overview(db_path)

    st.subheader("Veritabanı özeti")
    st.write(
        {
            "db": overview["db_path"],
            "toplam_kayit": overview["total_documents"],
            "ozel_tablo_sayilari": overview["specialized_counts"],
        }
    )

    col1, col2 = st.columns(2)
    with col1:
        if overview["document_types"]:
            st.dataframe(pd.DataFrame(overview["document_types"]), use_container_width=True)
    with col2:
        if overview["batches"]:
            st.dataframe(pd.DataFrame(overview["batches"]), use_container_width=True)

    st.subheader("Arama / filtreleme paneli")
    search_text = st.text_input("Metin veya dosya adına göre ara")
    document_type_options = [row["document_type"] for row in overview["document_types"] if row["document_type"]]
    selected_types = st.multiselect("Belge tipi", document_type_options, default=document_type_options)
    selected_file_types = st.multiselect("Dosya tipi", ["image", "pdf"], default=["image", "pdf"])
    batch_options = [row["batch_name"] for row in overview["batches"] if row["batch_name"]]
    selected_batch = st.selectbox("Batch adı", [""] + batch_options)
    result_limit = st.slider("Sonuç limiti", min_value=10, max_value=500, value=100, step=10)

    results = query_database(
        db_path,
        search_text=search_text,
        document_types=selected_types or None,
        file_types=selected_file_types or None,
        batch_name=selected_batch or None,
        limit=result_limit,
    )

    summary_rows = []
    for item in results:
        summary_rows.append(
            {
                "ID": item["id"],
                "Dosya": item["file_name"],
                "Tür": item["file_type"],
                "Belge Tipi": item["document_type"],
                "Pipeline": item["pipeline_name"],
                "Batch": item["batch_name"],
                "Oluşturma": item["created_at"],
                "Önizleme": str(item["text_content"] or "")[:100],
            }
        )

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
        selected_id = st.selectbox("Detay için kayıt ID", [row["ID"] for row in summary_rows])
        selected_record = next(item for item in results if item["id"] == selected_id)
        st.subheader("Kayıt detayı")
        st.json(selected_record)
    else:
        st.info("Filtrelere uygun kayıt bulunamadı.")

    st.subheader("Belge türüne göre ayrı tablo şemaları")
    schema_table = st.selectbox(
        "Özel tablo seç",
        [
            "identity_documents",
            "invoice_documents",
            "form_documents",
            "table_documents",
            "generic_documents",
        ],
    )
    specialized_rows = get_specialized_records(db_path, schema_table, limit=result_limit)
    if specialized_rows:
        st.dataframe(pd.DataFrame(specialized_rows), use_container_width=True)
    else:
        st.info("Seçili özel tabloda henüz kayıt yok.")
    st.stop()

if source_type == "Toplu klasör":
    default_folder = str(IMAGES_DIR)
    folder_path = st.sidebar.text_input("Klasör yolu", value=default_folder)
    include_images = st.sidebar.checkbox("Görselleri işle", value=True)
    include_pdfs = st.sidebar.checkbox("PDF'leri işle", value=True)

    if not folder_path:
        st.info("Soldan işlenecek klasör yolunu girin.")
        st.stop()

    batch_report = batch_process_folder(
        folder_path,
        lang=lang,
        include_images=include_images,
        include_pdfs=include_pdfs,
    )
    excel_data = excel_bytes_from_report(batch_report)
    excel_path = save_excel_report(batch_report, OUTPUTS_DIR / "streamlit_toplu_klasor.xlsx")
    save_to_db = st.sidebar.checkbox("Veritabanına da kaydet", value=True)
    db_result = None
    if save_to_db:
        db_result = save_batch_to_database(batch_report, OUTPUTS_DIR / "ocr_results.db", batch_name="streamlit_batch")

    st.subheader("Toplu klasör özeti")
    st.write({
        "klasor": batch_report["folder"],
        "dosya_sayisi": batch_report["file_count"],
        "belge_turleri": batch_report["document_type_counts"],
    })

    rows = []
    for item in batch_report["items"]:
        rows.append(
            {
                "Dosya": item.get("file_name"),
                "Tür": item.get("file_type"),
                "Belge Tipi": item.get("document_type"),
                "Önizleme": str(item.get("text", ""))[:120],
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.download_button(
        label="Toplu Excel indir",
        data=excel_data,
        file_name="streamlit_toplu_klasor.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.sidebar.success(f"Excel kaydedildi: {excel_path.name}")
    if db_result is not None:
        st.sidebar.success(f"Veritabanına kaydedildi: {Path(db_result['db_path']).name}")
        st.sidebar.write(f"Eklenen kayıt: {db_result['inserted_count']}")
    st.sidebar.success(f"Çıktılar {OUTPUTS_DIR} klasörüne kaydedildi")
    st.stop()

image_bgr = None
image_label = "uploaded"
pdf_report = None

if source_type == "Örnek görsel seç":
    selected = st.sidebar.selectbox("Örnekler", sample_files)
    if selected:
        image_label = Path(selected).stem
        image_bgr = cv2.imread(str(IMAGES_DIR / selected))
elif source_type == "Görsel yükle":
    upload = st.sidebar.file_uploader("Görsel yükle", type=["png", "jpg", "jpeg"])
    if upload is not None:
        image_label = Path(upload.name).stem
        image_bgr = image_from_upload(upload)
elif source_type == "PDF yükle":
    upload = st.sidebar.file_uploader("PDF yükle", type=["pdf"])
    if upload is not None:
        image_label = Path(upload.name).stem
        pdf_images = pdf_bytes_to_images(upload.read())
        pdf_report = analyze_pdf_pages(pdf_images, lang=lang)
        page_number = st.sidebar.slider("PDF sayfası", 1, len(pdf_images), 1)
        image_bgr = pdf_images[page_number - 1]
else:
    camera_file = st.sidebar.camera_input("Kameradan çek")
    if camera_file is not None:
        image_label = "camera_capture"
        image_bgr = image_from_upload(camera_file)

if image_bgr is None:
    st.info("Soldan örnek, görsel, PDF, kamera veya klasör girişi seçin.")
    st.stop()

mode_type_map = {
    "CMR Çıkarımı": "cmr",
    "Form Otomasyonu": "form",
    "Tablo Tanıma": "tablo",
    "Alan Çıkarımı": "kimlik",
    "NLP Çıkarımı": "fatura_veya_fis"
}
forced_type = mode_type_map.get(mode) if mode != "Otomatik" else None
pipeline_result = run_specialized_pipeline(image_bgr, lang=lang, forced_type=forced_type)
edge_result = pipeline_result["edge_result"]
ocr_input = pipeline_result["ocr_source"]
analysis = pipeline_result
detected = analysis["classification"]
resolved_mode = resolve_auto_mode(detected["type"]) if mode == "Otomatik" else mode

st.sidebar.markdown(f"**Tahmin edilen belge tipi:** {detected['type']}")
st.sidebar.json(detected["scores"])
st.sidebar.write(f"Belge kenarı bulundu: {'Evet' if edge_result['found'] else 'Hayır'}")
if mode == "Otomatik":
    st.sidebar.success(f"Önerilen akış: {resolved_mode}")

left, right = st.columns([1, 1])
with left:
    st.subheader("Girdi")
    st.image(to_rgb(image_bgr), use_container_width=True)

with right:
    st.subheader("Belge kenarı / düzeltilmiş görünüm")
    preview = edge_result["annotated"] if edge_result["found"] else ocr_input
    st.image(to_rgb(preview), use_container_width=True)

st.subheader("OCR giriş görüntüsü")
st.image(to_rgb(ocr_input), use_container_width=True)

report: dict[str, object] = {
    "source": image_label,
    "document_type": detected["type"],
    "mode": resolved_mode,
    "pipeline_name": analysis["pipeline_name"],
    "specialized": analysis["specialized"],
}

st.subheader("Özel pipeline özeti")
st.write({
    "pipeline": analysis["pipeline_name"],
    "belge_tipi": detected["type"],
    "kenar_bulundu": analysis["specialized"].get("document_edges_found"),
})
st.json(analysis["specialized"])

if resolved_mode == "Temel OCR":
    st.subheader("Ham OCR metni")
    st.text_area("Ham metin", analysis["raw_text"], height=220)
    save_text(f"{image_label}_basic_streamlit.txt", analysis["raw_text"])
    report["text"] = analysis["raw_text"]
elif resolved_mode == "OCR Temizleme":
    st.subheader("Temizlenmiş OCR metni")
    st.text_area("Temiz metin", analysis["cleaned_text"], height=220)
    save_text(f"{image_label}_clean_streamlit.txt", analysis["cleaned_text"])
    report["text"] = analysis["cleaned_text"]
elif resolved_mode == "Post-Processing":
    st.subheader("Düzeltilmiş metin")
    st.text_area("Düzeltilmiş", analysis["corrected_text"], height=220)
    st.subheader("Düzeltmeler")
    st.json(analysis["corrections"])
    save_text(f"{image_label}_postprocess_streamlit.txt", analysis["corrected_text"])
    report["text"] = analysis["corrected_text"]
    report["corrections"] = analysis["corrections"]
elif resolved_mode == "Alan Çıkarımı":
    st.subheader("Alanlar")
    st.json(analysis["fields"])
    save_text(f"{image_label}_fields_streamlit.txt", str(analysis["fields"]))
    report["fields"] = analysis["fields"]
elif resolved_mode == "NLP Çıkarımı":
    st.subheader("Anlamlı bilgiler")
    st.json(analysis["insights"])
    save_text(f"{image_label}_nlp_streamlit.txt", str(analysis["insights"]))
    report["insights"] = analysis["insights"]
    report["text"] = analysis["corrected_text"]
elif resolved_mode == "Form Otomasyonu":
    form_result = extract_form_fields(ocr_input, lang=lang)
    annotated = draw_boxes(ocr_input, form_result["data"], color=(255, 0, 0))
    st.subheader("Algılanan satırlar")
    st.image(to_rgb(annotated), use_container_width=True)
    st.json(form_result["fields"])
    save_output(f"{image_label}_form_streamlit.png", annotated)
    save_text(f"{image_label}_form_streamlit.txt", str(form_result["fields"]))
    report["fields"] = form_result["fields"]
elif resolved_mode == "Tablo Tanıma":
    table_result = detect_table(ocr_input, lang=lang)
    st.subheader("Tablo hücreleri")
    st.image(to_rgb(table_result["annotated"]), use_container_width=True)
    st.write(table_result["rows"])
    save_output(f"{image_label}_table_streamlit.png", table_result["annotated"])
    save_table_csv(table_result["rows"], OUTPUTS_DIR / f"{image_label}_table_streamlit.csv")
    report["rows"] = table_result["rows"]
elif resolved_mode == "CMR Çıkarımı":
    st.subheader("CMR Çıkarım sonucu")
    cmr_fields = analysis["specialized"].get("cmr_fields", {})
    if cmr_fields:
        st.json(cmr_fields)
        save_text(f"{image_label}_cmr_streamlit.txt", str(cmr_fields))
        report["fields"] = cmr_fields
    else:
        st.info("CMR alanları bulunamadı.")
elif resolved_mode == "PDF OCR":
    if pdf_report is None:
        st.warning("PDF OCR modu için soldan bir PDF yükleyin.")
    else:
        st.subheader("PDF OCR özeti")
        st.write({"sayfa": pdf_report["page_count"], "belge_tipi": pdf_report["document_type"]})
        st.text_area("Birleşik metin", str(pdf_report["full_text"]), height=220)
        st.subheader("Sayfa bazlı sonuçlar")
        st.json(pdf_report["pages"])
        table_pages = [table for table in pdf_report["tables"] if table["row_count"] > 0]
        if table_pages:
            st.subheader("PDF tablo ayrıştırma")
            st.json(table_pages)
        report = pdf_report
else:
    preview = annotate_ocr_boxes(ocr_input, lang=lang)
    st.subheader("OCR önizleme")
    st.image(to_rgb(preview["annotated"]), use_container_width=True)
    st.text_area("Metin", analysis["corrected_text"], height=220)
    report["text"] = analysis["corrected_text"]

excel_file_name = f"{image_label}_{resolved_mode.lower().replace(' ', '_')}.xlsx"
excel_data = excel_bytes_from_report(report)
excel_path = save_excel_report(report, OUTPUTS_DIR / excel_file_name)

st.download_button(
    label="Excel indir",
    data=excel_data,
    file_name=excel_file_name,
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.sidebar.success(f"Excel kaydedildi: {excel_path.name}")
st.sidebar.success(f"Çıktılar {OUTPUTS_DIR} klasörüne kaydedildi")
