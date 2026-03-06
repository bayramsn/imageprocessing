from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import pytesseract
from rapidfuzz import fuzz, process

from common import (
    clean_ocr_text,
    configure_tesseract,
    extract_text_lines,
    normalize_whitespace,
    preprocess_for_ocr,
)

DEFAULT_TERMS = [
    "T.C.",
    "KIMLIK",
    "KİMLİK",
    "ADI",
    "SOYADI",
    "DOGUM",
    "DOĞUM",
    "TARIHI",
    "TARİHİ",
    "CINSIYET",
    "CİNSİYET",
    "FATURA",
    "VERGI",
    "VERGİ",
    "TOPLAM",
    "TUTAR",
    "MAHALLESI",
    "MAHALLESİ",
    "SOKAK",
    "CADDE",
    "NO",
    "TURKIYE",
    "TÜRKİYE",
]

CHARACTER_FIXES = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B"})

FIELD_PATTERNS = {
    "tc_kimlik_no": r"\b\d{11}\b",
    "tarih": r"\b\d{2}[./-]\d{2}[./-]\d{4}\b",
    "isim": r"(?:ADI|İSİM|ISIM)\s*[:\-]?\s*([A-ZÇĞİÖŞÜ]{2,}(?:\s+[A-ZÇĞİÖŞÜ]{2,}){0,3})",
    "soyisim": r"(?:SOYADI|SOYİSİM|SOYISIM)\s*[:\-]?\s*([A-ZÇĞİÖŞÜ]{2,}(?:\s+[A-ZÇĞİÖŞÜ]{2,}){0,3})",
    "belge_no": r"(?:BELGE NO|FATURA NO|NO)\s*[:\-]?\s*([A-Z0-9-]{3,})",
}

ADDRESS_PATTERN = re.compile(
    r"([A-ZÇĞİÖŞÜ0-9\s]+(?:MAH\.?|MAHALLESİ|SOK\.?|SOKAK|CAD\.?|CADDE|BLV\.?|NO:?|APT\.?|DAİRE)"
    r"[A-ZÇĞİÖŞÜ0-9\s:./\-]+)",
    re.IGNORECASE,
)
INSTITUTION_PATTERN = re.compile(
    r"\b([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s]{2,}(?:A\.Ş\.|LTD\. ŞTİ\.|ÜNİVERSİTESİ|BELEDİYESİ|BAKANLIĞI|HASTANESİ))\b"
)
AMOUNT_PATTERN = re.compile(r"\b\d+[.,]\d{2}\s?(?:TL|TRY)\b", re.IGNORECASE)
DATE_PATTERN = re.compile(r"\b\d{2}[./-]\d{2}[./-]\d{4}\b")
PHONE_PATTERN = re.compile(r"(?:\+90|0)?\s*\(?5\d{2}\)?\s*\d{3}\s*\d{2}\s*\d{2}")


def configure() -> None:
    configure_tesseract()



def correct_token(token: str, threshold: int = 78) -> tuple[str, int]:
    if len(token) <= 2 or any(char.isdigit() for char in token):
        return token, 100

    normalized = token.upper().translate(CHARACTER_FIXES)
    match = process.extractOne(normalized, DEFAULT_TERMS, scorer=fuzz.ratio)
    if not match:
        return token, 0

    candidate, score, _ = match
    if score < threshold:
        return token, score
    return candidate, score



def ocr_text(image, lang: str = "tur", psm: int = 6) -> dict[str, Any]:
    processed = preprocess_for_ocr(image)
    raw_text = pytesseract.image_to_string(processed, lang=lang, config=f"--psm {psm}")
    cleaned_text = clean_ocr_text(raw_text)
    return {
        "processed": processed,
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
    }


def ocr_data(image, lang: str = "tur", psm: int = 6) -> dict[str, list[Any]]:
    processed = preprocess_for_ocr(image)
    return pytesseract.image_to_data(
        processed,
        lang=lang,
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT,
    )



def postprocess_text(text: str) -> dict[str, Any]:
    corrections: list[dict[str, Any]] = []
    corrected_lines: list[str] = []

    for line in normalize_whitespace(text).splitlines():
        corrected_tokens: list[str] = []
        for token in line.split():
            corrected, score = correct_token(token)
            corrected_tokens.append(corrected)
            if corrected != token:
                corrections.append({"original": token, "corrected": corrected, "score": score})
        corrected_lines.append(" ".join(corrected_tokens))

    corrected_text = "\n".join(corrected_lines)
    return {"corrected_text": corrected_text, "corrections": corrections}



def extract_fields(cleaned_text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field_name, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, cleaned_text, flags=re.IGNORECASE)
        if not match:
            result[field_name] = None
            continue
        result[field_name] = match.group(1) if match.lastindex else match.group(0)
    return result



def extract_insights(cleaned_text: str) -> dict[str, Any]:
    insights = {
        "kurum_adlari": sorted(set(INSTITUTION_PATTERN.findall(cleaned_text))),
        "adres_adaylari": [match.strip() for match in ADDRESS_PATTERN.findall(cleaned_text)],
        "tutarlar": AMOUNT_PATTERN.findall(cleaned_text),
        "tarihler": DATE_PATTERN.findall(cleaned_text),
        "telefonlar": PHONE_PATTERN.findall(cleaned_text),
    }
    if insights["kurum_adlari"]:
        insights["olasi_belge_turu"] = "kurumsal_belge"
    elif insights["tutarlar"]:
        insights["olasi_belge_turu"] = "fatura_veya_fis"
    else:
        insights["olasi_belge_turu"] = "genel_belge"
    return insights


def classify_document_type(cleaned_text: str, image: np.ndarray | None = None) -> dict[str, Any]:
    upper_text = cleaned_text.upper()
    scores = {
        "kimlik": 0,
        "fatura_veya_fis": 0,
        "form": 0,
        "tablo": 0,
        "genel_belge": 1,
    }

    kimlik_terms = ["T.C", "KIMLIK", "KİMLİK", "TC KIMLIK NO", "DOGUM", "SOYADI"]
    invoice_terms = ["FATURA", "TOPLAM", "TUTAR", "KDV", "VERGI", "VERGİ", "FIYAT"]
    form_terms = ["FORM", "BASVURU", "BAŞVURU", "AD SOYAD", "TELEFON", "E-POSTA"]

    for term in kimlik_terms:
        if term in upper_text:
            scores["kimlik"] += 3
    for term in invoice_terms:
        if term in upper_text:
            scores["fatura_veya_fis"] += 3
    for term in form_terms:
        if term in upper_text:
            scores["form"] += 2

    if re.search(r"\b\d{11}\b", cleaned_text):
        scores["kimlik"] += 5
    if AMOUNT_PATTERN.search(cleaned_text):
        scores["fatura_veya_fis"] += 4
    if cleaned_text.count(":") >= 3:
        scores["form"] += 4

    if image is not None:
        table_preview = detect_table(image)
        if len(table_preview["rows"]) >= 2 and any(len(row) >= 2 for row in table_preview["rows"]):
            scores["tablo"] += 6

    predicted = max(scores.items(), key=lambda item: item[1])[0]
    return {"type": predicted, "scores": scores}


def analyze_document(image: np.ndarray, lang: str = "tur") -> dict[str, Any]:
    base = ocr_text(image, lang=lang, psm=6)
    cleaned_text = base["cleaned_text"]
    post = postprocess_text(cleaned_text)
    fields = extract_fields(post["corrected_text"])
    insights = extract_insights(post["corrected_text"])
    classification = classify_document_type(post["corrected_text"], image=image)
    return {
        **base,
        "corrected_text": post["corrected_text"],
        "corrections": post["corrections"],
        "fields": fields,
        "insights": insights,
        "classification": classification,
    }


def run_specialized_pipeline(
    image: np.ndarray,
    lang: str = "tur",
    forced_type: str | None = None,
) -> dict[str, Any]:
    edge_result = detect_document_edges(image)
    ocr_source = edge_result["warped"] if edge_result["found"] else image
    analysis = analyze_document(ocr_source, lang=lang)

    document_type = forced_type or analysis["classification"]["type"]
    pipeline_name = f"{document_type}_pipeline"
    specialized: dict[str, Any] = {
        "document_edges_found": edge_result["found"],
        "corners": edge_result["corners"],
    }

    if document_type == "kimlik":
        relevant_fields = {
            key: analysis["fields"].get(key)
            for key in ["tc_kimlik_no", "isim", "soyisim", "tarih"]
        }
        specialized.update(
            {
                "identity_fields": relevant_fields,
                "address_candidates": analysis["insights"].get("adres_adaylari", []),
            }
        )
    elif document_type == "fatura_veya_fis":
        table_result = detect_table(ocr_source, lang=lang)
        specialized.update(
            {
                "amounts": analysis["insights"].get("tutarlar", []),
                "dates": analysis["insights"].get("tarihler", []),
                "table_rows": table_result["rows"],
            }
        )
    elif document_type == "form":
        form_result = extract_form_fields(ocr_source, lang=lang)
        specialized.update(
            {
                "form_fields": form_result["fields"],
                "line_count": len(form_result["lines"]),
            }
        )
    elif document_type == "tablo":
        table_result = detect_table(ocr_source, lang=lang)
        specialized.update(
            {
                "table_rows": table_result["rows"],
                "row_count": len(table_result["rows"]),
            }
        )
    else:
        specialized.update(
            {
                "text_length": len(analysis["corrected_text"]),
                "institution_candidates": analysis["insights"].get("kurum_adlari", []),
            }
        )

    return {
        **analysis,
        "ocr_source": ocr_source,
        "edge_result": edge_result,
        "pipeline_name": pipeline_name,
        "specialized": specialized,
    }


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)

    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def four_point_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = order_points(points)
    top_left, top_right, bottom_right, bottom_left = rect

    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    max_width = max(int(width_top), int(width_bottom))

    height_right = np.linalg.norm(top_right - bottom_right)
    height_left = np.linalg.norm(top_left - bottom_left)
    max_height = max(int(height_right), int(height_left))

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def detect_document_edges(image: np.ndarray) -> dict[str, Any]:
    resized = image.copy()
    ratio = 1.0
    if image.shape[0] > 1200:
        ratio = image.shape[0] / 1200
        new_width = int(image.shape[1] / ratio)
        resized = cv2.resize(image, (new_width, 1200))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]

    document_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) == 4:
            document_contour = approximation.reshape(4, 2).astype("float32")
            break

    annotated = image.copy()
    warped = image.copy()
    found = False
    corners: list[list[int]] = []

    if document_contour is not None:
        scaled_contour = document_contour * ratio
        contour_int = scaled_contour.astype(int)
        cv2.polylines(annotated, [contour_int], True, (0, 255, 255), 3)
        warped = four_point_transform(image, scaled_contour)
        corners = contour_int.tolist()
        found = True

    return {
        "found": found,
        "annotated": annotated,
        "warped": warped,
        "edges": edged,
        "corners": corners,
    }



def extract_form_fields(image, lang: str = "tur") -> dict[str, Any]:
    processed = preprocess_for_ocr(image)
    data = pytesseract.image_to_data(
        processed,
        lang=lang,
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )
    lines = extract_text_lines(data)
    fields: dict[str, str] = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            key = clean_ocr_text(key).lower().replace(" ", "_")
            value = clean_ocr_text(value)
            if key:
                fields[key] = value
    return {"lines": lines, "fields": fields, "data": data}



def draw_boxes(image, data: dict[str, list[Any]], color: tuple[int, int, int] = (0, 255, 0)):
    annotated = image.copy()
    for i in range(len(data["text"])):
        text = str(data["text"][i]).strip()
        if not text:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 1)
    return annotated


def annotate_ocr_boxes(image, lang: str = "tur", min_confidence: int = 40):
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )
    annotated = image.copy()
    words: list[dict[str, Any]] = []
    for i in range(len(data["text"])):
        text = str(data["text"][i]).strip()
        conf_raw = str(data["conf"][i])
        conf = int(float(conf_raw)) if conf_raw != "-1" else -1
        if not text or conf < min_confidence:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            text,
            (x, max(20, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
        words.append({"text": text, "conf": conf, "box": [x, y, w, h]})
    return {"annotated": annotated, "words": words, "data": data}



def detect_table(image, lang: str = "tur") -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        ~gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        -2,
    )

    horizontal = binary.copy()
    vertical = binary.copy()
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, vertical_kernel)
    grid = cv2.add(horizontal, vertical)

    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(contour) for contour in contours]
    boxes = [box for box in boxes if box[2] > 20 and box[3] > 15]
    boxes.sort(key=lambda box: (box[1], box[0]))

    rows: list[list[tuple[int, int, int, int]]] = []
    for box in boxes:
        x, y, w, h = box
        placed = False
        for row in rows:
            if abs(row[0][1] - y) <= 15:
                row.append(box)
                row.sort(key=lambda item: item[0])
                placed = True
                break
        if not placed:
            rows.append([box])

    extracted_rows: list[list[str]] = []
    annotated = image.copy()
    for row in rows:
        values: list[str] = []
        for x, y, w, h in row:
            if w > image.shape[1] * 0.95 and h > image.shape[0] * 0.95:
                continue
            roi = image[y : y + h, x : x + w]
            roi_processed = preprocess_for_ocr(roi)
            text = pytesseract.image_to_string(roi_processed, lang=lang, config="--psm 7")
            values.append(clean_ocr_text(text))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if any(values):
            extracted_rows.append(values)

    return {"rows": extracted_rows, "annotated": annotated, "grid": grid}


def extract_tables_from_pdf_pages(images: list[np.ndarray], lang: str = "tur") -> list[dict[str, Any]]:
    page_tables: list[dict[str, Any]] = []
    for page_number, image in enumerate(images, start=1):
        table_result = detect_table(image, lang=lang)
        meaningful_rows = [row for row in table_result["rows"] if any(cell.strip() for cell in row)]
        page_tables.append(
            {
                "page": page_number,
                "row_count": len(meaningful_rows),
                "rows": meaningful_rows,
            }
        )
    return page_tables


def pdf_bytes_to_images(pdf_bytes: bytes, scale: float = 2.0) -> list[np.ndarray]:
    document = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    images: list[np.ndarray] = []
    for page_index in range(len(document)):
        page = document[page_index]
        bitmap = page.render(scale=scale).to_pil()
        images.append(cv2.cvtColor(np.array(bitmap), cv2.COLOR_RGB2BGR))
    return images


def pdf_file_to_images(pdf_path: str | Path, scale: float = 2.0) -> list[np.ndarray]:
    return pdf_bytes_to_images(Path(pdf_path).read_bytes(), scale=scale)


def analyze_pdf_pages(images: list[np.ndarray], lang: str = "tur") -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    aggregated_text: list[str] = []
    classifications: list[str] = []

    page_tables = extract_tables_from_pdf_pages(images, lang=lang)

    for page_number, image in enumerate(images, start=1):
        analysis = run_specialized_pipeline(image, lang=lang)
        page_table = next((item for item in page_tables if item["page"] == page_number), None)
        pages.append(
            {
                "page": page_number,
                "text": analysis["corrected_text"],
                "fields": analysis["fields"],
                "insights": analysis["insights"],
                "classification": analysis["classification"],
                "pipeline_name": analysis["pipeline_name"],
                "specialized": analysis["specialized"],
                "table": page_table,
            }
        )
        aggregated_text.append(analysis["corrected_text"])
        classifications.append(analysis["classification"]["type"])

    final_type = "genel_belge"
    if classifications:
        final_type = max(set(classifications), key=classifications.count)

    return {
        "page_count": len(images),
        "document_type": final_type,
        "full_text": "\n\n".join(aggregated_text),
        "pages": pages,
        "tables": page_tables,
    }


def batch_process_folder(
    folder_path: str | Path,
    lang: str = "tur",
    include_images: bool = True,
    include_pdfs: bool = True,
    recursive: bool = False,
) -> dict[str, Any]:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Klasör bulunamadı: {folder}")

    image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    items: list[dict[str, Any]] = []

    def collect_files(pattern: str) -> list[Path]:
        return sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))

    if include_images:
        image_files: list[Path] = []
        for pattern in image_patterns:
            image_files.extend(collect_files(pattern))

        for image_path in image_files:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            analysis = run_specialized_pipeline(image, lang=lang)
            table_result = detect_table(analysis["ocr_source"], lang=lang)
            items.append(
                {
                    "file_name": image_path.name,
                    "file_type": "image",
                    "document_type": analysis["classification"]["type"],
                    "pipeline_name": analysis["pipeline_name"],
                    "text": analysis["corrected_text"],
                    "fields": analysis["fields"],
                    "insights": analysis["insights"],
                    "specialized": analysis["specialized"],
                    "rows": table_result["rows"],
                }
            )

    if include_pdfs:
        for pdf_path in collect_files("*.pdf"):
            pdf_images = pdf_file_to_images(pdf_path)
            report = analyze_pdf_pages(pdf_images, lang=lang)
            items.append(
                {
                    "file_name": pdf_path.name,
                    "file_type": "pdf",
                    "document_type": report["document_type"],
                    "pipeline_name": "pdf_multipage_pipeline",
                    "text": report["full_text"],
                    "pages": report["pages"],
                    "tables": report["tables"],
                }
            )

    counts: dict[str, int] = {}
    for item in items:
        doc_type = str(item.get("document_type", "genel_belge"))
        counts[doc_type] = counts.get(doc_type, 0) + 1

    return {
        "folder": str(folder),
        "file_count": len(items),
        "document_type_counts": counts,
        "items": items,
    }


def initialize_database(db_path: str | Path) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_name TEXT,
                source_folder TEXT,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                document_type TEXT,
                pipeline_name TEXT,
                text_content TEXT,
                fields_json TEXT,
                insights_json TEXT,
                specialized_json TEXT,
                pages_json TEXT,
                tables_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS identity_documents (
                document_id INTEGER PRIMARY KEY,
                tc_kimlik_no TEXT,
                isim TEXT,
                soyisim TEXT,
                dogum_tarihi TEXT,
                adres TEXT,
                FOREIGN KEY(document_id) REFERENCES ocr_documents(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS invoice_documents (
                document_id INTEGER PRIMARY KEY,
                belge_no TEXT,
                toplam_tutar TEXT,
                tarih TEXT,
                tutarlar_json TEXT,
                tablo_satirlari_json TEXT,
                FOREIGN KEY(document_id) REFERENCES ocr_documents(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS form_documents (
                document_id INTEGER PRIMARY KEY,
                alan_sayisi INTEGER,
                alanlar_json TEXT,
                satir_sayisi INTEGER,
                FOREIGN KEY(document_id) REFERENCES ocr_documents(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS table_documents (
                document_id INTEGER PRIMARY KEY,
                satir_sayisi INTEGER,
                tablo_json TEXT,
                FOREIGN KEY(document_id) REFERENCES ocr_documents(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS generic_documents (
                document_id INTEGER PRIMARY KEY,
                metin_uzunlugu INTEGER,
                kurum_adaylari_json TEXT,
                FOREIGN KEY(document_id) REFERENCES ocr_documents(id)
            )
            """
        )
    return path


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _insert_specialized_document(connection: sqlite3.Connection, document_id: int, item: dict[str, Any]) -> None:
    document_type = item.get("document_type")
    fields = item.get("fields") or {}
    insights = item.get("insights") or {}
    specialized = item.get("specialized") or {}

    if document_type == "kimlik":
        identity_fields = specialized.get("identity_fields") or {}
        connection.execute(
            """
            INSERT OR REPLACE INTO identity_documents (
                document_id,
                tc_kimlik_no,
                isim,
                soyisim,
                dogum_tarihi,
                adres
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                identity_fields.get("tc_kimlik_no") or fields.get("tc_kimlik_no"),
                identity_fields.get("isim") or fields.get("isim"),
                identity_fields.get("soyisim") or fields.get("soyisim"),
                identity_fields.get("tarih") or fields.get("tarih"),
                ", ".join(specialized.get("address_candidates") or insights.get("adres_adaylari") or []),
            ),
        )
    elif document_type == "fatura_veya_fis":
        amounts = specialized.get("amounts") or insights.get("tutarlar") or []
        connection.execute(
            """
            INSERT OR REPLACE INTO invoice_documents (
                document_id,
                belge_no,
                toplam_tutar,
                tarih,
                tutarlar_json,
                tablo_satirlari_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                fields.get("belge_no"),
                amounts[0] if amounts else None,
                (specialized.get("dates") or insights.get("tarihler") or [None])[0],
                _json_dump(amounts),
                _json_dump(specialized.get("table_rows") or item.get("rows") or []),
            ),
        )
    elif document_type == "form":
        form_fields = specialized.get("form_fields") or fields
        connection.execute(
            """
            INSERT OR REPLACE INTO form_documents (
                document_id,
                alan_sayisi,
                alanlar_json,
                satir_sayisi
            ) VALUES (?, ?, ?, ?)
            """,
            (
                document_id,
                len(form_fields),
                _json_dump(form_fields),
                specialized.get("line_count"),
            ),
        )
    elif document_type == "tablo":
        table_rows = specialized.get("table_rows") or item.get("rows") or []
        connection.execute(
            """
            INSERT OR REPLACE INTO table_documents (
                document_id,
                satir_sayisi,
                tablo_json
            ) VALUES (?, ?, ?)
            """,
            (
                document_id,
                specialized.get("row_count") or len(table_rows),
                _json_dump(table_rows),
            ),
        )
    else:
        connection.execute(
            """
            INSERT OR REPLACE INTO generic_documents (
                document_id,
                metin_uzunlugu,
                kurum_adaylari_json
            ) VALUES (?, ?, ?)
            """,
            (
                document_id,
                specialized.get("text_length") or len(str(item.get("text") or "")),
                _json_dump(specialized.get("institution_candidates") or insights.get("kurum_adlari") or []),
            ),
        )


def save_batch_to_database(
    report: dict[str, Any],
    db_path: str | Path,
    batch_name: str | None = None,
) -> dict[str, Any]:
    path = initialize_database(db_path)
    inserted = 0
    with sqlite3.connect(path) as connection:
        for item in report.get("items", []):
            cursor = connection.execute(
                """
                INSERT INTO ocr_documents (
                    batch_name,
                    source_folder,
                    file_name,
                    file_type,
                    document_type,
                    pipeline_name,
                    text_content,
                    fields_json,
                    insights_json,
                    specialized_json,
                    pages_json,
                    tables_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_name or Path(str(report.get("folder", "batch"))).name,
                    str(report.get("folder", "")),
                    item.get("file_name"),
                    item.get("file_type"),
                    item.get("document_type"),
                    item.get("pipeline_name"),
                    item.get("text"),
                    _json_dump(item.get("fields")),
                    _json_dump(item.get("insights")),
                    _json_dump(item.get("specialized")),
                    _json_dump(item.get("pages")),
                    _json_dump(item.get("tables") or item.get("rows")),
                ),
            )
            _insert_specialized_document(connection, int(cursor.lastrowid), item)
            inserted += 1
        connection.commit()

    return {"db_path": str(path), "inserted_count": inserted}


def get_database_overview(db_path: str | Path) -> dict[str, Any]:
    path = initialize_database(db_path)
    with sqlite3.connect(path) as connection:
        total_documents = connection.execute("SELECT COUNT(*) FROM ocr_documents").fetchone()[0]
        type_rows = connection.execute(
            "SELECT document_type, COUNT(*) FROM ocr_documents GROUP BY document_type ORDER BY COUNT(*) DESC"
        ).fetchall()
        batch_rows = connection.execute(
            "SELECT batch_name, COUNT(*) FROM ocr_documents GROUP BY batch_name ORDER BY COUNT(*) DESC"
        ).fetchall()

        specialized_counts = {
            "identity_documents": connection.execute("SELECT COUNT(*) FROM identity_documents").fetchone()[0],
            "invoice_documents": connection.execute("SELECT COUNT(*) FROM invoice_documents").fetchone()[0],
            "form_documents": connection.execute("SELECT COUNT(*) FROM form_documents").fetchone()[0],
            "table_documents": connection.execute("SELECT COUNT(*) FROM table_documents").fetchone()[0],
            "generic_documents": connection.execute("SELECT COUNT(*) FROM generic_documents").fetchone()[0],
        }

    return {
        "db_path": str(path),
        "total_documents": total_documents,
        "document_types": [{"document_type": row[0], "count": row[1]} for row in type_rows],
        "batches": [{"batch_name": row[0], "count": row[1]} for row in batch_rows],
        "specialized_counts": specialized_counts,
    }


def query_database(
    db_path: str | Path,
    search_text: str = "",
    document_types: list[str] | None = None,
    file_types: list[str] | None = None,
    batch_name: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    path = initialize_database(db_path)
    query = """
        SELECT
            id,
            batch_name,
            source_folder,
            file_name,
            file_type,
            document_type,
            pipeline_name,
            text_content,
            fields_json,
            insights_json,
            specialized_json,
            created_at
        FROM ocr_documents
        WHERE 1 = 1
    """
    params: list[Any] = []

    if search_text:
        query += " AND (file_name LIKE ? OR text_content LIKE ? OR pipeline_name LIKE ?)"
        pattern = f"%{search_text}%"
        params.extend([pattern, pattern, pattern])

    if document_types:
        placeholders = ", ".join("?" for _ in document_types)
        query += f" AND document_type IN ({placeholders})"
        params.extend(document_types)

    if file_types:
        placeholders = ", ".join("?" for _ in file_types)
        query += f" AND file_type IN ({placeholders})"
        params.extend(file_types)

    if batch_name:
        query += " AND batch_name = ?"
        params.append(batch_name)

    query += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, params).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        results.append(
            {
                "id": row["id"],
                "batch_name": row["batch_name"],
                "source_folder": row["source_folder"],
                "file_name": row["file_name"],
                "file_type": row["file_type"],
                "document_type": row["document_type"],
                "pipeline_name": row["pipeline_name"],
                "text_content": row["text_content"],
                "fields": json.loads(row["fields_json"] or "null"),
                "insights": json.loads(row["insights_json"] or "null"),
                "specialized": json.loads(row["specialized_json"] or "null"),
                "created_at": row["created_at"],
            }
        )
    return results


def get_specialized_records(db_path: str | Path, table_name: str, limit: int = 100) -> list[dict[str, Any]]:
    allowed_tables = {
        "identity_documents",
        "invoice_documents",
        "form_documents",
        "table_documents",
        "generic_documents",
    }
    if table_name not in allowed_tables:
        raise ValueError(f"Desteklenmeyen tablo: {table_name}")

    path = initialize_database(db_path)
    query = f"SELECT * FROM {table_name} ORDER BY document_id DESC LIMIT ?"
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, (limit,)).fetchall()

    return [dict(row) for row in rows]


def excel_bytes_from_report(report: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_rows = []
        for key, value in report.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                summary_rows.append({"alan": key, "deger": value})
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="ozet", index=False)

        if isinstance(report.get("fields"), dict):
            pd.DataFrame(
                [{"alan": key, "deger": value} for key, value in report["fields"].items()]
            ).to_excel(writer, sheet_name="alanlar", index=False)

        if isinstance(report.get("insights"), dict):
            rows: list[dict[str, Any]] = []
            for key, value in report["insights"].items():
                if isinstance(value, list):
                    rows.append({"alan": key, "deger": ", ".join(map(str, value))})
                else:
                    rows.append({"alan": key, "deger": value})
            pd.DataFrame(rows).to_excel(writer, sheet_name="icgoruler", index=False)

        if isinstance(report.get("pages"), list):
            page_rows: list[dict[str, Any]] = []
            for page in report["pages"]:
                page_rows.append(
                    {
                        "page": page.get("page"),
                        "classification": page.get("classification", {}).get("type"),
                        "text": page.get("text"),
                    }
                )
            if page_rows:
                pd.DataFrame(page_rows).to_excel(writer, sheet_name="pdf_sayfalari", index=False)

        if isinstance(report.get("tables"), list):
            table_rows: list[dict[str, Any]] = []
            for table in report["tables"]:
                for row_index, row in enumerate(table.get("rows", []), start=1):
                    table_rows.append(
                        {
                            "page": table.get("page"),
                            "row": row_index,
                            "values": " | ".join(row),
                        }
                    )
            if table_rows:
                pd.DataFrame(table_rows).to_excel(writer, sheet_name="pdf_tablolar", index=False)

        if isinstance(report.get("rows"), list):
            row_lengths = [len(row) for row in report["rows"]] or [0]
            max_cols = max(row_lengths)
            columns = [f"kolon_{index + 1}" for index in range(max_cols)]
            normalized_rows = [row + [""] * (max_cols - len(row)) for row in report["rows"]]
            pd.DataFrame(normalized_rows, columns=columns).to_excel(
                writer,
                sheet_name="tablo",
                index=False,
            )

        if isinstance(report.get("items"), list):
            batch_rows: list[dict[str, Any]] = []
            for item in report["items"]:
                batch_rows.append(
                    {
                        "file_name": item.get("file_name"),
                        "file_type": item.get("file_type"),
                        "document_type": item.get("document_type"),
                        "pipeline_name": item.get("pipeline_name"),
                        "text_preview": str(item.get("text", ""))[:250],
                    }
                )
            if batch_rows:
                pd.DataFrame(batch_rows).to_excel(writer, sheet_name="toplu_ozet", index=False)

    return buffer.getvalue()


def save_excel_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path.write_bytes(excel_bytes_from_report(report))
    return output_path



def save_table_csv(rows: list[list[str]], output_path: Path) -> Path:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)
    return output_path
