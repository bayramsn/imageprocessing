from __future__ import annotations

import argparse
import re
from typing import Any

import cv2
import numpy as np
import pytesseract

from common import OCR_PSM, clean_ocr_text, configure_tesseract, extract_text_lines, preprocess_for_ocr, read_image, save_json

FIELD_PATTERNS = {
    "tc_kimlik_no": [
        r"(?:TC\s*KIMLIK\s*NO|T\.C\.\s*KIMLIK\s*NO)\s*[:\-]?\s*(\d{11})\b",
        r"\b(\d{11})\b",
    ],
    "tarih": [
        r"(?:DOGUM\s*TARIHI|DOGUM\s*TARIH|TARIH)\s*[:\-]?\s*(\d{2}[./-]\d{2}[./-]\d{4})\b",
        r"\b(\d{2}[./-]\d{2}[./-]\d{4})\b",
    ],
    "isim": [
        r"(?:^|\b)(?:ADI|İSİM|ISIM)\b\s*[:\-]?\s*([A-ZÇĞİÖŞÜ]{2,}(?:\s+[A-ZÇĞİÖŞÜ]{2,}){0,2})$",
    ],
    "soyisim": [
        r"(?:^|\b)(?:SOYADI|SOYİSİM|SOYISIM)\b\s*[:\-]?\s*([A-ZÇĞİÖŞÜ]{2,}(?:\s+[A-ZÇĞİÖŞÜ]{2,}){0,2})$",
    ],
}

ANCHOR_FIELDS = {
    "tc_kimlik_no": (300, 120, 420, 70),
    "isim": (300, 220, 320, 70),
    "soyisim": (300, 320, 320, 70),
    "tarih": (300, 420, 280, 70),
}

LABEL_MAP = {
    "TC KIMLIK NO": "tc_kimlik_no",
    "ADI": "isim",
    "SOYADI": "soyisim",
    "DOGUM TARIHI": "tarih",
}

FIELD_DEFAULTS = {"tc_kimlik_no": None, "isim": None, "soyisim": None, "tarih": None}


def normalize_field_value(field_name: str, value: str | None) -> str | None:
    if not value:
        return None

    cleaned = clean_ocr_text(value).upper().strip(" -:")
    cleaned = re.sub(
        r"\b(TC\s*KIMLIK\s*NO|T\.C\.\s*KIMLIK\s*NO|ADI|İSİM|ISIM|SOYADI|SOYİSİM|SOYISIM|DOGUM\s*TARIHI|DOGUM\s*TARIH|TARIH)\b\s*[:\-]?",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if field_name == "tc_kimlik_no":
        match = re.search(r"\b\d{11}\b", cleaned)
        return match.group(0) if match else None

    if field_name == "tarih":
        match = re.search(r"\b\d{2}[./-]\d{2}[./-]\d{4}\b", cleaned)
        return match.group(0).replace("/", ".").replace("-", ".") if match else None

    cleaned = re.sub(r"[^A-ZÇĞİÖŞÜ\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def is_label_line(line: str) -> bool:
    return any(re.search(rf"\b{re.escape(label)}\b", line, flags=re.IGNORECASE) for label in LABEL_MAP)


def extract_with_regex(text: str) -> dict[str, Any]:
    result: dict[str, Any] = FIELD_DEFAULTS.copy()
    lines = [clean_ocr_text(line) for line in text.splitlines() if line.strip()]

    for field_name, patterns in FIELD_PATTERNS.items():
        search_lines = lines if field_name in {"isim", "soyisim"} else lines + [clean_ocr_text(text)]
        for line in search_lines:
            for pattern in patterns:
                match = re.search(pattern, line, flags=re.IGNORECASE)
                if not match:
                    continue
                value = match.group(1) if match.lastindex else match.group(0)
                normalized = normalize_field_value(field_name, value)
                if normalized:
                    result[field_name] = normalized
                    break
            if result[field_name]:
                break
    return result


def extract_from_lines(lines: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = FIELD_DEFAULTS.copy()
    upper_lines = [clean_ocr_text(line).upper() for line in lines]

    for index, line in enumerate(upper_lines):
        for label, field_name in LABEL_MAP.items():
            label_pattern = rf"\b{re.escape(label)}\b"
            if not re.search(label_pattern, line):
                continue

            inline_match = re.search(rf"{label_pattern}\s*[:\-]?\s*(.+)$", line)
            value = inline_match.group(1).strip() if inline_match and inline_match.group(1).strip() else None
            if (not value) and index + 1 < len(upper_lines) and not is_label_line(upper_lines[index + 1]):
                value = upper_lines[index + 1].strip()
            normalized = normalize_field_value(field_name, value)
            if normalized:
                result[field_name] = normalized

    if not result["tc_kimlik_no"]:
        match = re.search(FIELD_PATTERNS["tc_kimlik_no"][1], "\n".join(lines))
        if match:
            result["tc_kimlik_no"] = match.group(0)

    if not result["tarih"]:
        match = re.search(FIELD_PATTERNS["tarih"][1], "\n".join(lines))
        if match:
            result["tarih"] = normalize_field_value("tarih", match.group(0))

    return result


def match_anchor(document: np.ndarray, template: np.ndarray) -> tuple[int, int, float]:
    gray_doc = cv2.cvtColor(document, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray_doc, gray_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_loc[0], max_loc[1], float(max_val)


def extract_with_anchor(document: np.ndarray, template_name: str, lang: str) -> dict[str, Any]:
    template = read_image(template_name)
    anchor_x, anchor_y, score = match_anchor(document, template)

    extracted: dict[str, Any] = {"anchor_score": round(score, 4)}
    for field_name, (offset_x, offset_y, width, height) in ANCHOR_FIELDS.items():
        x1 = max(anchor_x + offset_x, 0)
        y1 = max(anchor_y + offset_y, 0)
        roi = document[y1 : y1 + height, x1 : x1 + width]
        roi_processed = preprocess_for_ocr(roi)
        text = pytesseract.image_to_string(roi_processed, lang=lang, config=f"--psm {OCR_PSM['line']}")
        extracted[field_name] = normalize_field_value(field_name, text)
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Alan bazlı veri çıkarımı")
    parser.add_argument("image", help="images klasörü altındaki belge dosyası")
    parser.add_argument("--lang", default="tur", help="Tesseract dil kodu")
    parser.add_argument(
        "--template",
        help="Anchor olarak kullanılacak küçük şablon görseli. Örn: tc_anchor.png",
    )
    args = parser.parse_args()

    configure_tesseract()
    image = read_image(args.image)
    processed = preprocess_for_ocr(image)
    data = pytesseract.image_to_data(
        processed,
        lang=args.lang,
        config=f"--psm {OCR_PSM['block']}",
        output_type=pytesseract.Output.DICT,
    )
    text = pytesseract.image_to_string(processed, lang=args.lang, config=f"--psm {OCR_PSM['block']}")
    cleaned_text = clean_ocr_text(text)
    line_result = extract_from_lines(extract_text_lines(data))

    regex_result = extract_with_regex(cleaned_text)
    anchor_result = None
    if args.template:
        anchor_result = extract_with_anchor(image, args.template, args.lang)

    payload = {
        "cleaned_text": cleaned_text,
        "line_result": line_result,
        "regex_result": regex_result,
        "anchor_result": anchor_result,
    }
    output_path = save_json("08_field_extraction.json", payload)

    print(f"Alan çıkarım sonucu kaydedildi: {output_path}")
    print(payload)


if __name__ == "__main__":
    main()
