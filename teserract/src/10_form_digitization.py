from __future__ import annotations

import argparse
import re

import cv2
import pytesseract

from common import OCR_PSM, clean_ocr_text, configure_tesseract, extract_text_lines, preprocess_for_ocr, read_image, save_json, save_output

KNOWN_KEYS = [
    "ad soyad",
    "telefon",
    "e-posta",
    "e posta",
    "eposta",
    "bolum",
    "adres",
    "aciklama",
]

KEY_ALIASES = {
    "ad_soyad": "ad_soyad",
    "telefon": "telefon",
    "e_posta": "e_posta",
    "eposta": "e_posta",
    "e-posta": "e_posta",
    "e_posta_adresi": "e_posta",
    "bolum": "bolum",
    "adres": "adres",
    "aciklama": "aciklama",
}

EMAIL_PATTERN = re.compile(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", flags=re.IGNORECASE)


def normalize_key(key: str) -> str:
    normalized = clean_ocr_text(key).lower().strip(" -:")
    normalized = normalized.replace("e posta", "e-posta")
    normalized = re.sub(r"[^a-zçğıöşü0-9\- ]", " ", normalized)
    normalized = re.sub(r"\s+", "_", normalized).strip("_")
    return KEY_ALIASES.get(normalized, normalized)


def repair_email_value(value: str) -> str:
    cleaned = clean_ocr_text(value).lower()
    replacements = {
        "(at)": "@",
        "[at]": "@",
        "{at}": "@",
        " at ": "@",
        " arroba ": "@",
        " © ": "@",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)

    cleaned = re.sub(r"(?<=\w)\s*[(\[{<]\s*(?=[a-z0-9])", "@", cleaned)
    cleaned = re.sub(r"(?<=\w)\s+[a]t\s+(?=[a-z0-9])", "@", cleaned)
    cleaned = re.sub(r"\s*@\s*", "@", cleaned)
    cleaned = re.sub(r"\s*\.\s*", ".", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)

    if "@" not in cleaned:
        compact_match = re.match(r"^([a-z0-9._%+\-]+)([a-z0-9.-]+\.[a-z]{2,})$", cleaned)
        if compact_match:
            cleaned = f"{compact_match.group(1)}@{compact_match.group(2)}"

    match = EMAIL_PATTERN.search(cleaned)
    return match.group(0) if match else clean_ocr_text(value)


def normalize_value(key: str, value: str) -> str:
    cleaned = clean_ocr_text(value)
    if key == "e_posta":
        return repair_email_value(cleaned)
    return cleaned


def infer_email_from_lines(lines: list[str]) -> str | None:
    for line in lines:
        repaired = repair_email_value(line)
        match = EMAIL_PATTERN.search(repaired)
        if match:
            return match.group(0)
    return None


def lines_to_fields(lines: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in lines:
        normalized = clean_ocr_text(line)
        if ":" in normalized:
            key, value = normalized.split(":", 1)
        else:
            key = ""
            value = ""
            lower_line = normalized.lower()
            for known_key in KNOWN_KEYS:
                if lower_line.startswith(known_key):
                    key = known_key
                    value = normalized[len(known_key) :].strip(" -:")
                    break

            if not key:
                match = re.match(r"^([A-Za-zÇĞİÖŞÜçğıöşü\- ]{3,20})\s+(.+)$", normalized)
                if match:
                    key, value = match.group(1), match.group(2)

        key = normalize_key(key)
        value = normalize_value(key, value)
        if key and value:
            current = fields.get(key)
            if not current or len(value) > len(current):
                fields[key] = value

    if "e_posta" not in fields:
        inferred_email = infer_email_from_lines(lines)
        if inferred_email:
            fields["e_posta"] = inferred_email

    return fields


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR ile form otomasyonu")
    parser.add_argument("image", help="images klasörü altındaki form dosyası")
    parser.add_argument("--lang", default="tur", help="Tesseract dil kodu")
    args = parser.parse_args()

    configure_tesseract()
    image = read_image(args.image)
    processed = preprocess_for_ocr(image)
    raw_text = pytesseract.image_to_string(processed, lang=args.lang, config=f"--psm {OCR_PSM['block']}")
    sparse_text = pytesseract.image_to_string(image, lang=args.lang, config=f"--psm {OCR_PSM['sparse']}")
    data = pytesseract.image_to_data(
        image,
        lang=args.lang,
        config=f"--psm {OCR_PSM['sparse']}",
        output_type=pytesseract.Output.DICT,
    )

    data_lines = extract_text_lines(data)
    text_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    sparse_lines = [line.strip() for line in sparse_text.splitlines() if line.strip()]
    lines = list(dict.fromkeys(data_lines + sparse_lines + text_lines))
    fields = lines_to_fields(lines)

    annotated = image.copy()
    for i in range(len(data["text"])):
        text = str(data["text"][i]).strip()
        if not text:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 1)

    image_path = save_output("10_form_digitization_boxes.png", annotated)
    json_path = save_json(
        "10_form_digitization.json",
        {
            "raw_text": raw_text,
            "sparse_text": sparse_text,
            "lines": lines,
            "fields": fields,
        },
    )

    print(f"Form kutuları kaydedildi: {image_path}")
    print(f"Form JSON kaydedildi: {json_path}")
    print(fields)


if __name__ == "__main__":
    main()
