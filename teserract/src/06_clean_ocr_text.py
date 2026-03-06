from __future__ import annotations

import argparse

import pytesseract

from common import clean_ocr_text, configure_tesseract, extract_text_lines, preprocess_for_ocr, read_image, save_json, save_text


def clean_lines(lines: list[str]) -> list[str]:
    cleaned_lines: list[str] = []
    for line in lines:
        cleaned = clean_ocr_text(line)
        if cleaned:
            cleaned_lines.append(cleaned)
    return cleaned_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR çıktısını temizleme örneği")
    parser.add_argument("image", help="images klasörü altındaki dosya adı")
    parser.add_argument("--lang", default="tur", help="Tesseract dil kodu")
    parser.add_argument("--psm", default="6", help="Page segmentation mode")
    args = parser.parse_args()

    configure_tesseract()
    image = read_image(args.image)
    processed = preprocess_for_ocr(image)

    data = pytesseract.image_to_data(
        processed,
        lang=args.lang,
        config=f"--psm {args.psm}",
        output_type=pytesseract.Output.DICT,
    )
    raw_text = pytesseract.image_to_string(
        processed,
        lang=args.lang,
        config=f"--psm {args.psm}",
    )
    line_based_text = "\n".join(clean_lines(extract_text_lines(data)))
    cleaned_text = line_based_text or clean_ocr_text(raw_text)

    raw_path = save_text("06_raw_ocr_text.txt", raw_text)
    clean_path = save_text("06_cleaned_ocr_text.txt", cleaned_text)
    json_path = save_json(
        "06_cleaned_ocr_text.json",
        {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "lines": clean_lines(extract_text_lines(data)),
        },
    )

    print(f"Ham çıktı kaydedildi: {raw_path}")
    print(f"Temiz çıktı kaydedildi: {clean_path}")
    print(f"Temizleme raporu kaydedildi: {json_path}")
    print("\nTemizlenmiş metin:\n")
    print(cleaned_text)


if __name__ == "__main__":
    main()
