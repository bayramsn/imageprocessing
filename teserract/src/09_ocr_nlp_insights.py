from __future__ import annotations

import argparse
import re

import pytesseract

from common import clean_ocr_text, configure_tesseract, preprocess_for_ocr, read_image, save_json

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


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR + NLP ile anlam çıkarımı")
    parser.add_argument("image", help="images klasörü altındaki dosya adı")
    parser.add_argument("--lang", default="tur", help="Tesseract dil kodu")
    args = parser.parse_args()

    configure_tesseract()
    image = read_image(args.image)
    processed = preprocess_for_ocr(image)

    raw_text = pytesseract.image_to_string(processed, lang=args.lang, config="--psm 6")
    cleaned_text = clean_ocr_text(raw_text)

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

    output_path = save_json(
        "09_ocr_nlp_insights.json",
        {
            "text": cleaned_text,
            "insights": insights,
        },
    )

    print(f"Anlam çıkarımı kaydedildi: {output_path}")
    print(insights)


if __name__ == "__main__":
    main()
