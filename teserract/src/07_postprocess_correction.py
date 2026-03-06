from __future__ import annotations

import argparse
from collections.abc import Iterable

import pytesseract
from rapidfuzz import fuzz, process

from common import clean_ocr_text, configure_tesseract, preprocess_for_ocr, read_image, save_json, save_text

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

CHARACTER_FIXES = str.maketrans(
    {
        "0": "O",
        "1": "I",
        "5": "S",
        "8": "B",
    }
)


def correct_token(token: str, vocabulary: Iterable[str], threshold: int = 78) -> tuple[str, int]:
    if len(token) <= 2 or any(char.isdigit() for char in token):
        return token, 100

    normalized = token.upper().translate(CHARACTER_FIXES)
    match = process.extractOne(normalized, vocabulary, scorer=fuzz.ratio)
    if not match:
        return token, 0

    candidate, score, _ = match
    if score < threshold:
        return token, score
    return candidate, score


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR post-processing ve düzeltme")
    parser.add_argument("image", help="images klasörü altındaki dosya adı")
    parser.add_argument("--lang", default="tur", help="Tesseract dil kodu")
    args = parser.parse_args()

    configure_tesseract()
    image = read_image(args.image)
    processed = preprocess_for_ocr(image)

    raw_text = pytesseract.image_to_string(processed, lang=args.lang, config="--psm 6")
    cleaned_text = clean_ocr_text(raw_text)

    corrections: list[dict[str, object]] = []
    corrected_lines: list[str] = []

    for line in cleaned_text.splitlines():
        corrected_tokens: list[str] = []
        for token in line.split():
            corrected, score = correct_token(token, DEFAULT_TERMS)
            corrected_tokens.append(corrected)
            if corrected != token:
                corrections.append(
                    {
                        "original": token,
                        "corrected": corrected,
                        "score": score,
                    }
                )
        corrected_lines.append(" ".join(corrected_tokens))

    corrected_text = "\n".join(corrected_lines)

    text_path = save_text("07_postprocessed_text.txt", corrected_text)
    json_path = save_json(
        "07_postprocessed_text.json",
        {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "corrected_text": corrected_text,
            "corrections": corrections,
        },
    )

    print(f"Düzeltilmiş metin kaydedildi: {text_path}")
    print(f"Düzeltme raporu kaydedildi: {json_path}")
    print("\nDüzeltilmiş metin:\n")
    print(corrected_text)


if __name__ == "__main__":
    main()
