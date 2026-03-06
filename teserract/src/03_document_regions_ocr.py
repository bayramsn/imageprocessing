from __future__ import annotations

import argparse
from dataclasses import dataclass

import pytesseract

from common import configure_tesseract, preprocess_for_ocr, read_image, save_json


@dataclass
class Region:
    name: str
    x: int
    y: int
    w: int
    h: int


# `document_sample.png` örneğine göre ayarlanmış bölgeler.
REGIONS = [
    Region("tc_kimlik_no", 350, 180, 360, 70),
    Region("isim", 350, 280, 260, 70),
    Region("soyisim", 350, 380, 260, 70),
    Region("dogum_tarihi", 350, 480, 260, 70),
    Region("adres", 350, 580, 900, 90),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Belge üzerindeki sabit bölgelerden OCR okuma")
    parser.add_argument("--image", default="document_sample.png", help="images klasörü altındaki belge adı")
    parser.add_argument("--lang", default="tur+eng", help="Tesseract dil kodu")
    args = parser.parse_args()

    configure_tesseract()
    image = read_image(args.image)
    payload: dict[str, str] = {}

    for region in REGIONS:
        roi = image[region.y : region.y + region.h, region.x : region.x + region.w]
        processed_roi = preprocess_for_ocr(roi)
        text = pytesseract.image_to_string(processed_roi, lang=args.lang, config="--psm 7")
        cleaned = " ".join(text.split())
        payload[region.name] = cleaned
        print(f"{region.name}: {cleaned}")

    output_path = save_json("03_document_regions_ocr.json", payload)
    print(f"\nBölgesel OCR JSON kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
