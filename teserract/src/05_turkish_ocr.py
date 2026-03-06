from __future__ import annotations

import argparse

import pytesseract

from common import configure_tesseract, preprocess_for_ocr, read_image, save_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Türkçe OCR örneği")
    parser.add_argument("image", help="images klasörü altındaki dosya adı")
    args = parser.parse_args()

    configure_tesseract()
    image = read_image(args.image)
    processed = preprocess_for_ocr(image)

    text = pytesseract.image_to_string(processed, lang="tur", config="--psm 6")
    output_path = save_text("turkish_ocr_result.txt", text)

    print("Türkçe OCR sonucu:\n")
    print(text)
    print(f"\nMetin kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
