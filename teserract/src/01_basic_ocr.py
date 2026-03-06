from __future__ import annotations

import argparse

import pytesseract

from common import configure_tesseract, read_image, save_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Basit OCR örneği")
    parser.add_argument("image", help="images klasörü altındaki dosya adı")
    args = parser.parse_args()

    configure_tesseract()
    image = read_image(args.image)
    text = pytesseract.image_to_string(image)
    output_path = save_text("basic_ocr_result.txt", text)

    print("Çıkarılan metin:\n")
    print(text)
    print(f"\nMetin kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
