from __future__ import annotations

import argparse

import cv2
import pytesseract

from common import configure_tesseract, preprocess_for_ocr, read_image, save_output, save_text


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR öncesi görüntü temizleme")
    parser.add_argument("image", help="images klasörü altındaki dosya adı")
    parser.add_argument("--show", action="store_true", help="İşlenmiş görüntüyü pencerede göster")
    args = parser.parse_args()

    configure_tesseract()
    image = read_image(args.image)
    processed = preprocess_for_ocr(image)

    cleaned_path = save_output("preprocessed.png", processed)
    text = pytesseract.image_to_string(processed)
    text_path = save_text("preprocessed_ocr_result.txt", text)

    print(f"Temizlenmiş görüntü kaydedildi: {cleaned_path}")
    print(f"OCR metni kaydedildi: {text_path}")
    print("\nOCR sonucu:\n")
    print(text)

    if args.show:
        cv2.imshow("Processed", processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
