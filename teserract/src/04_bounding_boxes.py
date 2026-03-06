from __future__ import annotations

import argparse

import cv2
import pytesseract

from common import configure_tesseract, read_image, save_output


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR bounding box görselleştirme")
    parser.add_argument("--show", action="store_true", help="İşaretlenmiş görüntüyü pencerede göster")
    args = parser.parse_args()

    configure_tesseract()
    image = read_image("document_sample.png")

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(float(data["conf"][i])) if data["conf"][i] != "-1" else -1

        if text and conf > 40:
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                image,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

    output_path = save_output("bounding_boxes.png", image)
    print(f"İşaretlenmiş görüntü kaydedildi: {output_path}")

    if args.show:
        cv2.imshow("Bounding Boxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
