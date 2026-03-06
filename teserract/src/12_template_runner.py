from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pytesseract

from common import PROJECT_ROOT, clean_ocr_text, configure_tesseract, preprocess_for_ocr, read_image, save_json


def load_template(template_name: str) -> dict[str, Any]:
    template_path = PROJECT_ROOT / "templates" / template_name
    return json.loads(template_path.read_text(encoding="utf-8"))



def run_template(template_name: str, image_name: str | None = None) -> dict[str, Any]:
    template = load_template(template_name)
    image_file = image_name or Path(template["image"]).name
    image = read_image(image_file)
    ocr_conf = template.get("ocr", {})
    lang = ocr_conf.get("lang", "tur")
    psm = ocr_conf.get("psm", 6)

    processed = preprocess_for_ocr(image)
    full_text = clean_ocr_text(
        pytesseract.image_to_string(processed, lang=lang, config=f"--psm {psm}")
    )

    fields: dict[str, Any] = {}
    for field_name, field_conf in template.get("fields", {}).items():
        roi = field_conf.get("roi")
        regex = field_conf.get("regex")
        value = None

        if roi:
            x, y, w, h = roi
            crop = image[y : y + h, x : x + w]
            crop_processed = preprocess_for_ocr(crop)
            value = clean_ocr_text(
                pytesseract.image_to_string(crop_processed, lang=lang, config="--psm 7")
            )
        elif regex:
            match = re.search(regex, full_text, flags=re.IGNORECASE)
            if match:
                value = match.group(0)

        if value and regex:
            match = re.search(regex, value, flags=re.IGNORECASE)
            if match:
                value = match.group(0)

        fields[field_name] = value

    return {
        "template": template["name"],
        "image": image_file,
        "text": full_text,
        "fields": fields,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="JSON template ile OCR alan çıkarımı")
    parser.add_argument("template", help="templates klasörü altındaki template dosyası")
    parser.add_argument("--image", help="Template içindeki varsayılan görsel yerine kullanılacak görsel")
    args = parser.parse_args()

    configure_tesseract()
    payload = run_template(args.template, args.image)
    output_path = save_json("12_template_runner.json", payload)

    print(f"Template sonucu kaydedildi: {output_path}")
    print(payload)


if __name__ == "__main__":
    main()
