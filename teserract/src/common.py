from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pytesseract

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    PROJECT_ROOT = Path(getattr(sys, "_MEIPASS"))
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TESSDATA_DIR = PROJECT_ROOT / "tessdata"

OCR_PSM = {
    "block": "6",
    "line": "7",
    "sparse": "11",
}

TABLE_SETTINGS = {
    "adaptive_block_size": 15,
    "adaptive_c": -2,
    "horizontal_kernel": 40,
    "vertical_kernel": 40,
    "column_tolerance": 28,
    "row_tolerance": 15,
}

# Windows üzerinde Tesseract kuruluysa ve PATH'e ekli değilse bu alanı doldur.
# Örnek: r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
TESSERACT_CMD: Optional[str] = None


def configure_tesseract() -> None:
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    else:
        env_cmd = os.environ.get("TESSERACT_CMD")
        if env_cmd and Path(env_cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = env_cmd
        else:
            windows_candidates = [
                Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
                Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
            ]
            for candidate in windows_candidates:
                if candidate.exists():
                    pytesseract.pytesseract.tesseract_cmd = str(candidate)
                    break

    if TESSDATA_DIR.exists():
        os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)


def read_image(image_name: str) -> np.ndarray:
    image_path = IMAGES_DIR / image_name
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(
            f"Görüntü bulunamadı: {image_path}. Örnek görselleri images klasörüne ekleyin."
        )
    return image


def save_output(file_name: str, image: np.ndarray) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / file_name
    cv2.imwrite(str(output_path), image)
    return output_path


def save_text(file_name: str, text: str) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / file_name
    output_path.write_text(text, encoding="utf-8")
    return output_path


def save_json(file_name: str, payload: Any) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / file_name
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    return cleaned


def normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def clean_ocr_text(text: str) -> str:
    substitutions = {
        "|": "I",
        "¦": "I",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
        "…": "...",
    }

    cleaned = text
    for source, target in substitutions.items():
        cleaned = cleaned.replace(source, target)

    cleaned = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ.,:/\-()%@]", " ", cleaned)
    cleaned = re.sub(r"(?<=\d)\s+(?=\d)", "", cleaned)
    cleaned = re.sub(r"(?<=[a-zçğıöşü0-9])(?=[A-ZÇĞİÖŞÜ])", " ", cleaned)
    cleaned = re.sub(r"(?<=[A-ZÇĞİÖŞÜ]{2})(?=NO\b)", " ", cleaned)

    known_fixes = {
        "TCKIMLIKNO": "TC KIMLIK NO",
        "DOGUMTARIHI": "DOGUM TARIHI",
        "TOPLAMTUTAR": "TOPLAM TUTAR",
        "ADSOYAD": "AD SOYAD",
        "ADRESATATURK": "ADRES ATATURK",
        "TARIHI": "TARIHI",
    }
    for source, target in known_fixes.items():
        cleaned = cleaned.replace(source, target)

    cleaned = re.sub(r"\bTL(?=[A-ZÇĞİÖŞÜ])", "TL ", cleaned)
    cleaned = re.sub(r"\s*:\s*", ": ", cleaned)
    return normalize_whitespace(cleaned)


def extract_text_lines(data: dict[str, list[str | int]]) -> list[str]:
    line_map: dict[tuple[int, int, int], list[tuple[int, str]]] = {}

    for i in range(len(data["text"])):
        raw_text = str(data["text"][i]).strip()
        if not raw_text:
            continue

        key = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
        )
        line_map.setdefault(key, []).append((int(data["left"][i]), raw_text))

    lines: list[str] = []
    for _, tokens in sorted(line_map.items()):
        ordered = " ".join(word for _, word in sorted(tokens, key=lambda item: item[0]))
        normalized = normalize_whitespace(ordered)
        if normalized:
            lines.append(normalized)

    return lines
