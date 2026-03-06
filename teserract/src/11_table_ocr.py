from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

import cv2
import pytesseract

from common import OCR_PSM, TABLE_SETTINGS, clean_ocr_text, configure_tesseract, preprocess_for_ocr, read_image, save_output


def sort_cells(contours: list, tolerance: int = 15) -> list[list[tuple[int, int, int, int]]]:
    boxes = [cv2.boundingRect(contour) for contour in contours]
    boxes = [box for box in boxes if box[2] > 20 and box[3] > 15]
    if not boxes:
        return []

    median_height = statistics.median(box[3] for box in boxes)
    median_width = statistics.median(box[2] for box in boxes)
    boxes = [
        box
        for box in boxes
        if box[2] <= median_width * 2.2 and box[3] <= median_height * 2.2
    ]

    deduped: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: (item[1], item[0])):
        if any(abs(box[0] - existing[0]) < 8 and abs(box[1] - existing[1]) < 8 for existing in deduped):
            continue
        deduped.append(box)
    boxes = deduped
    boxes.sort(key=lambda box: (box[1], box[0]))

    rows: list[list[tuple[int, int, int, int]]] = []
    for box in boxes:
        x, y, w, h = box
        placed = False
        for row in rows:
            if abs(row[0][1] - y) <= tolerance:
                row.append(box)
                row.sort(key=lambda item: item[0])
                placed = True
                break
        if not placed:
            rows.append([box])
    return rows


def build_column_centers(rows: list[list[tuple[int, int, int, int]]], tolerance: int) -> list[int]:
    clusters: list[dict[str, int]] = []
    for row in rows:
        for x, _, w, _ in row:
            center = x + w // 2
            matched_cluster = next(
                (cluster for cluster in clusters if abs(cluster["center"] - center) <= tolerance),
                None,
            )
            if matched_cluster is None:
                clusters.append({"center": center, "count": 1})
                continue

            total = matched_cluster["center"] * matched_cluster["count"] + center
            matched_cluster["count"] += 1
            matched_cluster["center"] = int(total / matched_cluster["count"])

    min_support = max(1, len(rows) // 3)
    return [
        cluster["center"]
        for cluster in sorted(clusters, key=lambda item: item["center"])
        if cluster["count"] >= min_support
    ]


def save_csv(rows: list[list[str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def looks_numeric_like(value: str) -> bool:
    cleaned = clean_ocr_text(value).upper()
    return bool(re.search(r"\d", cleaned) or any(token in cleaned for token in ["TL", ",", ".", "%"]))


def normalize_cell_value(value: str, column_index: int) -> str:
    cleaned = clean_ocr_text(value).replace("_", "").strip("- ")
    if column_index == 1 and cleaned in {"D", "S", "O"}:
        cleaned = cleaned.replace("D", "2") if cleaned == "D" else cleaned
        cleaned = cleaned.replace("S", "5") if cleaned == "S" else cleaned
        cleaned = cleaned.replace("O", "0") if cleaned == "O" else cleaned
    if column_index >= 2 and looks_numeric_like(cleaned):
        cleaned = cleaned.replace("O", "0")
        cleaned = re.sub(r"(?<=\d)\s+(?=\d)", "", cleaned)
    return cleaned


def normalize_header_value(value: str) -> str:
    cleaned = clean_ocr_text(value).upper()
    cleaned = cleaned.replace("0", "O")
    cleaned = cleaned.replace("1", "I")
    compact = cleaned.replace(" ", "")
    known_headers = {
        "TOPLAM": "TOPLAM",
        "BIRIMFIYAT": "BIRIM FIYAT",
        "URUN": "URUN",
        "ADET": "ADET",
        "MIKTAR": "MIKTAR",
    }
    if compact in known_headers:
        return known_headers[compact]
    cleaned = re.sub(r"\bBIRIMFIYAT\b", "BIRIM FIYAT", cleaned)
    cleaned = re.sub(r"\bB1RIM\b", "BIRIM", cleaned)
    cleaned = re.sub(r"\bFIYATI\b", "FIYAT", cleaned)
    return cleaned


def is_header_row(row_values: list[str]) -> bool:
    joined = " ".join(row_values)
    alpha_count = sum(char.isalpha() for char in joined)
    digit_count = sum(char.isdigit() for char in joined)
    return alpha_count > 0 and alpha_count >= digit_count * 2


def stabilize_header_row(row_values: list[str], body_rows: list[list[str]]) -> list[str]:
    stabilized = [normalize_header_value(value) for value in row_values]

    first_column_is_text = any(row and row[0] and not looks_numeric_like(row[0]) for row in body_rows)
    second_column_is_quantity = any(len(row) > 1 and row[1].isdigit() for row in body_rows)

    if stabilized and (not stabilized[0] or len(stabilized[0]) <= 1) and first_column_is_text:
        stabilized[0] = "URUN"
    if len(stabilized) > 1 and (not stabilized[1] or len(stabilized[1]) <= 1) and second_column_is_quantity:
        stabilized[1] = "ADET"
    if len(stabilized) > 2 and not stabilized[2]:
        stabilized[2] = "BIRIM FIYAT"
    if len(stabilized) > 3 and not stabilized[3]:
        stabilized[3] = "TOPLAM"

    return stabilized


def fill_down_sparse_cells(rows: list[list[str]], column_indexes: tuple[int, ...] = (0,)) -> list[list[str]]:
    if not rows:
        return rows

    filled_rows = [row[:] for row in rows]
    for row_index in range(1, len(filled_rows)):
        current_row = filled_rows[row_index]
        previous_row = filled_rows[row_index - 1]

        populated_cells = sum(1 for cell in current_row if cell)
        if populated_cells < max(2, len(current_row) // 2):
            continue

        for column_index in column_indexes:
            if column_index >= len(current_row) or column_index >= len(previous_row):
                continue
            if current_row[column_index]:
                continue

            previous_value = previous_row[column_index].strip()
            if previous_value and not looks_numeric_like(previous_value):
                current_row[column_index] = previous_value

    return filled_rows


def ocr_cell(image, box: tuple[int, int, int, int], lang: str, column_index: int) -> str:
    x, y, w, h = box
    roi = image[y : y + h, x : x + w]
    roi_processed = preprocess_for_ocr(roi)
    text = pytesseract.image_to_string(roi_processed, lang=lang, config=f"--psm {OCR_PSM['line']}")
    return normalize_cell_value(text, column_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR ile tablo tanıma")
    parser.add_argument("image", help="images klasörü altındaki tablo görseli")
    parser.add_argument("--lang", default="tur", help="Tesseract dil kodu")
    args = parser.parse_args()

    configure_tesseract()
    image = read_image(args.image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        ~gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        TABLE_SETTINGS["adaptive_block_size"],
        TABLE_SETTINGS["adaptive_c"],
    )

    horizontal = binary.copy()
    vertical = binary.copy()

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (TABLE_SETTINGS["horizontal_kernel"], 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, TABLE_SETTINGS["vertical_kernel"]))

    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, vertical_kernel)
    grid = cv2.add(horizontal, vertical)

    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rows = sort_cells(contours, tolerance=TABLE_SETTINGS["row_tolerance"])
    column_centers = build_column_centers(rows, TABLE_SETTINGS["column_tolerance"])

    annotated = image.copy()
    extracted_rows: list[list[str]] = []

    for row_index, row in enumerate(rows):
        grouped_values = {index: [] for index in range(len(column_centers))}
        for x, y, w, h in row:
            if w > image.shape[1] * 0.95 and h > image.shape[0] * 0.95:
                continue
            column_index = len(grouped_values)
            if column_centers:
                center = x + w // 2
                column_index = min(range(len(column_centers)), key=lambda index: abs(column_centers[index] - center))
            value = ocr_cell(image, (x, y, w, h), args.lang, column_index)
            if value:
                grouped_values.setdefault(column_index, []).append((x, value))
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        row_values: list[str] = []
        for column_index in range(len(grouped_values)):
            tokens = [value for _, value in sorted(grouped_values[column_index], key=lambda item: item[0]) if value]
            merged = " ".join(dict.fromkeys(tokens)).strip()
            row_values.append(merged)

        non_empty_ratio = sum(1 for cell in row_values if cell) / max(len(row_values), 1)
        if sum(1 for cell in row_values if cell) >= 2 and non_empty_ratio >= 0.5:
            extracted_rows.append(row_values)

    if extracted_rows and is_header_row(extracted_rows[0]):
        extracted_rows[0] = stabilize_header_row(extracted_rows[0], extracted_rows[1:])
        extracted_rows[1:] = fill_down_sparse_cells(extracted_rows[1:], column_indexes=(0,))
    else:
        extracted_rows = fill_down_sparse_cells(extracted_rows, column_indexes=(0,))

    image_path = save_output("11_table_structure.png", annotated)
    csv_path = Path(image_path.parent / "11_table_ocr.csv")
    save_csv(extracted_rows, csv_path)

    print(f"Tablo yapısı kaydedildi: {image_path}")
    print(f"CSV çıktısı kaydedildi: {csv_path}")
    print(extracted_rows)


if __name__ == "__main__":
    main()
