from __future__ import annotations

import argparse
from pathlib import Path

from common import OUTPUTS_DIR, PROJECT_ROOT, save_json, save_text
from pipelines import analyze_pdf_pages, configure, pdf_file_to_images, save_excel_report


def resolve_pdf_path(pdf_argument: str) -> Path:
    candidate = Path(pdf_argument)
    if candidate.exists():
        return candidate

    workspace_candidate = PROJECT_ROOT / "pdfs" / pdf_argument
    if workspace_candidate.exists():
        return workspace_candidate

    raise FileNotFoundError(f"PDF bulunamadı: {pdf_argument}")



def main() -> None:
    parser = argparse.ArgumentParser(description="PDF OCR ve Excel aktarımı")
    parser.add_argument("pdf", help="PDF dosya yolu veya pdfs klasörü altındaki dosya adı")
    parser.add_argument("--lang", default="tur", help="Tesseract dil kodu")
    parser.add_argument("--scale", type=float, default=2.0, help="PDF render ölçeği")
    args = parser.parse_args()

    configure()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = resolve_pdf_path(args.pdf)
    images = pdf_file_to_images(pdf_path, scale=args.scale)
    report = analyze_pdf_pages(images, lang=args.lang)

    stem = pdf_path.stem
    json_path = save_json(f"14_{stem}_pdf_ocr.json", report)
    text_path = save_text(f"14_{stem}_pdf_ocr.txt", report["full_text"])
    excel_path = save_excel_report(report, OUTPUTS_DIR / f"14_{stem}_pdf_ocr.xlsx")

    print(f"JSON kaydedildi: {json_path}")
    print(f"Metin kaydedildi: {text_path}")
    print(f"Excel kaydedildi: {excel_path}")
    print(f"Belge tipi: {report['document_type']}")
    print(f"Sayfa sayısı: {report['page_count']}")
    table_pages = [table for table in report.get("tables", []) if table.get("row_count", 0) > 0]
    print(f"Tablo tespit edilen sayfa sayısı: {len(table_pages)}")


if __name__ == "__main__":
    main()
