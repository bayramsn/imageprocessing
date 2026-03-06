from __future__ import annotations

import argparse
from pathlib import Path

from common import OUTPUTS_DIR, save_json
from pipelines import batch_process_folder, configure, save_batch_to_database, save_excel_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Klasör içindeki görsel ve PDF dosyalarını toplu OCR işleme")
    parser.add_argument("folder", help="İşlenecek klasör yolu")
    parser.add_argument("--lang", default="tur", help="Tesseract dil kodu")
    parser.add_argument("--images-only", action="store_true", help="Sadece görseller işlensin")
    parser.add_argument("--pdfs-only", action="store_true", help="Sadece PDF dosyaları işlensin")
    parser.add_argument("--recursive", action="store_true", help="Alt klasörleri de işle")
    parser.add_argument(
        "--db-path",
        default=str(OUTPUTS_DIR / "ocr_results.db"),
        help="SQLite veritabanı dosya yolu",
    )
    parser.add_argument("--skip-db", action="store_true", help="Veritabanına kayıt yapma")
    args = parser.parse_args()

    configure()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    include_images = not args.pdfs_only
    include_pdfs = not args.images_only
    target_folder = args.folder
    if args.recursive:
        print("Recursive mod etkin. Alt klasörler de taranacak.")
    report = batch_process_folder(
        target_folder,
        lang=args.lang,
        include_images=include_images,
        include_pdfs=include_pdfs,
        recursive=args.recursive,
    )

    folder_name = Path(args.folder).resolve().name
    json_path = save_json(f"15_batch_{folder_name}.json", report)
    excel_path = save_excel_report(report, OUTPUTS_DIR / f"15_batch_{folder_name}.xlsx")
    db_result = None
    if not args.skip_db:
        db_result = save_batch_to_database(report, args.db_path, batch_name=folder_name)

    print(f"Toplu işlem JSON kaydedildi: {json_path}")
    print(f"Toplu işlem Excel kaydedildi: {excel_path}")
    if db_result is not None:
        print(f"Veritabanına kaydedildi: {db_result['db_path']}")
        print(f"Eklenen kayıt sayısı: {db_result['inserted_count']}")
    print(f"İşlenen dosya sayısı: {report['file_count']}")
    print(report['document_type_counts'])
    for item in report.get("items", []):
        print(f"- {item.get('file_name')} -> {item.get('document_type')} ({item.get('pipeline_name')})")


if __name__ == "__main__":
    main()
