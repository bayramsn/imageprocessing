from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_module(file_name: str, module_name: str):
    file_path = SRC_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Modül yüklenemedi: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


field_extraction = load_module("08_field_extraction.py", "field_extraction_mod")
form_digitization = load_module("10_form_digitization.py", "form_digitization_mod")
table_ocr = load_module("11_table_ocr.py", "table_ocr_mod")


class QualityImprovementsTests(unittest.TestCase):
    def test_regex_extraction_is_label_aware(self) -> None:
        text = "TC KIMLIK NO: 12345678901\nADI: AHMET\nSOYADI: YILMAZ\nDOGUM TARIHI: 12.05.1998"
        result = field_extraction.extract_with_regex(text)
        self.assertEqual(result["tc_kimlik_no"], "12345678901")
        self.assertEqual(result["isim"], "AHMET")
        self.assertEqual(result["soyisim"], "YILMAZ")
        self.assertEqual(result["tarih"], "12.05.1998")

    def test_email_repair_fixes_common_ocr_mistakes(self) -> None:
        repaired = form_digitization.repair_email_value("elif.demir( example.com")
        self.assertEqual(repaired, "elif.demir@example.com")

    def test_table_normalization_keeps_headers_textual(self) -> None:
        self.assertEqual(table_ocr.normalize_header_value("T0PLAM"), "TOPLAM")
        self.assertEqual(table_ocr.normalize_cell_value("45,0O TL", 2), "45,00 TL")

    def test_fill_down_sparse_first_column(self) -> None:
        rows = [
            ["DEFTER", "2", "45,00 TL", "90,00 TL"],
            ["", "5", "12,50 TL", "62,50 TL"],
        ]
        filled = table_ocr.fill_down_sparse_cells(rows)
        self.assertEqual(filled[1][0], "DEFTER")


if __name__ == "__main__":
    unittest.main()
