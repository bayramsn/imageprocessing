from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "images"
PDFS_DIR = PROJECT_ROOT / "pdfs"
TEMPLATES_DIR = PROJECT_ROOT / "templates"


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


FONT_26 = get_font(26)
FONT_32 = get_font(32)
FONT_38 = get_font(38)
FONT_44 = get_font(44)


def save(image: Image.Image, name: str) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image.save(IMAGES_DIR / name)



def draw_multiline(draw: ImageDraw.ImageDraw, start: tuple[int, int], lines: list[str], font, step: int = 44) -> None:
    x, y = start
    for line in lines:
        draw.text((x, y), line, fill="black", font=font)
        y += step



def create_sample_text() -> None:
    image = Image.new("RGB", (1400, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 60), "OCR TEST METNI", fill="black", font=FONT_44)
    lines = [
        "Bu ornek belge OCR denemesi icin hazirlandi.",
        "Belge No: DOC-2026-001",
        "Tarih: 06.03.2026",
        "Toplam Tutar: 2450,75 TL",
        "Adres: Ataturk Mah. Inonu Cad. No: 18 Kadikoy Istanbul",
    ]
    draw_multiline(draw, (60, 150), lines, FONT_32, step=60)
    save(image, "sample.png")



def create_turkish_text() -> None:
    image = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 40), "TURKCE OCR ORNEGI", fill="black", font=FONT_44)
    lines = [
        "Musteri Adi: Cagla Ozturk",
        "Adres: Cumhuriyet Mahallesi Sehitler Caddesi No: 25 Izmir",
        "Aciklama: Ogrenci kayit belgesi ve ucret bilgisi",
        "Telefon: 0532 111 22 33",
        "Odeme Tarihi: 06/03/2026",
    ]
    draw_multiline(draw, (50, 140), lines, FONT_32, step=64)
    save(image, "turkish_sample.png")



def create_document_sample() -> None:
    image = Image.new("RGB", (1600, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 40, 1560, 960), radius=18, outline="black", width=4)
    draw.rectangle((70, 70, 340, 150), outline="black", width=2)
    draw.text((90, 90), "T.C. KIMLIK", fill="black", font=FONT_38)
    draw.text((70, 200), "TC KIMLIK NO", fill="black", font=FONT_32)
    draw.text((370, 200), "12345678901", fill="black", font=FONT_32)
    draw.text((70, 300), "ADI", fill="black", font=FONT_32)
    draw.text((370, 300), "AHMET", fill="black", font=FONT_32)
    draw.text((70, 400), "SOYADI", fill="black", font=FONT_32)
    draw.text((370, 400), "YILMAZ", fill="black", font=FONT_32)
    draw.text((70, 500), "DOGUM TARIHI", fill="black", font=FONT_32)
    draw.text((370, 500), "12.05.1998", fill="black", font=FONT_32)
    draw.text((70, 600), "ADRES", fill="black", font=FONT_32)
    draw.text((370, 600), "ATATURK MAH. CUMHURIYET CAD. NO: 45 ANKARA", fill="black", font=FONT_26)
    save(image, "document_sample.png")
    image.crop((70, 70, 340, 150)).save(IMAGES_DIR / "tc_anchor.png")



def create_form_sample() -> None:
    image = Image.new("RGB", (1500, 1100), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 50), "BASVURU FORMU", fill="black", font=FONT_44)
    fields = [
        "Ad Soyad: Elif Demir",
        "Telefon: 0555 333 44 55",
        "E-Posta: elif.demir@example.com",
        "Bolum: Bilgisayar Muhendisligi",
        "Adres: Bahcelievler Mah. 12. Sokak No: 9 Bursa",
        "Aciklama: OCR form otomasyonu deneme verisi",
    ]
    y = 170
    for field in fields:
        draw.rectangle((50, y - 20, 1450, y + 40), outline="black", width=2)
        draw.text((70, y - 10), field, fill="black", font=FONT_32)
        y += 120
    save(image, "form_sample.png")



def create_table_sample() -> None:
    image = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 40), "URUN LISTESI", fill="black", font=FONT_44)
    left, top = 60, 140
    col_widths = [420, 220, 260, 260]
    row_height = 100
    headers = ["URUN", "ADET", "BIRIM FIYAT", "TOPLAM"]
    data_rows = [
        ["DEFTER", "2", "45,00 TL", "90,00 TL"],
        ["KALEM", "5", "12,50 TL", "62,50 TL"],
        ["SILGI", "3", "9,00 TL", "27,00 TL"],
    ]
    x = left
    for width in col_widths:
        draw.line((x, top, x, top + row_height * (len(data_rows) + 1)), fill="black", width=3)
        x += width
    draw.line((x, top, x, top + row_height * (len(data_rows) + 1)), fill="black", width=3)
    for row_index in range(len(data_rows) + 2):
        y = top + row_index * row_height
        draw.line((left, y, left + sum(col_widths), y), fill="black", width=3)

    x = left + 20
    for index, header in enumerate(headers):
        draw.text((x, top + 28), header, fill="black", font=FONT_26)
        x += col_widths[index]

    for row_index, row_values in enumerate(data_rows, start=1):
        x = left + 20
        y = top + row_index * row_height + 28
        for col_index, value in enumerate(row_values):
            draw.text((x, y), value, fill="black", font=FONT_26)
            x += col_widths[col_index]
    save(image, "table_sample.png")


def create_cmr_sample() -> None:
    image = Image.new("RGB", (1500, 1100), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 40), "INTERNATIONAL CONSIGNMENT NOTE (CMR)", fill="black", font=FONT_44)

    fields = [
        "Sender: XYZ Logistics Company",
        "Consignee: ABC Trading Ltd.",
        "Place of delivery: Berlin, Germany",
        "Place and date of taking in charge: Istanbul, Turkey - 12.04.2024",
        "Gross weight: 24500 kg",
        "Nature of goods: Electronic components, auto parts and textiles",
    ]

    y = 150
    for field in fields:
        draw.text((60, y), field, fill="black", font=FONT_32)
        y += 80

    save(image, "cmr_sample.png")

def create_pdf_sample() -> None:
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    first = Image.open(IMAGES_DIR / "sample.png").convert("RGB")
    second = Image.open(IMAGES_DIR / "form_sample.png").convert("RGB")
    third = Image.open(IMAGES_DIR / "table_sample.png").convert("RGB")
    first.save(PDFS_DIR / "ocr_demo_pack.pdf", save_all=True, append_images=[second, third])



def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    create_sample_text()
    create_turkish_text()
    create_document_sample()
    create_form_sample()
    create_table_sample()
    create_cmr_sample()
    create_pdf_sample()
    print("Ornek gorseller olusturuldu.")


if __name__ == "__main__":
    main()
