import argparse  # Komut satiri argumanlarini okumak icin argparse kutuphanesi
import os  # Dosya yollarini ve dosya varligini kontrol etmek icin os modulu
from typing import Dict  # Sozluk tip ipucunu kullanmak icin typing modulu

import cv2  # OpenCV fonksiyonlarini kullanmak icin cv2 modulu
import numpy as np  # Sayisal islemler ve matrisler icin NumPy


DEFAULT_IMAGE = r"C:\opencv yakalayıcı\05_gaussian_blur_manual\ai.jpg"  # Varsayilan gorsel dosya yolu


def load_image(path: str) -> np.ndarray:  # Verilen yoldan gorseli okuyup donduren fonksiyon
    if not os.path.isfile(path):  # Dosya mevcut degilse
        raise FileNotFoundError(f"Image not found: {path}")  # Dosya bulunamadi hatasi firlat
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # OpenCV ile gorseli renkli olarak yuklemeye calis
    if img is None:  # Okuma basarisizsa
        with open(path, "rb") as f:  # Dosyayi ikili modda ac
            data = np.asarray(bytearray(f.read()), dtype=np.uint8)  # Veriyi byte dizisine cevir
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # Ham byte'lardan gorseli dekode et
    if img is None:  # Hala okunamadiysa
        raise ValueError(f"Unable to read image: {path}")  # Okuma hatasi bildir
    return img  # Yuklenen gorseli geri dondur


def preprocess(gray: np.ndarray, blur_ksize: int, blur_sigma: float) -> np.ndarray:  # Gri goruntuyu bulaniklastirarak on isleme tabi tut
    ksize = max(3, blur_ksize | 1)  # Cekirdegi en az 3 ve tek sayi olacak sekilde ayarla
    return cv2.GaussianBlur(gray, (ksize, ksize), blur_sigma)  # Gauss bulaniklastirmasi uygula ve sonucu dondur


def apply_clahe(gray: np.ndarray, clip: float, grid: int) -> np.ndarray:  # Kontrast artirimi icin CLAHE uygulayan fonksiyon
    grid = max(2, grid)  # Izgara boyutunu en az 2'ye sabitle
    clahe = cv2.createCLAHE(clipLimit=max(1.0, clip), tileGridSize=(grid, grid))  # CLAHE nesnesini belirtilen ayarlarla olustur
    return clahe.apply(gray)  # CLAHE uygulanmis gri goruntuyu dondur


def threshold_image(gray_blurred: np.ndarray, thresh: int, adaptive: bool, block: int, c: int, adaptive_merge: bool) -> np.ndarray:  # Esikleme stratejisini belirleyen fonksiyon
    binary = None  # Ikili goruntuyu tutacak degiskeni baslat
    if adaptive or adaptive_merge:  # Uyarlamali esikleme kullanilacaksa
        blk = max(3, block | 1)  # Blok boyutunu tek sayi ve en az 3 olarak zorunlu kil
        adaptive_bin = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, c)  # Uyarlamali esikleme uygula
        if not adaptive:  # Sadece birlestirme modu secildiyse
            binary = adaptive_bin  # Uyarlamali sonucu dogrudan kullan
    if binary is None:  # Henuz ikili goruntu olusmadiysa
        _, binary = cv2.threshold(gray_blurred, thresh, 255, cv2.THRESH_BINARY | (cv2.THRESH_OTSU if thresh == 0 else 0))  # Sabit veya Otsu esigi uygula
    if adaptive_merge and adaptive:  # Uyarlamali ve birlestirme birlikte istendiyse
        binary = cv2.bitwise_or(binary, adaptive_bin)  # Uyarlamali sonucu mevcut maskeyle OR islemi yap
    return binary  # Esiklenmis ikili goruntuyu dondur


def morphology(binary: np.ndarray, ksize: int, op: str, seq: bool = False) -> np.ndarray:  # Morfolojik islemleri uygulayan fonksiyon
    k = max(1, ksize)  # Cekirdek boyutunu sifirdan buyuk olacak sekilde ayarla
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))  # Eliptik yapisal eleman olustur
    ops = {  # Desteklenen morfoloji islemlerini tanimla
        "open": cv2.MORPH_OPEN,  # Gurultuyu temizlemek icin acma islemi
        "close": cv2.MORPH_CLOSE,  # Bosluklari kapatmak icin kapama islemi
        "erode": cv2.MORPH_ERODE,  # Asindirma islemi
        "dilate": cv2.MORPH_DILATE,  # Genisletme islemi
    }
    morph_type = ops.get(op, cv2.MORPH_OPEN)  # Kullanici secimine gore morfoloji turunu al, yoksa acma kullan
    result = cv2.morphologyEx(binary, morph_type, kernel)  # Secilen morfoloji islemini uygula
    if seq:  # Ek ardil siralama istenirse
        # Kucuk bosluklari kapatip gurultuyu temizlemek icin close -> open sirasini uygula
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)  # Once kapama uygula
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)  # Sonra acma uygula
    return result  # Morfolojik sonucu dondur


def watershed_split(binary: np.ndarray, min_distance: int = 3) -> np.ndarray:  # Dokunan paralar icin watershed bolme islemi
    # Separate touching coins using distance transform + watershed
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)  # Mesafe donusumunu hesapla
    if dist.max() == 0:  # Bos maskede bolme yapilamazsa
        return binary  # Girilen maskeyi oldugu gibi dondur
    markers = (dist > 0.25 * dist.max()).astype(np.uint8)  # Mesafe haritasindan tohum bolgeleri cikart
    markers = cv2.morphologyEx(markers, cv2.MORPH_ERODE, np.ones((max(1, min_distance), max(1, min_distance)), np.uint8))  # Tohumlari birbirinden ayirmak icin asindir
    num_labels, markers = cv2.connectedComponents(markers)  # Bagli bilesenleri etiketle
    markers = markers + 1  # Arka planin 1 olmasi icin etiketleri kaydir
    unknown = cv2.subtract(binary, (markers > 1).astype(np.uint8) * 255)  # Bilinmeyen bolgeleri cikart
    markers[unknown == 255] = 0  # Bilinmeyenleri 0 ile isaretle
    bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)  # Watershed icin gri maskeyi BGR formata cevir
    cv2.watershed(bgr, markers)  # Watershed algoritmasini uygula
    out = np.zeros_like(binary)  # Cikis maskesi icin bos matris olustur
    out[markers > 1] = 255  # Nesne bolgelerini beyaza boya
    return out  # Bolunmus maskeyi dondur


def detect_edges(img: np.ndarray, low: int, high: int) -> np.ndarray:  # Canny ile kenar tespiti yapan fonksiyon
    return cv2.Canny(img, low, high)  # Belirtilen alt ve ust esiklerle Canny uygula


def count_coins(binary: np.ndarray, min_area: float) -> Dict[str, np.ndarray]:  # Konturlari sayip maske olusturan fonksiyon
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Dis konturlari bul
    mask = np.zeros(binary.shape, dtype=np.uint8)  # Cizim icin bos maske olustur
    kept = 0  # Sayilacak kontur adedini baslat
    for cnt in contours:  # Her konturu isleme al
        area = cv2.contourArea(cnt)  # Kontur alanini hesapla
        if area < min_area:  # Minimum alanin altinda ise
            continue  # Bu konturu atla
        kept += 1  # Kabul edilen kontur sayisini artir
        cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)  # Konturu dolu olarak maskeye ciz
    return {"count": kept, "mask": mask, "areas": [cv2.contourArea(c) for c in contours]}  # Toplam sayi ve alan listelerini dondur


def estimate_min_area(binary: np.ndarray, pct: float, floor: float) -> float:  # Dinamik minimum alan limitini tahmin eden fonksiyon
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Konturlari bul
    areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 1]  # Onemsiz kucuk alanlari eleyerek liste olustur
    if not areas:  # Hic alan yoksa
        return floor  # Taban degerini dondur
    target = np.percentile(areas, pct * 100)  # Alanlarin belirtilen yuzdelik degerini al
    return max(floor, target)  # Yuzdelik ve taban arasindan buyuk olani sec


def stack_images(images: Dict[str, np.ndarray]) -> np.ndarray:  # Gorselleri tabloda birlestiren fonksiyon
    # Convert to BGR for uniform stacking
    prepared = []  # Birlestirmeye hazir goruntulerin listesi
    for name, img in images.items():  # Her isim-goruntu ciftini incele
        if img is None:  # Gorsel eksikse
            continue  # Atla
        if img.ndim == 2:  # Tek kanalli gri goruntu ise
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # BGR formatina donustur
        prepared.append(img)  # Islenmis goruntuyu listeye ekle
    # match sizes to first image
    base_h, base_w = prepared[0].shape[:2]  # Referans yukseklik ve genisligi al
    resized = [cv2.resize(im, (base_w, base_h), interpolation=cv2.INTER_AREA) for im in prepared]  # Tum goruntuleri referans boyutlara getir
    rows = []  # Satirlari tutacak liste
    cols = 3  # Her satirdaki kolon sayisi
    for i in range(0, len(resized), cols):  # Goruntuleri uc'erli gruplar halinde dolas
        rows.append(np.hstack(resized[i : i + cols]))  # Gruplari yatayda istifle ve satir listesine ekle
    max_w = max(r.shape[1] for r in rows)  # En genis satirin genisligini bul
    padded_rows = []  # Pad'lenmis satirlari toplayacak liste
    for r in rows:  # Her satir icin
        if r.shape[1] < max_w:  # Daha dar ise
            pad_w = max_w - r.shape[1]  # Eksik genisligi hesapla
            pad = np.zeros((r.shape[0], pad_w, r.shape[2]), dtype=r.dtype)  # Siyah pad olustur
            r = np.hstack((r, pad))  # Satiri pad ile genislet
        padded_rows.append(r)  # Satiri pad'lenmis listeye ekle
    return np.vstack(padded_rows)  # Tum satirlari dikeyde birlestirip dondur


def parse_args() -> argparse.Namespace:  # Komut satiri parametrelerini toplayan fonksiyon
    parser = argparse.ArgumentParser(description="Coin counting tool")  # Aciklama ile parser olustur
    parser.add_argument("--image", "-i", default=DEFAULT_IMAGE, help="Input image path")  # Girdi gorsel yolunu al
    parser.add_argument("--blur-ksize", type=int, default=5, help="Gaussian blur kernel size (odd)")  # Gauss bulaniklastirma cekirdek boyutu
    parser.add_argument("--blur-sigma", type=float, default=1.0, help="Gaussian sigma")  # Gauss sigma degeri
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE before blur")  # Bulaniklastirmadan once CLAHE uygulanacak mi
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit")  # CLAHE clip limiti
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE grid size")  # CLAHE grid boyutu
    parser.add_argument("--thresh", type=int, default=0, help="Threshold value (0=auto)")  # Sabit esik degeri
    parser.add_argument("--adaptive", action="store_true", help="Use adaptive threshold")  # Uyarlamali esikleme kullanilsin mi
    parser.add_argument("--adaptive-merge", action="store_true", help="Combine adaptive and auto threshold")  # Uyarlamali sonucu Otsu ile birlestir
    parser.add_argument("--block", type=int, default=11, help="Adaptive block size (odd)")  # Uyarlamali esik blok boyutu
    parser.add_argument("--c", type=int, default=2, help="Adaptive constant C")  # Uyarlamali esik sabiti
    parser.add_argument("--morph", choices=["open", "close", "erode", "dilate"], default="close", help="Morphology operation")  # Morfoloji islemi secimi
    parser.add_argument("--morph-ksize", type=int, default=5, help="Morphology kernel size")  # Morfoloji cekirdek boyutu
    parser.add_argument("--morph-seq", action="store_true", help="Apply additional morphology sequence")  # Morfoloji sonrasi close-open sirasi
    parser.add_argument("--canny-low", type=int, default=80, help="Canny lower")  # Canny alt esik degeri
    parser.add_argument("--canny-high", type=int, default=160, help="Canny upper")  # Canny ust esik degeri
    parser.add_argument("--min-area", type=float, default=30.0, help="Min area for counting")  # Kontur sayimi icin minimum alan
    parser.add_argument("--min-area-pct", type=float, default=0.005, help="Min area as percent image")  # Goruntu yuzdesi olarak alt alan limiti
    parser.add_argument("--area-pctile", type=float, default=0.2, help="Use this percentile of detected areas as dynamic min-area (0.2 = 20th percentile)")  # Alan yuzdeligi ile dinamik alt limit
    parser.add_argument("--watershed", action="store_true", help="Use watershed splitting")  # Watershed bolme islemi kullanilsin mi
    parser.add_argument("--min-distance", type=int, default=3, help="Min distance kernel for watershed seeds")  # Watershed tohumlari icin minimum mesafe cekirdegi
    return parser.parse_args()  # Argumanlari isle ve sonuc nesnesini dondur


def main() -> None:  # Programin ana calisma fonksiyonu
    args = parse_args()  # Komut satiri argumanlarini al
    bgr = load_image(args.image)  # Girdiyi oku ve BGR olarak al

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # Renkli goruntuyu gri tonlamaya cevir
    if args.clahe:  # CLAHE bayragi aciksa
        gray = apply_clahe(gray, clip=args.clahe_clip, grid=args.clahe_grid)  # CLAHE uygula

    gray_blur = preprocess(gray, args.blur_ksize, args.blur_sigma)  # Gri goruntuyu bulaniklastir
    binary = threshold_image(gray_blur, args.thresh, args.adaptive, args.block, args.c, args.adaptive_merge)  # Esikleme islemini yap
    binary_clean = morphology(binary, args.morph_ksize, args.morph, seq=args.morph_seq)  # Morfolojik temizlik uygula
    if args.watershed:  # Watershed isteniyorsa
        binary_clean = watershed_split(binary_clean, min_distance=args.min_distance)  # Dokunan nesneleri ayir
    edges = detect_edges(gray_blur, args.canny_low, args.canny_high)  # Kenarlari Canny ile tespit et

    img_area = binary.shape[0] * binary.shape[1]  # Goruntu alanini hesapla
    min_area_floor = max(args.min_area, img_area * (args.min_area_pct / 100))  # Sabit limit ile yuzde limitten buyuk olani al
    min_area_dyn = estimate_min_area(binary_clean, args.area_pctile, min_area_floor) if args.area_pctile > 0 else min_area_floor  # Dinamik minimum alani hesapla
    coins = count_coins(binary_clean, min_area_dyn)  # Konturlari say ve maske uret
    count = coins["count"]  # Toplam sayiyi al
    mask = coins["mask"]  # Cizilen maskeyi al

    annotated = bgr.copy()  # Cizimler icin orijinalin kopyasini olustur
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Konturlari yeniden bul
    for cnt in contours:  # Her kontur uzerinde dolas
        area = cv2.contourArea(cnt)  # Kontur alanini hesapla
        if area < min_area_dyn:  # Dinamik alt limiti saglamiyorsa
            continue  # Bu konturu es gec
        (x, y, w, h) = cv2.boundingRect(cnt)  # Konturu sinirlayan dikdortgeni bul
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dikdortgeni yesil olarak ciz
    cv2.putText(annotated, f"Coins: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)  # Toplam sayiyi ekrana yaz

    stacked = stack_images({  # Sonuclari bir tabloda birlestir
        "original": bgr,  # Orijinal BGR goruntu
        "gray_blur": gray_blur,  # Bulaniklastirilmis gri goruntu
        "binary": binary,  # Ilk esiklenmis maske
        "morph": binary_clean,  # Morfoloji sonrasi temiz maske
        "edges": edges,  # Kenar tespiti sonucu
        "mask": mask,  # Boyanmis kontur maskesi
        "annotated": annotated,  # Dikdortgenlerle anotlu goruntu
    })  # Stack islemini tamamla

    cv2.namedWindow("Coin Analyzer", cv2.WINDOW_NORMAL)  # Sonuc penceresini ac ve yeniden boyutlandirilabilir yap
    cv2.imshow("Coin Analyzer", stacked)  # Birlestirilmis goruntuleri pencerede goster
    print(f"Detected coins: {count}")  # Tespit edilen coin sayisini konsola yaz
    print("Press q or Esc to exit")  # Cikis talimatini konsola yaz
    while True:  # Pencere acik oldugu surece dongude kal
        if cv2.getWindowProperty("Coin Analyzer", cv2.WND_PROP_VISIBLE) < 1:  # Pencere kapandiysa
            break  # Donguden cik
        key = cv2.waitKey(30)  # Kullanicidan tus girdisi bekle
        if key is None or key == -1:  # Girdi yoksa
            continue  # Donguye devam et
        key &= 0xFF  # Tus degerini 8-bit araliginda tut
        if key in (27, ord("q")):  # Esc veya q tusuna basildiysa
            break  # Donguyu bitir
    cv2.destroyAllWindows()  # Tum OpenCV pencerelerini kapat


if __name__ == "__main__":  # Betik dogrudan calistiriliyorsa
    main()  # Ana fonksiyonu calistir
