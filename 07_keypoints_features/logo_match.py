import argparse  # Komut satırı argümanlarını işler
import os  # Dosya varlığını kontrol eder
from typing import List, Tuple  # Tip ipuçları için

import cv2  # OpenCV ile özellik çıkarımı ve eşleştirme
import numpy as np  # Sayısal matris işlemleri


def create_detector(kind: str):  # Özellik dedektörünü seçer/kurar
    kind = kind.lower()  # Girdiyi küçük harfe indirger
    if kind == "sift":  # SIFT seçildiyse
        if not hasattr(cv2, "SIFT_create"):  # Derlemede SIFT yoksa hata ver
            raise RuntimeError("OpenCV build lacks SIFT; choose ORB or install opencv-contrib-python")
        return cv2.SIFT_create()  # SIFT dedektör + tanımlayıcıyı oluşturur
    return cv2.ORB_create(nfeatures=2000)  # Varsayılan olarak ORB dedektörü döner


def load_gray(path: str) -> np.ndarray:  # Görseli gri tonlamada yükler
    if not os.path.isfile(path):  # Yol gerçekten dosya mı kontrol et
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Dosyayı gri olarak oku
    if img is None:  # Standart okuma başarısızsa (özellikle Unicode yollarda)
        with open(path, "rb") as f:  # Dosyayı ikili moda aç
            data = np.asarray(bytearray(f.read()), dtype=np.uint8)  # Baytları numpy dizisine çevir
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)  # Manuel decode ile yeniden dene
    if img is None:  # Hâlâ okunamadıysa hata fırlat
        raise ValueError(f"Unable to read image: {path}")
    return img  # Gri görseli döndür


def match_features(desc1: np.ndarray, desc2: np.ndarray, kind: str, ratio: float) -> List[cv2.DMatch]:  # Özellik eşleştirir
    if kind == "sift":  # SIFT için L2 uzaklığı kullan
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # Çapraz kontrolsüz BF eşleyici
    else:  # ORB için Hamming uzaklığı kullan
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # Çapraz kontrolsüz BF eşleyici
    knn = matcher.knnMatch(desc1, desc2, k=2)  # Her tanımlayıcı için en iyi iki eşleşmeyi bul
    good = []  # Filtrelenmiş iyi eşleşmeler listesi
    for m, n in knn:  # Lowe oran testi uygula
        if m.distance < ratio * n.distance:  # Oran eşiği sağlanıyorsa kabul et
            good.append(m)  # İyi eşleşmelere ekle
    return good  # İyi eşleşmeleri döndür


def find_homography(kp1, kp2, matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:  # Homografi hesaplar
    if len(matches) < 4:  # En az 4 eşleşme yoksa mümkün değil
        return None, None  # Homografi üretilemedi
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)  # Şablon noktaları
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)  # Sahne noktaları
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)  # RANSAC ile sağlam homografi hesapla
    return H, mask  # Matris ve uyum maskesini döndür


def draw_box(scene_color: np.ndarray, template_shape: Tuple[int, int], H: np.ndarray) -> np.ndarray:  # Homografi ile kutu çizer
    h, w = template_shape  # Şablon yüksekliği ve genişliği
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)  # Köşe noktaları
    dst = cv2.perspectiveTransform(pts, H)  # Köşeleri sahneye yansıt
    out = scene_color.copy()  # Orijinal görüntüyü bozma
    cv2.polylines(out, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)  # Yeşil kutuyu çiz
    return out  # Çizilmiş görüntüyü döndür


def main():  # Uygulamanın giriş noktası
    parser = argparse.ArgumentParser(description="Logo detection with ORB/SIFT feature matching")  # Argüman ayrıştırıcıyı kur
    parser.add_argument(  # Şablon görsel yolu argümanı
        "--template",
        default=r"C:\opencv yakalayıcı\07_keypoints_features\ChatGPT Image 10 Ara 2025 15_10_57.png",
        help="Template logo image path",
    )  # Varsayılan örnek dosya
    parser.add_argument(  # Sahne görsel yolu argümanı
        "--scene",
        default=r"C:\opencv yakalayıcı\07_keypoints_features\ChatGPT Image 10 Ara 2025 15_10_57.png",
        help="Scene image path",
    )  # Varsayılan sahne dosyası
    parser.add_argument("--feature", choices=["orb", "sift"], default="orb", help="Feature type")  # Kullanılacak dedektör
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test threshold")  # Oran eşiği
    parser.add_argument("--min-good", type=int, default=15, help="Min good matches to declare detection")  # Tespit eşiği
    parser.add_argument("--max-draw", type=int, default=50, help="Max matches to draw")  # Çizilecek eşleşme sayısı
    parser.add_argument("--output", default="", help="Optional path to save visualization")  # Kaydedilecek dosya yolu
    args = parser.parse_args()  # Argümanları oku

    template = load_gray(args.template)  # Şablon görselini gri yükle
    scene = load_gray(args.scene)  # Sahne görselini gri yükle

    detector = create_detector(args.feature)  # Seçilen dedektörü oluştur
    kp1, desc1 = detector.detectAndCompute(template, None)  # Şablon için anahtar noktalar ve tanımlayıcılar
    kp2, desc2 = detector.detectAndCompute(scene, None)  # Sahne için anahtar noktalar ve tanımlayıcılar
    if desc1 is None or desc2 is None:  # Tanımlayıcı yoksa
        raise RuntimeError("No descriptors found; check images or try SIFT")  # Hata ver

    good = match_features(desc1, desc2, args.feature, args.ratio)  # Eşleşmeleri oran testinden geçir
    good_sorted = sorted(good, key=lambda m: m.distance)  # En düşük mesafeden sırala
    good_draw = good_sorted[: args.max_draw]  # Görselleştirilecek alt küme

    detected = len(good) >= args.min_good  # Yeterli iyi eşleşme var mı karar ver
    msg = f"Logo bulundu ({len(good)} good matches)" if detected else f"Logo bulunamadi ({len(good)} good matches)"  # Durum mesajı hazırla

    H, mask = find_homography(kp1, kp2, good_sorted)  # Homografiyi hesapla
    scene_color = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)  # Çizim için sahneyi renklendir
    if detected and H is not None:  # Hem yeterli eşleşme hem homografi varsa
        scene_color = draw_box(scene_color, template.shape, H)  # Kutuyu çiz
        cv2.putText(scene_color, "Logo bulundu", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3, cv2.LINE_AA)  # Başarılı metin yaz
    else:  # Aksi halde başarısız metni yaz
        cv2.putText(scene_color, "Bulunamadi", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)  # Başarısız metin yaz

    vis = cv2.drawMatches(template, kp1, scene_color, kp2, good_draw, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # Eşleşmeleri görselleştir
    print(msg)  # Konsola durum yazdır

    cv2.namedWindow("Logo Match", cv2.WINDOW_NORMAL)  # Yeniden boyutlanabilir pencere aç
    cv2.imshow("Logo Match", vis)  # Pencereyi göster
    if args.output:  # Çıktı yolu verildiyse
        cv2.imwrite(args.output, vis)  # Görselleştirmeyi kaydet
        print(f"Saved: {args.output}")  # Kaydedildiğini bildir
    print("Press q or Esc to exit")  # Çıkış talimatı yaz
    while True:  # Sonsuz döngüyle tuş bekle
        key = cv2.waitKey(1) & 0xFF  # 1 ms bekle ve tuşu al
        if key in (27, ord("q")):  # Esc veya q basıldıysa
            break  # Döngüden çık
    cv2.destroyAllWindows()  # Açık tüm pencereleri kapat


if __name__ == "__main__":  # Betik doğrudan çalıştırıldıysa
    main()  # Ana fonksiyonu çalıştır
