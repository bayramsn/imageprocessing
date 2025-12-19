import argparse  # Komut satırı argümanlarını okumak için
import cv2  # OpenCV fonksiyonları için
from utils import load_image_bgr, bgr_to_rgb  # Yardımcı yükleme ve renk dönüşümü


def main():  # Programın giriş noktası
    parser = argparse.ArgumentParser(description="ORB ile iki görüntü arasındaki benzerliği ölç")  # Açıklamalı argüman ayrıştırıcı
    parser.add_argument("image1", help="Birinci görüntü yolu")  # İlk görüntü yolu argümanı
    parser.add_argument("image2", help="İkinci görüntü yolu")  # İkinci görüntü yolu argümanı
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe oran testi eşiği")  # Oran testi eşiği
    parser.add_argument("--min-matches", type=int, default=20, help="Benzer sayılmak için minimum iyi eşleşme")  # Eşik değer
    parser.add_argument("--show", action="store_true", help="Ekranda eşleşmeleri göster")  # Görselleştirme bayrağı
    args = parser.parse_args()  # Argümanları oku

    img1 = load_image_bgr(args.image1)  # İlk görüntüyü BGR olarak yükle
    img2 = load_image_bgr(args.image2)  # İkinci görüntüyü BGR olarak yükle

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # Birinciyi griye çevir
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # İkinciyi griye çevir

    orb = cv2.ORB_create(nfeatures=2000)  # ORB dedektörünü oluştur
    kp1, desc1 = orb.detectAndCompute(gray1, None)  # İlk görüntüde anahtar noktaları ve tanımlayıcıları bul
    kp2, desc2 = orb.detectAndCompute(gray2, None)  # İkinci görüntüde aynı işlemi yap

    if desc1 is None or desc2 is None:  # Tanımlayıcı yoksa
        raise RuntimeError("Görüntülerden özellik çıkarılamadı; farklı görüntüler deneyin")  # Hata ver

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # Hamming mesafesiyle BF eşleyici
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)  # Her tanımlayıcı için en iyi iki eşleşmeyi bul

    good = []  # İyi eşleşmeleri tutacak liste
    for m, n in knn_matches:  # Tüm eşleşmeler üzerinde dolaş
        if m.distance < args.ratio * n.distance:  # Lowe oran testi koşulu
            good.append(m)  # Koşulu sağlayanı ekle

    similarity_score = len(good)  # Skor olarak iyi eşleşme sayısı
    print(f"İyi eşleşme sayısı: {similarity_score}")  # Skoru yazdır
    if similarity_score >= args.min_matches:  # Eşik kontrolü
        print("Sonuç: BENZER")  # Benzer mesajı
    else:  # Eşik altı ise
        print("Sonuç: BENZEMİYOR")  # Benzemiyor mesajı

    if args.show:  # Görselleştirme istenmişse
        good_sorted = sorted(good, key=lambda m: m.distance)  # Eşleşmeleri mesafeye göre sırala
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_sorted, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # Eşleşmeleri çiz
        cv2.namedWindow("ORB Matches", cv2.WINDOW_NORMAL)  # Pencere aç
        cv2.imshow("ORB Matches", img_matches)  # Görüntüyü göster
        print("Pencereyi kapatmak için bir tuşa basın")  # Bilgi mesajı
        cv2.waitKey(0)  # Tuş bekle
        cv2.destroyAllWindows()  # Pencereyi kapat


if __name__ == "__main__":  # Dosya doğrudan çalıştırıldığında
    main()  # Ana fonksiyonu çağır
