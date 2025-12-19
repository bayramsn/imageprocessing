"""
ORB ile görüntü benzerliği ölçümü
İki resim arasında keypoint karşılaştırması yapıyor
"""
import argparse
import cv2
from utils import load_image_bgr, bgr_to_rgb


def main():
    parser = argparse.ArgumentParser(description="ORB ile iki görüntü arasındaki benzerliği ölç")
    parser.add_argument("image1", help="Birinci görüntü yolu")
    parser.add_argument("image2", help="İkinci görüntü yolu")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe oran testi eşiği")
    parser.add_argument("--min-matches", type=int, default=20, help="Benzer sayılmak için minimum iyi eşleşme")
    parser.add_argument("--show", action="store_true", help="Ekranda eşleşmeleri göster")
    args = parser.parse_args()

    # resimleri yükle
    img1 = load_image_bgr(args.image1)
    img2 = load_image_bgr(args.image2)

    # griye çevir - orb gri istiyor
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ORB dedektörü oluştur
    # nfeatures'ı artırınca daha fazla keypoint buluyor ama yavaşlıyor
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, desc1 = orb.detectAndCompute(gray1, None)
    kp2, desc2 = orb.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None:
        raise RuntimeError("Görüntülerden özellik çıkarılamadı; farklı görüntüler deneyin")

    # eşleştirici - ORB için hamming mesafesi kullanılıyor
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    # lowe ratio test - bu kısım önemli
    # en iyi 2 eşleşme arasındaki oran belirli bir değerin altındaysa kabul et
    good = []
    for m, n in knn_matches:
        if m.distance < args.ratio * n.distance:
            good.append(m)

    # sonucu yazdır
    similarity_score = len(good)
    print(f"İyi eşleşme sayısı: {similarity_score}")
    if similarity_score >= args.min_matches:
        print("Sonuç: BENZER")
    else:
        print("Sonuç: BENZEMİYOR")

    # görselleştirme
    if args.show:
        good_sorted = sorted(good, key=lambda m: m.distance)
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_sorted, None, 
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.namedWindow("ORB Matches", cv2.WINDOW_NORMAL)
        cv2.imshow("ORB Matches", img_matches)
        print("Pencereyi kapatmak için bir tuşa basın")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
