"""
Kenar tespiti ile basit sınıflandırma
rafta ürün var mı yok mu gibi basit sorular için

NOT: Bu yöntem çok basit, gerçek projelerde CNN kullanılmalı
     ama geleneksel CV yöntemlerini anlamak için güzel örnek
"""
import argparse
import cv2
import numpy as np
from utils import load_image_bgr


def main():
    parser = argparse.ArgumentParser(description="Kenar sayısına göre EMPTY / NOT EMPTY sınıflandırması")
    parser.add_argument("image", help="Sınıflandırılacak görüntü yolu")
    parser.add_argument("--blur", type=int, default=5, help="Gaussian blur çekirdek boyutu (tek sayı)")
    parser.add_argument("--canny-low", type=int, default=50, help="Canny alt eşiği")
    parser.add_argument("--canny-high", type=int, default=150, help="Canny üst eşiği")
    parser.add_argument("--edge-thresh", type=int, default=500, help="EDGE piksel eşiği")
    parser.add_argument("--show", action="store_true", help="Ara adımları ekranda göster")
    args = parser.parse_args()

    img = load_image_bgr(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur uygula - gürültüyü azaltır
    ksize = max(3, args.blur | 1)  # tek sayı olmalı
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # canny edge detection
    edges = cv2.Canny(blurred, args.canny_low, args.canny_high)
    
    # kenar piksellerini say
    edge_pixels = int(np.count_nonzero(edges))

    # basit kural: çok kenar varsa dolu, yoksa boş
    # FIXME: bu eşik değeri her resim için farklı olmalı
    label = "NOT EMPTY" if edge_pixels > args.edge_thresh else "EMPTY"
    
    print(f"Kenar piksel sayısı: {edge_pixels}")
    print(f"Sonuç: {label}")

    if args.show:
        vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, f"Edges: {edge_pixels}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
        cv2.imshow("Edges", vis)
        print("Pencereyi kapatmak için tuşa basın")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
