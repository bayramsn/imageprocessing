import argparse  # Komut satırı argümanlarını okumak için
import cv2  # OpenCV işlemleri için
import numpy as np  # Piksel sayımı için NumPy
from utils import load_image_bgr  # Güvenli görüntü yükleme yardımcı fonksiyonu


def main():  # Uygulama giriş noktası
    parser = argparse.ArgumentParser(description="Kenar sayısına göre EMPTY / NOT EMPTY sınıflandırması")  # Açıklamalı parser
    parser.add_argument("image", help="Sınıflandırılacak görüntü yolu")  # Girdi görüntüsü argümanı
    parser.add_argument("--blur", type=int, default=5, help="Gaussian blur çekirdek boyutu (tek sayı)")  # Blur boyutu
    parser.add_argument("--canny-low", type=int, default=50, help="Canny alt eşiği")  # Canny alt eşik
    parser.add_argument("--canny-high", type=int, default=150, help="Canny üst eşiği")  # Canny üst eşik
    parser.add_argument("--edge-thresh", type=int, default=500, help="EDGE piksel eşiği (üstü ise NOT EMPTY)")  # Sınıflandırma eşiği
    parser.add_argument("--show", action="store_true", help="Ara adımları ekranda göster")  # Görselleştirme bayrağı
    args = parser.parse_args()  # Argümanları oku

    img = load_image_bgr(args.image)  # Görüntüyü yükle
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gri tona çevir

    ksize = max(3, args.blur | 1)  # Çekirdeği tek ve en az 3 yap
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)  # Gaussian blur uygula

    edges = cv2.Canny(blurred, args.canny_low, args.canny_high)  # Canny kenar tespiti yap
    edge_pixels = int(np.count_nonzero(edges))  # Kenar piksel sayısını hesapla

    label = "NOT EMPTY" if edge_pixels > args.edge_thresh else "EMPTY"  # Basit kural ile etiket seç
    print(f"Kenar piksel sayısı: {edge_pixels}")  # Sayıyı yazdır
    print(f"Sonuç: {label}")  # Sonucu yazdır

    if args.show:  # Görselleştirme istenirse
        vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Kenarları BGR yap
        cv2.putText(vis, f"Edges: {edge_pixels}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)  # Yazı ekle
        cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)  # Pencere aç
        cv2.imshow("Edges", vis)  # Göster
        print("Pencereyi kapatmak için tuşa basın")  # Bilgi ver
        cv2.waitKey(0)  # Bekle
        cv2.destroyAllWindows()  # Kapat


if __name__ == "__main__":  # Doğrudan çalıştırıldığında
    main()  # Ana fonksiyonu çağır
