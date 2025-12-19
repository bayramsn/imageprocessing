import argparse  # argparse ile komut satırı argümanlarını okuyoruz
import os  # Dosya yolu ve kontrol işlemleri için

import cv2  # OpenCV ana kütüphanesi
import numpy as np  # Numpy, görüntüleri yanyana birleştirmek için


def parse_args() -> argparse.Namespace:  # Komut satırı argümanlarını hazırlayan fonksiyon
    parser = argparse.ArgumentParser(  # Açıklama metni ile argüman parser'ı
        description="Gaussian blur trackbar demo: adjust kernel size and sigma live."
    )
    default_image = r"C:\opencv yakalayıcı\05_gaussian_blur_manual\ai.jpg"
    parser.add_argument(  # Görüntü dosya yolunu zorunlu argüman olarak ekliyoruz
        "--image",
        "-i",
        default=default_image,
        help=f"Path to the input image to blur (default: {default_image})",
    )
    return parser.parse_args()  # Argümanları parse edip döndürüyoruz


def main() -> None:  # Uygulamanın ana fonksiyonu
    args = parse_args()  # Argümanları oku

    if not os.path.isfile(args.image):  # Dosya mevcut mu kontrolü
        raise FileNotFoundError(f"Image not found: {args.image}")  # Yoksa hata ver

    original = cv2.imread(args.image)  # Görüntüyü oku
    if original is None:  # Okuma başarısızsa
        # Unicode yol problemleri için manuel okuma denemesi
        with open(args.image, "rb") as f:  # Dosyayı ikili modda oku
            data = np.asarray(bytearray(f.read()), dtype=np.uint8)  # Baytları numpy dizisine çevir
        original = cv2.imdecode(data, cv2.IMREAD_COLOR)  # OpenCV ile decode et
    if original is None:  # Hâlâ okunamadıysa
        raise ValueError(f"Unable to read image: {args.image}")  # Hata fırlat

    window_name = "Gaussian Blur Explorer"  # Pencere adı
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Yeniden boyutlanabilir pencere oluştur

    max_kernel_slider = 15  # Kernel kaydırıcısı için üst değer (2*n+1 ile 31'e çıkar)
    max_sigma_slider = 50   # Sigma kaydırıcısı için üst değer (0.0 - 5.0 aralığı)

    cv2.createTrackbar("Kernel (odd)", window_name, 2, max_kernel_slider, lambda _=None: None)  # Kernel trackbar'ı
    cv2.createTrackbar("Sigma x10", window_name, 10, max_sigma_slider, lambda _=None: None)  # Sigma trackbar'ı

    while True:  # Sürekli döngü: trackbar değerlerini oku ve görüntüyü güncelle
        kernel_slider = cv2.getTrackbarPos("Kernel (odd)", window_name)  # Kernel slider değeri
        sigma_slider = cv2.getTrackbarPos("Sigma x10", window_name)  # Sigma slider değeri

        kernel_size = max(1, kernel_slider) * 2 + 1  # En az 3 olacak şekilde tek sayı kernel hesabı
        sigma = sigma_slider / 10.0  # Sigma'yı slider'dan yüzen sayıya çevir

        blurred = cv2.GaussianBlur(original, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)  # Gaussian blur uygula
        stacked = np.hstack((original, blurred))  # Orijinal ve blur görüntüyü yanyana birleştir

        cv2.imshow(window_name, stacked)  # Pencereye göster

        key = cv2.waitKey(30) & 0xFF  # 30 ms bekleyip tuş oku
        if key in (27, ord("q")):  # ESC veya q'ya basılırsa çık
            break  # Döngüden çık

    cv2.destroyAllWindows()  # Açık tüm pencereleri kapat


if __name__ == "__main__":  # Dosya doğrudan çalıştırıldığında main'i çağır
    main()
