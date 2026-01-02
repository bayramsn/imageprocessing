"""
Tilt-Shift Efekti (Minyatür Dünyalar)

Bu betik, bir fotoğrafa "Tilt-Shift" efekti uygulayarak sahnenin minyatür bir
maket gibi görünmesini sağlar.
Bunu, görüntünün üst ve alt kısımlarını bulanıklaştırıp, orta kısmı odaklı (net) bırakarak yapar.

Kullanım:
    python tilt_shift_effect.py sehir.jpg --output minyatur.png
"""

import cv2
import numpy as np
import argparse

def create_tilt_shift(image_path, output_path=None, focus_height=0.3, blur_strength=15):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: Görüntü okunamadı: {image_path}")
        return

    h, w = img.shape[:2]
    
    # Bulanık görüntü oluştur
    # Gaussian blur için kernel boyutu tek sayı olmalı
    k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
    blurred = cv2.GaussianBlur(img, (k, k), 0)

    # Maske oluştur (Linear Gradient)
    # 0 = tamamen bulanık, 1 = tamamen net
    mask = np.zeros((h, w), dtype=np.float32)
    
    # Odak bölgesi (Görüntünün ortasında dikey olarak)
    # y1: Odak başlangıcı, y2: Odak bitişi
    center_y = h // 2
    span = int(h * focus_height) // 2
    y1, y2 = center_y - span, center_y + span
    
    # Maskeyi doldur
    # Üst kısım geçişi (0 -> 1)
    for y in range(y1):
        mask[y, :] = y / y1  # Kademeli artış
        
    # Orta kısım (1)
    mask[y1:y2, :] = 1.0
    
    # Alt kısım geçişi (1 -> 0)
    for y in range(y2, h):
        mask[y, :] = 1.0 - ((y - y2) / (h - y2))

    # Maskeyi 3 kanala çıkar (BGR ile işlem yapabilmek için)
    mask_3ch = cv2.merge([mask, mask, mask])

    # Görüntüleri birleştir
    # Sonuç = Net * Maske + Bulanık * (1 - Maske)
    img_float = img.astype(np.float32)
    blurred_float = blurred.astype(np.float32)
    
    result = img_float * mask_3ch + blurred_float * (1.0 - mask_3ch)
    result = result.astype(np.uint8)

    # Doygunluğu (Saturation) artır - Minyatür etkisi için önemlidir
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.4 # Saturation artır
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Tilt-Shift sonucu kaydedildi: {output_path}")
    else:
        cv2.imshow("Original", img)
        cv2.imshow("Tilt-Shift Effect", result)
        print("Kapatmak için bir tuşa basın...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tilt-Shift Efekti Oluşturucu")
    parser.add_argument("image", help="İşlenecek görüntü")
    parser.add_argument("--output", help="Çıktı dosyası")
    parser.add_argument("--focus", type=float, default=0.3, help="Odak alanı yüksekliği (0.0 - 1.0 arası)")
    
    args = parser.parse_args()
    create_tilt_shift(args.image, args.output, args.focus)
