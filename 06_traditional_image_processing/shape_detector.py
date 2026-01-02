"""
Geometrik Şekil Tespiti ve Sınıflandırma Uygulaması

Bu betik, görüntüdeki temel geometrik şekilleri (üçgen, kare, dikdörtgen, daire, beşgen) tespit eder.
OpenCV'nin kontur bulma ve çokgen yaklaştırma (polygon approximation) yöntemlerini kullanır.

Kullanım:
    python shape_detector.py resim.jpg --output sonuc.png
"""

import cv2
import numpy as np
import argparse
import os

def detect_shapes(image_path, output_path=None):
    # Görüntüyü oku
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: Görüntü okunamadı: {image_path}")
        return

    # Ön işleme: Gri tonlama -> Blur -> Threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Beyaz arka plandaki şekilleri bulmak için THRESH_BINARY_INV kullanıyoruz
    # Arka plan (255) -> Siyah (0), Şekiller (<240) -> Beyaz (255)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)

    # Konturları bul
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Toplam {len(contours)} kontur bulundu.")

    for contour in contours:
        # Alan çok küçükse gürültü olabilir, atla
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        # Şekli yaklaştır (approximate)
        epsilon = 0.04 * cv2.arcLength(contour, True) # Çevrenin %4'ü kadar tolerans
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Köşe sayısı bul
        vertices = len(approx)
        shape_name = "Bilinmeyen"

        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h

        if vertices == 3:
            shape_name = "Ucgen"
        elif vertices == 4:
            # Kare mi dikdörtgen mi?
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Kare"
            else:
                shape_name = "Dikdortgen"
        elif vertices == 5:
            shape_name = "Besgen"
        else:
            # 5'ten fazlaysa muhtemelen daire (veya çok kenarlı bir şekil)
            shape_name = "Daire"

        # Şeklin merkezini bul (yazıyı oraya yazmak için)
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
        else:
            cX, cY = x, y

        # Konturu çiz ve ismi yaz
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
        cv2.putText(img, shape_name, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print(f"Şekil: {shape_name}, Alan: {area:.2f}, Köşe: {vertices}")

    # Sonucu göster veya kaydet
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Sonuç kaydedildi: {output_path}")
    else:
        cv2.imshow("Shape Detector", img)
        print("Çıkmak için bir tuşa basın...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basit Geometrik Şekil Tespit Aracı")
    parser.add_argument("image", help="İşlenecek görüntü yolu")
    parser.add_argument("--output", help="Sonucun kaydedileceği dosya yolu")
    
    args = parser.parse_args()
    detect_shapes(args.image, args.output)
