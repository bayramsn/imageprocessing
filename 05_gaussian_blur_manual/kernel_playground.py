"""
Kernel Bahçesi (Custom Filter Playground)

Bu betik, manuel olarak tanımlanan konvolüsyon matrislerinin (kernel)
görüntü üzerindeki etkisini gösterir.
Keskinleştirme, Kabartma (Emboss), Kenar Bulma gibi filtreler içerir.

OpenCV'nin `filter2D` fonksiyonunu kullanır.

Kullanım:
    python kernel_playground.py resim.jpg --output sonuc_klasoru/
"""

import cv2
import numpy as np
import argparse
import os

def apply_kernels(image_path, output_dir=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: Görüntü okunamadı: {image_path}")
        return

    # Farklı kernel tanımları
    kernels = {
        "Identity (Etkisiz)": np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]),
        "Sharpen (Keskinlestir)": np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]),
        "Strong Sharpen": np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ]),
        "Edge Detect (Kenar)": np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]),
        "Emboss (Kabartma)": np.array([
            [-2, -1, 0],
            [-1,  1, 1],
            [ 0,  1, 2]
        ]),
        "Motion Blur (Hareket)": np.eye(9) / 9.0  # 9x9 diyagonal matris
    }

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Filtreler uygulanıyor...")
    
    # Ekranda göstermek için bir grid hazırla (opsiyonel)
    row1 = []
    row2 = []
    
    i = 0
    for name, kernel in kernels.items():
        # Kernel uygula (ddepth=-1: çıktı görüntüsü girdiyle aynı derinlikte olsun)
        filtered = cv2.filter2D(img, -1, kernel)
        
        print(f"Uygulandı: {name}")
        
        # Dosyaya kaydet
        if output_dir:
            file_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".png"
            path = os.path.join(output_dir, file_name)
            cv2.imwrite(path, filtered)
        else:
             # Görselleştirme için başlık ekle
            display = filtered.copy()
            cv2.putText(display, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)
            cv2.putText(display, "Kernel:", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Kernel'i string olarak yazdır (sadece ilk satırı örnek)
            k_str = str(kernel[0]) + "..."
            cv2.putText(display, k_str, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if i < 3:
                row1.append(display)
            else:
                row2.append(display)
        i += 1

    if output_dir:
        print(f"Tüm sonuçlar kaydedildi: {output_dir}")
    elif row1 and row2:
        # Sonuçları birleştirip göster
        # Görüntüleri yeniden boyutlandır (ekrana sığması için)
        h, w = img.shape[:2]
        new_w = 400
        new_h = int(h * (new_w / w))
        
        final_row1 = np.hstack([cv2.resize(im, (new_w, new_h)) for im in row1])
        final_row2 = np.hstack([cv2.resize(im, (new_w, new_h)) for im in row2])
        
        final_grid = np.vstack([final_row1, final_row2])
        
        cv2.imshow("Kernel Playground", final_grid)
        print("Kapatmak için bir tuşa basın...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manuel Kernel Filtre Deneyimi")
    parser.add_argument("image", help="İşlenecek görüntü")
    parser.add_argument("--output", help="Sonuçların kaydedileceği klasör")
    
    args = parser.parse_args()
    apply_kernels(args.image, args.output)
