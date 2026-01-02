"""
3D Renk Uzayı Görselleştirme Aracı

Bir görüntünün RGB renk dağılımını 3 boyutlu uzayda görselleştirir.
Her piksel, 3D uzayda (Red, Green, Blue) koordinatlarında bir nokta olarak çizilir.
Noktanın rengi, pikselin kendi rengini temsil eder.

Kullanım:
    python color_distribution_3d.py resim.jpg --sample 5000
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

def visualize_3d_colors(image_path, num_samples=5000, output_path=None):
    # Görüntüyü oku
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: Görüntü okunamadı: {image_path}")
        return

    # BGR'den RGB'ye çevir (Matplotlib için)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resmi düzleştir (Her satır bir piksel olacak şekilde)
    # Shape: (H*W, 3)
    pixels = img_rgb.reshape(-1, 3)
    
    # Performans için örneklem al (Tüm pikselleri çizmek çok yavaş olur)
    if num_samples < len(pixels):
        indices = np.random.choice(len(pixels), num_samples, replace=False)
        sample_pixels = pixels[indices]
    else:
        sample_pixels = pixels

    # Renkleri 0-1 aralığına normalize et (Matplotlib için)
    colors = sample_pixels / 255.0

    # 3D Grafik oluştur
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    r = sample_pixels[:, 0]
    g = sample_pixels[:, 1]
    b = sample_pixels[:, 2]

    ax.scatter(r, g, b, c=colors, marker='o', s=5, alpha=0.5)

    ax.set_xlabel('R (Kırmızı)')
    ax.set_ylabel('G (Yeşil)')
    ax.set_zlabel('B (Mavi)')
    ax.set_title(f'3D Renk Dağılımı ({os.path.basename(image_path)})')

    # Kaydet veya göster
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Grafik kaydedildi: {output_path}")
    else:
        plt.show()

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D RGB Renk Dağılımı Görselleştirici")
    parser.add_argument("image", help="Analiz edilecek görüntü")
    parser.add_argument("--sample", type=int, default=5000, help="Örnek alınacak piksel sayısı")
    parser.add_argument("--output", help="Grafiği kaydet (örn: grafik.png)")

    args = parser.parse_args()
    visualize_3d_colors(args.image, args.sample, args.output)
