"""
Blur Karşılaştırma Aracı

Aynı görüntüye farklı blur türleri uygulayıp karşılaştır.
Kernel size ve sigma değiştikçe farkı gör.
"""
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_image(path):
    """Görüntüyü yükle"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Görüntü bulunamadı: {path}")
    
    img = cv2.imread(path)
    if img is None:
        with open(path, "rb") as f:
            data = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def add_noise(img, noise_type='gaussian'):
    """Görüntüye gürültü ekle - test için"""
    noisy = img.copy().astype(np.float32)
    
    if noise_type == 'gaussian':
        # Gaussian gürültü
        noise = np.random.normal(0, 25, img.shape)
        noisy = noisy + noise
    elif noise_type == 'salt_pepper':
        # Tuz-biber gürültüsü
        prob = 0.02
        salt = np.random.random(img.shape[:2]) < prob
        pepper = np.random.random(img.shape[:2]) < prob
        noisy[salt] = 255
        noisy[pepper] = 0
    
    return np.clip(noisy, 0, 255).astype(np.uint8)


def compare_blur_types(img, kernel_size=7):
    """Farklı blur türlerini karşılaştır"""
    
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    results = {
        'Orijinal': img,
        f'Gaussian Blur ({k}x{k})': cv2.GaussianBlur(img, (k, k), 0),
        f'Median Blur ({k})': cv2.medianBlur(img, k),
        f'Bilateral Filter': cv2.bilateralFilter(img, 9, 75, 75),
        f'Box Filter ({k}x{k})': cv2.blur(img, (k, k)),
    }
    
    return results


def compare_kernel_sizes(img, sizes=[3, 7, 15, 31]):
    """Farklı kernel boyutlarını karşılaştır"""
    
    results = {'Orijinal': img}
    for k in sizes:
        k = k if k % 2 == 1 else k + 1
        results[f'Kernel {k}x{k}'] = cv2.GaussianBlur(img, (k, k), 0)
    
    return results


def compare_sigma_values(img, sigmas=[0.5, 1, 2, 5]):
    """Farklı sigma değerlerini karşılaştır"""
    
    results = {'Orijinal': img}
    for sigma in sigmas:
        # Kernel boyutunu sigma'ya göre hesapla (6*sigma + 1)
        k = int(6 * sigma + 1)
        k = k if k % 2 == 1 else k + 1
        results[f'Sigma={sigma}'] = cv2.GaussianBlur(img, (k, k), sigma)
    
    return results


def plot_comparison(results, title="Blur Karşılaştırması", save_path=None):
    """Sonuçları yan yana göster"""
    
    n = len(results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for ax, (name, img) in zip(axes, results.items()):
        ax.imshow(img)
        ax.set_title(name)
        ax.axis('off')
    
    # Boş subplot'ları gizle
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Kaydedildi: {save_path}")
    
    plt.show()


def interactive_blur(img):
    """Trackbar ile interaktif blur"""
    
    cv2.namedWindow('Blur Comparison', cv2.WINDOW_NORMAL)
    
    # Trackbar'lar
    cv2.createTrackbar('Kernel', 'Blur Comparison', 3, 31, lambda x: None)
    cv2.createTrackbar('Type', 'Blur Comparison', 0, 3, lambda x: None)
    
    blur_types = ['Gaussian', 'Median', 'Bilateral', 'Box']
    
    print("ESC tuşu ile çıkış")
    
    while True:
        k = cv2.getTrackbarPos('Kernel', 'Blur Comparison')
        blur_type = cv2.getTrackbarPos('Type', 'Blur Comparison')
        
        # Kernel tek sayı olmalı
        k = max(3, k)
        k = k if k % 2 == 1 else k + 1
        
        # Blur uygula
        if blur_type == 0:  # Gaussian
            blurred = cv2.GaussianBlur(img, (k, k), 0)
        elif blur_type == 1:  # Median
            blurred = cv2.medianBlur(img, k)
        elif blur_type == 2:  # Bilateral
            blurred = cv2.bilateralFilter(img, k, 75, 75)
        else:  # Box
            blurred = cv2.blur(img, (k, k))
        
        # Bilgi ekle
        display = blurred.copy()
        cv2.putText(display, f"{blur_types[blur_type]} - Kernel: {k}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Blur Comparison', display)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Blur Karşılaştırma Aracı")
    parser.add_argument("image", help="Görüntü yolu")
    parser.add_argument("--kernel", type=int, default=7, help="Kernel boyutu")
    parser.add_argument("--interactive", action="store_true", help="İnteraktif mod")
    parser.add_argument("--noise", choices=['gaussian', 'salt_pepper'], 
                        help="Test için gürültü ekle")
    parser.add_argument("--save", help="Sonucu kaydet")
    parser.add_argument("--mode", choices=['types', 'kernels', 'sigma'], 
                        default='types', help="Karşılaştırma modu")
    args = parser.parse_args()
    
    print(f"Görüntü yükleniyor: {args.image}")
    img = load_image(args.image)
    
    # Gürültü ekle (isteğe bağlı)
    if args.noise:
        print(f"Gürültü ekleniyor: {args.noise}")
        img = add_noise(img, args.noise)
    
    if args.interactive:
        # BGR'ye çevir (OpenCV için)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        interactive_blur(img_bgr)
    else:
        if args.mode == 'types':
            results = compare_blur_types(img, args.kernel)
            title = "Blur Türleri Karşılaştırması"
        elif args.mode == 'kernels':
            results = compare_kernel_sizes(img)
            title = "Kernel Boyutu Karşılaştırması"
        else:
            results = compare_sigma_values(img)
            title = "Sigma Değeri Karşılaştırması"
        
        plot_comparison(results, title, args.save)


if __name__ == "__main__":
    main()
