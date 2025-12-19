"""
Custom Gaussian Filter - Kendin Yaz

Gaussian Blur'u sıfırdan implemente et.
OpenCV ile karşılaştır, hız farkını gör.

Bu kodu anlayan CNN'i de anlar!
"""
import argparse
import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import time
import os


def load_image(path):
    """Görüntüyü yükle"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Görüntü bulunamadı: {path}")
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        with open(path, "rb") as f:
            data = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    
    return img


def create_gaussian_kernel(size, sigma):
    """
    Gaussian kernel oluştur
    
    Formül: G(x,y) = (1 / 2πσ²) * exp(-(x² + y²) / 2σ²)
    """
    # Kernel tek sayı olmalı
    size = size if size % 2 == 1 else size + 1
    
    # Merkez noktası
    center = size // 2
    
    # Kernel matrisi oluştur
    kernel = np.zeros((size, size), dtype=np.float64)
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            # Gaussian formülü
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize et (toplam = 1)
    kernel = kernel / kernel.sum()
    
    return kernel


def create_gaussian_kernel_fast(size, sigma):
    """
    Gaussian kernel - Vektörize versiyon (daha hızlı)
    
    1D Gaussian'ların outer product'ı = 2D Gaussian
    """
    size = size if size % 2 == 1 else size + 1
    
    # 1D koordinatlar
    x = np.arange(size) - size // 2
    
    # 1D Gaussian
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    
    # 2D Gaussian = outer product
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    return kernel_2d / kernel_2d.sum()


def convolve2d_manual(image, kernel):
    """
    Elle convolution - Eğitim amaçlı, yavaş!
    
    Her piksel için:
    1. Kernel'i üzerine koy
    2. Elemanları çarp
    3. Topla
    """
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # Kenarlar için padding ekle (yansıma modu)
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    
    # Sonuç matrisi
    output = np.zeros_like(image, dtype=np.float64)
    
    # Her piksel için convolution
    for i in range(h):
        for j in range(w):
            # Pikselin etrafındaki bölgeyi al
            region = padded[i:i+kh, j:j+kw]
            # Kernel ile çarp ve topla
            output[i, j] = np.sum(region * kernel)
    
    return output.astype(np.uint8)


def convolve2d_vectorized(image, kernel):
    """
    Vektörize convolution - Daha hızlı ama bellek yoğun
    """
    from numpy.lib.stride_tricks import sliding_window_view
    
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    
    # Sliding window ile tüm bölgeleri al
    windows = sliding_window_view(padded, (kh, kw))
    
    # Tüm bölgeleri kernel ile çarp ve topla
    output = np.sum(windows * kernel, axis=(2, 3))
    
    return output.astype(np.uint8)


def gaussian_blur_separable(image, kernel_size, sigma):
    """
    Ayrılabilir Gaussian - En verimli elle yazım
    
    2D Gaussian = 1D yatay * 1D dikey
    Hesaplama: O(n²) → O(2n)
    """
    size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # 1D kernel
    x = np.arange(size) - size // 2
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Önce yatay, sonra dikey
    temp = ndimage.convolve1d(image.astype(np.float64), kernel_1d, axis=1, mode='reflect')
    output = ndimage.convolve1d(temp, kernel_1d, axis=0, mode='reflect')
    
    return output.astype(np.uint8)


def benchmark(image, kernel_size=7, sigma=1.5):
    """Farklı yöntemlerin hızını karşılaştır"""
    
    # Küçük görüntü ile test (elle convolution yavaş)
    test_size = 256
    if image.shape[0] > test_size or image.shape[1] > test_size:
        test_img = cv2.resize(image, (test_size, test_size))
    else:
        test_img = image
    
    kernel = create_gaussian_kernel_fast(kernel_size, sigma)
    
    results = {}
    
    print(f"\nBenchmark: {test_img.shape[0]}x{test_img.shape[1]} görüntü, {kernel_size}x{kernel_size} kernel")
    print("=" * 60)
    
    # 1. Elle convolution (çok yavaş)
    print("Elle convolution...", end=" ", flush=True)
    start = time.time()
    result1 = convolve2d_manual(test_img, kernel)
    t1 = time.time() - start
    results['Elle (loop)'] = (t1, result1)
    print(f"{t1:.3f} saniye")
    
    # 2. Vektörize
    print("Vektörize...", end=" ", flush=True)
    start = time.time()
    result2 = convolve2d_vectorized(test_img, kernel)
    t2 = time.time() - start
    results['Vektörize'] = (t2, result2)
    print(f"{t2:.3f} saniye")
    
    # 3. Ayrılabilir (separable)
    print("Ayrılabilir...", end=" ", flush=True)
    start = time.time()
    result3 = gaussian_blur_separable(test_img, kernel_size, sigma)
    t3 = time.time() - start
    results['Ayrılabilir'] = (t3, result3)
    print(f"{t3:.3f} saniye")
    
    # 4. SciPy
    print("SciPy...", end=" ", flush=True)
    start = time.time()
    result4 = ndimage.gaussian_filter(test_img, sigma)
    t4 = time.time() - start
    results['SciPy'] = (t4, result4)
    print(f"{t4:.3f} saniye")
    
    # 5. OpenCV
    print("OpenCV...", end=" ", flush=True)
    start = time.time()
    result5 = cv2.GaussianBlur(test_img, (kernel_size, kernel_size), sigma)
    t5 = time.time() - start
    results['OpenCV'] = (t5, result5)
    print(f"{t5:.3f} saniye")
    
    print("=" * 60)
    print(f"OpenCV vs Elle: {t1/t5:.0f}x daha hızlı!")
    
    return results


def visualize_kernel(kernel):
    """Kernel'i 3D olarak göster"""
    
    fig = plt.figure(figsize=(10, 4))
    
    # 2D görünüm
    ax1 = fig.add_subplot(121)
    im = ax1.imshow(kernel, cmap='hot', interpolation='nearest')
    ax1.set_title(f'Gaussian Kernel {kernel.shape[0]}x{kernel.shape[1]}')
    plt.colorbar(im, ax=ax1)
    
    # 3D görünüm
    ax2 = fig.add_subplot(122, projection='3d')
    x = np.arange(kernel.shape[0])
    y = np.arange(kernel.shape[1])
    X, Y = np.meshgrid(x, y)
    ax2.plot_surface(X, Y, kernel, cmap='viridis', edgecolor='none')
    ax2.set_title('3D Görünüm (Çan Eğrisi)')
    
    plt.tight_layout()
    plt.show()


def compare_results(image, kernel_size=7, sigma=1.5, save_path=None):
    """Elle yazılan ile OpenCV sonuçlarını karşılaştır"""
    
    kernel = create_gaussian_kernel_fast(kernel_size, sigma)
    
    # Elle blur
    manual_result = gaussian_blur_separable(image, kernel_size, sigma)
    
    # OpenCV blur
    opencv_result = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Fark
    diff = cv2.absdiff(manual_result, opencv_result)
    
    # Görselleştir
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'Custom vs OpenCV Gaussian Blur (kernel={kernel_size}, sigma={sigma})', 
                 fontsize=12, fontweight='bold')
    
    # Orijinal
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Orijinal')
    axes[0, 0].axis('off')
    
    # Elle yazılan
    axes[0, 1].imshow(manual_result, cmap='gray')
    axes[0, 1].set_title('Custom (Elle Yazılan)')
    axes[0, 1].axis('off')
    
    # OpenCV
    axes[0, 2].imshow(opencv_result, cmap='gray')
    axes[0, 2].set_title('OpenCV')
    axes[0, 2].axis('off')
    
    # Kernel
    axes[1, 0].imshow(kernel, cmap='hot')
    axes[1, 0].set_title(f'Gaussian Kernel ({kernel_size}x{kernel_size})')
    
    # Fark
    axes[1, 1].imshow(diff, cmap='hot')
    axes[1, 1].set_title(f'Fark (Max: {diff.max()}, Mean: {diff.mean():.2f})')
    
    # Histogram of difference
    axes[1, 2].hist(diff.ravel(), bins=50, color='red', alpha=0.7)
    axes[1, 2].set_title('Fark Histogramı')
    axes[1, 2].set_xlabel('Piksel Farkı')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Kaydedildi: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Custom Gaussian Filter")
    parser.add_argument("image", help="Görüntü yolu")
    parser.add_argument("--kernel", type=int, default=7, help="Kernel boyutu")
    parser.add_argument("--sigma", type=float, default=1.5, help="Sigma değeri")
    parser.add_argument("--benchmark", action="store_true", help="Hız testi yap")
    parser.add_argument("--show-kernel", action="store_true", help="Kernel'i görselleştir")
    parser.add_argument("--save", help="Sonucu kaydet")
    args = parser.parse_args()
    
    print("Görüntü yükleniyor...")
    image = load_image(args.image)
    print(f"Boyut: {image.shape}")
    
    if args.show_kernel:
        kernel = create_gaussian_kernel_fast(args.kernel, args.sigma)
        print(f"\nGaussian Kernel ({args.kernel}x{args.kernel}, sigma={args.sigma}):")
        print(kernel.round(4))
        visualize_kernel(kernel)
    
    if args.benchmark:
        benchmark(image, args.kernel, args.sigma)
    else:
        compare_results(image, args.kernel, args.sigma, args.save)


if __name__ == "__main__":
    main()
