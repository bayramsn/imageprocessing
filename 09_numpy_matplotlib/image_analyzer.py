"""
NumPy & Matplotlib ile Görüntü ve Matris Analiz Aracı

Bu proje ile öğrenecekleriniz:
- NumPy slicing, reshape işlemleri
- Matris mantığı = Görüntü mantığı
- Matplotlib ile subplot & grafik çizimi
- Histogram analizi ve threshold uygulaması
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def load_image_as_array(path: str) -> np.ndarray:
    """Görüntüyü NumPy array olarak yükle"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Görüntü bulunamadı: {path}")
    
    # OpenCV ile oku (BGR formatında)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    # Türkçe karakterli yollarda sorun olursa
    if img is None:
        with open(path, "rb") as f:
            data = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
    
    # BGR -> RGB dönüşümü (matplotlib için)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def analyze_pixels(img: np.ndarray) -> dict:
    """Piksel değerlerini analiz et"""
    
    # Temel bilgiler
    info = {
        "shape": img.shape,                    # (yükseklik, genişlik, kanal)
        "dtype": str(img.dtype),               # veri tipi (uint8 genelde)
        "size": img.size,                      # toplam piksel sayısı
        "ndim": img.ndim,                      # boyut sayısı
    }
    
    # Gri seviye için dönüştür
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # İstatistikler
    info["min"] = int(np.min(gray))
    info["max"] = int(np.max(gray))
    info["mean"] = float(np.mean(gray))
    info["std"] = float(np.std(gray))
    info["median"] = float(np.median(gray))
    
    # Kanal bazlı analiz (RGB)
    if img.ndim == 3:
        channels = ["Red", "Green", "Blue"]
        for i, ch in enumerate(channels):
            info[f"{ch}_mean"] = float(np.mean(img[:, :, i]))
            info[f"{ch}_std"] = float(np.std(img[:, :, i]))
    
    return info


def apply_threshold(gray: np.ndarray, thresh_value: int = 127) -> np.ndarray:
    """Basit threshold uygula - binary görüntü oluştur"""
    # Piksel > thresh_value ise 255, değilse 0
    _, binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    return binary


def apply_adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Adaptif threshold - farklı aydınlatma koşullarında daha iyi"""
    return cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 
        11, 2  # blok boyutu ve sabit
    )


def plot_analysis(img_rgb: np.ndarray, save_path: str = None):
    """Görüntü analizini görselleştir"""
    
    # Gri seviye dönüşümü
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Threshold uygulamaları
    binary = apply_threshold(gray, thresh_value=127)
    adaptive = apply_adaptive_threshold(gray)
    
    # Figure oluştur - 3x3 subplot
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle("NumPy & Matplotlib ile Görüntü Analizi", fontsize=14, fontweight='bold')
    
    # --- Satır 1: Orijinal görüntüler ---
    
    # 1. Orijinal RGB
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Orijinal (RGB)")
    axes[0, 0].axis('off')
    
    # 2. Gri Seviye
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title("Gri Seviye")
    axes[0, 1].axis('off')
    
    # 3. Piksel Yoğunluk Haritası (heatmap)
    axes[0, 2].imshow(gray, cmap='hot')
    axes[0, 2].set_title("Yoğunluk Haritası (Hot)")
    axes[0, 2].axis('off')
    
    # --- Satır 2: Histogramlar ---
    
    # 4. Gri Histogram
    axes[1, 0].hist(gray.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    axes[1, 0].set_title("Gri Seviye Histogramı")
    axes[1, 0].set_xlabel("Piksel Değeri (0-255)")
    axes[1, 0].set_ylabel("Frekans")
    axes[1, 0].axvline(x=np.mean(gray), color='red', linestyle='--', label=f'Ortalama: {np.mean(gray):.1f}')
    axes[1, 0].legend()
    
    # 5. RGB Histogram
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        axes[1, 1].hist(img_rgb[:, :, i].ravel(), bins=256, range=(0, 256), 
                        color=color, alpha=0.5, label=color.upper())
    axes[1, 1].set_title("RGB Kanalları Histogramı")
    axes[1, 1].set_xlabel("Piksel Değeri")
    axes[1, 1].set_ylabel("Frekans")
    axes[1, 1].legend()
    
    # 6. Kümülatif Histogram
    hist, bins = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    cumulative = np.cumsum(hist)
    axes[1, 2].plot(cumulative, color='blue')
    axes[1, 2].set_title("Kümülatif Histogram")
    axes[1, 2].set_xlabel("Piksel Değeri")
    axes[1, 2].set_ylabel("Kümülatif Frekans")
    axes[1, 2].fill_between(range(256), cumulative, alpha=0.3)
    
    # --- Satır 3: Threshold sonuçları ---
    
    # 7. Binary Threshold
    axes[2, 0].imshow(binary, cmap='gray')
    axes[2, 0].set_title("Binary Threshold (127)")
    axes[2, 0].axis('off')
    
    # 8. Adaptif Threshold
    axes[2, 1].imshow(adaptive, cmap='gray')
    axes[2, 1].set_title("Adaptif Threshold")
    axes[2, 1].axis('off')
    
    # 9. İstatistik Kutusu
    info = analyze_pixels(img_rgb)
    stats_text = f"""
    Boyut: {info['shape']}
    Veri Tipi: {info['dtype']}
    
    Min: {info['min']}
    Max: {info['max']}
    Ortalama: {info['mean']:.2f}
    Std: {info['std']:.2f}
    Medyan: {info['median']:.1f}
    
    R Ortalama: {info['Red_mean']:.1f}
    G Ortalama: {info['Green_mean']:.1f}
    B Ortalama: {info['Blue_mean']:.1f}
    """
    axes[2, 2].text(0.1, 0.5, stats_text, transform=axes[2, 2].transAxes, 
                    fontsize=11, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[2, 2].set_title("İstatistikler")
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Analiz kaydedildi: {save_path}")
    
    plt.show()


def demonstrate_numpy_operations(img: np.ndarray):
    """NumPy işlemlerini göster - eğitim amaçlı"""
    
    print("\n" + "=" * 50)
    print("NumPy Matris İşlemleri Demonstrasyonu")
    print("=" * 50)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 1. Temel bilgiler
    print(f"\n📐 Matris Şekli (shape): {img.shape}")
    print(f"   -> {img.shape[0]} satır (yükseklik)")
    print(f"   -> {img.shape[1]} sütun (genişlik)")
    print(f"   -> {img.shape[2]} kanal (RGB)")
    
    # 2. Slicing örnekleri
    print(f"\n🔪 NumPy Slicing Örnekleri:")
    print(f"   img[0, 0]      -> İlk piksel (RGB): {img[0, 0]}")
    print(f"   img[0, 0, 0]   -> İlk piksel Red değeri: {img[0, 0, 0]}")
    print(f"   img[:, :, 0]   -> Sadece Red kanalı, shape: {img[:, :, 0].shape}")
    print(f"   img[100:200, 150:250] -> 100x100 bölge, shape: {img[100:200, 150:250].shape}")
    
    # 3. Reshape örneği
    h, w = gray.shape
    flat = gray.reshape(-1)  # 1D array'e çevir
    print(f"\n🔄 Reshape İşlemi:")
    print(f"   Orijinal: {gray.shape} -> Düzleştirilmiş: {flat.shape}")
    print(f"   Toplam piksel: {flat.shape[0]} = {h} x {w}")
    
    # 4. Boolean indexing
    bright_pixels = gray > 200
    dark_pixels = gray < 50
    print(f"\n🎯 Boolean Indexing:")
    print(f"   Parlak piksel sayısı (>200): {np.sum(bright_pixels)}")
    print(f"   Koyu piksel sayısı (<50): {np.sum(dark_pixels)}")
    print(f"   Parlak/Toplam oranı: {np.sum(bright_pixels) / gray.size * 100:.1f}%")
    
    # 5. Matematiksel işlemler
    print(f"\n📊 Matematiksel İşlemler:")
    print(f"   np.min(gray)    = {np.min(gray)}")
    print(f"   np.max(gray)    = {np.max(gray)}")
    print(f"   np.mean(gray)   = {np.mean(gray):.2f}")
    print(f"   np.std(gray)    = {np.std(gray):.2f}")
    print(f"   np.median(gray) = {np.median(gray)}")
    
    # 6. Negatif görüntü
    negative = 255 - gray
    print(f"\n🔄 Negatif Görüntü:")
    print(f"   negative = 255 - gray")
    print(f"   Yeni ortalama: {np.mean(negative):.2f} (önceki: {np.mean(gray):.2f})")


def main():
    parser = argparse.ArgumentParser(
        description="NumPy & Matplotlib ile Görüntü Analiz Aracı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek Kullanım:
  python image_analyzer.py resim.jpg
  python image_analyzer.py resim.jpg --save analiz.png
  python image_analyzer.py resim.jpg --demo
        """
    )
    parser.add_argument("image", help="Analiz edilecek görüntü yolu")
    parser.add_argument("--save", help="Analiz grafiğini kaydet (örn: analiz.png)")
    parser.add_argument("--demo", action="store_true", 
                        help="NumPy işlemlerini konsola yazdır")
    parser.add_argument("--no-plot", action="store_true", 
                        help="Grafik gösterme, sadece istatistikleri yazdır")
    args = parser.parse_args()
    
    # Görüntüyü yükle
    print(f"Görüntü yükleniyor: {args.image}")
    img_rgb = load_image_as_array(args.image)
    
    # Temel analiz
    info = analyze_pixels(img_rgb)
    
    print("\n" + "=" * 50)
    print("GÖRÜNTÜ ANALİZİ")
    print("=" * 50)
    print(f"Boyut: {info['shape'][1]}x{info['shape'][0]} piksel")
    print(f"Kanal sayısı: {info['shape'][2]}")
    print(f"Veri tipi: {info['dtype']}")
    print(f"\nGri Seviye İstatistikleri:")
    print(f"  Min: {info['min']}, Max: {info['max']}")
    print(f"  Ortalama: {info['mean']:.2f}")
    print(f"  Standart Sapma: {info['std']:.2f}")
    print(f"  Medyan: {info['median']:.1f}")
    
    # NumPy demo
    if args.demo:
        demonstrate_numpy_operations(img_rgb)
    
    # Görselleştirme
    if not args.no_plot:
        plot_analysis(img_rgb, save_path=args.save)


if __name__ == "__main__":
    main()
