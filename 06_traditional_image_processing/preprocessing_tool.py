"""
Geleneksel Görüntü İşleme - Preprocessing Tool

Threshold, Canny Edge, Morphological işlemler.
Belge/plaka tanıma ön işleme sistemi.
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
    
    return img


def threshold_comparison(gray):
    """Farklı threshold yöntemlerini karşılaştır"""
    
    results = {
        'Orijinal (Gri)': gray,
        'Binary (127)': cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
        'Binary Inv': cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1],
        'Otsu': cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        'Adaptive Mean': cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 11, 2),
        'Adaptive Gaussian': cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 2),
    }
    
    return results


def edge_detection_comparison(gray):
    """Farklı kenar tespit yöntemlerini karşılaştır"""
    
    # Blur uygula (gürültü azaltma)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Sobel X ve Y
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x = np.uint8(np.absolute(sobel_x))
    sobel_y = np.uint8(np.absolute(sobel_y))
    sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)
    
    # Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    results = {
        'Orijinal': gray,
        'Sobel X': sobel_x,
        'Sobel Y': sobel_y,
        'Sobel Combined': sobel_combined,
        'Laplacian': laplacian,
        'Canny (50-150)': cv2.Canny(blurred, 50, 150),
        'Canny (100-200)': cv2.Canny(blurred, 100, 200),
    }
    
    return results


def morphology_comparison(binary):
    """Morphological işlemleri karşılaştır"""
    
    kernel = np.ones((5, 5), np.uint8)
    
    results = {
        'Orijinal': binary,
        'Erosion': cv2.erode(binary, kernel, iterations=1),
        'Dilation': cv2.dilate(binary, kernel, iterations=1),
        'Opening': cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel),
        'Closing': cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel),
        'Gradient': cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel),
        'Top Hat': cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel),
        'Black Hat': cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel),
    }
    
    return results


def document_preprocessing(img):
    """Belge tarama için ön işleme pipeline"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    steps = {
        '1. Orijinal': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        '2. Gri Seviye': gray,
    }
    
    # Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    steps['3. Gaussian Blur'] = blurred
    
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    steps['4. Adaptive Threshold'] = binary
    
    # Morphological temizlik
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    steps['5. Morph Open'] = opened
    
    # Closing
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    steps['6. Morph Close'] = closed
    
    return steps


def plate_detection_preprocessing(img):
    """Plaka tanıma için ön işleme"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    steps = {
        '1. Orijinal': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        '2. Gri': gray,
    }
    
    # Bilateral filter (kenarları koruyarak blur)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    steps['3. Bilateral Filter'] = filtered
    
    # Canny
    edges = cv2.Canny(filtered, 30, 200)
    steps['4. Canny Edges'] = edges
    
    # Dilate (kenarları kalınlaştır)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    steps['5. Dilate'] = dilated
    
    # Konturları bul
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dikdörtgen konturları filtrele
    result = img.copy()
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
        
        if len(approx) == 4:
            cv2.drawContours(result, [approx], -1, (0, 255, 0), 3)
    
    steps['6. Plaka Adayları'] = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return steps


def plot_results(results, title="Sonuçlar", save_path=None):
    """Sonuçları göster"""
    
    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for ax, (name, img) in zip(axes, results.items()):
        if len(img.shape) == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(name, fontsize=10)
        ax.axis('off')
    
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Kaydedildi: {save_path}")
    
    plt.show()


def interactive_preprocessing(img):
    """Trackbar ile interaktif ön işleme"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cv2.namedWindow('Preprocessing', cv2.WINDOW_NORMAL)
    
    cv2.createTrackbar('Blur', 'Preprocessing', 1, 15, lambda x: None)
    cv2.createTrackbar('Thresh', 'Preprocessing', 127, 255, lambda x: None)
    cv2.createTrackbar('Canny Low', 'Preprocessing', 50, 200, lambda x: None)
    cv2.createTrackbar('Canny High', 'Preprocessing', 150, 300, lambda x: None)
    cv2.createTrackbar('Mode', 'Preprocessing', 0, 3, lambda x: None)
    
    modes = ['Threshold', 'Canny', 'Adaptive', 'Morphology']
    
    print("ESC ile çıkış")
    
    while True:
        blur = cv2.getTrackbarPos('Blur', 'Preprocessing')
        thresh = cv2.getTrackbarPos('Thresh', 'Preprocessing')
        canny_low = cv2.getTrackbarPos('Canny Low', 'Preprocessing')
        canny_high = cv2.getTrackbarPos('Canny High', 'Preprocessing')
        mode = cv2.getTrackbarPos('Mode', 'Preprocessing')
        
        blur = blur * 2 + 1  # Tek sayı yap
        
        # Blur
        blurred = cv2.GaussianBlur(gray, (blur, blur), 0)
        
        # İşlem
        if mode == 0:  # Threshold
            _, result = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
        elif mode == 1:  # Canny
            result = cv2.Canny(blurred, canny_low, canny_high)
        elif mode == 2:  # Adaptive
            result = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
        else:  # Morphology
            _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Bilgi ekle
        display = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        cv2.putText(display, f"Mode: {modes[mode]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Preprocessing', display)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Geleneksel Görüntü İşleme Aracı")
    parser.add_argument("image", help="Görüntü yolu")
    parser.add_argument("--mode", choices=['threshold', 'edge', 'morphology', 
                                           'document', 'plate', 'all'],
                        default='all', help="İşlem modu")
    parser.add_argument("--interactive", action="store_true", help="İnteraktif mod")
    parser.add_argument("--save", help="Sonucu kaydet")
    args = parser.parse_args()
    
    print(f"Görüntü yükleniyor: {args.image}")
    img = load_image(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if args.interactive:
        interactive_preprocessing(img)
        return
    
    if args.mode == 'threshold':
        results = threshold_comparison(gray)
        plot_results(results, "Threshold Karşılaştırması", args.save)
    
    elif args.mode == 'edge':
        results = edge_detection_comparison(gray)
        plot_results(results, "Kenar Tespiti Karşılaştırması", args.save)
    
    elif args.mode == 'morphology':
        # Önce binary yap
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        results = morphology_comparison(binary)
        plot_results(results, "Morphological İşlemler", args.save)
    
    elif args.mode == 'document':
        results = document_preprocessing(img)
        plot_results(results, "Belge Tarama Pipeline", args.save)
    
    elif args.mode == 'plate':
        results = plate_detection_preprocessing(img)
        plot_results(results, "Plaka Tanıma Pipeline", args.save)
    
    else:  # all
        print("\n1. Threshold Karşılaştırması")
        results = threshold_comparison(gray)
        plot_results(results, "Threshold Karşılaştırması")
        
        print("\n2. Kenar Tespiti Karşılaştırması")
        results = edge_detection_comparison(gray)
        plot_results(results, "Kenar Tespiti Karşılaştırması")
        
        print("\n3. Belge Tarama Pipeline")
        results = document_preprocessing(img)
        plot_results(results, "Belge Tarama Pipeline")


if __name__ == "__main__":
    main()
