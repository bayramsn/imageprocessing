"""
Görüntü Eşleştirme ve Takip Sistemi

SIFT, ORB, AKAZE ile keypoint bulma ve eşleştirme.
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


def detect_features(gray, method='orb'):
    """Keypoint ve descriptor çıkar"""
    
    if method == 'orb':
        detector = cv2.ORB_create(nfeatures=2000)
    elif method == 'sift':
        detector = cv2.SIFT_create()
    elif method == 'akaze':
        detector = cv2.AKAZE_create()
    elif method == 'brisk':
        detector = cv2.BRISK_create()
    else:
        raise ValueError(f"Bilinmeyen metod: {method}")
    
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    return keypoints, descriptors


def match_features(desc1, desc2, method='orb', ratio=0.75):
    """İki görüntünün descriptor'larını eşleştir"""
    
    # Matcher seç
    if method in ['sift']:
        # Float descriptor - L2 norm
        bf = cv2.BFMatcher(cv2.NORM_L2)
    else:
        # Binary descriptor - Hamming
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # KNN match
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < ratio * n.distance:
                good_matches.append(m)
    
    return good_matches


def draw_matches_custom(img1, kp1, img2, kp2, matches, max_matches=50):
    """Eşleşmeleri özel görselleştirme ile çiz"""
    
    # En iyi eşleşmeleri al
    matches = sorted(matches, key=lambda x: x.distance)[:max_matches]
    
    # Görüntüleri yan yana koy
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w = w1 + w2
    
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[:h1, :w1] = img1
    result[:h2, w1:w1+w2] = img2
    
    # Eşleşme çizgileri
    for match in matches:
        pt1 = tuple(map(int, kp1[match.queryIdx].pt))
        pt2 = tuple(map(int, kp2[match.trainIdx].pt))
        pt2 = (pt2[0] + w1, pt2[1])
        
        # Rastgele renk
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        
        cv2.circle(result, pt1, 5, color, -1)
        cv2.circle(result, pt2, 5, color, -1)
        cv2.line(result, pt1, pt2, color, 1)
    
    return result


def find_homography(kp1, kp2, matches, min_matches=4):
    """Homography matrisi bul"""
    
    if len(matches) < min_matches:
        return None, None
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H, mask


def compare_methods(img1, img2):
    """Farklı feature detection yöntemlerini karşılaştır"""
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    methods = ['orb', 'sift', 'akaze', 'brisk']
    results = {}
    
    for method in methods:
        try:
            kp1, desc1 = detect_features(gray1, method)
            kp2, desc2 = detect_features(gray2, method)
            
            if desc1 is None or desc2 is None:
                results[method] = {
                    'keypoints1': 0,
                    'keypoints2': 0,
                    'matches': 0,
                    'time': 0
                }
                continue
            
            import time
            start = time.time()
            matches = match_features(desc1, desc2, method)
            elapsed = time.time() - start
            
            results[method] = {
                'keypoints1': len(kp1),
                'keypoints2': len(kp2),
                'matches': len(matches),
                'time': elapsed,
                'kp1': kp1,
                'kp2': kp2,
                'good_matches': matches
            }
        except Exception as e:
            print(f"{method} hatası: {e}")
            results[method] = {'keypoints1': 0, 'keypoints2': 0, 'matches': 0, 'time': 0}
    
    return results


def plot_comparison(img1, img2, results, save_path=None):
    """Karşılaştırma sonuçlarını göster"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Feature Detection Karşılaştırması', fontsize=14, fontweight='bold')
    
    methods = list(results.keys())
    
    for ax, method in zip(axes.flatten(), methods):
        result = results[method]
        
        if 'kp1' in result and result['matches'] > 0:
            # Eşleşmeleri çiz
            match_img = draw_matches_custom(
                cv2.cvtColor(img1, cv2.COLOR_BGR2RGB),
                result['kp1'],
                cv2.cvtColor(img2, cv2.COLOR_BGR2RGB),
                result['kp2'],
                result['good_matches'],
                max_matches=30
            )
            ax.imshow(match_img)
        else:
            # Sadece görüntüleri göster
            combined = np.hstack([
                cv2.cvtColor(img1, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            ])
            ax.imshow(combined)
        
        ax.set_title(f"{method.upper()}\n"
                     f"KP: {result['keypoints1']}/{result['keypoints2']} | "
                     f"Matches: {result['matches']} | "
                     f"Time: {result['time']*1000:.1f}ms")
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Kaydedildi: {save_path}")
    
    plt.show()


def visualize_keypoints(img, method='orb'):
    """Tek görüntüdeki keypoint'leri görselleştir"""
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, desc = detect_features(gray, method)
    
    # Keypoint'leri çiz
    img_kp = cv2.drawKeypoints(img, kp, None, 
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return img_kp, len(kp)


def main():
    parser = argparse.ArgumentParser(description="Görüntü Eşleştirme Sistemi")
    parser.add_argument("image1", help="Birinci görüntü")
    parser.add_argument("image2", nargs='?', help="İkinci görüntü (opsiyonel)")
    parser.add_argument("--method", choices=['orb', 'sift', 'akaze', 'brisk', 'all'],
                        default='orb', help="Feature detection yöntemi")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio testi eşiği")
    parser.add_argument("--show-keypoints", action="store_true", help="Sadece keypoint'leri göster")
    parser.add_argument("--save", help="Sonucu kaydet")
    args = parser.parse_args()
    
    print(f"Görüntü 1 yükleniyor: {args.image1}")
    img1 = load_image(args.image1)
    
    # Sadece keypoint görselleştirme
    if args.show_keypoints or args.image2 is None:
        if args.method == 'all':
            methods = ['orb', 'sift', 'akaze', 'brisk']
        else:
            methods = [args.method]
        
        fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 5))
        if len(methods) == 1:
            axes = [axes]
        
        for ax, method in zip(axes, methods):
            img_kp, count = visualize_keypoints(img1, method)
            ax.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{method.upper()}\n{count} keypoint")
            ax.axis('off')
        
        plt.tight_layout()
        
        if args.save:
            plt.savefig(args.save, dpi=150)
            print(f"Kaydedildi: {args.save}")
        
        plt.show()
        return
    
    # İki görüntü eşleştirme
    print(f"Görüntü 2 yükleniyor: {args.image2}")
    img2 = load_image(args.image2)
    
    if args.method == 'all':
        # Tüm yöntemleri karşılaştır
        print("\nTüm yöntemler karşılaştırılıyor...")
        results = compare_methods(img1, img2)
        
        print("\n" + "=" * 60)
        print(f"{'Yöntem':<10} {'KP1':>8} {'KP2':>8} {'Matches':>10} {'Time':>10}")
        print("=" * 60)
        for method, result in results.items():
            print(f"{method.upper():<10} {result['keypoints1']:>8} {result['keypoints2']:>8} "
                  f"{result['matches']:>10} {result['time']*1000:>9.1f}ms")
        
        plot_comparison(img1, img2, results, args.save)
    else:
        # Tek yöntem
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        print(f"\n{args.method.upper()} ile özellik çıkarılıyor...")
        kp1, desc1 = detect_features(gray1, args.method)
        kp2, desc2 = detect_features(gray2, args.method)
        
        print(f"Görüntü 1: {len(kp1)} keypoint")
        print(f"Görüntü 2: {len(kp2)} keypoint")
        
        if desc1 is None or desc2 is None:
            print("Descriptor çıkarılamadı!")
            return
        
        matches = match_features(desc1, desc2, args.method, args.ratio)
        print(f"İyi eşleşme: {len(matches)}")
        
        # Homography
        H, mask = find_homography(kp1, kp2, matches)
        if H is not None:
            inliers = np.sum(mask)
            print(f"Homography bulundu - Inliers: {inliers}/{len(matches)}")
        
        # Görselleştir
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure(figsize=(14, 7))
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.title(f'{args.method.upper()} - {len(matches)} iyi eşleşme')
        plt.axis('off')
        
        if args.save:
            plt.savefig(args.save, dpi=150)
            print(f"Kaydedildi: {args.save}")
        
        plt.show()


if __name__ == "__main__":
    main()
