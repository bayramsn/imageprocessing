"""
Otomatik Panorama Oluşturucu (Image Stitching)

Bu betik, birden fazla örtüşen fotoğrafı birleştirerek tek bir panorama görüntüsü oluşturur.
OpenCV'nin güçlü `Stitcher` sınıfını kullanır.
Geleneksel yöntemlerle (Keypoint matching -> Homography -> Warping) yapılan işlemin otomatik halidir.

Kullanım:
    python panorama_maker.py resim1.jpg resim2.jpg --output panorama.png
"""

import cv2
import argparse
import sys

def create_panorama(images, output_path="panorama_result.png"):
    print(f"{len(images)} adet görüntü birleştiriliyor...")
    
    # Görüntüleri oku
    imgs = []
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"UYARI: Görüntü okunamadı: {img_path}")
            continue
        imgs.append(img)
    
    if len(imgs) < 2:
        print("HATA: Panorama için en az 2 geçerli görüntü gerekli.")
        return

    # Stitcher oluştur
    # OpenCV sürümüne göre farklılık gösterebilir
    try:
        stitcher = cv2.Stitcher_create() 
    except AttributeError:
        stitcher = cv2.createStitcher() # Eski sürümler için

    # Birleştirme işlemini yap
    status, panorama = stitcher.stitch(imgs)

    if status == cv2.Stitcher_OK:
        print("Panorama başarıyla oluşturuldu!")
        cv2.imwrite(output_path, panorama)
        print(f"Kaydedildi: {output_path}")
        
        # Sonucu göster (boyutunu küçülterek)
        h, w = panorama.shape[:2]
        if w > 1000:
            scale = 1000 / w
            display_img = cv2.resize(panorama, (0, 0), fx=scale, fy=scale)
        else:
            display_img = panorama
            
        cv2.imshow("Panorama Sonucu", display_img)
        print("Kapatmak için bir tuşa basın...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        error_msg = "Bilinmeyen Hata"
        if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
            error_msg = "Yeterli görüntü yok veya örtüşme yetersiz"
        elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            error_msg = "Homografi hesaplanamadı (Ortak noktalar bulunamadı)"
        elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            error_msg = "Kamera parametreleri ayarlanamadı"
            
        print(f"HATA: Panorama oluşturulamadı. Kod: {status}")
        print(f"Hata Detayı: {error_msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Otomatik Panorama Oluşturucu")
    parser.add_argument("images", nargs='+', help="Birleştirilecek görüntü yolları (en az 2 tane)")
    parser.add_argument("--output", default="panorama_result.png", help="Çıktı dosya adı")
    
    args = parser.parse_args()
    create_panorama(args.images, args.output)
