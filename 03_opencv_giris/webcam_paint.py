"""
Sanal Çizim Tahtası (Webcam Paint)

Bu uygulama, webcam görüntüsü üzerinde sanal olarak çizim yapmanıza olanak tanır.
OpenCV'nin çizim fonksiyonlarını ve fare olaylarını (mouse events) kullanır.

Kullanım:
    python webcam_paint.py
    
Kontroller:
    - Sol Tık Sürükle: Çizim yap
    - 'c': Ekranı temizle
    - 'r': Rengi Kırmızı yap
    - 'g': Rengi Yeşil yap
    - 'b': Rengi Mavi yap
    - 'q': Çıkış
"""

import cv2
import numpy as np

# Global değişkenler
drawing = False
ix, iy = -1, -1
color = (0, 0, 255) # Varsayılan: Kırmızı (BGR)
thickness = 5
canvas = None # Çizim katmanı

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, canvas, color, thickness

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Maske (Canvas) üzerine çiz
            cv2.line(canvas, (ix, iy), (x, y), color, thickness)
            ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(canvas, (ix, iy), (x, y), color, thickness)

def main():
    global canvas, color
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Hata: Webcam açılamadı!")
        return

    # Pencere oluştur ve mouse callback ata
    cv2.namedWindow('Webcam Paint')
    cv2.setMouseCallback('Webcam Paint', draw_circle)

    # Canvas'ı ilk frame boyutunda başlatmak için bir kare oku
    ret, frame = cap.read()
    if not ret:
        return
    
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    print("Program çalışıyor... Çıkmak için 'q' tuşuna basın.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # Ayna etkisi

        # Canvas siyah değilse çizimi frame üzerine ekle
        # Maskeleme yöntemi:
        # 1. Canvas'ı griye çevir ve maske oluştur (çizim olan yerler beyaz)
        img2gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # 2. Frame'in çizim gelecek yerlerini karart
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # 3. Canvas'tan sadece çizimi al
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

        # 4. İkisini topla
        dst = cv2.add(frame_bg, canvas_fg)

        # Renk ve durumu gösteren arayüz
        # 1. Bilgi çubuğu (Arka plan)
        cv2.rectangle(dst, (0, 0), (w, 60), (50, 50, 50), -1)
        
        # 2. Seçili renk göstergesi (Sol üst köşe)
        cv2.circle(dst, (30, 30), 20, color, -1)
        cv2.circle(dst, (30, 30), 22, (255, 255, 255), 2) # Çerçeve
        
        # 3. Metin
        info_text = f"Renk: {color} | Temizle: C | Cikis: Q"
        cv2.putText(dst, info_text, (70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 4. Yardım metni
        cv2.putText(dst, "Renk degistirmek icin R, G, B tuslarina basin", (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow('Webcam Paint', dst)

        k = cv2.waitKey(1) & 0xFF
        
        # Tuş kontrolleri (Büyük/Küçük harf duyarlı olmaması için)
        if k == ord('q') or k == ord('Q'):
            break
        elif k == ord('c') or k == ord('C'):
            canvas = np.zeros((h, w, 3), dtype=np.uint8) # Temizle
        elif k == ord('r') or k == ord('R'):
            color = (0, 0, 255) # Kırmızı
            print("Renk: Kırmızı")
        elif k == ord('g') or k == ord('G'):
            color = (0, 255, 0) # Yeşil
            print("Renk: Yeşil")
        elif k == ord('b') or k == ord('B'):
            color = (255, 0, 0) # Mavi
            print("Renk: Mavi")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
