"""
Webcam Görüntü Deney Laboratuvarı - FPS Özellikli

Geliştirilmiş versiyon:
- FPS gösterimi
- Daha fazla filtre
- Ekran görüntüsü kaydetme
"""
import cv2
import time
import numpy as np


def apply_edge_filter(frame):
    """Canny kenar tespiti"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def apply_cartoon_filter(frame):
    """Karikatür efekti"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def apply_negative_filter(frame):
    """Negatif görüntü"""
    return 255 - frame


def apply_sepia_filter(frame):
    """Sepia (eski fotoğraf) efekti"""
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(frame, kernel)
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    return sepia


def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return
    
    # Kamera ayarları
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    filter_mode = 'c'
    mode_labels = {
        'c': 'Normal',
        'g': 'Gri',
        'b': 'Blur',
        'r': 'Resize (1/2)',
        'e': 'Edge (Canny)',
        'k': 'Cartoon',
        'n': 'Negative',
        's': 'Sepia',
    }
    
    prev_time = time.time()
    frame_count = 0
    fps = 0
    screenshot_count = 0
    
    print("=" * 50)
    print("WEBCAM GÖRÜNTÜ DENEY LABORATUVARI")
    print("=" * 50)
    print("Tuşlar:")
    for key, label in mode_labels.items():
        print(f"  {key} -> {label}")
    print("  p -> Ekran görüntüsü kaydet")
    print("  q -> Çıkış")
    print("=" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # FPS hesapla (her 10 frame'de bir güncelle)
        frame_count += 1
        if frame_count >= 10:
            curr_time = time.time()
            fps = frame_count / (curr_time - prev_time)
            prev_time = curr_time
            frame_count = 0
        
        # Filtre uygula
        if filter_mode == 'g':
            processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        elif filter_mode == 'b':
            processed = cv2.GaussianBlur(frame, (21, 21), 0)
        elif filter_mode == 'r':
            h, w = frame.shape[:2]
            processed = cv2.resize(frame, (w // 2, h // 2))
        elif filter_mode == 'e':
            processed = apply_edge_filter(frame)
        elif filter_mode == 'k':
            processed = apply_cartoon_filter(frame)
        elif filter_mode == 'n':
            processed = apply_negative_filter(frame)
        elif filter_mode == 's':
            processed = apply_sepia_filter(frame)
        else:
            processed = frame
        
        # Bilgi ekle
        info_frame = processed.copy()
        cv2.putText(info_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(info_frame, f"Mode: {mode_labels.get(filter_mode, 'Normal')}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(info_frame, "q: quit | p: screenshot", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Webcam Lab', info_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            # Ekran görüntüsü kaydet
            filename = f"screenshot_{screenshot_count}.png"
            cv2.imwrite(filename, processed)
            print(f"Kaydedildi: {filename}")
            screenshot_count += 1
        elif chr(key) in mode_labels:
            filter_mode = chr(key)
            print(f"Mod değişti: {mode_labels[filter_mode]}")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
