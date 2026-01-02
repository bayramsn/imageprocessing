"""
Yüz ve Göz Tespiti (Haar Cascades)

Bu betik, OpenCV'nin önceden eğitilmiş Haar Cascade sınıflandırıcılarını kullanarak
görüntüdeki yüzleri ve gözleri tespit eder.
Derin öğrenme modellerine göre daha hafiftir ve CPU üzerinde hızlı çalışır.

Kullanım:
    python face_eye_detector.py resim.jpg --output sonuc.png
"""

import cv2
import argparse
import os

def detect_faces_eyes(image_path, output_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: Görüntü okunamadı: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OpenCV ile gelen Haar Cascade dosyalarını yükle
    # Genellikle cv2.data.haarcascades altında bulunur
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

    if not os.path.exists(face_cascade_path):
        print("HATA: Haar cascade dosyaları bulunamadı. Lütfen opencv-python kütüphanesini kontrol edin.")
        return

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    # Yüz tespiti
    # scaleFactor: Her ölçekte görüntünün ne kadar küçüleceği (1.1 = %10)
    # minNeighbors: Bir bölgenin yüz sayılması için kaç komşu dikdörtgene ihtiyacı olduğu
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    print(f"Tespit edilen yüz sayısı: {len(faces)}")

    for (x, y, w, h) in faces:
        # Yüzü dikdörtgen içine al (Mavi)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Gözleri sadece yüz bölgesinde ara (Performans ve doğruluk için)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
        
        for (ex, ey, ew, eh) in eyes:
            # Gözü dikdörtgen içine al (Yeşil)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Sonucu kaydet veya göster
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Sonuç kaydedildi: {output_path}")
    else:
        cv2.imshow('Yuz ve Goz Tespiti', img)
        print("Kapatmak için bir tuşa basın...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Haar Cascade ile Yüz ve Göz Tespiti")
    parser.add_argument("image", help="İşlenecek görüntü yolu")
    parser.add_argument("--output", help="Sonuç görüntüsünün kaydedileceği yol")
    
    args = parser.parse_args()
    detect_faces_eyes(args.image, args.output)
