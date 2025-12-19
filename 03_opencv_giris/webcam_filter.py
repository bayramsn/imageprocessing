import cv2  # OpenCV kütüphanesini içe aktar

# --- Proje Fikri ---
# Webcam'den görüntü alıp basit filtrelerle oynayabileceğin bir uygulama:
# c → normal renkli
# g → gri ton
# b → bulanık (blur)
# r → çözünürlük yarıya düşürülmüş

# --- Öğreneceklerin ---
# cv2.VideoCapture ile kamera kullanma
# cv2.cvtColor (BGR → GRAY)
# cv2.resize
# Klavyeden tuş yakalama: cv2.waitKey()

# Kamerayı başlat
cap = cv2.VideoCapture(0)  # Varsayılan kamerayı aç

if not cap.isOpened():  # Kamera başarıyla açıldı mı kontrol et
    print("Kamera açılamadı. Lütfen bir webcam bağlı olduğundan emin olun.")  # Hata mesajı yazdır
    raise SystemExit(1)  # Programı sonlandır

# Filtre modunu başlangıçta 'c' (normal) olarak ayarla
filter_mode = 'c'  # Varsayılan mod
mode_labels = {  # Tuşlara karşılık gelen mod adları
    'c': 'Normal',  # c tuşu: normal renkli
    'g': 'Gri',  # g tuşu: gri ton
    'b': 'Blur',  # b tuşu: bulanık
    'r': 'Yarı Çözünürlük',  # r tuşu: yarı çözünürlük
}

while True:  # Sürekli kare okumak için döngü
    ret, frame = cap.read()  # Kameradan kare oku

    if not ret:  # Kare okunamadıysa
        break  # Döngüyü bitir

    # Filtre moduna göre kareyi işle
    if filter_mode == 'g':  # Gri ton modu
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # BGR'den griye çevir
    elif filter_mode == 'b':  # Blur modu
        processed_frame = cv2.GaussianBlur(frame, (15, 15), 0)  # Gauss blur uygula
    elif filter_mode == 'r':  # Yarı çözünürlük modu
        height, width, _ = frame.shape  # Orijinal boyutları al
        processed_frame = cv2.resize(frame, (width // 2, height // 2))  # Boyutu yarıya indir
    else:  # Normal mod
        processed_frame = frame  # Değişiklik yapma

    text_color = (255, 255, 255) if len(processed_frame.shape) == 3 else 255  # Yazı rengi seç
    cv2.putText(  # Mod bilgisini kareye yaz
        processed_frame,
        f"Mode: {mode_labels.get(filter_mode, 'Normal')} (c/g/b/r)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        text_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(  # Çıkış bilgisini kareye yaz
        processed_frame,
        "Quit: q",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2,
        cv2.LINE_AA,
    )

    cv2.imshow('Webcam Filter Playground', processed_frame)  # Kareyi ekranda göster

    key = cv2.waitKey(1) & 0xFF  # Klavyeden tuş oku

    if key == ord('q'):  # q tuşu basıldıysa
        break  # Döngüden çık
    elif key == ord('c'):  # c tuşu basıldıysa
        filter_mode = 'c'  # Normal mod seç
    elif key == ord('g'):  # g tuşu basıldıysa
        filter_mode = 'g'  # Gri mod seç
    elif key == ord('b'):  # b tuşu basıldıysa
        filter_mode = 'b'  # Blur mod seç
    elif key == ord('r'):  # r tuşu basıldıysa
        filter_mode = 'r'  # Yarı çözünürlük mod seç

cap.release()  # Kamerayı serbest bırak
cv2.destroyAllWindows()  # Açık pencereleri kapat
