import os  # Dosya yolları için os modülü
import cv2  # Görüntü okumak ve çizim yapmak için OpenCV
import numpy as np  # Sayısal işlemler için NumPy
import torch  # Tensör işlemleri ve modeller için PyTorch


def load_image_bgr(path: str) -> np.ndarray:  # BGR formatında güvenli görüntü yükler
    if not os.path.isfile(path):  # Yol geçerli mi kontrol et
        raise FileNotFoundError(f"Image not found: {path}")  # Bulunamadıysa hata ver
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # OpenCV ile renkli oku
    if img is None:  # Okuma başarısızsa
        with open(path, "rb") as f:  # Dosyayı ikili modda aç
            data = np.asarray(bytearray(f.read()), dtype=np.uint8)  # Baytları NumPy dizisine çevir
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # Baytlardan yeniden decode et
    if img is None:  # Hala yoksa
        raise ValueError(f"Unable to read image: {path}")  # Hata ver
    return img  # BGR görüntüyü döndür


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:  # BGR'yi RGB'ye çevirir
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Renk kanallarını ters çevir


def to_torch_tensor(img_rgb: np.ndarray) -> torch.Tensor:  # RGB görüntüyü torch tensöre çevirir
    tensor = torch.from_numpy(img_rgb).float()  # NumPy'den float tensör oluştur
    tensor = tensor.permute(2, 0, 1)  # HWC'den CHW formatına geç
    tensor = tensor / 255.0  # 0-1 aralığına ölçekle
    return tensor  # Tensörü döndür


def normalize_tensor(tensor: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:  # Tensörü normalize eder
    mean_t = torch.tensor(mean).view(3, 1, 1)  # Mean'i şekle uydur
    std_t = torch.tensor(std).view(3, 1, 1)  # Std'yi şekle uydur
    return (tensor - mean_t) / std_t  # Normalizasyon uygula


def resize_rgb(img_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:  # RGB görüntüyü istenen boyuta getirir
    return cv2.resize(img_rgb, size, interpolation=cv2.INTER_LINEAR)  # Doğrusal interpolasyonla yeniden boyutlandır


def put_label(img: np.ndarray, text: str, pos=(10, 30), color=(0, 255, 0)) -> np.ndarray:  # Görüntüye basit metin yazar
    out = img.copy()  # Kopya oluştur
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)  # Metni çiz
    return out  # Çıkışı döndür


def draw_boxes(img: np.ndarray, boxes: list, labels: list[str], scores: list[float], score_thresh: float = 0.5) -> np.ndarray:  # Kutu ve etiket çizer
    out = img.copy()  # Kopya al
    for box, label, score in zip(boxes, labels, scores):  # Her kutu için döngü
        if score < score_thresh:  # Eşik altı ise atla
            continue  # Sonraki elemana geç
        x1, y1, x2, y2 = map(int, box)  # Kutu koordinatlarını tam sayıya çevir
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dikdörtgen çiz
        caption = f"{label}: {score:.2f}"  # Etiket metni hazırla
        cv2.putText(out, caption, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)  # Metni yaz
    return out  # Sonucu döndür


def show_image(title: str, img: np.ndarray, wait: int = 0) -> None:  # Görüntüyü ekranda gösterir
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # Yeniden boyutlanabilir pencere aç
    cv2.imshow(title, img)  # Görüntüyü göster
    cv2.waitKey(wait)  # Belirtilen süre kadar bekle (0 ise bekler)
    if wait == 0:  # Sonsuz beklediyse
        cv2.waitKey(1)  # Pencereyi yenile

