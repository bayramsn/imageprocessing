"""
Ortak yardımcı fonksiyonlar
bu dosyayı diğer projelerden import ediyorum
"""
import os
import cv2
import numpy as np
import torch


def load_image_bgr(path: str) -> np.ndarray:
    """Görüntüyü yükler - unicode path'lerde sorun çıkınca imdecode kullanıyorum"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    # cv2.imread Türkçe karakterli yollarda çalışmıyor bazen
    # stackoverflow'dan buldum bu çözümü
    if img is None:
        with open(path, "rb") as f:
            data = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"Unable to read image: {path}")
    return img


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """BGR -> RGB dönüşümü"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_torch_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    # RGB görüntüyü pytorch tensöre çevir
    # HWC -> CHW formatı lazım pytorch için
    tensor = torch.from_numpy(img_rgb).float()
    tensor = tensor.permute(2, 0, 1)  # kanal sırası değişiyor
    tensor = tensor / 255.0
    return tensor


def normalize_tensor(tensor: torch.Tensor, mean: list[float], std: list[float]) -> torch.Tensor:
    """ImageNet normalizasyonu - mean ve std değerleri modelden geliyor"""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def resize_rgb(img_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    # boyut değiştir - size = (width, height)
    return cv2.resize(img_rgb, size, interpolation=cv2.INTER_LINEAR)


def put_label(img: np.ndarray, text: str, pos=(10, 30), color=(0, 255, 0)) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out


def draw_boxes(img: np.ndarray, boxes: list, labels: list[str], scores: list[float], score_thresh: float = 0.5) -> np.ndarray:
    """Detection sonuçlarını çiz
    
    TODO: renkleri farklı sınıflara göre ayarla
    """
    out = img.copy()
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        caption = f"{label}: {score:.2f}"
        # y1-10 bazen negatif olabiliyor, max ile düzelttim
        cv2.putText(out, caption, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def show_image(title: str, img: np.ndarray, wait: int = 0) -> None:
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)  # resize edilebilsin
    cv2.imshow(title, img)
    cv2.waitKey(wait)
    if wait == 0:
        cv2.waitKey(1)  # pencereyi yenile
