import argparse  # Komut satırı argümanlarını okumak için
import cv2  # Görüntü işlemleri için OpenCV
import torch  # Model çalıştırmak için PyTorch
from torchvision import models  # Hazır modeller için torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights  # Detection ağırlıkları
from utils import load_image_bgr, bgr_to_rgb, resize_rgb, to_torch_tensor, normalize_tensor, draw_boxes  # Yardımcı fonksiyonlar


def run_classification(img_rgb):  # Sınıflandırma kısmını çalıştırır
    weights = models.ResNet18_Weights.DEFAULT  # ResNet18 ön eğitimli ağırlıklarını al
    model = models.resnet18(weights=weights)  # Modeli yükle
    model.eval()  # Çıkarım moduna al
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Cihazı seç
    model.to(device)  # Modeli cihaza taşı
    target_hw = (weights.meta["input_size"][1], weights.meta["input_size"][2])  # Hedef yükseklik ve genişlik
    img_resized = resize_rgb(img_rgb, target_hw[::-1])  # OpenCV sırası ile yeniden boyutlandır
    tensor = to_torch_tensor(img_resized)  # Tensöre çevir
    tensor = normalize_tensor(tensor, weights.meta["mean"], weights.meta["std"])  # Normalizasyon uygula
    batch = tensor.unsqueeze(0).to(device)  # Batch boyutu ekle ve cihaza taşı
    with torch.no_grad():  # Gradientsiz çıkarım
        probs = torch.nn.functional.softmax(model(batch), dim=1)[0]  # Olasılıkları hesapla
    score, idx = torch.max(probs, dim=0)  # En yüksek olasılığı ve sınıfı al
    label = weights.meta["categories"][idx.item()]  # Sınıf adını al
    exists = bool(score.item() > 0.25)  # Basit var/yok kararı için eşik
    return label, score.item(), exists  # Sonuçları döndür


def run_detection(img_rgb):  # Nesne tespiti kısmını çalıştırır
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT  # Detection ağırlıklarını al
    model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)  # Modeli yükle
    model.eval()  # Çıkarım moduna al
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Cihazı seç
    model.to(device)  # Cihaza taşı
    tensor = to_torch_tensor(img_rgb)  # Tensöre çevir (0-1 ölçekli)
    tensor = normalize_tensor(tensor, weights.meta["mean"], weights.meta["std"])  # Normalizasyon uygula
    batch = [tensor.to(device)]  # Detection modeli list bekler
    with torch.no_grad():  # Gradientsiz çıkarım
        outputs = model(batch)  # Çıktıları al
    out = outputs[0]  # Tek görüntü olduğu için ilk elemanı al
    boxes = out["boxes"].cpu().numpy().tolist()  # Kutuları CPU'ya al
    scores = out["scores"].cpu().numpy().tolist()  # Skorları al
    labels_idx = out["labels"].cpu().numpy().tolist()  # Etiket indekslerini al
    labels = [weights.meta["categories"][i] for i in labels_idx]  # İsimleri eşleştir
    return boxes, labels, scores  # Sonuçları döndür


def main():  # Programın giriş noktası
    parser = argparse.ArgumentParser(description="Sınıflandırma vs Nesne Tespiti karşılaştırması")  # Açıklamalı parser
    parser.add_argument("image", help="İşlenecek görüntü yolu")  # Girdi görüntüsü argümanı
    parser.add_argument("--score", type=float, default=0.5, help="Detection skor eşiği")  # Skor eşiği
    parser.add_argument("--show", action="store_true", help="Görselleri ekranda göster")  # Gösterim bayrağı
    args = parser.parse_args()  # Argümanları oku

    img_bgr = load_image_bgr(args.image)  # Görüntüyü yükle
    img_rgb = bgr_to_rgb(img_bgr)  # RGB'ye çevir

    cls_label, cls_score, exists = run_classification(img_rgb)  # Sınıflandırma çalıştır
    print(f"Sınıflandırma: {cls_label} ({cls_score:.2f}) -> {'OBJECT EXISTS' if exists else 'NO OBJECT'}")  # Sonucu yaz

    boxes, labels, scores = run_detection(img_rgb)  # Detection çalıştır
    print("Detection sonuçları (eşik filtreli):")  # Başlık yaz
    for b, l, s in zip(boxes, labels, scores):  # Her tespit için döngü
        if s < args.score:  # Eşik altı ise geç
            continue  # Sonrakine geç
        print(f"- {l}: {s:.2f} bbox={b}")  # Bilgiyi yaz

    drawn = draw_boxes(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), boxes, labels, scores, score_thresh=args.score)  # Kutuları çiz

    if args.show:  # Gösterim istenirse
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)  # Pencere aç
        cv2.imshow("Detection", drawn)  # Görüntüyü göster
        print("Pencereyi kapatmak için tuşa basın")  # Bilgi ver
        cv2.waitKey(0)  # Bekle
        cv2.destroyAllWindows()  # Pencereyi kapat


if __name__ == "__main__":  # Dosya doğrudan çalıştırıldığında
    main()  # Ana fonksiyonu çağır
