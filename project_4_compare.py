"""
Sınıflandırma vs Nesne Tespiti karşılaştırması

Sınıflandırma: bu ne? -> tek cevap
Detection: nerede ne var? -> birden fazla kutu
"""
import argparse
import cv2
import torch
from torchvision import models
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from utils import (load_image_bgr, bgr_to_rgb, resize_rgb, 
                   to_torch_tensor, normalize_tensor, draw_boxes)


def run_classification(img_rgb):
    """ResNet ile sınıflandırma yap"""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # resize
    target_hw = (weights.meta["input_size"][1], weights.meta["input_size"][2])
    img_resized = resize_rgb(img_rgb, target_hw[::-1])
    
    # tensor hazırla
    tensor = to_torch_tensor(img_resized)
    tensor = normalize_tensor(tensor, weights.meta["mean"], weights.meta["std"])
    batch = tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(batch), dim=1)[0]
    
    score, idx = torch.max(probs, dim=0)
    label = weights.meta["categories"][idx.item()]
    
    # basit eşik - aslında pek anlamlı değil ama örnek olsun
    exists = bool(score.item() > 0.25)
    return label, score.item(), exists


def run_detection(img_rgb):
    """Faster R-CNN ile nesne tespiti"""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # detection modeli resize istemiyor, orijinal boyut kullanılabilir
    tensor = to_torch_tensor(img_rgb)
    tensor = normalize_tensor(tensor, weights.meta["mean"], weights.meta["std"])
    
    # detection modeli list of tensor bekliyor, classification'dan farklı
    batch = [tensor.to(device)]
    
    with torch.no_grad():
        outputs = model(batch)
    
    out = outputs[0]  # tek resim var
    
    # sonuçları çıkar
    boxes = out["boxes"].cpu().numpy().tolist()
    scores = out["scores"].cpu().numpy().tolist()
    labels_idx = out["labels"].cpu().numpy().tolist()
    labels = [weights.meta["categories"][i] for i in labels_idx]
    
    return boxes, labels, scores


def main():
    parser = argparse.ArgumentParser(description="Sınıflandırma vs Nesne Tespiti")
    parser.add_argument("image", help="İşlenecek görüntü yolu")
    parser.add_argument("--score", type=float, default=0.5, help="Detection skor eşiği")
    parser.add_argument("--show", action="store_true", help="Görselleri ekranda göster")
    args = parser.parse_args()

    img_bgr = load_image_bgr(args.image)
    img_rgb = bgr_to_rgb(img_bgr)

    # sınıflandırma
    print("=" * 40)
    print("SINIFLANDIRMA (ResNet18)")
    print("=" * 40)
    cls_label, cls_score, exists = run_classification(img_rgb)
    print(f"Tahmin: {cls_label}")
    print(f"Güven: {cls_score*100:.1f}%")
    print(f"Sonuç: {'OBJECT EXISTS' if exists else 'NO OBJECT'}")

    # detection
    print("\n" + "=" * 40)
    print("NESNE TESPİTİ (Faster R-CNN)")
    print("=" * 40)
    boxes, labels, scores = run_detection(img_rgb)
    
    count = 0
    for b, l, s in zip(boxes, labels, scores):
        if s < args.score:
            continue
        count += 1
        print(f"- {l}: {s*100:.1f}%")
    print(f"\nToplam tespit: {count} nesne")

    # görselleştir
    if args.show:
        drawn = draw_boxes(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 
                          boxes, labels, scores, score_thresh=args.score)
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Detection", drawn)
        print("\nPencereyi kapatmak için tuşa basın")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
