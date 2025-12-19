"""
Tek Görüntü - Üç Yaklaşım

Classification vs Detection vs Segmentation karşılaştırması.
Aynı görüntü üzerinde üç farklı görevin çıktısını gösterir.
"""
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
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
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_classification(img_rgb, device):
    """
    CLASSIFICATION: "Bu ne?"
    Tek etiket + güven skoru döndürür
    """
    # Model yükle
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval().to(device)
    
    # Preprocessing
    preprocess = weights.transforms()
    
    # Tensor'a çevir
    img_tensor = preprocess(torch.from_numpy(img_rgb).permute(2, 0, 1))
    batch = img_tensor.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Top-5 sonuç
    top5_probs, top5_indices = torch.topk(probs, 5)
    categories = weights.meta["categories"]
    
    results = []
    for prob, idx in zip(top5_probs.tolist(), top5_indices.tolist()):
        results.append((categories[idx], prob * 100))
    
    return results


def run_detection(img_rgb, device, score_thresh=0.5):
    """
    DETECTION: "Nerede ne var?"
    Bounding box + etiket + skor döndürür
    """
    # Model yükle
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.eval().to(device)
    
    # Tensor'a çevir
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    
    # Normalize
    mean = torch.tensor(weights.meta["mean"]).view(3, 1, 1)
    std = torch.tensor(weights.meta["std"]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    # Inference
    with torch.no_grad():
        outputs = model([img_tensor.to(device)])[0]
    
    # Sonuçları filtrele
    boxes = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    
    categories = weights.meta["categories"]
    
    results = []
    for box, label, score in zip(boxes, labels, scores):
        if score >= score_thresh:
            results.append({
                "box": box.tolist(),
                "label": categories[label],
                "score": score * 100
            })
    
    return results


def run_segmentation(img_rgb, device):
    """
    SEGMENTATION: "Hangi piksel neye ait?"
    Piksel bazlı maske döndürür
    """
    # Model yükle
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet50(weights=weights)
    model.eval().to(device)
    
    # Preprocessing
    preprocess = weights.transforms()
    
    # Tensor'a çevir
    img_tensor = preprocess(torch.from_numpy(img_rgb).permute(2, 0, 1))
    batch = img_tensor.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(batch)["out"]
    
    # Argmax ile sınıf maskesi
    mask = outputs.argmax(1).squeeze().cpu().numpy()
    
    # Sınıf isimleri
    categories = weights.meta["categories"]
    
    # Benzersiz sınıflar
    unique_classes = np.unique(mask)
    detected_classes = [categories[c] for c in unique_classes if c > 0]  # 0 = background
    
    return mask, detected_classes, categories


def visualize_classification(ax, img, results):
    """Classification sonuçlarını görselleştir"""
    ax.imshow(img)
    ax.set_title("CLASSIFICATION\n\"Bu ne?\"", fontsize=12, fontweight='bold')
    
    # Sonuçları yazdır
    text = "\n".join([f"{label}: {score:.1f}%" for label, score in results[:3]])
    ax.text(10, 30, text, fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.8),
            verticalalignment='top')
    ax.axis('off')


def visualize_detection(ax, img, results):
    """Detection sonuçlarını görselleştir"""
    img_copy = img.copy()
    
    # Kutuları çiz
    for det in results:
        x1, y1, x2, y2 = map(int, det["box"])
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"{det['label']}: {det['score']:.0f}%"
        cv2.putText(img_copy, label, (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    ax.imshow(img_copy)
    ax.set_title(f"DETECTION\n\"Nerede ne var?\" ({len(results)} nesne)", 
                 fontsize=12, fontweight='bold')
    ax.axis('off')


def visualize_segmentation(ax, img, mask, categories):
    """Segmentation sonuçlarını görselleştir"""
    # Renk paleti oluştur
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(categories), 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background siyah
    
    # Renkli maske oluştur
    colored_mask = colors[mask]
    
    # Orijinal görüntü ile karıştır
    alpha = 0.5
    blended = (img * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    
    ax.imshow(blended)
    ax.set_title("SEGMENTATION\n\"Hangi piksel neye ait?\"", fontsize=12, fontweight='bold')
    ax.axis('off')


def compare_all(img_rgb, device, save_path=None):
    """Üç görevi karşılaştır"""
    
    print("1/3 Classification çalışıyor...")
    cls_results = run_classification(img_rgb, device)
    
    print("2/3 Detection çalışıyor...")
    det_results = run_detection(img_rgb, device)
    
    print("3/3 Segmentation çalışıyor...")
    seg_mask, seg_classes, categories = run_segmentation(img_rgb, device)
    
    # Görselleştir
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Classification vs Detection vs Segmentation", 
                 fontsize=14, fontweight='bold')
    
    # Orijinal
    axes[0].imshow(img_rgb)
    axes[0].set_title("ORİJİNAL", fontsize=12)
    axes[0].axis('off')
    
    # Classification
    visualize_classification(axes[1], img_rgb, cls_results)
    
    # Detection
    visualize_detection(axes[2], img_rgb, det_results)
    
    # Segmentation
    visualize_segmentation(axes[3], img_rgb, seg_mask, categories)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nKaydedildi: {save_path}")
    
    plt.show()
    
    # Özet yazdır
    print("\n" + "=" * 60)
    print("SONUÇ ÖZETİ")
    print("=" * 60)
    
    print("\n📌 CLASSIFICATION (Bu ne?):")
    for label, score in cls_results[:3]:
        print(f"   {label}: {score:.1f}%")
    
    print(f"\n📍 DETECTION (Nerede ne var?): {len(det_results)} nesne bulundu")
    for det in det_results[:5]:
        print(f"   {det['label']}: {det['score']:.0f}%")
    
    print(f"\n🎨 SEGMENTATION (Hangi piksel?): {len(seg_classes)} sınıf bulundu")
    for cls in seg_classes[:5]:
        print(f"   {cls}")


def main():
    parser = argparse.ArgumentParser(
        description="Classification vs Detection vs Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnek Kullanım:
  python compare_tasks.py resim.jpg
  python compare_tasks.py resim.jpg --save sonuc.png
        """
    )
    parser.add_argument("image", help="Görüntü yolu")
    parser.add_argument("--save", help="Sonucu kaydet")
    parser.add_argument("--device", choices=['cuda', 'cpu'], 
                        help="Cihaz seçimi (varsayılan: otomatik)")
    args = parser.parse_args()
    
    # Cihaz seçimi
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    
    # Görüntü yükle
    print(f"Görüntü yükleniyor: {args.image}")
    img_rgb = load_image(args.image)
    print(f"Boyut: {img_rgb.shape}")
    
    # Karşılaştır
    compare_all(img_rgb, device, args.save)


if __name__ == "__main__":
    main()
