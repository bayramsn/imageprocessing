"""
Hazır CNN modeli ile görüntü sınıflandırma
MobileNet veya ResNet kullanıyor
Eğitim yok, sadece inference (çıkarım)
"""
import argparse
import cv2
import torch
from torchvision import models
from utils import load_image_bgr, bgr_to_rgb, resize_rgb, to_torch_tensor, normalize_tensor


def main():
    parser = argparse.ArgumentParser(description="Hazır CNN ile görüntü sınıflandırma")
    parser.add_argument("image", help="Sınıflandırılacak görüntü yolu")
    parser.add_argument("--model", choices=["mobilenet", "resnet"], default="mobilenet", 
                        help="Kullanılacak model")
    parser.add_argument("--topk", type=int, default=3, help="Kaç tahmini yazdırsın")
    args = parser.parse_args()

    # gpu varsa kullan
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    # model yükle
    if args.model == "mobilenet":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
    else:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)

    model.eval()  # inference mode - dropout vs kapanıyor
    model.to(device)

    # görüntüyü hazırla
    img_bgr = load_image_bgr(args.image)
    img_rgb = bgr_to_rgb(img_bgr)
    
    # modelin beklediği boyut
    input_size = weights.meta.get("input_size", (3, 224, 224))
    target_hw = (input_size[1], input_size[2])
    
    # resize - opencv (w,h) sırası kullanıyor
    img_resized = resize_rgb(img_rgb, target_hw[::-1])

    # tensöre çevir ve normalize et
    tensor = to_torch_tensor(img_resized)
    tensor = normalize_tensor(tensor, weights.meta["mean"], weights.meta["std"])
    
    # batch dimension ekle - model (N,C,H,W) bekliyor
    batch = tensor.unsqueeze(0).to(device)

    # inference
    with torch.no_grad():  # gradient hesaplama - memory için önemli
        outputs = model(batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

    # top-k sonuçları al
    topk = min(args.topk, probs.numel())
    scores, indices = torch.topk(probs, topk)
    categories = weights.meta["categories"]

    print("\n--- Tahmin Sonuçları ---")
    for rank, (score, idx) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
        label = categories[idx]
        print(f"{rank}. {label}: {score*100:.1f}%")


if __name__ == "__main__":
    main()
