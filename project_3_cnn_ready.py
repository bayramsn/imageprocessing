import argparse  # Komut satırı argümanlarını okumak için
import cv2  # Görüntü okuma için OpenCV
import torch  # Model çalıştırma için PyTorch
from torchvision import models  # Hazır modeller için torchvision
from utils import load_image_bgr, bgr_to_rgb, resize_rgb, to_torch_tensor, normalize_tensor  # Yardımcı fonksiyonlar


def main():  # Uygulama giriş noktası
    parser = argparse.ArgumentParser(description="Hazır CNN ile görüntü sınıflandırma (yalnızca çıkarım)")  # Açıklama ekle
    parser.add_argument("image", help="Sınıflandırılacak görüntü yolu")  # Girdi görüntü argümanı
    parser.add_argument("--model", choices=["mobilenet", "resnet"], default="mobilenet", help="Kullanılacak hazır model")  # Model seçimi
    parser.add_argument("--topk", type=int, default=3, help="Kaç tahmini yazdırayım")  # Top-k çıktısı
    args = parser.parse_args()  # Argümanları oku

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU varsa kullan, yoksa CPU

    if args.model == "mobilenet":  # Mobilenet seçildiyse
        weights = models.MobileNet_V3_Large_Weights.DEFAULT  # Önceden eğitilmiş ağırlıkları al
        model = models.mobilenet_v3_large(weights=weights)  # Modeli ağırlıklarla yükle
    else:  # Resnet seçimi
        weights = models.ResNet18_Weights.DEFAULT  # ResNet18 ağırlıkları
        model = models.resnet18(weights=weights)  # Modeli yükle

    model.eval()  # Modeli çıkarım moduna al
    model.to(device)  # Modeli cihaza taşı

    img_bgr = load_image_bgr(args.image)  # Görüntüyü yükle
    img_rgb = bgr_to_rgb(img_bgr)  # RGB'ye çevir
    input_size = weights.meta.get("input_size", (3, 224, 224))  # Model giriş boyutunu al
    target_hw = (input_size[1], input_size[2])  # Hedef yükseklik ve genişlik
    img_resized = resize_rgb(img_rgb, target_hw[::-1])  # OpenCV sırası (width, height) ile yeniden boyutlandır

    tensor = to_torch_tensor(img_resized)  # Tensöre çevir
    tensor = normalize_tensor(tensor, weights.meta["mean"], weights.meta["std"])  # Normalize et
    batch = tensor.unsqueeze(0).to(device)  # Batch boyutu ekle ve cihaza taşı

    with torch.no_grad():  # Gradientsiz çıkarım bloğu
        outputs = model(batch)  # Modelden ham skorları al
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]  # Olasılık dağılımını hesapla

    topk = min(args.topk, probs.numel())  # Top-k sınırını belirle
    scores, indices = torch.topk(probs, topk)  # En yüksek olasılıkları al
    categories = weights.meta["categories"]  # Sınıf isimlerini al

    print("--- Tahmin Sonuçları ---")  # Başlık yaz
    for rank, (score, idx) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):  # Her tahmin için döngü
        label = categories[idx]  # Etiketi al
        print(f"{rank}. {label}: {score:.3f}")  # Sıra, etiket ve olasılığı yaz


if __name__ == "__main__":  # Dosya doğrudan çalıştırıldığında
    main()  # Ana fonksiyonu çağır
