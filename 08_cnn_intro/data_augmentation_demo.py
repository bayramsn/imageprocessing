"""
Veri Çoğaltma (Data Augmentation) Demo

CNN eğitimi için eldeki veri setini yapay olarak çoğaltır.
Görüntüleri döndürme, kaydırma, kesme ve parlaklık değiştirme işlemlerini gösterir.
PyTorch `torchvision.transforms` kütüphanesini kullanır.

Kullanım:
    python data_augmentation_demo.py resim.jpg
"""

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

def demo_augmentation(image_path):
    if not os.path.exists(image_path):
        print(f"Dosya bulunamadı: {image_path}")
        return

    # Görüntüyü PIL formatında yükle (Torchvision PIL bekler)
    img = Image.open(image_path)

    # Augmentation İşlemleri Tanımla
    transform_definitions = {
        "Orijinal": transforms.Compose([]),
        
        "Random Rotation (30°)": transforms.RandomRotation(degrees=30),
        
        "Random Horizontal Flip": transforms.RandomHorizontalFlip(p=1.0),
        
        "Center Crop": transforms.CenterCrop(size=(img.height//2, img.width//2)),
        
        "Color Jitter (Parlaklik/Kontrast)": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        
        "Random Affine (Kaydirma)": transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        
        "Grayscale": transforms.Grayscale(num_output_channels=3),
        
        "Gaussian Blur": transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    }

    # Gridi ayarla
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    print("Augmentation işlemleri uygulanıyor...")

    for i, (name, transform) in enumerate(transform_definitions.items()):
        if i >= len(axes): break
        
        # Dönüşümü uygula
        augmented_img = transform(img)
        
        axes[i].imshow(augmented_img)
        axes[i].set_title(name)
        axes[i].axis('off')

    plt.tight_layout()
    
    output_file = "augmentation_demo_result.png"
    plt.savefig(output_file)
    print(f"Sonuç kaydedildi: {output_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Augmentation Demo")
    parser.add_argument("image", help="Kaynak görüntü")
    
    args = parser.parse_args()
    demo_augmentation(args.image)
