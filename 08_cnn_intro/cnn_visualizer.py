"""
CNN Görselleştirici - Feature Maps

Her katmandan çıkan feature map'leri görselleştir.
Kernel'lerin ne öğrendiğini anla.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cv2
import os


def load_model_safe(model_path):
    """Modeli güvenli şekilde yükle"""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")
    
    return keras.models.load_model(model_path)


def load_sample_image(image_path=None):
    """Test görüntüsü yükle veya MNIST'ten al"""
    
    if image_path and os.path.isfile(image_path):
        # Dış görüntü yükle
        img = keras.preprocessing.image.load_img(
            image_path, color_mode="grayscale", target_size=(28, 28)
        )
        arr = keras.preprocessing.image.img_to_array(img)
        arr = 1.0 - (arr / 255.0)  # Invert (MNIST formatı)
        return np.expand_dims(arr, 0)
    else:
        # MNIST'ten örnek al
        (_, _), (x_test, _) = keras.datasets.mnist.load_data()
        x_test = x_test.astype("float32") / 255.0
        x_test = np.expand_dims(x_test, -1)
        # Rastgele bir örnek
        idx = np.random.randint(0, len(x_test))
        return np.expand_dims(x_test[idx], 0)


def get_feature_maps(model, input_image, layer_indices=None):
    """Belirtilen katmanlardan feature map'leri çıkar"""
    
    # Tüm Conv2D katmanlarını bul
    conv_layers = []
    for i, layer in enumerate(model.layers):
        if 'conv2d' in layer.name.lower():
            conv_layers.append((i, layer.name, layer))
    
    if layer_indices is None:
        layer_indices = [l[0] for l in conv_layers]
    
    feature_maps = {}
    
    for idx in layer_indices:
        layer = model.layers[idx]
        # Intermediate model oluştur
        intermediate_model = keras.Model(
            inputs=model.input,
            outputs=layer.output
        )
        # Feature map al
        fmap = intermediate_model.predict(input_image, verbose=0)
        feature_maps[layer.name] = fmap[0]  # Batch boyutunu kaldır
    
    return feature_maps


def visualize_feature_maps(feature_maps, max_filters=16, save_path=None):
    """Feature map'leri görselleştir"""
    
    n_layers = len(feature_maps)
    
    fig, axes = plt.subplots(n_layers, max_filters, 
                              figsize=(max_filters * 1.5, n_layers * 1.5))
    fig.suptitle('CNN Feature Maps', fontsize=14, fontweight='bold')
    
    for i, (layer_name, fmaps) in enumerate(feature_maps.items()):
        n_filters = min(fmaps.shape[-1], max_filters)
        
        for j in range(max_filters):
            ax = axes[i, j] if n_layers > 1 else axes[j]
            
            if j < n_filters:
                ax.imshow(fmaps[:, :, j], cmap='viridis')
            ax.axis('off')
            
            if j == 0:
                ax.set_ylabel(layer_name, fontsize=8, rotation=0, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Kaydedildi: {save_path}")
    
    plt.show()


def visualize_kernels(model, layer_index=0, save_path=None):
    """İlk Conv katmanının kernel'lerini görselleştir"""
    
    # Conv katmanını bul
    conv_layer = None
    for layer in model.layers:
        if 'conv2d' in layer.name.lower():
            conv_layer = layer
            break
    
    if conv_layer is None:
        print("Conv2D katmanı bulunamadı!")
        return
    
    # Kernel ağırlıklarını al
    kernels, biases = conv_layer.get_weights()
    
    # Kernels shape: (kernel_h, kernel_w, input_channels, n_filters)
    n_filters = kernels.shape[-1]
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    fig.suptitle(f'Öğrenilen Kernel\'ler ({conv_layer.name})', fontsize=12, fontweight='bold')
    
    for i, ax in enumerate(axes.flatten()):
        if i < n_filters:
            kernel = kernels[:, :, 0, i]  # İlk kanal
            ax.imshow(kernel, cmap='gray')
            ax.set_title(f'F{i}', fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Kaydedildi: {save_path}")
    
    plt.show()


def visualize_activations_grid(model, input_image, save_path=None):
    """Tüm katman aktivasyonlarını grid olarak göster"""
    
    # Tüm katman çıkışlarını al
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name or 'pool' in layer.name]
    layer_names = [layer.name for layer in model.layers if 'conv' in layer.name or 'pool' in layer.name]
    
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(input_image, verbose=0)
    
    # Her katman için ayrı figür
    for layer_name, activation in zip(layer_names, activations):
        n_features = activation.shape[-1]
        size = activation.shape[1]
        
        n_cols = 8
        n_rows = min(4, (n_features + n_cols - 1) // n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 1.5))
        fig.suptitle(f'{layer_name} - {n_features} feature maps ({size}x{size})', fontsize=10)
        
        for i, ax in enumerate(axes.flatten()):
            if i < n_features:
                ax.imshow(activation[0, :, :, i], cmap='viridis')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()


def model_summary_visual(model):
    """Model yapısını görsel olarak göster"""
    
    print("\n" + "=" * 60)
    print("MODEL YAPISI")
    print("=" * 60)
    
    total_params = 0
    
    for i, layer in enumerate(model.layers):
        params = layer.count_params()
        total_params += params
        
        # Layer tipi ve şekli
        output_shape = layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'
        
        print(f"{i+1:2}. {layer.name:<25} {str(output_shape):<20} Params: {params:,}")
    
    print("=" * 60)
    print(f"Toplam parametre: {total_params:,}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="CNN Görselleştirici")
    parser.add_argument("model", help="Model dosyası (.h5 veya .keras)")
    parser.add_argument("--image", help="Test görüntüsü (opsiyonel)")
    parser.add_argument("--kernels", action="store_true", help="Kernel'leri göster")
    parser.add_argument("--save", help="Sonuçları kaydet")
    args = parser.parse_args()
    
    print(f"Model yükleniyor: {args.model}")
    model = load_model_safe(args.model)
    
    # Model özeti
    model_summary_visual(model)
    
    # Test görüntüsü yükle
    print("\nTest görüntüsü hazırlanıyor...")
    input_image = load_sample_image(args.image)
    
    # Girdi görüntüsünü göster
    plt.figure(figsize=(3, 3))
    plt.imshow(input_image[0, :, :, 0], cmap='gray')
    plt.title('Giriş Görüntüsü')
    plt.axis('off')
    plt.show()
    
    # Feature maps
    print("\nFeature map'ler çıkarılıyor...")
    feature_maps = get_feature_maps(model, input_image)
    visualize_feature_maps(feature_maps, max_filters=16, 
                           save_path=args.save + '_fmaps.png' if args.save else None)
    
    # Kernel'ler
    if args.kernels:
        print("\nKernel'ler görselleştiriliyor...")
        visualize_kernels(model, save_path=args.save + '_kernels.png' if args.save else None)
    
    # Tahmin
    pred = model.predict(input_image, verbose=0)
    pred_class = np.argmax(pred)
    confidence = pred[0][pred_class] * 100
    
    print(f"\nTahmin: {pred_class} (Güven: {confidence:.1f}%)")


if __name__ == "__main__":
    main()
