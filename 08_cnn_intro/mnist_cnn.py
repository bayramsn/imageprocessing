import argparse  # Komut satiri argumanlarini islemek icin argparse modulu
import os  # Ortam degiskenleri ve temel isletim islemleri icin os modulu
from pathlib import Path  # Dosya ve klasor yollarini guvenli sekilde temsil etmek icin Path sinifi
from typing import Tuple  # Fonksiyon tip ipuclari icin Tuple tipi

import matplotlib.pyplot as plt  # Egitim grafigini cizmek icin matplotlib pyplot
import numpy as np  # Sayisal islemler ve matrisler icin NumPy
import tensorflow as tf  # TensorFlow kutuphanesi (cekirdek)
from tensorflow import keras  # TensorFlow uzerinde Keras API'sini kullanmak icin


def build_model(input_shape: Tuple[int, int, int] = (28, 28, 1)) -> keras.Model:  # Varsayilan giris boyutuyla CNN modelini kur
    model = keras.Sequential(  # Katmanlari sirayla ekleyen Sequential model olustur
        [
            keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),  # 32 filtreli konvolusyon katmani
            keras.layers.MaxPool2D((2, 2)),  # 2x2 max pooling ile boyutlari yarimla
            keras.layers.Conv2D(64, (3, 3), activation="relu"),  # 64 filtreli ikinci konvolusyon katmani
            keras.layers.MaxPool2D((2, 2)),  # Tekrar 2x2 max pooling uygula
            keras.layers.Flatten(),  # Ozellik haritalarini tek boyutlu vektore ac
            keras.layers.Dense(128, activation="relu"),  # 128 noronlu tam baglantili gizli katman
            keras.layers.Dense(10, activation="softmax"),  # 10 sinif icin olasilik veren cikis katmani
        ]  # Katman listesi sonu
    )  # Sequential olusturma satiri sonu
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])  # Modeli optimizer, kayip ve dogruluk metrikleri ile derle
    return model  # Kurulan modeli geri dondur


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # MNIST verisini indirip isleyen fonksiyon
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()  # MNIST veri setini indir ve egitim/test olarak ayir
    x_train = x_train.astype("float32") / 255.0  # Egitim verisini 0-1 araligina normalize et
    x_test = x_test.astype("float32") / 255.0  # Test verisini 0-1 araligina normalize et
    x_train = np.expand_dims(x_train, -1)  # Egitim verisine kanal boyutu ekle (28,28,1)
    x_test = np.expand_dims(x_test, -1)  # Test verisine kanal boyutu ekle
    return x_train, y_train, x_test, y_test  # Hazirlanmis verileri dondur


def plot_history(history: keras.callbacks.History, out_path: str) -> None:  # Egitim/dogrulama kayip ve dogruluk grafigini kaydet
    plt.figure(figsize=(8, 4))  # Grafik boyutlarini ayarla
    plt.subplot(1, 2, 1)  # Birinci subplot: kayip grafigi
    plt.plot(history.history["loss"], label="train_loss")  # Egitim kaybini ciz
    plt.plot(history.history["val_loss"], label="val_loss")  # Dogrulama kaybini ciz
    plt.legend()  # Aciklama kutusunu goster
    plt.title("Loss")  # Baslik ekle
    plt.subplot(1, 2, 2)  # Ikinci subplot: dogruluk grafigi
    plt.plot(history.history["accuracy"], label="train_acc")  # Egitim dogrulugunu ciz
    plt.plot(history.history["val_accuracy"], label="val_acc")  # Dogrulama dogrulugunu ciz
    plt.legend()  # Aciklama kutusunu goster
    plt.title("Accuracy")  # Baslik ekle
    plt.tight_layout()  # Alt grafigi duzenli yerlestir
    plt.savefig(out_path)  # Olusan grafigi belirtilen yola kaydet
    plt.close()  # Figure nesnesini kapat
    print(f"Saved plot to {out_path}")  # Kaydedilen dosya yolunu konsola yaz


def predict_custom(model: keras.Model, image_path: str) -> int:  # Disaridan verilen 28x28 gri goruntu icin sinif tahmini yap
    img = keras.preprocessing.image.load_img(image_path, color_mode="grayscale", target_size=(28, 28))  # Goruntuyu gri tonlamada yukle ve 28x28'e olce
    arr = keras.preprocessing.image.img_to_array(img)  # Goruntuyu NumPy dizisine cevir
    arr = 1.0 - (arr / 255.0)  # Rakamlar koyu varsayildigi icin degerleri tersine cevir
    arr = np.expand_dims(arr, 0)  # Batch boyutunu ekle (1,28,28,1)
    preds = model.predict(arr, verbose=0)  # Modelden olasilik tahminlerini al
    return int(np.argmax(preds, axis=1)[0])  # En yuksek olasilikli sinif indeksini tam sayi olarak dondur


def main():  # Komut satiri argumanlarini okuyup egitimi ve opsiyonel tahmini yurut
    parser = argparse.ArgumentParser(description="Train a simple CNN on MNIST and optionally predict a custom digit")  # Arguman ayristirici olustur
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")  # Epoch sayisi icin parametre ekle
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")  # Batch boyutu parametresi ekle
    parser.add_argument("--model-out", default="mnist_cnn.h5", help="Path to save trained model")  # Model kayit yolu parametresi ekle
    parser.add_argument("--plot-out", default="training_curve.png", help="Path to save training curves")  # Grafik kayit yolu parametresi ekle
    parser.add_argument("--predict", default="", help="Path to a custom digit image (grayscale) for prediction")  # Tahmin icin dis goruntu yolunu al
    args = parser.parse_args()  # Komut satiri argumanlarini parse et

    x_train, y_train, x_test, y_test = load_data()  # MNIST verisini indir ve hazirla
    model = build_model()  # CNN modelini olustur

    history = model.fit(  # Modeli egit ve gecmisi kaydet
        x_train,  # Egitim verisi girdisi
        y_train,  # Egitim etiketleri
        validation_split=0.1,  # Verinin %10'unu dogrulama icin ayir
        epochs=args.epochs,  # Epoch sayisini argumandan al
        batch_size=args.batch_size,  # Batch boyutunu argumandan al
        verbose=2,  # Egitim log seviyesini ayarla
    )  # fit cagrisinin sonu

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)  # Modeli test verisi uzerinde degerlendir
    print(f"Test accuracy: {test_acc:.4f}")  # Test dogruluk degerini konsola yaz

    if args.model_out:  # Model kaydedilecekse
        model.save(args.model_out)  # Egitilmis modeli belirtilen yola kaydet
        print(f"Model saved to {args.model_out}")  # Kayit yolunu bildir

    if args.plot_out:  # Grafik kaydedilecekse
        plot_history(history, args.plot_out)  # Egitim/dogrulama grafiklerini kaydet

    if args.predict:  # Dis goruntu ile tahmin isteniyorsa
        if not Path(args.predict).is_file():  # Dosya varligini kontrol et
            raise FileNotFoundError(f"Prediction image not found: {args.predict}")  # Dosya yoksa hata firlat
        pred = predict_custom(model, args.predict)  # Goruntu icin sinif tahmini yap
        print(f"Custom image prediction: {pred}")  # Tahmin sonucunu yazdir


if __name__ == "__main__":  # Betik dogrudan calistirildiginda
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TensorFlow log seviyesini uyarilara sinirla
    main()  # Ana fonksiyonu cagir
