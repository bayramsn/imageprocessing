import argparse
import os

import cv2
import numpy as np


# 2D Gaussian kernel üretimi (standart formül)

def make_gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError("Kernel size must be odd and positive")
    radius = ksize // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * (sigma**2)))  # e^{-(x^2+y^2)/(2σ^2)}
    kernel /= kernel.sum()  # normalize et ki toplam 1 olsun
    return kernel.astype(np.float32)


# Basit 2D konvolüsyon (color destekli)

def convolve2d(image: np.ndarray, kernel: np.ndarray, padding: str = "reflect") -> np.ndarray:
    if image.ndim == 2:
        image = image[..., None]
    h, w, c = image.shape
    kh, kw = kernel.shape
    pad = kh // 2

    if padding == "reflect":
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    else:  # zero padding
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="constant")

    out = np.zeros_like(image, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            region = padded[y : y + kh, x : x + kw]
            out[y, x] = (region * kernel[..., None]).sum(axis=(0, 1))
    return out.squeeze()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual Gaussian blur vs OpenCV comparison")
    default_image = r"C:\opencv yakalayıcı\05_gaussian_blur_manual\ai.jpg"
    parser.add_argument("--image", "-i", default=default_image, help=f"Path to input image (default: {default_image})")
    parser.add_argument("--ksize", type=int, default=5, help="Odd kernel size (e.g., 3,5,7)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma (float)")
    parser.add_argument("--padding", choices=["reflect", "zero"], default="reflect", help="Padding mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:  # Unicode yol problemi için yedek okuma
        with open(args.image, "rb") as f:
            data = np.asarray(bytearray(f.read()), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to read image: {args.image}")

    kernel = make_gaussian_kernel(args.ksize, args.sigma)

    # Manuel konvolüsyon (float32 hesap, sonra clip+uint8)
    manual = convolve2d(bgr.astype(np.float32), kernel, padding=args.padding)
    manual_clipped = np.clip(manual, 0, 255).astype(np.uint8)

    # OpenCV Gaussian blur ile kıyas
    opencv = cv2.GaussianBlur(bgr, (args.ksize, args.ksize), sigmaX=args.sigma, sigmaY=args.sigma, borderType=cv2.BORDER_REFLECT)

    # Fark metriği
    diff = np.abs(manual_clipped.astype(np.float32) - opencv.astype(np.float32)).mean()
    print(f"Mean abs difference (manual vs OpenCV): {diff:.6f}")

    stacked = np.hstack((bgr, manual_clipped, opencv))
    cv2.namedWindow("Manual vs OpenCV", cv2.WINDOW_NORMAL)
    cv2.imshow("Manual vs OpenCV", stacked)
    print("Press q or Esc to exit")
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
