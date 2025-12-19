import argparse
import os

import cv2
import numpy as np

try:
    from scipy.signal import convolve2d
except ImportError as exc:  # SciPy yoksa kullanıcıya net hata ver
    raise SystemExit("SciPy gerekli: pip install scipy") from exc


def make_gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError("Kernel size must be odd and positive")
    radius = ksize // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * (sigma**2)))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def scipy_conv_color(image: np.ndarray, kernel: np.ndarray, boundary: str = "symm") -> np.ndarray:
    if image.ndim == 2:
        image = image[..., None]
    channels = []
    for c in range(image.shape[2]):
        channels.append(convolve2d(image[..., c], kernel, mode="same", boundary=boundary))
    out = np.stack(channels, axis=2)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gaussian blur with SciPy vs OpenCV")
    default_image = r"C:\opencv yakalayıcı\05_gaussian_blur_manual\ai.jpg"
    parser.add_argument("--image", "-i", default=default_image, help=f"Path to input image (default: {default_image})")
    parser.add_argument("--ksize", type=int, default=5, help="Odd kernel size (3,5,7...)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma")
    parser.add_argument("--boundary", choices=["symm", "fill", "wrap"], default="symm", help="SciPy boundary mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        with open(args.image, "rb") as f:
            data = np.asarray(bytearray(f.read()), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to read image: {args.image}")

    kernel = make_gaussian_kernel(args.ksize, args.sigma)

    manual = scipy_conv_color(bgr.astype(np.float32), kernel, boundary=args.boundary)
    manual_clipped = np.clip(manual, 0, 255).astype(np.uint8)

    opencv = cv2.GaussianBlur(bgr, (args.ksize, args.ksize), sigmaX=args.sigma, sigmaY=args.sigma, borderType=cv2.BORDER_REFLECT)

    diff = np.abs(manual_clipped.astype(np.float32) - opencv.astype(np.float32)).mean()
    print(f"Mean abs difference (SciPy vs OpenCV): {diff:.6f}")

    stacked = np.hstack((bgr, manual_clipped, opencv))
    cv2.namedWindow("SciPy vs OpenCV", cv2.WINDOW_NORMAL)
    cv2.imshow("SciPy vs OpenCV", stacked)
    print("Press q or Esc to exit")
    while True:
        # Pencere kapandıysa çık
        if cv2.getWindowProperty("SciPy vs OpenCV", cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(30)
        if key is None or key == -1:
            continue
        key &= 0xFF
        if key in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
