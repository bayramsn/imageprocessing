import argparse
import os
from typing import Optional

import cv2
import numpy as np

# SciPy opsiyonel
try:
    from scipy.signal import convolve2d
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def load_image(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        with open(path, "rb") as f:
            data = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read image: {path}")
    return img


def make_gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError("Kernel size must be odd and positive")
    radius = ksize // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * (sigma**2)))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def convolve2d_manual(image: np.ndarray, kernel: np.ndarray, padding: str = "reflect") -> np.ndarray:
    if image.ndim == 2:
        image = image[..., None]
    h, w, c = image.shape
    kh, kw = kernel.shape
    pad = kh // 2
    if padding == "reflect":
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    else:
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    out = np.zeros_like(image, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            region = padded[y : y + kh, x : x + kw]
            out[y, x] = (region * kernel[..., None]).sum(axis=(0, 1))
    return out.squeeze()


def convolve2d_scipy(image: np.ndarray, kernel: np.ndarray, boundary: str = "symm") -> np.ndarray:
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required for scipy mode. Install with: pip install scipy")
    if image.ndim == 2:
        image = image[..., None]
    channels = [convolve2d(image[..., c], kernel, mode="same", boundary=boundary) for c in range(image.shape[2])]
    return np.stack(channels, axis=2)


def run_all_filters(img: np.ndarray, ksize: int, sigma: float, padding: str, boundary: str) -> dict:
    kernel = make_gaussian_kernel(ksize, sigma)

    # OpenCV Gaussian
    cv_gauss = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)

    # Manuel numpy konvolüsyon
    manual = convolve2d_manual(img.astype(np.float32), kernel, padding=padding)
    manual = np.clip(manual, 0, 255).astype(np.uint8)

    # SciPy konvolüsyon (opsiyonel)
    scipy_img: Optional[np.ndarray]
    if SCIPY_AVAILABLE:
        scipy_raw = convolve2d_scipy(img.astype(np.float32), kernel, boundary=boundary)
        scipy_img = np.clip(scipy_raw, 0, 255).astype(np.uint8)
    else:
        scipy_img = None

    # Diğer filtreler
    blur = cv2.blur(img, (ksize, ksize))
    median = cv2.medianBlur(img, ksize)
    bilateral = cv2.bilateralFilter(img, d=ksize, sigmaColor=sigma * 20, sigmaSpace=sigma * 10)

    return {
        "original": img,
        "cv_gauss": cv_gauss,
        "manual": manual,
        "scipy": scipy_img,
        "blur": blur,
        "median": median,
        "bilateral": bilateral,
    }


def summarize(reference: np.ndarray, others: dict) -> None:
    ref_f = reference.astype(np.float32)
    for name, img in others.items():
        if img is None:
            print(f"{name}: skipped (not available)")
            continue
        diff = np.abs(ref_f - img.astype(np.float32)).mean()
        print(f"Mean abs difference vs OpenCV Gaussian ({name}): {diff:.6f}")


def stack_and_show(results: dict) -> None:
    rows = []
    row1 = [results["original"], results["cv_gauss"], results["manual"]]
    row1 = [img for img in row1 if img is not None]
    rows.append(np.hstack(row1))

    row2 = [results.get("scipy"), results["blur"], results["median"], results["bilateral"]]
    row2 = [img for img in row2 if img is not None]
    if row2:
        rows.append(np.hstack(row2))

    stacked = np.vstack(rows)
    cv2.namedWindow("All Filters", cv2.WINDOW_NORMAL)
    cv2.imshow("All Filters", stacked)
    print("Press q or Esc to exit")
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test multiple blur filters and compare")
    default_image = r"C:\opencv yakalayıcı\05_gaussian_blur_manual\ai.jpg"
    parser.add_argument("--image", "-i", default=default_image, help=f"Path to input image (default: {default_image})")
    parser.add_argument("--ksize", type=int, default=5, help="Odd kernel size (3,5,7...)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma")
    parser.add_argument("--padding", choices=["reflect", "zero"], default="reflect", help="Manual padding mode")
    parser.add_argument("--boundary", choices=["symm", "fill", "wrap"], default="symm", help="SciPy boundary mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img = load_image(args.image)

    results = run_all_filters(img, ksize=args.ksize, sigma=args.sigma, padding=args.padding, boundary=args.boundary)

    summarize(results["cv_gauss"], {
        "manual": results["manual"],
        "scipy": results.get("scipy"),
        "blur": results["blur"],
        "median": results["median"],
        "bilateral": results["bilateral"],
    })

    stack_and_show(results)


if __name__ == "__main__":
    main()
