import argparse
import os

import cv2
import numpy as np


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


def apply_filters(img: np.ndarray) -> dict:
    filters = {}
    # Box filter (normalized)
    filters["box"] = cv2.boxFilter(img, ddepth=-1, ksize=(5, 5), normalize=True)
    # Average blur
    filters["blur"] = cv2.blur(img, (5, 5))
    # Gaussian blur
    filters["gaussian"] = cv2.GaussianBlur(img, (5, 5), sigmaX=1.0, sigmaY=1.0)
    # Median blur
    filters["median"] = cv2.medianBlur(img, 5)
    # Bilateral filter
    filters["bilateral"] = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    # Sobel gradients
    filters["sobel_x"] = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    filters["sobel_y"] = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # Magnitude from Sobel
    mag = cv2.magnitude(filters["sobel_x"].astype(np.float32), filters["sobel_y"].astype(np.float32))
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    filters["sobel_mag"] = mag
    # Laplacian
    filters["laplacian"] = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    # Sharpen (unsharp masking style)
    gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0, sigmaY=2.0)
    sharpen = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    filters["sharpen"] = sharpen
    return filters


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img


def _resize_to(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def stack_results(original: np.ndarray, filters: dict) -> np.ndarray:
    base_wh = (original.shape[1], original.shape[0])  # (w, h)
    all_images = [
        ("original", original),
        ("box", filters.get("box")),
        ("blur", filters.get("blur")),
        ("gaussian", filters.get("gaussian")),
        ("median", filters.get("median")),
        ("bilateral", filters.get("bilateral")),
        ("sobel_x", filters.get("sobel_x")),
        ("sobel_y", filters.get("sobel_y")),
        ("sobel_mag", filters.get("sobel_mag")),
        ("laplacian", filters.get("laplacian")),
        ("sharpen", filters.get("sharpen")),
    ]
    prepared = []
    for name, img in all_images:
        if img is None:
            continue
        u8 = _to_uint8(img)
        resized = _resize_to(u8, base_wh)
        prepared.append(resized)

    # Grid halinde: 3 sütun
    cols = 3
    rows = []
    for i in range(0, len(prepared), cols):
        row_imgs = prepared[i : i + cols]
        rows.append(np.hstack(row_imgs))

    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad_w = max_w - r.shape[1]
            pad = np.zeros((r.shape[0], pad_w, r.shape[2]), dtype=r.dtype)
            r = np.hstack((r, pad))
        padded_rows.append(r)

    return np.vstack(padded_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply multiple OpenCV filters (box, blur, gaussian, median, bilateral, sobel, laplacian, sharpen)")
    default_image = os.path.join(os.path.dirname(__file__), "ai.jpg")
    parser.add_argument("--image", "-i", default=default_image, help="Path to input image (default: ai.jpg in this folder)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img = load_image(args.image)
    filters = apply_filters(img)
    stacked = stack_results(img, filters)
    cv2.namedWindow("Filters Demo", cv2.WINDOW_NORMAL)
    cv2.imshow("Filters Demo", stacked)
    print("Press q or Esc to exit")
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
