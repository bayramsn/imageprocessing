from __future__ import annotations

import argparse

import cv2

from common import OUTPUTS_DIR, save_output, save_text
from pipelines import analyze_document, annotate_ocr_boxes, configure, detect_document_edges, save_excel_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Webcam ile canlı OCR")
    parser.add_argument("--camera", type=int, default=0, help="Kamera index değeri")
    parser.add_argument("--lang", default="tur", help="Tesseract dil kodu")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=15,
        help="OCR işleminin kaç karede bir çalışacağı",
    )
    args = parser.parse_args()

    configure()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError("Kamera açılamadı. Kamera index değerini kontrol edin.")

    frame_count = 0
    last_overlay = None
    last_text = ""
    last_report = None
    last_edges = None

    print("Canlı OCR başladı. Çıkmak için q, anlık kayıt için s tuşuna basın.")

    while True:
        success, frame = capture.read()
        if not success:
            break

        display_frame = frame.copy()
        if frame_count % max(args.frame_skip, 1) == 0:
            edges = detect_document_edges(frame)
            ocr_source = edges["warped"] if edges["found"] else frame
            overlay = annotate_ocr_boxes(ocr_source, lang=args.lang, min_confidence=45)
            report = analyze_document(ocr_source, lang=args.lang)
            last_overlay = overlay["annotated"]
            last_text = report["corrected_text"]
            last_report = report
            last_edges = edges

        if last_overlay is not None:
            if last_edges is not None and last_edges["found"]:
                display_frame = last_edges["annotated"].copy()
                preview = cv2.resize(last_overlay, (320, 220))
                display_frame[20:240, 20:340] = preview
                cv2.putText(
                    display_frame,
                    "Duzenlenmis belge",
                    (20, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            else:
                display_frame = last_overlay.copy()
            lines = [line for line in last_text.splitlines() if line][:3]
            for index, line in enumerate(lines):
                cv2.putText(
                    display_frame,
                    line[:80],
                    (20, 30 + index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )
            if last_report is not None:
                label = f"Belge tipi: {last_report['classification']['type']}"
                cv2.putText(
                    display_frame,
                    label,
                    (20, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 128, 255),
                    2,
                )
            if last_edges is not None:
                edge_label = "Kenar bulundu" if last_edges["found"] else "Kenar bulunamadi"
                cv2.putText(
                    display_frame,
                    edge_label,
                    (20, display_frame.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if last_edges["found"] else (0, 0, 255),
                    2,
                )

        cv2.imshow("Canli OCR", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and last_report is not None and last_overlay is not None:
            image_path = save_output("13_webcam_snapshot.png", last_overlay)
            text_path = save_text("13_webcam_snapshot.txt", last_text)
            excel_path = save_excel_report(last_report, OUTPUTS_DIR / "13_webcam_snapshot.xlsx")
            print(f"Kaydedildi: {image_path}")
            print(f"Kaydedildi: {text_path}")
            print(f"Kaydedildi: {excel_path}")

        frame_count += 1

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
