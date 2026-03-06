"""Unified video source – webcam, video file, or RTSP stream."""

from __future__ import annotations

import logging
import sys
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_IS_MACOS = sys.platform == "darwin"


def _request_macos_camera_permission() -> None:
    """On macOS, trigger the camera permission dialog via AVFoundation."""
    if not _IS_MACOS:
        return
    try:
        import subprocess
        script = (
            "import objc\n"
            "from AVFoundation import AVCaptureDevice, AVMediaTypeVideo\n"
            "status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)\n"
            "print('status:', status)\n"
            "if status == 0:\n"
            "    AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeVideo, lambda x: None)\n"
        )
        subprocess.run([sys.executable, "-c", script], timeout=5)
    except Exception:
        pass  # pyobjc not available – skip


class VideoSource:
    """Iterable frame generator that wraps OpenCV ``VideoCapture``.

    Supports:
    * ``0``, ``1``, …  – webcam index
    * ``"/path/to/video.mp4"`` – local file
    * ``"rtsp://..."`` – RTSP / HTTP stream

    Usage::

        for frame_idx, frame in VideoSource(0):
            ...
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        stride: int = 1,
        imgsz: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            source: Webcam index, video path, or stream URL.
            stride: Yield every *stride*-th frame (1 = every frame).
            imgsz:  Optional ``(width, height)`` to resize frames.
        """
        self._source = source
        self._stride = max(1, stride)
        self._imgsz = imgsz
        self._cap: Optional[cv2.VideoCapture] = None

    # ── context manager ─────────────────────────────────────────────

    def open(self) -> "VideoSource":
        """Open the underlying ``VideoCapture``.

        On macOS webcam sources, ``cv2.CAP_AVFOUNDATION`` is tried first so
        that the OS permission dialog is properly triggered and AVFoundation is
        used instead of the FFMPEG avdevice path (which does NOT trigger the
        macOS camera-permission dialog).
        """
        src = self._source
        # Try to interpret string digits as int (webcam index)
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        is_webcam = isinstance(src, int)

        if _IS_MACOS and is_webcam:
            # Try AVFoundation first – this triggers the macOS permission dialog
            _request_macos_camera_permission()
            logger.debug("Trying AVFoundation backend for webcam index %s", src)
            self._cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
            if not self._cap.isOpened():
                logger.warning(
                    "AVFoundation failed for index %s – falling back to default backend", src
                )
                self._cap.release()
                self._cap = cv2.VideoCapture(src)
        else:
            self._cap = cv2.VideoCapture(src)

        if not self._cap.isOpened():
            hint = ""
            if _IS_MACOS and is_webcam:
                hint = (
                    "\n\n  ╔══ macOS Camera Permission Required ══════════════════════════╗"
                    "\n  ║  System Settings → Privacy & Security → Camera                 ║"
                    "\n  ║  Enable camera access for Terminal (or your terminal app).      ║"
                    "\n  ║  Then restart the terminal and run again.                       ║"
                    "\n  ╚══════════════════════════════════════════════════════════════════╝"
                )
            raise IOError(f"Cannot open video source: {self._source}{hint}")

        logger.info(
            "Opened source=%s  fps=%.1f  size=%dx%d",
            self._source,
            self._cap.get(cv2.CAP_PROP_FPS),
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── properties ──────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        return float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        """(width, height)"""
        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def total_frames(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── iterator ────────────────────────────────────────────────────

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield ``(frame_index, frame)`` respecting *stride*."""
        if self._cap is None:
            self.open()
        assert self._cap is not None

        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            if idx % self._stride == 0:
                if self._imgsz is not None:
                    frame = cv2.resize(frame, self._imgsz)
                yield idx, frame
            idx += 1

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        return self.frames()
