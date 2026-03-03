"""Camera capture for USB webcams and RTSP/IP cameras."""

import time
from typing import Optional

import cv2

from .base import BaseCapture


class CameraCapture(BaseCapture):
    """
    Video capture from USB webcam or RTSP/IP camera.

    Supports:
    - USB/Integrated webcams (by index: 0, 1, 2, ...)
    - RTSP streams (rtsp://...)
    - HTTP streams (http://...)
    - IP cameras (e.g., rtsp://192.168.1.100:554/stream)
    """

    def __init__(
        self,
        source: str | int = "0",
        fps: int = 30,
        resolution: tuple[int, int] = (1280, 720),
        rtsp_timeout: float = 5.0,
    ):
        """
        Initialize camera capture.

        Args:
            source: Camera index (0, 1, 2...) or RTSP URL
            fps: Target frames per second
            resolution: Target resolution (width, height)
            rtsp_timeout: Timeout for RTSP connection in seconds
        """
        super().__init__(str(source), fps, resolution)
        self.rtsp_timeout = rtsp_timeout
        self._cap: Optional[cv2.VideoCapture] = None

        # Convert string "0" to integer for OpenCV
        self._cv2_source = int(source) if str(source).isdigit() else source

    def open(self) -> bool:
        """
        Open the camera capture.

        Returns:
            True if successfully opened, False otherwise
        """
        try:
            self._cap = cv2.VideoCapture(self._cv2_source)

            # Set RTSP timeout for network streams
            if isinstance(self._cv2_source, str) and self._cv2_source.startswith("rtsp://"):
                # Set buffer size to reduce latency
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))

            if not self._cap.isOpened():
                return False

            # Set resolution
            width, height = self.resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Set FPS
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Verify connection by reading a test frame
            ret, _ = self._cap.read()
            if not ret:
                self.close()
                return False

            self._is_opened = True
            return True

        except Exception as e:
            print(f"Error opening camera: {e}")
            return False

    def close(self) -> None:
        """Close the camera capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False

    def read(self):
        """
        Read a single frame from the camera.

        Returns:
            Frame as numpy array or None if read failed
        """
        if not self.is_opened():
            return None

        ret, frame = self._cap.read()
        return frame if ret else None

    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    @property
    def width(self) -> int:
        """Get actual frame width."""
        if self._cap:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return self.resolution[0]

    @property
    def height(self) -> int:
        """Get actual frame height."""
        if self._cap:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self.resolution[1]

    @property
    def fps_actual(self) -> float:
        """Get actual FPS being captured."""
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_FPS)
        return float(self.fps)

    def set_backend(self, backend: int) -> bool:
        """
        Set the CV2 backend for capture.

        Args:
            backend: cv2.CAP_* backend constant

        Returns:
            True if successful
        """
        if self._cap and not self._is_opened:
            self._cap.set(cv2.CAP_PROP_BACKEND, backend)
            return True
        return False


class RTSPCapture(CameraCapture):
    """
    Specialized capture for RTSP streams with reconnection support.

    Features:
    - Automatic reconnection on stream failure
    - Configurable timeout and retry logic
    - Optimized buffer settings for low latency
    """

    def __init__(
        self,
        rtsp_url: str,
        fps: int = 30,
        resolution: tuple[int, int] = (1280, 720),
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize RTSP capture.

        Args:
            rtsp_url: Full RTSP URL (e.g., rtsp://192.168.1.100:554/stream)
            fps: Target frames per second
            resolution: Target resolution
            max_retries: Maximum connection retry attempts
            retry_delay: Delay between retries in seconds
        """
        super().__init__(rtsp_url, fps, resolution)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._retry_count = 0

    def open(self) -> bool:
        """Open RTSP stream with retry logic."""
        for attempt in range(self.max_retries):
            if super().open():
                self._retry_count = 0
                return True

            self._retry_count += 1
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        return False

    def read(self):
        """Read frame with automatic reconnection on failure."""
        frame = super().read()

        if frame is None and self._retry_count < self.max_retries:
            # Attempt reconnection
            print(f"RTSP stream lost, attempting reconnection... (attempt {self._retry_count + 1})")
            self.close()
            if self.open():
                self._retry_count += 1
                return super().read()

        return frame
