"""Threaded frame queue for non-blocking video capture."""

import queue
import threading
from typing import Optional

import cv2
import numpy as np


class FrameQueue:
    """
    Thread-safe queue for video frames.

    Runs video capture in a separate thread to prevent blocking
    during inference. Implements frame dropping to prevent
    backlog when processing is slower than capture rate.
    """

    def __init__(
        self,
        capture,
        queue_size: int = 10,
        drop_frames: bool = True,
    ):
        """
        Initialize the frame queue.

        Args:
            capture: BaseCapture instance (camera, file, etc.)
            queue_size: Maximum number of frames to buffer
            drop_frames: If True, drop old frames when queue is full
        """
        self.capture = capture
        self.queue_size = queue_size
        self.drop_frames = drop_frames

        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0

    def start(self) -> bool:
        """
        Start the capture thread.

        Returns:
            True if successfully started
        """
        if not self.capture.open():
            return False

        self._stopped = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop the capture thread."""
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.capture.close()

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        while not self._stopped:
            frame = self.capture.read()

            if frame is None:
                # End of stream or connection lost
                if isinstance(self.capture, VideoFileCapture):
                    if self.capture.loop:
                        self.capture.seek(0)
                        continue
                    else:
                        break
                else:
                    # For cameras, retry
                    continue

            self._frame_count += 1

            if self.drop_frames:
                # Drop oldest frame if queue is full
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self._queue.get_nowait()  # Remove oldest
                        self._queue.put_nowait(frame)  # Add new
                    except queue.Empty:
                        pass
            else:
                # Block until space available
                try:
                    self._queue.put(frame, timeout=1.0)
                except queue.Full:
                    pass

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Read a frame from the queue.

        Args:
            timeout: Maximum time to wait for frame. None = wait forever

        Returns:
            Frame as numpy array or None if timeout/empty
        """
        try:
            return self._queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def read_latest(self) -> Optional[np.ndarray]:
        """
        Read the latest frame, discarding any queued frames.

        Returns:
            Most recent frame or None
        """
        frame = None
        while True:
            try:
                frame = self._queue.get_nowait()
            except queue.Empty:
                break
        return frame

    def clear(self) -> None:
        """Clear all frames from the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @property
    def frame_count(self) -> int:
        """Get total frames captured."""
        return self._frame_count

    @property
    def queue_size_current(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """Check if capture thread is running."""
        return self._thread is not None and self._thread.is_alive() and not self._stopped

    @property
    def fps(self) -> float:
        """Get the capture's FPS."""
        if hasattr(self.capture, "fps_actual"):
            return self.capture.fps_actual
        return float(self.capture.fps)

    @property
    def width(self) -> int:
        """Get frame width."""
        return self.capture.width

    @property
    def height(self) -> int:
        """Get frame height."""
        return self.capture.height

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Import VideoFileCapture for type checking
from .video_file import VideoFileCapture
