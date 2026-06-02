import time

import cv2

from config import PipelineConfig


class FrameProducer:
    """Wraps cv2.VideoCapture with deferred open to eliminate import-time side effects."""

    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg
        self._cap = None

    def open(self):
        self._cap = cv2.VideoCapture(self._cfg.camera_index)
        time.sleep(2)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.frame_height)

    def read(self):
        """Read a frame, flip horizontally. Returns (success, frame)."""
        success, frame = self._cap.read()
        if success:
            frame = cv2.flip(frame, 1)
        return success, frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
