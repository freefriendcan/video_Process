import os
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from capture.frame_packet import FramePacket
from capture.ir_detector import IRDetector
from config import PipelineConfig


class FrameProducer:
    """RTSP capture with dedicated thread, bounded queue, and auto-reconnect.

    Reads from Tapo C225 stream2 (480p) continuously on a daemon thread.
    Produces ``FramePacket`` objects with pre-computed RGB and IR flag.

    On-demand ``capture_hires()`` opens stream1 (2K) for a single frame
    then releases the connection immediately.
    """

    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg
        self._ir_detector = IRDetector(cfg.ir_std_threshold)
        self._queue: queue.Queue[FramePacket] = queue.Queue(maxsize=2)
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self):
        """Start the capture thread. Blocks briefly to validate the URL."""
        if not self._cfg.rtsp_url:
            raise RuntimeError(
                "RTSP URL not configured. Set TAPO_IP, TAPO_USER, TAPO_PASS "
                "environment variables or provide rtsp_url in PipelineConfig."
            )
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="rtsp-capture"
        )
        self._thread.start()
        logger.info("Capture thread started for {}", self._cfg.rtsp_url)

    def read_latest(self) -> Optional[FramePacket]:
        """Non-blocking read of the most recent frame.

        Drains the queue and returns only the newest ``FramePacket``,
        or ``None`` if the queue is empty.
        """
        pkt: Optional[FramePacket] = None
        while not self._queue.empty():
            try:
                pkt = self._queue.get_nowait()
            except queue.Empty:
                break
        return pkt

    def capture_hires(self) -> Optional[np.ndarray]:
        """On-demand: grab a single frame from the 2K main stream.

        Opens a temporary connection, reads one frame, and releases.
        Returns the BGR ndarray or ``None`` on failure.
        """
        url = self._cfg.rtsp_hires_url
        if not url:
            logger.warning("No hires RTSP URL configured, skipping capture_hires()")
            return None

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error("Failed to open hires stream: {}", url)
            return None

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        cap.release()

        if not ok:
            logger.error("Failed to read frame from hires stream")
            return None

        logger.debug("Captured hires frame {}×{}", frame.shape[1], frame.shape[0])
        return frame

    def release(self):
        """Stop capture thread and release resources.

        Only sets the stop flag and waits for the capture thread to
        finish.  The capture thread exclusively owns ``self._cap`` to
        avoid race conditions with ``cv2.VideoCapture`` which is NOT
        thread-safe at the C++ level.
        """
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                logger.warning(
                    "Capture thread did not exit within 10s — "
                    "daemon thread will be cleaned up on process exit"
                )
        logger.info("FrameProducer released")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self):
        """Dedicated capture thread — reads RTSP, produces FramePackets.

        This thread exclusively owns ``self._cap``.  The main thread
        MUST NOT call ``self._cap.read()`` or ``self._cap.release()``.
        """
        delay = self._cfg.rtsp_reconnect_delay

        try:
            while self._running:
                self._cap = self._connect()
                if self._cap is None:
                    logger.warning(
                        "RTSP connection failed, retrying in {:.1f}s...", delay
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, self._cfg.rtsp_max_reconnect_delay)
                    continue

                # Connection established — reset backoff
                delay = self._cfg.rtsp_reconnect_delay
                logger.info("RTSP connected: {}", self._cfg.rtsp_url)

                self._read_loop()

                # _read_loop exited → connection lost
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
        finally:
            # Guarantee cleanup on any exit path
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            logger.debug("Capture thread exiting")

    def _read_loop(self):
        """Read frames until the stream drops or we're told to stop."""
        consecutive_failures = 0

        while self._running:
            ok, bgr = self._cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= 10:
                    logger.warning(
                        "RTSP: {} consecutive read failures, reconnecting...",
                        consecutive_failures,
                    )
                    return
                time.sleep(0.01)
                continue

            consecutive_failures = 0
            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ir = self._ir_detector.is_ir_mode(bgr)

            pkt = FramePacket(
                bgr=bgr, rgb=rgb, ir_mode=ir,
                timestamp=time.time(), width=w, height=h,
            )

            # Drop-oldest when full — always keep the freshest frame
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(pkt)

    def _connect(self) -> Optional[cv2.VideoCapture]:
        """Open an RTSP VideoCapture with TCP transport and minimal buffering.

        Forces TCP via ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` to prevent
        UDP packet loss artefacts (grey frames, green pixels) that
        corrupt downstream ML model inputs over Wi-Fi.
        """
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self._cfg.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
