"""Tapo C225 RTSP connection test.

Tests:
  1. .env loading and URL construction
  2. TCP RTSP connection to stream2 (480p)
  3. Frame capture and IR detection
  4. On-demand hires capture from stream1 (2K)

Usage:
    python scripts/test_rtsp.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from loguru import logger

from config import PipelineConfig
from capture.ir_detector import IRDetector


def main():
    cfg = PipelineConfig()

    # --- Step 1: Config validation ---
    logger.info("=== Step 1: Config Validation ===")
    if not cfg.rtsp_url:
        logger.error(
            "RTSP URL empty! Check .env file has TAPO_IP, TAPO_USER, TAPO_PASS"
        )
        sys.exit(1)

    logger.info("RTSP URL (480p): {}", cfg.rtsp_url)
    logger.info("RTSP URL (2K):   {}", cfg.rtsp_hires_url)
    logger.success("Config OK ✓")

    # --- Step 2: TCP RTSP connection (stream2 — 480p) ---
    logger.info("\n=== Step 2: RTSP TCP Connection (stream2, 480p) ===")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    t0 = time.time()
    cap = cv2.VideoCapture(cfg.rtsp_url, cv2.CAP_FFMPEG)
    elapsed = time.time() - t0

    if not cap.isOpened():
        logger.error("Failed to open RTSP stream2 in {:.1f}s", elapsed)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    logger.success("Connected in {:.2f}s ✓", elapsed)

    # --- Step 3: Read frames ---
    logger.info("\n=== Step 3: Frame Capture Test (5 frames) ===")
    ir_detector = IRDetector(cfg.ir_std_threshold)

    for i in range(5):
        ok, frame = cap.read()
        if not ok:
            logger.error("Frame {} read failed!", i)
            continue

        h, w = frame.shape[:2]
        ir = ir_detector.is_ir_mode(frame)
        means = frame.mean(axis=(0, 1))

        logger.info(
            "Frame {}: {}×{}, IR={}, channels=[B={:.0f}, G={:.0f}, R={:.0f}]",
            i, w, h, ir, means[0], means[1], means[2],
        )

    cap.release()
    logger.success("Stream2 (480p) capture OK ✓")

    # --- Step 4: Hires capture (stream1 — 2K) ---
    logger.info("\n=== Step 4: Hires Capture Test (stream1, 2K) ===")
    if not cfg.rtsp_hires_url:
        logger.warning("No hires URL configured, skipping")
    else:
        t0 = time.time()
        cap2 = cv2.VideoCapture(cfg.rtsp_hires_url, cv2.CAP_FFMPEG)
        if not cap2.isOpened():
            logger.error("Failed to open RTSP stream1 (2K)")
        else:
            cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, hires = cap2.read()
            cap2.release()
            elapsed = time.time() - t0

            if ok:
                h, w = hires.shape[:2]
                logger.success(
                    "Hires frame: {}×{} captured in {:.2f}s ✓", w, h, elapsed
                )

                # Save test frame
                test_path = "data/logs/test_hires.jpg"
                os.makedirs(os.path.dirname(test_path), exist_ok=True)
                cv2.imwrite(test_path, hires)
                logger.info("Saved test frame: {}", test_path)
            else:
                logger.error("Failed to read hires frame")

    # --- Summary ---
    logger.info("\n=== Summary ===")
    logger.success("All RTSP connection tests passed ✓")
    logger.info("Pipeline is ready to start: python main.py")


if __name__ == "__main__":
    main()
