#!/usr/bin/env python3
"""Capture local webcam enrollment frames into ``data/enroll/<name>/``."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

# Make repo-root modules importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PipelineConfig  # noqa: E402
from detection.face_detector import FaceDetection, FaceDetector  # noqa: E402
from quality.face_quality_gate import FaceQualityGate  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture good-quality webcam face frames for local enrollment.",
    )
    parser.add_argument("--name", required=True, help="Enrollment label / output folder name.")
    parser.add_argument("--count", type=int, default=10, help="Number of good frames to save.")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Minimum seconds between automatically saved good frames.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Mac webcam index.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/enroll"),
        help="Enrollment root directory.",
    )
    parser.add_argument(
        "--keypress",
        action="store_true",
        help="Save only when a good frame is visible and Space is pressed.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the OpenCV preview window for headless capture.",
    )
    parser.add_argument(
        "--enroll",
        action="store_true",
        help="Run scripts/enroll_faces.py after capture completes.",
    )
    return parser.parse_args()


def _open_camera(camera_index: int) -> cv2.VideoCapture:
    backend = getattr(cv2, "CAP_AVFOUNDATION", 0)
    capture = cv2.VideoCapture(camera_index, backend)
    if capture.isOpened():
        return capture

    capture.release()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open webcam index {camera_index}")
    return capture


def _preview_enabled(no_preview: bool) -> bool:
    if no_preview:
        return False
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    return True


def _single_quality_face(
    bgr_frame: np.ndarray,
    detector: FaceDetector,
    gate: FaceQualityGate,
) -> tuple[FaceDetection | None, str]:
    frame_h, frame_w = bgr_frame.shape[:2]
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect(rgb_frame)
    if len(faces) != 1:
        return None, f"expected exactly one face, got {len(faces)}"

    face = faces[0]
    x, y, w, h, keypoints = face
    face_roi = bgr_frame[max(0, y) : y + h, max(0, x) : x + w]
    if face_roi.size == 0:
        return None, "empty face crop"

    passed, _score = gate.check(
        face_roi,
        keypoints,
        frame_w,
        frame_h,
        (x, y, w, h),
        ir_mode=False,
    )
    if not passed:
        return None, "quality gate rejected face"
    return face, "ok"


def _next_output_path(output_dir: Path, name: str, saved_count: int) -> Path:
    index = saved_count + 1
    while True:
        candidate = output_dir / f"{name}_{index:02d}.jpg"
        if not candidate.exists():
            return candidate
        index += 1


def _run_enrollment() -> int:
    script_path = Path(__file__).resolve().parent / "enroll_faces.py"
    completed = subprocess.run([sys.executable, str(script_path)], check=False)
    return int(completed.returncode)


def main() -> int:
    args = _parse_args()
    if args.count <= 0:
        logger.error("--count must be positive")
        return 2
    if args.interval < 0:
        logger.error("--interval must be non-negative")
        return 2

    output_dir = args.output_root / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig()
    detector = FaceDetector(cfg)
    gate = FaceQualityGate(cfg)
    capture = _open_camera(args.camera_index)
    preview = _preview_enabled(bool(args.no_preview))

    logger.info(
        "Capturing {} good face frames for '{}' into {}",
        args.count,
        args.name,
        output_dir,
    )
    logger.info("Use small head turns and expression changes between saved frames.")
    if args.keypress:
        logger.info("Press Space to save each accepted frame; press q to quit.")

    saved = 0
    last_save_time = 0.0
    try:
        while saved < args.count:
            ok, frame = capture.read()
            if not ok or frame is None:
                logger.warning("Webcam read failed; retrying")
                time.sleep(0.05)
                continue

            face, reason = _single_quality_face(frame, detector, gate)
            now = time.time()

            key = -1
            if preview:
                try:
                    preview_frame = frame.copy()
                    if face is not None:
                        x, y, w, h, _keypoints = face
                        cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imshow("Enrollment Capture", preview_frame)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as exc:
                    logger.warning("Preview unavailable; continuing headless: {}", exc)
                    preview = False

            if key == ord("q"):
                logger.info("Capture cancelled by user")
                break

            should_save = face is not None and (
                (args.keypress and key == ord(" "))
                or (not args.keypress and now - last_save_time >= args.interval)
            )
            if not should_save:
                if face is None:
                    logger.debug("Skipping frame: {}", reason)
                time.sleep(0.01)
                continue

            out_path = _next_output_path(output_dir, args.name, saved)
            if not cv2.imwrite(str(out_path), frame):
                logger.error("Failed to write {}", out_path)
                return 1
            saved += 1
            last_save_time = now
            logger.success("Saved {}/{}: {}", saved, args.count, out_path)

    finally:
        capture.release()
        if preview:
            cv2.destroyAllWindows()

    if saved != args.count:
        logger.error("Captured {}/{} required frames", saved, args.count)
        return 1

    if args.enroll:
        return _run_enrollment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
