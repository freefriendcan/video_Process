#!/usr/bin/env python3
"""Re-enroll the local face gallery (``data/embeddings/faces.pkl``).

Builds the gallery with the **exact same** detect -> align -> embed path the live
pipeline uses (YOLOv11n-face + insightface ``buffalo_l``), so gallery and runtime
share one embedding space. The previous gallery was built with DeepFace and is
incompatible with the current insightface pipeline — delete it and re-run this.

Layout expected:

    data/enroll/
        ogulcan/   img1.jpg img2.jpg ...
        OG/        img1.png ...

Each subfolder name becomes a gallery label. One clear, roughly frontal face per
image works best; if an image has several faces the largest is used.

Usage:
    uv run python scripts/enroll_faces.py
    uv run python scripts/enroll_faces.py \
        --enroll-dir data/enroll --output data/embeddings/faces.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

# Make repo-root modules importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PipelineConfig  # noqa: E402
from detection.face_detector import FaceDetection, FaceDetector  # noqa: E402
from identification.face_identifier import FaceIdentifier  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _largest_face(faces: list[FaceDetection]) -> FaceDetection | None:
    """Pick the detection with the largest bbox area, or None if empty."""
    if not faces:
        return None
    return max(faces, key=lambda f: f[2] * f[3])  # f = (x, y, w, h, keypoints)


def enroll_person(
    person_dir: Path,
    face_det: FaceDetector,
    identifier: FaceIdentifier,
) -> np.ndarray | None:
    """Return an (N, 512) float32 array of embeddings for one person, or None."""
    embeddings: list[np.ndarray] = []
    images = sorted(p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        logger.warning("[{}] no images found, skipping", person_dir.name)
        return None

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.warning("[{}] unreadable image: {}", person_dir.name, img_path.name)
            continue

        frame_h, frame_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        face = _largest_face(face_det.detect(rgb))
        if face is None:
            logger.warning("[{}] no face detected: {}", person_dir.name, img_path.name)
            continue

        x, y, w, h, keypoints = face
        face_roi = bgr[max(0, y) : y + h, max(0, x) : x + w]  # BGR crop — matches live path
        if face_roi.size == 0:
            logger.warning("[{}] empty crop: {}", person_dir.name, img_path.name)
            continue

        crop_kps = FaceIdentifier.crop_keypoints(keypoints, (x, y, w, h), (frame_w, frame_h))
        embedding = identifier.embed_face(face_roi, crop_kps)  # (512,) L2-normalized
        embeddings.append(embedding)
        logger.info("[{}] embedded {} ({}x{} face)", person_dir.name, img_path.name, w, h)

    if not embeddings:
        logger.warning("[{}] produced no embeddings, skipping", person_dir.name)
        return None
    stacked: np.ndarray = np.stack(embeddings).astype(np.float32)
    return stacked


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-enroll the local face gallery.")
    parser.add_argument("--enroll-dir", default="data/enroll", type=Path)
    parser.add_argument("--output", default=None, type=Path,
                        help="Gallery output path (default: config.gallery_path)")
    args = parser.parse_args()

    cfg = PipelineConfig()
    output = args.output or Path(cfg.gallery_path)

    if not args.enroll_dir.is_dir():
        logger.error(
            "Enrollment dir not found: {}. Create it with one subfolder per person "
            "(e.g. {}/ogulcan/*.jpg) and re-run.",
            args.enroll_dir, args.enroll_dir,
        )
        return 1

    person_dirs = sorted(p for p in args.enroll_dir.iterdir() if p.is_dir())
    if not person_dirs:
        logger.error("No person subfolders under {}.", args.enroll_dir)
        return 1

    logger.info("Loading detector + ArcFace (insightface {})...", cfg.arcface_model_name)
    face_det = FaceDetector(cfg)
    identifier = FaceIdentifier.for_enrollment(cfg)

    gallery: dict[str, np.ndarray] = {}
    for person_dir in person_dirs:
        arr = enroll_person(person_dir, face_det, identifier)
        if arr is not None:
            gallery[person_dir.name] = arr

    if not gallery:
        logger.error("No embeddings produced — gallery NOT written.")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(gallery, handle)

    logger.success("Gallery written: {}", output)
    for label, arr in gallery.items():
        logger.success("  {}: {} embeddings, shape {}", label, arr.shape[0], arr.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
