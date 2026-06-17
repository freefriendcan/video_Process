#!/usr/bin/env python3
"""Calibrate ArcFace cosine threshold from labeled Tapo face crops."""

from __future__ import annotations

import argparse
import statistics
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute genuine/impostor cosine distributions from labeled Tapo face crops. "
            "Use one subfolder per label; labels absent from the gallery are treated as strangers."
        ),
    )
    parser.add_argument(
        "crop_root",
        type=Path,
        help="Folder containing labeled Tapo crop subfolders.",
    )
    parser.add_argument(
        "--gallery",
        type=Path,
        default=None,
        help="Gallery pickle path (default: config.gallery_path).",
    )
    parser.add_argument(
        "--stranger-label",
        default="stranger",
        help="Case-insensitive substring that marks a folder as impostor/stranger.",
    )
    return parser.parse_args()


def _largest_face(faces: list[FaceDetection]) -> FaceDetection | None:
    if not faces:
        return None
    return max(faces, key=lambda face: face[2] * face[3])


def _image_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def _best_gallery_match(
    embedding: np.ndarray,
    gallery: dict[str, np.ndarray],
) -> tuple[str | None, float]:
    best_label: str | None = None
    best_score = -1.0
    for label, gallery_embeddings in gallery.items():
        score = float(np.max(gallery_embeddings @ embedding))
        if score > best_score:
            best_label = label
            best_score = score
    return best_label, best_score


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def _describe(label: str, values: list[float]) -> None:
    if not values:
        logger.warning("{}: no samples", label)
        return
    logger.info(
        "{}: n={} min={:.3f} p05={:.3f} mean={:.3f} p50={:.3f} p95={:.3f} max={:.3f}",
        label,
        len(values),
        min(values),
        _percentile(values, 5),
        statistics.fmean(values),
        _percentile(values, 50),
        _percentile(values, 95),
        max(values),
    )


def _recommend_threshold(genuine: list[float], impostor: list[float]) -> float | None:
    if not genuine or not impostor:
        return None

    genuine_floor = _percentile(genuine, 5)
    impostor_ceiling = _percentile(impostor, 95)
    if impostor_ceiling < genuine_floor:
        return (impostor_ceiling + genuine_floor) / 2.0

    return max(impostor_ceiling, min(genuine) - 0.01)


def main() -> int:
    args = _parse_args()
    if not args.crop_root.is_dir():
        logger.error("Crop root not found: {}", args.crop_root)
        return 2

    cfg = PipelineConfig()
    gallery_path = args.gallery or Path(cfg.gallery_path)
    identifier = FaceIdentifier.for_enrollment(cfg)
    gallery = identifier._load_gallery(gallery_path)
    detector = FaceDetector(cfg)

    genuine_scores: list[float] = []
    impostor_scores: list[float] = []

    label_dirs = sorted(path for path in args.crop_root.iterdir() if path.is_dir())
    if not label_dirs:
        logger.error("No labeled subfolders under {}", args.crop_root)
        return 2

    for label_dir in label_dirs:
        expected_label = label_dir.name
        is_stranger = args.stranger_label.lower() in expected_label.lower()
        paths = _image_paths(label_dir)
        if not paths:
            logger.warning("[{}] no images found", expected_label)
            continue

        for image_path in paths:
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                logger.warning("[{}] unreadable image: {}", expected_label, image_path.name)
                continue

            frame_h, frame_w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            face = _largest_face(detector.detect(rgb))
            if face is None:
                logger.warning("[{}] no face detected: {}", expected_label, image_path.name)
                continue

            x, y, w, h, keypoints = face
            face_roi = bgr[max(0, y) : y + h, max(0, x) : x + w]
            if face_roi.size == 0:
                logger.warning("[{}] empty crop: {}", expected_label, image_path.name)
                continue

            crop_keypoints = FaceIdentifier.crop_keypoints(
                keypoints,
                (x, y, w, h),
                (frame_w, frame_h),
            )
            embedding = identifier.embed_face(face_roi, crop_keypoints)
            best_label, score = _best_gallery_match(embedding, gallery)

            if not is_stranger and expected_label in gallery and best_label == expected_label:
                genuine_scores.append(score)
                bucket = "genuine"
            else:
                impostor_scores.append(score)
                bucket = "impostor"
            logger.info(
                "[{}] {} -> best={} cosine={:.3f} ({})",
                expected_label,
                image_path.name,
                best_label or "None",
                score,
                bucket,
            )

    _describe("genuine", genuine_scores)
    _describe("impostor", impostor_scores)

    threshold = _recommend_threshold(genuine_scores, impostor_scores)
    if threshold is None:
        logger.warning("Need both genuine and impostor samples to recommend a threshold.")
        return 1

    logger.success(
        "Recommended face_match_cosine_threshold: {:.3f} "
        "(midpoint/guard between Tapo genuine and impostor distributions)",
        threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
