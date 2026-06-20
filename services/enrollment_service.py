from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import PipelineConfig
from detection.face_detector import FaceDetection, FaceDetector
from identification.face_identifier import FaceIdentifier
from identification.gallery_store import GalleryStore
from repositories.face_repository import EnrolledUserSummary


@dataclass(frozen=True)
class EnrollmentImage:
    name: str
    field_name: str
    content_type: str | None
    data: bytes


@dataclass(frozen=True)
class PerImageEnrollmentResult:
    name: str
    angle: str | None
    ok: bool
    reason: str | None = None


@dataclass(frozen=True)
class EnrollmentResult:
    label: str
    embeddings_added: int
    per_image: tuple[PerImageEnrollmentResult, ...]
    total_embeddings: int


@dataclass(frozen=True)
class DeleteEnrollmentResult:
    status: str
    label: str


class EnrollmentService:
    """Enroll raw face photos into the local ArcFace gallery."""

    _KNOWN_ANGLES = {
        "front": "front",
        "left": "left",
        "right": "right",
        "up": "up",
        "down": "down",
        "upleft": "upLeft",
        "up_left": "upLeft",
        "up-left": "upLeft",
        "upRight".lower(): "upRight",
        "up_right": "upRight",
        "up-right": "upRight",
    }

    def __init__(
        self,
        cfg: PipelineConfig,
        face_detector: FaceDetector,
        identifier: FaceIdentifier,
        gallery_store: GalleryStore,
    ) -> None:
        self._cfg = cfg
        self._face_detector = face_detector
        self._identifier = identifier
        self._gallery_store = gallery_store

    def enroll(self, label: str, images: Sequence[EnrollmentImage]) -> EnrollmentResult:
        clean_label = label.strip()
        if not clean_label:
            raise ValueError("label must not be empty")
        if not 1 <= len(images) <= 7:
            raise ValueError("enrollment accepts between 1 and 7 images")

        embeddings: list[np.ndarray] = []
        source_angles: list[str | None] = []
        per_image: list[PerImageEnrollmentResult] = []

        for image in images:
            angle = self._source_angle(image)
            result = self._embed_image(image, angle)
            per_image.append(result[0])
            if result[1] is not None:
                embeddings.append(result[1])
                source_angles.append(angle)

        total_embeddings = self._gallery_store.total_embeddings(clean_label)
        if embeddings:
            total_embeddings = self._gallery_store.upsert(clean_label, embeddings, source_angles)

        return EnrollmentResult(
            label=clean_label,
            embeddings_added=len(embeddings),
            per_image=tuple(per_image),
            total_embeddings=total_embeddings,
        )

    def list_enrolled(self) -> list[EnrolledUserSummary]:
        return self._gallery_store.repository.list_users()

    def delete(self, label: str) -> DeleteEnrollmentResult:
        clean_label = label.strip()
        if not clean_label:
            raise ValueError("label must not be empty")
        removed = self._gallery_store.remove(clean_label)
        return DeleteEnrollmentResult(
            status="deleted" if removed else "not_found",
            label=clean_label,
        )

    def gallery_user_count(self) -> int:
        return self._gallery_store.user_count()

    def _embed_image(
        self,
        image: EnrollmentImage,
        angle: str | None,
    ) -> tuple[PerImageEnrollmentResult, np.ndarray | None]:
        bgr = self._decode_image(image.data)
        if bgr is None:
            return (
                PerImageEnrollmentResult(
                    name=image.name,
                    angle=angle,
                    ok=False,
                    reason="Image could not be decoded",
                ),
                None,
            )

        bgr = self._downscale(bgr)
        frame_h, frame_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        face = self._largest_face(self._face_detector.detect(rgb))
        if face is None:
            return (
                PerImageEnrollmentResult(
                    name=image.name,
                    angle=angle,
                    ok=False,
                    reason="No face detected",
                ),
                None,
            )

        x, y, w, h, _score, keypoints = face
        face_roi = bgr[max(0, y) : y + h, max(0, x) : x + w]
        if face_roi.size == 0:
            return (
                PerImageEnrollmentResult(
                    name=image.name,
                    angle=angle,
                    ok=False,
                    reason="Detected face crop is empty",
                ),
                None,
            )

        crop_keypoints = FaceIdentifier.crop_keypoints(keypoints, (x, y, w, h), (frame_w, frame_h))
        embedding = self._identifier.embed_face(face_roi, crop_keypoints)
        return (
            PerImageEnrollmentResult(name=image.name, angle=angle, ok=True),
            embedding,
        )

    @staticmethod
    def _decode_image(data: bytes) -> np.ndarray | None:
        if not data:
            return None
        encoded = np.frombuffer(data, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            return None
        return np.asarray(decoded)

    def _downscale(self, bgr: np.ndarray) -> np.ndarray:
        max_width = max(1, int(self._cfg.enrollment_max_width))
        height, width = bgr.shape[:2]
        if width <= max_width:
            return bgr
        scale = max_width / float(width)
        new_size = (max_width, max(1, int(round(height * scale))))
        resized = cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)
        return np.asarray(resized)

    @staticmethod
    def _largest_face(faces: Sequence[FaceDetection]) -> FaceDetection | None:
        if not faces:
            return None
        return max(faces, key=lambda face: face[2] * face[3])

    def _source_angle(self, image: EnrollmentImage) -> str | None:
        candidates = (
            image.field_name,
            Path(image.name).stem,
            image.name,
        )
        for candidate in candidates:
            normalized = candidate.strip().replace("[]", "")
            normalized = Path(normalized).stem if "." in normalized else normalized
            key = normalized.strip().lower()
            if key in {"files", "file", "images", "image"}:
                continue
            if key in self._KNOWN_ANGLES:
                return self._KNOWN_ANGLES[key]
            if key:
                return normalized
        return None
