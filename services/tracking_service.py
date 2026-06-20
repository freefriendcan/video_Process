from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from detection.onnx_runtime import BBox
from identification.face_identifier import FaceIdentifier
from services.enrollment_service import EnrollmentService
from tracking.tracker_manager import TrackerManager


@dataclass(frozen=True)
class ActiveFaceTrack:
    id: int
    user: str
    bbox: BBox


@dataclass(frozen=True)
class ActivePersonTrack:
    id: int
    user: str
    bbox: BBox
    reid_ok: bool
    confidence: float


@dataclass(frozen=True)
class ActiveTrackingSnapshot:
    faces: tuple[ActiveFaceTrack, ...]
    persons: tuple[ActivePersonTrack, ...]
    ts: float


@dataclass(frozen=True)
class HealthStatus:
    status: str
    gallery_users: int
    model_loaded: bool


class TrackingService:
    """Read-only adapter over TrackerManager's lock-safe snapshots."""

    def __init__(
        self,
        tracker_manager: TrackerManager,
        enrollment_service: EnrollmentService,
        identifier: FaceIdentifier,
    ) -> None:
        self._tracker_manager = tracker_manager
        self._enrollment_service = enrollment_service
        self._identifier = identifier

    def active(self) -> ActiveTrackingSnapshot:
        face_snapshot = self._tracker_manager.snapshot
        person_snapshot = self._tracker_manager.person_snapshot
        return ActiveTrackingSnapshot(
            faces=tuple(self._faces(face_snapshot)),
            persons=tuple(self._persons(person_snapshot)),
            ts=time.time(),
        )

    def health(self) -> HealthStatus:
        return HealthStatus(
            status="ok",
            gallery_users=self._enrollment_service.gallery_user_count(),
            model_loaded=self._identifier.model_loaded,
        )

    @staticmethod
    def _faces(snapshot: Mapping[int, Mapping[str, object]]) -> list[ActiveFaceTrack]:
        faces: list[ActiveFaceTrack] = []
        for track_id, data in sorted(snapshot.items()):
            faces.append(
                ActiveFaceTrack(
                    id=track_id,
                    user=str(data.get("user", "Unknown")),
                    bbox=cast(BBox, data["bbox"]),
                )
            )
        return faces

    @staticmethod
    def _persons(snapshot: Mapping[int, Mapping[str, object]]) -> list[ActivePersonTrack]:
        persons: list[ActivePersonTrack] = []
        for track_id, data in sorted(snapshot.items()):
            persons.append(
                ActivePersonTrack(
                    id=track_id,
                    user=str(data.get("user", "Unknown")),
                    bbox=cast(BBox, data["bbox"]),
                    reid_ok=bool(data.get("reid_ok", False)),
                    confidence=_float_value(data.get("confidence", 0.0)),
                )
            )
        return persons


def _float_value(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    return 0.0
