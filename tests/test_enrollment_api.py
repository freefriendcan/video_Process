from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import cast

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.app import create_app
from config import PipelineConfig
from detection.face_detector import FaceDetection, FaceDetector
from detection.onnx_runtime import NormalizedKeypoint
from identification.face_identifier import FaceIdentifier
from identification.gallery_store import GalleryStore
from repositories.face_repository import EnrolledUserSummary, FaceRepository
from services.enrollment_service import (
    DeleteEnrollmentResult,
    EnrollmentImage,
    EnrollmentResult,
    EnrollmentService,
    PerImageEnrollmentResult,
)
from services.tracking_service import (
    ActiveFaceTrack,
    ActivePersonTrack,
    ActiveTrackingSnapshot,
    HealthStatus,
    TrackingService,
)
from tracking.tracker_manager import TrackerManager


def test_face_repository_round_trips_embeddings_and_deletes_user(tmp_path):
    repository = FaceRepository(tmp_path / "faces.db")
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1.0

    total = repository.add_embeddings("Ada", [vector], ["front"])

    assert total == 1
    users = repository.list_users()
    assert [(user.label, user.num_embeddings) for user in users] == [("Ada", 1)]
    gallery = repository.get_all_embeddings()
    np.testing.assert_array_equal(gallery["Ada"], vector.reshape(1, 512))

    assert repository.delete_user("Ada") is True
    assert repository.list_users() == []
    assert repository.get_all_embeddings() == {}


def test_gallery_store_hot_reload_is_visible_to_face_identifier(tmp_path):
    repository = FaceRepository(tmp_path / "faces.db")
    store = GalleryStore(repository)
    identifier = FaceIdentifier.__new__(FaceIdentifier)
    identifier._cfg = PipelineConfig()
    identifier._gallery = {}
    identifier._gallery_store = store

    vector = np.zeros(512, dtype=np.float32)
    vector[12] = 1.0
    assert identifier._best_gallery_match(vector) == (None, -1.0)

    store.upsert("Ada", [vector], ["front"])

    assert identifier._best_gallery_match(vector) == ("Ada", 1.0)


class FakeFaceDetector:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def detect(self, rgb_frame: np.ndarray) -> list[FaceDetection]:
        self.frames.append(rgb_frame)
        keypoints = (
            NormalizedKeypoint(x=0.30, y=0.30),
            NormalizedKeypoint(x=0.50, y=0.30),
            NormalizedKeypoint(x=0.40, y=0.45),
            NormalizedKeypoint(x=0.32, y=0.62),
            NormalizedKeypoint(x=0.48, y=0.62),
        )
        return [(10, 10, 40, 30, 0.95, keypoints)]


class FakeIdentifier:
    def __init__(self) -> None:
        self.roi_shapes: list[tuple[int, int, int]] = []

    def embed_face(
        self,
        face_roi_raw: np.ndarray,
        keypoints: Sequence[NormalizedKeypoint],
    ) -> np.ndarray:
        self.roi_shapes.append(cast(tuple[int, int, int], face_roi_raw.shape))
        assert len(keypoints) == 5
        vector = np.zeros(512, dtype=np.float32)
        vector[3] = 1.0
        return vector


class FakeGalleryStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, int, tuple[str | None, ...]]] = []
        self._total = 0

    def total_embeddings(self, _label: str) -> int:
        return self._total

    def upsert(
        self,
        label: str,
        embeddings: Sequence[np.ndarray],
        source_angles: Sequence[str | None],
    ) -> int:
        self._total += len(embeddings)
        self.upserts.append((label, len(embeddings), tuple(source_angles)))
        return self._total


def _jpeg_bytes() -> bytes:
    bgr = np.full((120, 160, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok
    return encoded.tobytes()


def test_enrollment_service_embeds_batch_and_updates_gallery():
    detector = FakeFaceDetector()
    identifier = FakeIdentifier()
    gallery = FakeGalleryStore()
    service = EnrollmentService(
        cfg=PipelineConfig(enrollment_max_width=80),
        face_detector=cast(FaceDetector, detector),
        identifier=cast(FaceIdentifier, identifier),
        gallery_store=cast(GalleryStore, gallery),
    )

    result = service.enroll(
        "Ada",
        [
            EnrollmentImage(
                name="front.jpg",
                field_name="front",
                content_type="image/jpeg",
                data=_jpeg_bytes(),
            ),
            EnrollmentImage(
                name="upLeft.jpg",
                field_name="files[]",
                content_type="image/jpeg",
                data=_jpeg_bytes(),
            ),
        ],
    )

    assert result.label == "Ada"
    assert result.embeddings_added == 2
    assert result.total_embeddings == 2
    assert [item.ok for item in result.per_image] == [True, True]
    assert [item.angle for item in result.per_image] == ["front", "upLeft"]
    assert gallery.upserts == [("Ada", 2, ("front", "upLeft"))]
    assert detector.frames[0].shape[1] == 80
    assert identifier.roi_shapes[0][:2] == (30, 40)


class FakeEnrollmentService:
    def __init__(self) -> None:
        self.enroll_calls: list[tuple[str, tuple[EnrollmentImage, ...]]] = []

    def enroll(self, label: str, images: Sequence[EnrollmentImage]) -> EnrollmentResult:
        self.enroll_calls.append((label, tuple(images)))
        return EnrollmentResult(
            label=label,
            embeddings_added=len(images),
            per_image=tuple(
                PerImageEnrollmentResult(
                    name=image.name,
                    angle=image.field_name,
                    ok=True,
                )
                for image in images
            ),
            total_embeddings=len(images),
        )

    def list_enrolled(self) -> list[EnrolledUserSummary]:
        return [
            EnrolledUserSummary(
                label="Ada",
                num_embeddings=7,
                updated_at=datetime(2026, 6, 19, tzinfo=timezone.utc),
            )
        ]

    def delete(self, label: str) -> DeleteEnrollmentResult:
        return DeleteEnrollmentResult(status="deleted", label=label)

    def gallery_user_count(self) -> int:
        return 1


class FakeTrackingService:
    def active(self) -> ActiveTrackingSnapshot:
        return ActiveTrackingSnapshot(
            faces=(ActiveFaceTrack(id=3, user="Ada", bbox=(1, 2, 3, 4)),),
            persons=(
                ActivePersonTrack(
                    id=7,
                    user="Ada",
                    bbox=(10, 20, 30, 40),
                    reid_ok=True,
                    confidence=0.91,
                ),
            ),
            ts=123.0,
        )

    def health(self) -> HealthStatus:
        return HealthStatus(status="ok", gallery_users=1, model_loaded=True)


def test_enrollment_api_accepts_one_multipart_batch():
    enrollment_service = FakeEnrollmentService()
    app = create_app(
        enrollment_service=cast(EnrollmentService, enrollment_service),
        tracking_service=cast(TrackingService, FakeTrackingService()),
    )
    client = TestClient(app)

    response = client.post(
        "/enroll",
        data={"label": "Ada"},
        files=[
            ("front", ("front.jpg", b"front-bytes", "image/jpeg")),
            ("upLeft", ("upLeft.jpg", b"upleft-bytes", "image/jpeg")),
        ],
    )

    assert response.status_code == 200
    assert response.json()["embeddings_added"] == 2
    assert len(enrollment_service.enroll_calls) == 1
    label, images = enrollment_service.enroll_calls[0]
    assert label == "Ada"
    assert [(image.field_name, image.name, image.data) for image in images] == [
        ("front", "front.jpg", b"front-bytes"),
        ("upLeft", "upLeft.jpg", b"upleft-bytes"),
    ]


def test_enrollment_api_lists_deletes_tracks_and_health():
    app = create_app(
        enrollment_service=cast(EnrollmentService, FakeEnrollmentService()),
        tracking_service=cast(TrackingService, FakeTrackingService()),
    )
    client = TestClient(app)

    assert client.get("/enrolled").json()["users"][0]["label"] == "Ada"
    assert client.delete("/enrolled/Ada").json() == {"status": "deleted", "label": "Ada"}

    tracking = client.get("/tracking/active").json()
    assert tracking["faces"][0] == {"id": 3, "user": "Ada", "bbox": [1, 2, 3, 4]}
    assert tracking["persons"][0]["confidence"] == 0.91
    assert client.get("/healthz").json() == {
        "status": "ok",
        "gallery_users": 1,
        "model_loaded": True,
    }


def test_tracking_service_reads_tracker_snapshots():
    class FakeTrackerManager:
        @property
        def snapshot(self) -> dict[int, dict[str, object]]:
            return {3: {"user": "Ada", "bbox": (1, 2, 3, 4)}}

        @property
        def person_snapshot(self) -> dict[int, dict[str, object]]:
            return {
                7: {
                    "user": "Ada",
                    "bbox": (10, 20, 30, 40),
                    "reid_ok": True,
                    "confidence": 0.91,
                }
            }

    class FakeEnrollment:
        def gallery_user_count(self) -> int:
            return 1

    class FakeFaceIdentifier:
        model_loaded = True

    service = TrackingService(
        tracker_manager=cast(TrackerManager, FakeTrackerManager()),
        enrollment_service=cast(EnrollmentService, FakeEnrollment()),
        identifier=cast(FaceIdentifier, FakeFaceIdentifier()),
    )

    snapshot = service.active()

    assert snapshot.faces[0].user == "Ada"
    assert snapshot.persons[0].bbox == (10, 20, 30, 40)
    assert service.health().gallery_users == 1


def test_cors_preflight_delete_returns_allow_origin():
    app = create_app(
        enrollment_service=cast(EnrollmentService, FakeEnrollmentService()),
        tracking_service=cast(TrackingService, FakeTrackingService()),
    )
    client = TestClient(app)

    response = client.options(
        "/enrolled/Ada",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "DELETE",
        },
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers


def test_cors_simple_get_returns_allow_origin():
    app = create_app(
        enrollment_service=cast(EnrollmentService, FakeEnrollmentService()),
        tracking_service=cast(TrackingService, FakeTrackingService()),
    )
    client = TestClient(app)

    response = client.get(
        "/enrolled",
        headers={"Origin": "http://localhost:3000"},
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
