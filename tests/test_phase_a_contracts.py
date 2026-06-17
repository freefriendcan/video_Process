from __future__ import annotations

import pickle
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from config import PipelineConfig
from detection.face_detector import FaceDetector
from detection.fall_detector import FallDetector, FallTrackState
from detection.onnx_runtime import LetterboxMeta, NormalizedKeypoint, OnnxModel, YoloDetection
from detection.person_detector import PersonDetector
from events.dispatcher import EventDispatcher
from identification.face_identifier import FaceIdentifier
from main import VisionPipeline
from tracking.tracker_manager import PersonTrackRecord, TrackerManager


def test_onnx_model_uses_configured_coreml_provider(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")

    class FakeInput:
        name = "images"

    class FakeSession:
        def __init__(self, path: str, providers: list[str]) -> None:
            self.path = path
            self._providers = providers

        def get_inputs(self) -> list[FakeInput]:
            return [FakeInput()]

        def get_providers(self) -> list[str]:
            return self._providers

        def run(self, _outputs: object, _feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            return [np.zeros((1, 1, 84), dtype=np.float32)]

    fake_ort = types.SimpleNamespace(InferenceSession=FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    model = OnnxModel(str(model_path), ("CoreMLExecutionProvider", "CPUExecutionProvider"))

    assert model.providers == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_onnx_postprocess_filters_person_class_and_nms():
    output = np.zeros((1, 3, 84), dtype=np.float32)
    output[0, 0, :4] = [100.0, 100.0, 60.0, 80.0]
    output[0, 0, 4] = 0.95
    output[0, 1, :4] = [102.0, 102.0, 60.0, 80.0]
    output[0, 1, 4] = 0.80
    output[0, 2, :4] = [300.0, 300.0, 60.0, 80.0]
    output[0, 2, 5] = 0.99

    detections = OnnxModel.postprocess_yolo(
        output=output,
        meta=LetterboxMeta(
            original_width=640,
            original_height=640,
            input_size=640,
            scale=1.0,
            pad_x=0.0,
            pad_y=0.0,
        ),
        conf_threshold=0.5,
        nms_iou=0.45,
        class_id=0,
    )

    assert len(detections) == 1
    assert detections[0].bbox == (70, 60, 60, 80)
    assert detections[0].class_id == 0


def test_person_detector_returns_bbox_with_confidence():
    class FakeModel:
        def infer_rgb(self, _rgb_frame: np.ndarray):
            return np.empty((0,), dtype=np.float32), object()

        def postprocess_yolo(self, **_kwargs: object) -> list[YoloDetection]:
            return [YoloDetection(bbox=(10, 20, 30, 40), score=0.87, class_id=0)]

    detector = PersonDetector.__new__(PersonDetector)
    detector._cfg = PipelineConfig()
    detector._model = FakeModel()

    detections = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))

    assert detections == [((10, 20, 30, 40), 0.87)]


def test_face_detector_returns_bbox_confidence_and_keypoints():
    keypoints = (
        (20.0, 30.0, 1.0),
        (40.0, 30.0, 1.0),
        (30.0, 45.0, 1.0),
        (22.0, 58.0, 1.0),
        (38.0, 58.0, 1.0),
    )

    class FakeModel:
        def infer_rgb(self, _rgb_frame: np.ndarray):
            return np.empty((0,), dtype=np.float32), object()

        def postprocess_yolo(self, **_kwargs: object) -> list[YoloDetection]:
            return [
                YoloDetection(
                    bbox=(10, 20, 40, 60),
                    score=0.91,
                    class_id=0,
                    keypoints=keypoints,
                )
            ]

    detector = FaceDetector.__new__(FaceDetector)
    detector._cfg = PipelineConfig()
    detector._model = FakeModel()

    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(detections) == 1
    x, y, w, h, score, detection_keypoints = detections[0]
    assert (x, y, w, h) == (0, 0, 60, 100)
    assert score == 0.91
    assert detection_keypoints[0] == NormalizedKeypoint(x=0.2, y=0.3)


class FakeMotTracker:
    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = outputs
        self.calls: list[np.ndarray] = []

    def update(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray | None = None,
    ) -> np.ndarray:
        self.calls.append(dets.copy())
        if self.outputs:
            return self.outputs.pop(0)
        return np.empty((0, 8), dtype=np.float32)


def _face_manager(fake_mot: FakeMotTracker) -> TrackerManager:
    manager = TrackerManager.__new__(TrackerManager)
    manager._cfg = PipelineConfig()
    manager._lock = threading.Lock()
    manager._trackers = {}
    manager._face_mot = fake_mot
    return manager


def test_face_bytetrack_new_track_shape_and_keypoint_attachment():
    keypoints = (
        NormalizedKeypoint(x=0.2, y=0.3),
        NormalizedKeypoint(x=0.4, y=0.3),
    )
    fake_mot = FakeMotTracker(
        [
            np.asarray([[10, 20, 40, 60, 7, 0.88, 0, 0]], dtype=np.float32),
        ]
    )
    manager = _face_manager(fake_mot)

    new_faces, active = manager.update_face_tracks(
        [(10, 20, 30, 40, 0.88, keypoints)],
        np.zeros((100, 100, 3), dtype=np.uint8),
        current_time=10.0,
    )

    np.testing.assert_allclose(
        fake_mot.calls[0],
        np.asarray([[10.0, 20.0, 40.0, 60.0, 0.88, 0.0]], dtype=np.float32),
    )
    assert new_faces == [{"tracker_id": 7, "bbox": (10, 20, 30, 40), "keypoints": keypoints}]
    assert active[0]["id"] == 7
    assert active[0]["user"] == "Identifying..."
    assert active[0]["detection_keypoints"] == keypoints


def test_face_bytetrack_preserves_identity_across_updates():
    keypoints = (NormalizedKeypoint(x=0.2, y=0.3),)
    fake_mot = FakeMotTracker(
        [
            np.asarray([[10, 20, 40, 60, 7, 0.88, 0, 0]], dtype=np.float32),
            np.asarray([[12, 22, 42, 62, 7, 0.86, 0, 0]], dtype=np.float32),
        ]
    )
    manager = _face_manager(fake_mot)

    manager.update_face_tracks(
        [(10, 20, 30, 40, 0.88, keypoints)],
        np.zeros((100, 100, 3), dtype=np.uint8),
        current_time=10.0,
    )
    manager.set_user(7, "Ada", retry_count=0)
    new_faces, active = manager.update_face_tracks(
        [(12, 22, 30, 40, 0.86, keypoints)],
        np.zeros((100, 100, 3), dtype=np.uint8),
        current_time=10.1,
    )

    assert new_faces == []
    assert active[0]["id"] == 7
    assert active[0]["user"] == "Ada"
    assert active[0]["bbox"] == (12, 22, 30, 40)


def test_face_bytetrack_keypoints_fall_back_to_iou_when_det_index_absent():
    first_keypoints = (NormalizedKeypoint(x=0.1, y=0.1),)
    second_keypoints = (NormalizedKeypoint(x=0.8, y=0.8),)
    fake_mot = FakeMotTracker(
        [
            np.asarray([[100, 100, 150, 150, 9, 0.8, 0, -1]], dtype=np.float32),
        ]
    )
    manager = _face_manager(fake_mot)

    new_faces, active = manager.update_face_tracks(
        [
            (10, 10, 30, 30, 0.7, first_keypoints),
            (98, 99, 55, 52, 0.8, second_keypoints),
        ],
        np.zeros((200, 200, 3), dtype=np.uint8),
        current_time=11.0,
    )

    assert new_faces[0]["keypoints"] == second_keypoints
    assert active[0]["detection_keypoints"] == second_keypoints


def test_face_bytetrack_evicts_stale_tracks():
    fake_mot = FakeMotTracker([np.empty((0, 8), dtype=np.float32)])
    manager = _face_manager(fake_mot)
    manager._cfg.face_track_buffer = 1
    manager._cfg.face_tracker_frame_rate = 1
    manager._cfg.face_detection_interval = 0.1
    manager._trackers = {
        4: {
            "user": "Ada",
            "bbox": (1, 2, 3, 4),
            "retry_count": 0,
            "last_identify_time": 1.0,
            "last_json_time": 1.0,
            "best_quality_score": 0,
            "detection_keypoints": None,
            "last_seen": 0.0,
            "is_new": False,
        }
    }

    _new_faces, active = manager.update_face_tracks(
        [],
        np.zeros((100, 100, 3), dtype=np.uint8),
        current_time=2.1,
    )

    assert active == []
    assert manager.exists(4) is False


def test_fall_alert_payload_has_action_schema_without_top_level_confidence():
    detector = FallDetector.__new__(FallDetector)
    state = FallTrackState()
    state.fall_state = "on_floor"
    state.last_body_velocity = 0.041
    state.last_torso_angle_deg = 78.0

    payload = detector._build_alert_payload(
        track_id=7,
        bbox=(10, 20, 100, 200),
        frame_size=(1280, 720),
        state=state,
        screenshot_path="data/logs/screenshots/fall_7.jpg",
        inactivity_elapsed_s=3.0,
    )

    assert payload["schema_version"] == "1.0"
    assert payload["event_type"] == "fall"
    assert payload["track_id"] == 7
    assert payload["fall_state"] == "on_floor"
    assert payload["bbox"] == [10, 20, 100, 200]
    assert payload["centroid"] == [60.0, 120.0]
    assert payload["frame_size"] == [1280, 720]
    assert payload["diagnostics"]["body_velocity"] == 0.041
    assert payload["media"]["snapshot_path"] == "data/logs/screenshots/fall_7.jpg"
    assert "confidence" not in payload


def test_fall_track_states_are_independent():
    detector = FallDetector.__new__(FallDetector)
    detector._cfg = PipelineConfig()
    detector._track_states = {}

    first = detector._state_for(1)
    second = detector._state_for(2)
    first.fall_state = "falling"
    first.feature_buffer.append(np.ones(51, dtype=np.float32))

    assert detector._track_states[1].fall_state == "falling"
    assert detector._track_states[2].fall_state == "idle"
    assert len(second.feature_buffer) == 0


def test_dispatcher_sends_fall_alert_as_json_without_confidence(monkeypatch):
    sent: dict[str, object] = {}

    def fake_post(url: str, json: dict[str, object], timeout: float):
        sent["url"] = url
        sent["json"] = json
        sent["timeout"] = timeout
        return types.SimpleNamespace(status_code=200)

    monkeypatch.setattr("events.dispatcher.requests.post", fake_post)
    dispatcher = EventDispatcher(PipelineConfig())
    try:
        dispatcher.send_fall_alert(
            {
                "schema_version": "1.0",
                "event_type": "fall",
                "track_id": 3,
                "fall_state": "on_floor",
            }
        )
    finally:
        dispatcher.shutdown()

    assert sent["url"].endswith("/vision/fall_alert")
    assert sent["json"]["track_id"] == 3
    assert "confidence" not in sent["json"]


def test_gesture_payload_contract_unchanged(monkeypatch):
    sent: dict[str, object] = {}

    class FakeSession:
        def post(self, url: str, json: dict[str, object], timeout: float) -> None:
            sent["url"] = url
            sent["json"] = json
            sent["timeout"] = timeout

    dispatcher = EventDispatcher(PipelineConfig())
    dispatcher._presence_session = FakeSession()
    try:
        dispatcher.send_gesture_event("Victory", 1.25, "ogulcan")
    finally:
        dispatcher.shutdown()

    assert sent["url"].endswith("/vision/gesture")
    assert list(sent["json"].keys()) == ["gesture", "user", "location", "timestamp", "duration"]
    assert sent["json"]["gesture"] == "Victory"
    assert sent["json"]["user"] == "ogulcan"
    assert sent["json"]["location"] == "living_room"
    assert sent["json"]["duration"] == 1.25


def test_face_identifier_loads_and_normalizes_gallery(tmp_path: Path):
    gallery_path = tmp_path / "faces.pkl"
    with gallery_path.open("wb") as handle:
        pickle.dump({"person": np.ones((2, 512), dtype=np.float64)}, handle)

    identifier = FaceIdentifier.__new__(FaceIdentifier)
    gallery = identifier._load_gallery(gallery_path)

    assert gallery["person"].dtype == np.float32
    assert gallery["person"].shape == (2, 512)
    assert np.allclose(np.linalg.norm(gallery["person"], axis=1), 1.0)


def test_face_identifier_converts_full_frame_keypoints_to_crop_space():
    keypoints = (
        NormalizedKeypoint(x=0.25, y=0.25),
        NormalizedKeypoint(x=0.75, y=0.25),
    )

    cropped = FaceIdentifier.crop_keypoints(
        keypoints=keypoints,
        face_bbox=(100, 50, 200, 100),
        frame_size=(400, 200),
    )

    assert cropped[0] == NormalizedKeypoint(x=0.0, y=0.0)
    assert cropped[1] == NormalizedKeypoint(x=1.0, y=0.0)


def test_ir_mode_skips_local_identity_dispatch():
    class StubTracker:
        def __init__(self) -> None:
            self.users: list[tuple[int, str]] = []

        def set_user(
            self,
            tracker_id: int,
            user: str,
            retry_count: int | None = None,
            increment_retry: bool = False,
        ) -> None:
            self.users.append((tracker_id, user))

    class StubDispatcher:
        def __init__(self) -> None:
            self.submitted = False

        def submit(self, _fn: object, *_args: object, **_kwargs: object) -> None:
            self.submitted = True

    pipeline = VisionPipeline.__new__(VisionPipeline)
    tracker = StubTracker()
    dispatcher = StubDispatcher()
    pipeline._tracker_mgr = tracker
    pipeline._dispatcher = dispatcher

    pipeline._identify_new_faces(
        new_faces=(
            {
                "tracker_id": 9,
                "bbox": (10, 10, 100, 120),
                "keypoints": (),
            },
        ),
        frame=np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_size=(1280, 720),
        ir_mode=True,
    )

    assert dispatcher.submitted is False
    assert tracker.users == [(9, "Unknown")]


def test_tracker_manager_propagates_identified_face_to_containing_person():
    manager = TrackerManager.__new__(TrackerManager)
    manager._lock = threading.Lock()
    manager._trackers = {
        5: {
            "user": "Ada",
            "bbox": (40, 40, 20, 20),
        }
    }
    manager._person_tracks = {
        7: PersonTrackRecord(
            track_id=7,
            bbox=(0, 0, 140, 220),
            user="Unknown",
            reid_ok=True,
            last_seen=10.0,
            confidence=0.9,
        )
    }

    manager.propagate_identity()

    assert manager._person_tracks[7].user == "Ada"


def test_face_postprocess_parses_five_landmarks_from_20_channel_output():
    """YOLO11-pose face output is (1, 20, 8400) = 4 bbox + 1 score + 5*3 kps.

    Asserts the postprocessor extracts exactly 5 keypoints per detection from
    the 20-channel layout (regression guard for the landmark contract).
    """
    num_channels = 4 + 1 + 5 * 3  # == 20
    output = np.zeros((1, num_channels, 3), dtype=np.float32)
    # One confident face at letterbox center with 5 landmarks.
    output[0, :4, 0] = [320.0, 320.0, 80.0, 100.0]
    output[0, 4, 0] = 0.95
    landmark_xy = [(300, 300), (340, 300), (320, 330), (305, 355), (335, 355)]
    for i, (lx, ly) in enumerate(landmark_xy):
        output[0, 5 + i * 3, 0] = lx
        output[0, 5 + i * 3 + 1, 0] = ly
        output[0, 5 + i * 3 + 2, 0] = 1.0

    detections = OnnxModel.postprocess_yolo(
        output=output,
        meta=LetterboxMeta(
            original_width=640,
            original_height=640,
            input_size=640,
            scale=1.0,
            pad_x=0.0,
            pad_y=0.0,
        ),
        conf_threshold=0.5,
        nms_iou=0.45,
        keypoint_count=5,
    )

    assert len(detections) == 1
    assert len(detections[0].keypoints) == 5


@pytest.mark.skipif(
    not Path(PipelineConfig.face_det_model).exists(),
    reason="yolov11n-face.onnx not present (large, gitignored)",
)
def test_face_onnx_model_emits_20_channel_landmark_output():
    """The real face ONNX must expose 20 output channels (5 landmarks)."""
    import onnx

    model = onnx.load(PipelineConfig.face_det_model)
    out = model.graph.output[0]
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    # Expected (1, 20, 8400); the channel dim must be 20.
    assert 20 in dims, f"face ONNX output dims {dims} lack the 20-channel layout"
