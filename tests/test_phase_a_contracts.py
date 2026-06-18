from __future__ import annotations

import json
import pickle
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from config import PipelineConfig
from detection.face_detector import FaceDetector
from detection.fall_detector import FallDetector, FallTrackState
from detection.fall_geometry import fall_region_px
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


def test_fall_region_px_returns_centered_four_three_crop():
    assert fall_region_px(1280, 720) == (160, 0, 960, 720)

    portrait = fall_region_px(720, 1280)
    assert portrait == (0, 370, 720, 540)
    assert portrait[2] / portrait[3] == pytest.approx(4.0 / 3.0, abs=0.01)

    assert fall_region_px(1280, 720, aspect=0.0) == (0, 0, 1280, 720)
    assert fall_region_px(0, 720) == (0, 0, 0, 720)


@dataclass(frozen=True)
class FakePoseLandmark:
    x: float
    y: float
    visibility: float


class FakePoseDetector:
    def __init__(self, landmarks: list[FakePoseLandmark]) -> None:
        self._landmarks = landmarks
        self.images: list[np.ndarray] = []

    def process(self, image: np.ndarray) -> object:
        self.images.append(image)
        return types.SimpleNamespace(
            pose_landmarks=types.SimpleNamespace(landmark=self._landmarks),
        )


class FakeFallInterpreter:
    def allocate_tensors(self) -> None:
        pass

    def get_input_details(self) -> list[dict[str, object]]:
        return [{"index": 0}]

    def get_output_details(self) -> list[dict[str, object]]:
        return [{"index": 0}]

    def set_tensor(self, tensor_index: int, value: np.ndarray) -> None:
        pass

    def invoke(self) -> None:
        pass

    def get_tensor(self, tensor_index: int) -> np.ndarray:
        return np.asarray([[0.0]], dtype=np.float32)


def _fall_detector_for_crop_test(pose_detector: FakePoseDetector) -> FallDetector:
    import mediapipe as mp

    detector = FallDetector.__new__(FallDetector)
    detector._cfg = PipelineConfig()
    detector._mp_pose = mp.solutions.pose
    detector._pose_detector = pose_detector
    detector._SORT_KP_NAMES = sorted(
        [
            "Nose",
            "Left Eye",
            "Right Eye",
            "Left Ear",
            "Right Ear",
            "Left Shoulder",
            "Right Shoulder",
            "Left Elbow",
            "Right Elbow",
            "Left Wrist",
            "Right Wrist",
            "Left Hip",
            "Right Hip",
            "Left Knee",
            "Right Knee",
            "Left Ankle",
            "Right Ankle",
        ]
    )
    detector._SKP_IDX = {
        name: index for index, name in enumerate(detector._SORT_KP_NAMES)
    }
    detector._NUM_FEAT = len(detector._SORT_KP_NAMES) * 3
    detector._MP_LANDMARK_MAP = {
        detector._mp_pose.PoseLandmark.NOSE: "Nose",
        detector._mp_pose.PoseLandmark.LEFT_EYE: "Left Eye",
        detector._mp_pose.PoseLandmark.RIGHT_EYE: "Right Eye",
        detector._mp_pose.PoseLandmark.LEFT_EAR: "Left Ear",
        detector._mp_pose.PoseLandmark.RIGHT_EAR: "Right Ear",
        detector._mp_pose.PoseLandmark.LEFT_SHOULDER: "Left Shoulder",
        detector._mp_pose.PoseLandmark.RIGHT_SHOULDER: "Right Shoulder",
        detector._mp_pose.PoseLandmark.LEFT_ELBOW: "Left Elbow",
        detector._mp_pose.PoseLandmark.RIGHT_ELBOW: "Right Elbow",
        detector._mp_pose.PoseLandmark.LEFT_WRIST: "Left Wrist",
        detector._mp_pose.PoseLandmark.RIGHT_WRIST: "Right Wrist",
        detector._mp_pose.PoseLandmark.LEFT_HIP: "Left Hip",
        detector._mp_pose.PoseLandmark.RIGHT_HIP: "Right Hip",
        detector._mp_pose.PoseLandmark.LEFT_KNEE: "Left Knee",
        detector._mp_pose.PoseLandmark.RIGHT_KNEE: "Right Knee",
        detector._mp_pose.PoseLandmark.LEFT_ANKLE: "Left Ankle",
        detector._mp_pose.PoseLandmark.RIGHT_ANKLE: "Right Ankle",
    }
    detector._interpreter = FakeFallInterpreter()
    detector._input_details = [{"index": 0}]
    detector._output_details = [{"index": 0}]
    detector._track_states = {}
    detector.status = ""
    detector.confidence = 0.0
    return detector


def _fake_pose_landmarks() -> list[FakePoseLandmark]:
    import mediapipe as mp

    pose_enum = mp.solutions.pose.PoseLandmark
    landmarks = [FakePoseLandmark(0.0, 0.0, 0.0) for _ in range(33)]
    landmarks[pose_enum.NOSE.value] = FakePoseLandmark(0.25, 0.20, 1.0)
    landmarks[pose_enum.LEFT_SHOULDER.value] = FakePoseLandmark(0.45, 0.40, 1.0)
    landmarks[pose_enum.RIGHT_SHOULDER.value] = FakePoseLandmark(0.55, 0.40, 1.0)
    landmarks[pose_enum.LEFT_HIP.value] = FakePoseLandmark(0.45, 0.60, 1.0)
    landmarks[pose_enum.RIGHT_HIP.value] = FakePoseLandmark(0.55, 0.60, 1.0)
    landmarks[pose_enum.LEFT_WRIST.value] = FakePoseLandmark(0.50, 0.50, 1.0)
    landmarks[pose_enum.RIGHT_WRIST.value] = FakePoseLandmark(0.50, 0.50, 1.0)
    return landmarks


def test_fall_detector_feeds_pose_contiguous_four_three_crop():
    pose_detector = FakePoseDetector(_fake_pose_landmarks())
    detector = _fall_detector_for_crop_test(pose_detector)

    result = detector.process_track(
        track_id=3,
        rgb_frame=np.zeros((720, 1280, 3), dtype=np.uint8),
        bgr_frame=np.zeros((720, 1280, 3), dtype=np.uint8),
        bbox=(10, 20, 100, 200),
        frame_size=(1280, 720),
        current_time=10.0,
    )

    pose_input = pose_detector.images[0]
    assert pose_input.shape == (720, 960, 3)
    assert pose_input.flags.c_contiguous
    assert result.bbox == (10, 20, 100, 200)
    assert result.pose is not None
    assert result.pose.crop_bbox == (160, 0, 960, 720)
    assert result.pose.left_wrist == pytest.approx((640.0, 360.0))
    assert result.pose.right_wrist == pytest.approx((640.0, 360.0))
    assert result.pose.points is not None
    assert result.pose.points["Left Wrist"] == pytest.approx((0.5, 0.5, 1.0))
    assert result.pose.points["Nose"] == pytest.approx((0.3125, 0.2, 1.0))

    state = detector._track_states[3]
    model_features = state.feature_buffer[-1]
    nose_x, nose_y, _nose_conf = detector._kp_idx("Nose")

    expected_cropped_x_feature = (0.25 - 0.50) / 0.20
    pre_crop_full_frame_x = (160.0 + 0.25 * 960.0) / 1280.0
    pre_crop_x_feature = (pre_crop_full_frame_x - 0.50) / 0.20
    expected_y_feature = (0.20 - 0.60) / 0.20

    assert model_features[nose_x] == pytest.approx(expected_cropped_x_feature)
    assert abs(float(model_features[nose_x]) - pre_crop_x_feature) > 0.1
    assert model_features[nose_y] == pytest.approx(expected_y_feature)
    assert state.velocity_y_buffer[-1] == pytest.approx(0.50)


def test_fall_alert_payload_has_action_schema_without_top_level_confidence():
    detector = FallDetector.__new__(FallDetector)
    state = FallTrackState()
    state.fall_state = "on_floor"
    state.last_body_velocity = 0.041
    state.last_torso_angle_deg = 78.0
    state.confidence = 0.93456

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
    assert payload["diagnostics"]["fall_probability"] == 0.9346
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

    def fake_post(url: str, **kwargs: object):
        sent["url"] = url
        sent.update(kwargs)
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
    assert sent["timeout"] == 3.0
    assert "files" not in sent
    assert "data" not in sent
    assert "confidence" not in sent["json"]


def test_dispatcher_sends_fall_alert_as_multipart_with_screenshot(monkeypatch):
    sent: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object):
        sent["url"] = url
        sent.update(kwargs)
        return types.SimpleNamespace(status_code=200)

    payload = {
        "schema_version": "1.0",
        "event_type": "fall",
        "track_id": 3,
        "fall_state": "on_floor",
        "diagnostics": {"fall_probability": 0.93},
    }

    monkeypatch.setattr("events.dispatcher.requests.post", fake_post)
    dispatcher = EventDispatcher(PipelineConfig())
    try:
        dispatcher.send_fall_alert(payload, screenshot_bytes=b"jpeg-bytes")
    finally:
        dispatcher.shutdown()

    assert sent["url"].endswith("/vision/fall_alert")
    assert sent["timeout"] == 5.0
    assert "json" not in sent
    data = sent["data"]
    assert isinstance(data, dict)
    assert json.loads(data["payload"]) == payload
    files = sent["files"]
    assert isinstance(files, dict)
    assert files["screenshot"] == ("fall.jpg", b"jpeg-bytes", "image/jpeg")


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
    manager._cfg = PipelineConfig()
    manager._on_identity_event = None
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


def _identity_manager(events: list[dict[str, object]]) -> TrackerManager:
    manager = TrackerManager.__new__(TrackerManager)
    manager._cfg = PipelineConfig()
    manager._cfg.person_detection_interval = 0.15
    manager._cfg.tracker_track_buffer = 30
    manager._cfg.tracker_frame_rate = 30
    manager._cfg.person_identity_eviction_s = 5.0
    manager._lock = threading.Lock()
    manager._trackers = {}
    manager._person_tracks = {}
    manager._person_reid_enabled = True
    manager._on_identity_event = events.append
    return manager


def test_identity_event_emits_first_stamp_once_and_reidentify():
    events: list[dict[str, object]] = []
    manager = _identity_manager(events)
    manager._trackers = {5: {"user": "Ada", "bbox": (40, 40, 20, 20)}}
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
    manager.propagate_identity()
    manager._trackers[5]["user"] = "Bob"
    manager.propagate_identity()

    assert [event["event_type"] for event in events] == [
        "person_identified",
        "person_identified",
    ]
    assert events[0]["track_id"] == 7
    assert events[0]["user"] == "Ada"
    assert "dwell_s" not in events[0]
    assert events[1]["user"] == "Bob"
    assert manager._person_tracks[7].user == "Bob"
    assert manager._person_tracks[7].emitted_user == "Bob"
    assert manager._person_tracks[7].first_identified_at is not None


def test_identity_event_never_emits_or_downgrades_on_unknown_face():
    events: list[dict[str, object]] = []
    manager = _identity_manager(events)
    manager._trackers = {5: {"user": "Unknown", "bbox": (40, 40, 20, 20)}}
    manager._person_tracks = {
        7: PersonTrackRecord(
            track_id=7,
            bbox=(0, 0, 140, 220),
            user="Ada",
            reid_ok=True,
            last_seen=10.0,
            confidence=0.9,
            first_identified_at=4.0,
            emitted_user="Ada",
        )
    }

    manager.propagate_identity()

    assert events == []
    assert manager._person_tracks[7].user == "Ada"
    assert manager._person_tracks[7].emitted_user == "Ada"


def test_identity_event_emits_left_with_dwell_on_evict():
    events: list[dict[str, object]] = []
    manager = _identity_manager(events)
    manager._person_tracks = {
        7: PersonTrackRecord(
            track_id=7,
            bbox=(0, 0, 140, 220),
            user="Ada",
            reid_ok=True,
            last_seen=20.0,
            confidence=0.9,
            first_identified_at=12.0,
            emitted_user="Ada",
        )
    }

    manager._evict_stale_person_tracks(current_time=30.0)

    assert manager._person_tracks == {}
    assert len(events) == 1
    assert events[0]["event_type"] == "person_left"
    assert events[0]["track_id"] == 7
    assert events[0]["user"] == "Ada"
    assert events[0]["dwell_s"] == 8.0


def test_identified_person_track_survives_short_occlusion():
    events: list[dict[str, object]] = []
    manager = _identity_manager(events)
    manager._person_tracks = {
        7: PersonTrackRecord(
            track_id=7,
            bbox=(0, 0, 140, 220),
            user="Ada",
            reid_ok=True,
            last_seen=10.0,
            confidence=0.9,
            first_identified_at=4.0,
            emitted_user="Ada",
        )
    }

    manager._evict_stale_person_tracks(current_time=13.0)

    assert 7 in manager._person_tracks
    assert events == []


def test_identified_person_track_evicts_after_identity_ttl():
    events: list[dict[str, object]] = []
    manager = _identity_manager(events)
    manager._person_tracks = {
        7: PersonTrackRecord(
            track_id=7,
            bbox=(0, 0, 140, 220),
            user="Ada",
            reid_ok=True,
            last_seen=10.0,
            confidence=0.9,
            first_identified_at=4.0,
            emitted_user="Ada",
        )
    }

    manager._evict_stale_person_tracks(current_time=16.0)

    assert manager._person_tracks == {}
    assert len(events) == 1
    assert events[0]["event_type"] == "person_left"
    assert events[0]["track_id"] == 7
    assert events[0]["user"] == "Ada"
    assert events[0]["dwell_s"] == 6.0


def test_unidentified_person_track_keeps_short_eviction_ttl():
    events: list[dict[str, object]] = []
    manager = _identity_manager(events)
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

    manager._evict_stale_person_tracks(current_time=11.5)

    assert manager._person_tracks == {}
    assert events == []


def test_identified_person_redetection_preserves_session_without_reidentify():
    events: list[dict[str, object]] = []
    manager = _identity_manager(events)
    manager._person_tracks = {
        7: PersonTrackRecord(
            track_id=7,
            bbox=(0, 0, 140, 220),
            user="Ada",
            reid_ok=True,
            last_seen=10.0,
            confidence=0.9,
            first_identified_at=4.0,
            emitted_user="Ada",
        )
    }

    manager._evict_stale_person_tracks(current_time=13.0)
    manager._apply_person_results(
        np.asarray([[2, 3, 142, 223, 7, 0.91, 0, 0]], dtype=np.float32),
        current_time=13.1,
    )
    manager._trackers = {5: {"user": "Ada", "bbox": (40, 40, 20, 20)}}
    manager.propagate_identity()

    record = manager._person_tracks[7]
    assert record.user == "Ada"
    assert record.emitted_user == "Ada"
    assert record.first_identified_at == 4.0
    assert record.last_seen == 13.1
    assert events == []


def test_identity_event_does_not_emit_left_for_never_identified_track():
    events: list[dict[str, object]] = []
    manager = _identity_manager(events)
    manager._person_tracks = {
        7: PersonTrackRecord(
            track_id=7,
            bbox=(0, 0, 140, 220),
            user="Unknown",
            reid_ok=True,
            last_seen=20.0,
            confidence=0.9,
        )
    }

    manager._evict_stale_person_tracks(current_time=30.0)

    assert manager._person_tracks == {}
    assert events == []


def test_identity_event_callback_runs_outside_tracker_lock():
    events: list[dict[str, object]] = []
    manager = _identity_manager(events)

    def callback(event: dict[str, object]) -> None:
        lock_was_free = manager._lock.acquire(blocking=False)
        try:
            assert lock_was_free
            events.append(event)
        finally:
            if lock_was_free:
                manager._lock.release()

    manager._on_identity_event = callback
    manager._trackers = {5: {"user": "Ada", "bbox": (40, 40, 20, 20)}}
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

    assert len(events) == 1
    assert events[0]["event_type"] == "person_identified"


def test_dispatcher_identity_event_is_default_off(monkeypatch):
    monkeypatch.delenv("IDENTITY_EVENTS_ENABLED", raising=False)
    called = False

    def fake_post(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("events.dispatcher.requests.post", fake_post)
    cfg = PipelineConfig()
    cfg.identity_events_enabled = False
    dispatcher = EventDispatcher(cfg)
    try:
        dispatcher.send_identity_event({"event_type": "person_identified"})
    finally:
        dispatcher.shutdown()

    assert called is False


def test_dispatcher_identity_event_posts_when_enabled(monkeypatch):
    monkeypatch.delenv("IDENTITY_EVENTS_ENABLED", raising=False)
    sent: dict[str, object] = {}

    def fake_post(url: str, json: dict[str, object], timeout: float):
        sent["url"] = url
        sent["json"] = json
        sent["timeout"] = timeout
        return types.SimpleNamespace(status_code=200)

    monkeypatch.setattr("events.dispatcher.requests.post", fake_post)
    cfg = PipelineConfig()
    cfg.identity_events_enabled = True
    dispatcher = EventDispatcher(cfg)
    payload = {
        "schema_version": "1.0",
        "event_type": "person_identified",
        "track_id": 7,
        "user": "Ada",
        "zone": "living_room",
        "source": "mac_studio_living_room",
        "ts_wall": "2026-06-18T00:00:00+00:00",
    }
    try:
        dispatcher.send_identity_event(payload)
    finally:
        dispatcher.shutdown()

    assert sent["url"] == cfg.identity_event_url
    assert sent["json"] == payload
    assert sent["timeout"] == 1.0


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
