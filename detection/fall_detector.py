from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, cast

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger

from config import PipelineConfig
from detection.fall_geometry import fall_region_px

BBox = tuple[int, int, int, int]
Point = tuple[float, float]
PosePoint = tuple[float, float, float]
FallStateName = Literal["idle", "falling", "on_floor", "recovered"]


class _TFLiteInterpreter(Protocol):
    def allocate_tensors(self) -> None:
        ...

    def get_input_details(self) -> list[dict[str, Any]]:
        ...

    def get_output_details(self) -> list[dict[str, Any]]:
        ...

    def set_tensor(self, tensor_index: int, value: np.ndarray) -> None:
        ...

    def invoke(self) -> None:
        ...

    def get_tensor(self, tensor_index: int) -> np.ndarray:
        ...


class FallDiagnostics(TypedDict):
    body_velocity: float
    torso_angle_deg: float
    inactivity_elapsed_s: float


class FallMedia(TypedDict):
    snapshot_path: str


class FallAlertPayload(TypedDict, total=False):
    schema_version: str
    event_type: str
    track_id: int
    fall_state: FallStateName
    bbox: list[int]
    centroid: list[float]
    frame_size: list[int]
    diagnostics: FallDiagnostics
    ts_wall: str
    ts_monotonic: float
    source: str
    zone: str
    media: FallMedia


@dataclass(frozen=True)
class PoseTrackData:
    track_id: int
    bbox: BBox
    crop_bbox: BBox
    left_wrist: Point | None
    right_wrist: Point | None
    points: dict[str, PosePoint] | None = None


@dataclass(frozen=True)
class FallProcessingResult:
    track_id: int
    bbox: BBox
    pose: PoseTrackData | None = None
    alert: FallAlertPayload | None = None


@dataclass
class _PoseExtraction:
    features: np.ndarray
    pose: PoseTrackData
    torso_angle_deg: float


@dataclass
class FallTrackState:
    feature_buffer: deque[np.ndarray] = field(default_factory=deque)
    velocity_y_buffer: deque[float] = field(default_factory=deque)
    verification_state: str = "idle"
    verification_start: float = 0.0
    verification_prob: float = 0.0
    last_alert_time: float = 0.0
    last_process_time: float = 0.0
    status: str = ""
    confidence: float = 0.0
    fall_state: FallStateName = "idle"
    last_body_velocity: float = 0.0
    last_torso_angle_deg: float = 0.0


class FallDetector:
    _STATE_IDLE = "idle"
    _STATE_MONITORING = "monitoring"

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg

        self._mp_pose = mp.solutions.pose
        self._pose_detector = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._SORT_KP_NAMES = sorted(
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
        self._SKP_IDX = {name: index for index, name in enumerate(self._SORT_KP_NAMES)}
        self._NUM_FEAT = len(self._SORT_KP_NAMES) * 3

        self._MP_LANDMARK_MAP = {
            self._mp_pose.PoseLandmark.NOSE: "Nose",
            self._mp_pose.PoseLandmark.LEFT_EYE: "Left Eye",
            self._mp_pose.PoseLandmark.RIGHT_EYE: "Right Eye",
            self._mp_pose.PoseLandmark.LEFT_EAR: "Left Ear",
            self._mp_pose.PoseLandmark.RIGHT_EAR: "Right Ear",
            self._mp_pose.PoseLandmark.LEFT_SHOULDER: "Left Shoulder",
            self._mp_pose.PoseLandmark.RIGHT_SHOULDER: "Right Shoulder",
            self._mp_pose.PoseLandmark.LEFT_ELBOW: "Left Elbow",
            self._mp_pose.PoseLandmark.RIGHT_ELBOW: "Right Elbow",
            self._mp_pose.PoseLandmark.LEFT_WRIST: "Left Wrist",
            self._mp_pose.PoseLandmark.RIGHT_WRIST: "Right Wrist",
            self._mp_pose.PoseLandmark.LEFT_HIP: "Left Hip",
            self._mp_pose.PoseLandmark.RIGHT_HIP: "Right Hip",
            self._mp_pose.PoseLandmark.LEFT_KNEE: "Left Knee",
            self._mp_pose.PoseLandmark.RIGHT_KNEE: "Right Knee",
            self._mp_pose.PoseLandmark.LEFT_ANKLE: "Left Ankle",
            self._mp_pose.PoseLandmark.RIGHT_ANKLE: "Right Ankle",
        }

        self._interpreter: _TFLiteInterpreter | None = None
        self._input_details: list[dict[str, Any]] | None = None
        self._output_details: list[dict[str, Any]] | None = None
        self._track_states: dict[int, FallTrackState] = {}
        self._init_interpreter()

        self.status = ""
        self.confidence = 0.0

    def _init_interpreter(self) -> None:
        if not Path(self._cfg.fall_model).exists():
            logger.error(
                "Fall model not found at {}. Fall detection disabled.",
                self._cfg.fall_model,
            )
            return

        interpreter_cls: Any = None
        try:
            from ai_edge_litert.interpreter import Interpreter

            interpreter_cls = Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter

                interpreter_cls = Interpreter
            except ImportError:
                try:
                    from tensorflow.lite import Interpreter

                    interpreter_cls = Interpreter
                except ImportError:
                    logger.warning("No TFLite runtime found. Install: pip install ai-edge-litert")

        if interpreter_cls:
            interpreter = cast(_TFLiteInterpreter, interpreter_cls(model_path=self._cfg.fall_model))
            interpreter.allocate_tensors()
            self._interpreter = interpreter
            self._input_details = interpreter.get_input_details()
            self._output_details = interpreter.get_output_details()
            logger.info(
                "Transformer model loaded: {} (input: {})",
                self._cfg.fall_model,
                self._input_details[0]["shape"],
            )
            logger.success("Fall detector active: Transformer (TFLite)")
        else:
            logger.error("Transformer model unavailable. Fall detection is DISABLED.")

    @property
    def enabled(self) -> bool:
        return self._interpreter is not None

    @property
    def is_verifying(self) -> bool:
        return any(
            state.verification_state == self._STATE_MONITORING
            for state in self._track_states.values()
        )

    @property
    def verification_elapsed(self) -> float:
        elapsed_values = [
            time.time() - state.verification_start
            for state in self._track_states.values()
            if state.verification_state == self._STATE_MONITORING
        ]
        return max(elapsed_values, default=0.0)

    @property
    def track_statuses(self) -> dict[int, dict[str, object]]:
        return {
            track_id: {
                "status": state.status,
                "confidence": state.confidence,
                "fall_state": state.fall_state,
                "verifying": state.verification_state == self._STATE_MONITORING,
                "elapsed": (
                    time.time() - state.verification_start
                    if state.verification_state == self._STATE_MONITORING
                    else 0.0
                ),
            }
            for track_id, state in self._track_states.items()
        }

    def sync_tracks(self, active_track_ids: set[int]) -> None:
        stale_track_ids = [
            track_id for track_id in self._track_states if track_id not in active_track_ids
        ]
        for track_id in stale_track_ids:
            del self._track_states[track_id]

    def process_track(
        self,
        track_id: int,
        rgb_frame: np.ndarray,
        bgr_frame: np.ndarray,
        bbox: BBox,
        frame_size: tuple[int, int],
        current_time: float,
    ) -> FallProcessingResult:
        state = self._state_for(track_id)
        if not self.enabled:
            return FallProcessingResult(track_id=track_id, bbox=bbox)

        if current_time - state.last_process_time < self._cfg.fall_frame_time:
            return FallProcessingResult(track_id=track_id, bbox=bbox)
        state.last_process_time = current_time

        # Restore the legacy 4:3 Pose geometry used by the frozen transformer.
        # The person bbox still only gates fall execution and remains the alert
        # bbox; the crop box maps Pose wrists back into full-frame coordinates.
        frame_w, frame_h = frame_size
        rx, ry, rw, rh = fall_region_px(frame_w, frame_h, aspect=self._cfg.fall_pose_aspect)
        pose_rgb = np.ascontiguousarray(rgb_frame[ry : ry + rh, rx : rx + rw])
        pose_box: BBox = (rx, ry, rw, rh)
        if pose_rgb.size == 0:
            state.status = "Empty frame"
            return FallProcessingResult(track_id=track_id, bbox=bbox)

        pose: PoseTrackData | None = None
        alert: FallAlertPayload | None = None

        if state.verification_state == self._STATE_IDLE:
            is_fall, fall_prob, pose = self._run_detection(
                track_id=track_id,
                rgb_crop=pose_rgb,
                person_bbox=bbox,
                crop_bbox=pose_box,
                frame_size=frame_size,
                state=state,
                current_time=current_time,
            )
            if is_fall:
                state.verification_state = self._STATE_MONITORING
                state.verification_start = time.time()
                state.verification_prob = fall_prob
                state.fall_state = "falling"
                state.status = f"Verifying fall (0.0/{self._cfg.post_fall_wait}s)..."
                logger.info(
                    "[person:{}] Fall candidate (prob={:.2%}) - starting {:.0f}s inactivity check",
                    track_id,
                    fall_prob,
                    self._cfg.post_fall_wait,
                )

        elif state.verification_state == self._STATE_MONITORING:
            extraction = self._extract_and_normalize_pose(
                rgb_crop=pose_rgb,
                track_id=track_id,
                person_bbox=bbox,
                crop_bbox=pose_box,
                frame_size=frame_size,
                state=state,
            )
            if extraction is not None:
                pose = extraction.pose

            elapsed = time.time() - state.verification_start
            result = self._check_post_fall_inactivity(state)

            if result == "confirmed":
                state.verification_state = self._STATE_IDLE
                state.fall_state = "on_floor"
                state.status = "FALL CONFIRMED"
                state.confidence = state.verification_prob
                logger.critical(
                    "[person:{}] FALL CONFIRMED after {:.0f}s inactivity! prob={:.2%}",
                    track_id,
                    self._cfg.post_fall_wait,
                    state.verification_prob,
                )
                screenshot_path = self._save_screenshot(bgr_frame)
                alert = self._build_alert_payload(
                    track_id=track_id,
                    bbox=bbox,
                    frame_size=frame_size,
                    state=state,
                    screenshot_path=screenshot_path,
                    inactivity_elapsed_s=elapsed,
                )
            elif result == "cancelled":
                state.verification_state = self._STATE_IDLE
                state.fall_state = "recovered"
                state.status = "Stumble (recovered)"
                state.confidence = 0.0
                logger.info(
                    "[person:{}] Person recovered after {:.1f}s - fall cancelled",
                    track_id,
                    elapsed,
                )
            else:
                state.status = f"Verifying fall ({elapsed:.1f}/{self._cfg.post_fall_wait}s)..."

        self._refresh_public_status()
        return FallProcessingResult(track_id=track_id, bbox=bbox, pose=pose, alert=alert)

    def _state_for(self, track_id: int) -> FallTrackState:
        if track_id not in self._track_states:
            self._track_states[track_id] = FallTrackState(
                feature_buffer=deque(maxlen=self._cfg.fall_input_timesteps),
                velocity_y_buffer=deque(maxlen=self._cfg.velocity_window + 1),
            )
        return self._track_states[track_id]

    def _kp_idx(self, name: str) -> tuple[int, int, int]:
        index = self._SKP_IDX[name]
        return index * 3, index * 3 + 1, index * 3 + 2

    def _compute_body_velocity(self, state: FallTrackState) -> float:
        if len(state.velocity_y_buffer) < 2:
            return 0.0
        values = list(state.velocity_y_buffer)
        velocities = [values[index] - values[index - 1] for index in range(1, len(values))]
        return sum(velocities) / len(velocities)

    def _extract_and_normalize_pose(
        self,
        rgb_crop: np.ndarray,
        track_id: int,
        person_bbox: BBox,
        crop_bbox: BBox,
        frame_size: tuple[int, int],
        state: FallTrackState,
    ) -> _PoseExtraction | None:
        results = self._pose_detector.process(rgb_crop)
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        features = np.zeros(self._NUM_FEAT, dtype=np.float32)

        for mp_enum, kp_name in self._MP_LANDMARK_MAP.items():
            index = self._SKP_IDX[kp_name]
            landmark = landmarks[mp_enum.value]
            features[index * 3] = landmark.x
            features[index * 3 + 1] = landmark.y
            features[index * 3 + 2] = landmark.visibility

        self._append_body_y(features, state)
        torso_angle_deg = self._compute_torso_angle(features)
        state.last_torso_angle_deg = torso_angle_deg
        pose = self._pose_track_data(track_id, person_bbox, crop_bbox, frame_size, landmarks)

        normalized = self._hip_centered_features(features)
        return _PoseExtraction(features=normalized, pose=pose, torso_angle_deg=torso_angle_deg)

    def _append_body_y(self, features: np.ndarray, state: FallTrackState) -> None:
        _ls_x, ls_y, ls_c = self._kp_idx("Left Shoulder")
        _rs_x, rs_y, rs_c = self._kp_idx("Right Shoulder")
        _lh_x, lh_y, lh_c = self._kp_idx("Left Hip")
        _rh_x, rh_y, rh_c = self._kp_idx("Right Hip")

        body_y_pts: list[float] = []
        if features[ls_c] > 0.3:
            body_y_pts.append(float(features[ls_y]))
        if features[rs_c] > 0.3:
            body_y_pts.append(float(features[rs_y]))
        if features[lh_c] > 0.3:
            body_y_pts.append(float(features[lh_y]))
        if features[rh_c] > 0.3:
            body_y_pts.append(float(features[rh_y]))

        if body_y_pts:
            state.velocity_y_buffer.append(sum(body_y_pts) / len(body_y_pts))
            state.last_body_velocity = self._compute_body_velocity(state)

    def _compute_torso_angle(self, features: np.ndarray) -> float:
        ls_x, ls_y, ls_c = self._kp_idx("Left Shoulder")
        rs_x, rs_y, rs_c = self._kp_idx("Right Shoulder")
        lh_x, lh_y, lh_c = self._kp_idx("Left Hip")
        rh_x, rh_y, rh_c = self._kp_idx("Right Hip")

        shoulder_pts: list[tuple[float, float]] = []
        hip_pts: list[tuple[float, float]] = []
        if features[ls_c] > 0.3:
            shoulder_pts.append((float(features[ls_x]), float(features[ls_y])))
        if features[rs_c] > 0.3:
            shoulder_pts.append((float(features[rs_x]), float(features[rs_y])))
        if features[lh_c] > 0.3:
            hip_pts.append((float(features[lh_x]), float(features[lh_y])))
        if features[rh_c] > 0.3:
            hip_pts.append((float(features[rh_x]), float(features[rh_y])))
        if not shoulder_pts or not hip_pts:
            return 0.0

        shoulder_x = sum(point[0] for point in shoulder_pts) / len(shoulder_pts)
        shoulder_y = sum(point[1] for point in shoulder_pts) / len(shoulder_pts)
        hip_x = sum(point[0] for point in hip_pts) / len(hip_pts)
        hip_y = sum(point[1] for point in hip_pts) / len(hip_pts)
        dx = shoulder_x - hip_x
        dy = shoulder_y - hip_y
        return abs(math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-6))))

    def _hip_centered_features(self, features: np.ndarray) -> np.ndarray:
        ls_x, ls_y, ls_c = self._kp_idx("Left Shoulder")
        rs_x, rs_y, rs_c = self._kp_idx("Right Shoulder")
        lh_x, lh_y, lh_c = self._kp_idx("Left Hip")
        rh_x, rh_y, rh_c = self._kp_idx("Right Hip")

        valid_lh = features[lh_c] > 0.3
        valid_rh = features[rh_c] > 0.3
        if valid_lh and valid_rh:
            hip_x = (features[lh_x] + features[rh_x]) / 2
            hip_y = (features[lh_y] + features[rh_y]) / 2
        elif valid_lh:
            hip_x, hip_y = features[lh_x], features[lh_y]
        elif valid_rh:
            hip_x, hip_y = features[rh_x], features[rh_y]
        else:
            return features

        valid_ls = features[ls_c] > 0.3
        valid_rs = features[rs_c] > 0.3
        if valid_ls and valid_rs:
            shoulder_y = (features[ls_y] + features[rs_y]) / 2
        elif valid_ls:
            shoulder_y = features[ls_y]
        elif valid_rs:
            shoulder_y = features[rs_y]
        else:
            shoulder_y = None

        torso_h = abs(shoulder_y - hip_y) if shoulder_y is not None else None
        can_scale = torso_h is not None and torso_h > 1e-5

        normalized = features.copy()
        for kp_name in self._SORT_KP_NAMES:
            xi, yi, _confidence_index = self._kp_idx(kp_name)
            normalized[xi] -= hip_x
            normalized[yi] -= hip_y
            if can_scale:
                normalized[xi] /= torso_h
                normalized[yi] /= torso_h

        return normalized

    def _pose_track_data(
        self,
        track_id: int,
        person_bbox: BBox,
        crop_bbox: BBox,
        frame_size: tuple[int, int],
        landmarks: list[object],
    ) -> PoseTrackData:
        left_wrist = self._landmark_point(
            landmarks[self._mp_pose.PoseLandmark.LEFT_WRIST.value],
            crop_bbox,
        )
        right_wrist = self._landmark_point(
            landmarks[self._mp_pose.PoseLandmark.RIGHT_WRIST.value],
            crop_bbox,
        )
        return PoseTrackData(
            track_id=track_id,
            bbox=person_bbox,
            crop_bbox=crop_bbox,
            left_wrist=left_wrist,
            right_wrist=right_wrist,
            points=self._pose_points(landmarks, crop_bbox, frame_size),
        )

    @staticmethod
    def _landmark_point(landmark: object, crop_bbox: BBox) -> Point | None:
        visibility = float(getattr(landmark, "visibility", 0.0))
        if visibility <= 0.3:
            return None
        crop_x, crop_y, crop_w, crop_h = crop_bbox
        x = crop_x + float(getattr(landmark, "x")) * crop_w
        y = crop_y + float(getattr(landmark, "y")) * crop_h
        return x, y

    def _pose_points(
        self,
        landmarks: list[object],
        crop_bbox: BBox,
        frame_size: tuple[int, int],
    ) -> dict[str, PosePoint]:
        frame_w, frame_h = frame_size
        if frame_w <= 0 or frame_h <= 0:
            return {}

        crop_x, crop_y, crop_w, crop_h = crop_bbox
        points: dict[str, PosePoint] = {}
        for mp_enum, kp_name in self._MP_LANDMARK_MAP.items():
            landmark = landmarks[mp_enum.value]
            full_x = crop_x + float(getattr(landmark, "x")) * crop_w
            full_y = crop_y + float(getattr(landmark, "y")) * crop_h
            visibility = float(getattr(landmark, "visibility", 0.0))
            points[kp_name] = (full_x / frame_w, full_y / frame_h, visibility)
        return points

    def _run_detection(
        self,
        track_id: int,
        rgb_crop: np.ndarray,
        person_bbox: BBox,
        crop_bbox: BBox,
        frame_size: tuple[int, int],
        state: FallTrackState,
        current_time: float,
    ) -> tuple[bool, float, PoseTrackData | None]:
        interpreter = self._interpreter
        input_details = self._input_details
        output_details = self._output_details
        if interpreter is None or input_details is None or output_details is None:
            return False, 0.0, None

        extraction = self._extract_and_normalize_pose(
            rgb_crop=rgb_crop,
            track_id=track_id,
            person_bbox=person_bbox,
            crop_bbox=crop_bbox,
            frame_size=frame_size,
            state=state,
        )
        pose = extraction.pose if extraction is not None else None
        if extraction is not None:
            state.feature_buffer.append(extraction.features)
        elif len(state.feature_buffer) > 0:
            state.feature_buffer.append(state.feature_buffer[-1])
        else:
            state.feature_buffer.append(np.zeros(self._NUM_FEAT, dtype=np.float32))

        if len(state.feature_buffer) < self._cfg.fall_input_timesteps:
            state.status = (
                f"Buffering ({len(state.feature_buffer)}/{self._cfg.fall_input_timesteps})"
            )
            state.confidence = 0.0
            return False, 0.0, pose

        model_input = np.array(state.feature_buffer, dtype=np.float32)[np.newaxis, ...]
        try:
            interpreter.set_tensor(int(input_details[0]["index"]), model_input)
            interpreter.invoke()
            output = interpreter.get_tensor(int(output_details[0]["index"]))
            fall_prob = float(output[0][0])
        except Exception as exc:
            logger.error("[person:{}] Fall inference error: {}", track_id, exc)
            return False, 0.0, pose

        is_fall = fall_prob >= self._cfg.fall_confidence_threshold
        state.confidence = fall_prob

        if is_fall:
            body_velocity = self._compute_body_velocity(state)
            state.last_body_velocity = body_velocity
            if body_velocity < self._cfg.min_fall_velocity:
                state.status = f"Slow motion (v={body_velocity:.4f})"
                logger.debug(
                    "[person:{}] Velocity too low ({:.4f} < {}) - likely "
                    "sitting/bending (prob={:.2%})",
                    track_id,
                    body_velocity,
                    self._cfg.min_fall_velocity,
                    fall_prob,
                )
                return False, fall_prob, pose

            state.status = "FALL DETECTED"
            if (current_time - state.last_alert_time) > self._cfg.fall_alert_cooldown:
                state.last_alert_time = current_time
                logger.warning(
                    "[person:{}] FALL DETECTED! prob={:.2%} velocity={:.4f}",
                    track_id,
                    fall_prob,
                    body_velocity,
                )
                return True, fall_prob, pose
        else:
            state.status = "Standing"
            state.fall_state = "idle"

        return False, fall_prob, pose

    def _check_post_fall_inactivity(self, state: FallTrackState) -> str:
        elapsed = time.time() - state.verification_start
        if elapsed >= self._cfg.post_fall_wait:
            return "confirmed"
        body_velocity = self._compute_body_velocity(state)
        state.last_body_velocity = body_velocity
        if body_velocity < -self._cfg.post_fall_move_threshold:
            return "cancelled"
        return "monitoring"

    def _build_alert_payload(
        self,
        track_id: int,
        bbox: BBox,
        frame_size: tuple[int, int],
        state: FallTrackState,
        screenshot_path: str | None,
        inactivity_elapsed_s: float,
    ) -> FallAlertPayload:
        x, y, w, h = bbox
        payload: FallAlertPayload = {
            "schema_version": "1.0",
            "event_type": "fall",
            "track_id": track_id,
            "fall_state": state.fall_state,
            "bbox": [x, y, w, h],
            "centroid": [x + w / 2.0, y + h / 2.0],
            "frame_size": [frame_size[0], frame_size[1]],
            "diagnostics": {
                "body_velocity": float(state.last_body_velocity),
                "torso_angle_deg": float(state.last_torso_angle_deg),
                "inactivity_elapsed_s": float(inactivity_elapsed_s),
            },
            "ts_wall": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "ts_monotonic": time.monotonic(),
            "source": "mac_studio_living_room",
            "zone": "living_room",
        }
        if screenshot_path is not None:
            # TODO(Phase B): snapshot_path is a Mac-local filesystem path
            # (data/logs/screenshots/...). The Pi backend cannot fetch it. Phase B
            # should upload/serve the image (e.g. attach bytes or a reachable URL).
            payload["media"] = {"snapshot_path": screenshot_path}
        return payload

    def _save_screenshot(self, bgr_frame: np.ndarray) -> str | None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int(time.time() * 1000) % 1000
        filename = f"fall_{timestamp}_{millis:03d}.jpg"
        filepath = self._cfg.screenshot_dir / filename
        try:
            cv2.imwrite(str(filepath), bgr_frame)
            logger.info("Screenshot saved: {}", filepath)
            return str(filepath)
        except Exception as exc:
            logger.error("Screenshot save error: {}", exc)
            return None

    def _refresh_public_status(self) -> None:
        if not self._track_states:
            self.status = ""
            self.confidence = 0.0
            return
        states = list(self._track_states.values())
        self.confidence = max(state.confidence for state in states)
        priority = {"on_floor": 3, "falling": 2, "recovered": 1, "idle": 0}
        selected = max(states, key=lambda state: priority[state.fall_state])
        self.status = selected.status
