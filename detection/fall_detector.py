import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from loguru import logger

from config import PipelineConfig


class FallDetector:
    _STATE_IDLE = "idle"
    _STATE_MONITORING = "monitoring"

    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg

        self._mp_pose = mp.solutions.pose
        self._pose_detector = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._SORT_KP_NAMES = sorted([
            'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
            'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
            'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
            'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
        ])
        self._SKP_IDX = {n: i for i, n in enumerate(self._SORT_KP_NAMES)}
        self._NUM_FEAT = len(self._SORT_KP_NAMES) * 3  # 51

        self._MP_LANDMARK_MAP = {
            self._mp_pose.PoseLandmark.NOSE: 'Nose',
            self._mp_pose.PoseLandmark.LEFT_EYE: 'Left Eye',
            self._mp_pose.PoseLandmark.RIGHT_EYE: 'Right Eye',
            self._mp_pose.PoseLandmark.LEFT_EAR: 'Left Ear',
            self._mp_pose.PoseLandmark.RIGHT_EAR: 'Right Ear',
            self._mp_pose.PoseLandmark.LEFT_SHOULDER: 'Left Shoulder',
            self._mp_pose.PoseLandmark.RIGHT_SHOULDER: 'Right Shoulder',
            self._mp_pose.PoseLandmark.LEFT_ELBOW: 'Left Elbow',
            self._mp_pose.PoseLandmark.RIGHT_ELBOW: 'Right Elbow',
            self._mp_pose.PoseLandmark.LEFT_WRIST: 'Left Wrist',
            self._mp_pose.PoseLandmark.RIGHT_WRIST: 'Right Wrist',
            self._mp_pose.PoseLandmark.LEFT_HIP: 'Left Hip',
            self._mp_pose.PoseLandmark.RIGHT_HIP: 'Right Hip',
            self._mp_pose.PoseLandmark.LEFT_KNEE: 'Left Knee',
            self._mp_pose.PoseLandmark.RIGHT_KNEE: 'Right Knee',
            self._mp_pose.PoseLandmark.LEFT_ANKLE: 'Left Ankle',
            self._mp_pose.PoseLandmark.RIGHT_ANKLE: 'Right Ankle',
        }

        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._init_interpreter()

        self._feature_buffer = deque(maxlen=cfg.fall_input_timesteps)
        self._last_alert_time = 0.0
        self._velocity_y_buffer = deque(maxlen=cfg.velocity_window + 1)

        self._verification_state = self._STATE_IDLE
        self._verification_start = 0.0
        self._verification_prob = 0.0
        self._last_process_time = 0.0

        self.status = ""
        self.confidence = 0.0

    def _init_interpreter(self):
        if not Path(self._cfg.fall_model).exists():
            logger.error("Fall model not found at {}. Fall detection disabled.", self._cfg.fall_model)
            return

        TFLiteInterpreter = None
        try:
            from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
            except ImportError:
                try:
                    from tensorflow.lite import Interpreter as TFLiteInterpreter
                except ImportError:
                    logger.warning("No TFLite runtime found. Install: pip install ai-edge-litert")

        if TFLiteInterpreter:
            self._interpreter = TFLiteInterpreter(model_path=self._cfg.fall_model)
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            logger.info("Transformer model loaded: {} (input: {})",
                        self._cfg.fall_model, self._input_details[0]['shape'])
            logger.success("Fall detector active: Transformer (TFLite)")
        else:
            logger.error("Transformer model unavailable. Fall detection is DISABLED.")

    @property
    def enabled(self):
        return self._interpreter is not None

    @property
    def is_verifying(self):
        return self._verification_state == self._STATE_MONITORING

    @property
    def verification_elapsed(self):
        if self.is_verifying:
            return time.time() - self._verification_start
        return 0.0

    def _kp_idx(self, name):
        i = self._SKP_IDX[name]
        return i * 3, i * 3 + 1, i * 3 + 2

    def _compute_body_velocity(self):
        if len(self._velocity_y_buffer) < 2:
            return 0.0
        buf = list(self._velocity_y_buffer)
        velocities = [buf[i] - buf[i - 1] for i in range(1, len(buf))]
        return sum(velocities) / len(velocities)

    def _extract_and_normalize_pose(self, rgb_frame):
        results = self._pose_detector.process(rgb_frame)
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        features = np.zeros(self._NUM_FEAT, dtype=np.float32)

        for mp_enum, kp_name in self._MP_LANDMARK_MAP.items():
            idx = self._SKP_IDX[kp_name]
            lm = landmarks[mp_enum.value]
            features[idx * 3] = lm.x
            features[idx * 3 + 1] = lm.y
            features[idx * 3 + 2] = lm.visibility

        _ls_x, _ls_y, _ls_c = self._kp_idx('Left Shoulder')
        _rs_x, _rs_y, _rs_c = self._kp_idx('Right Shoulder')
        _lh_x, _lh_y, _lh_c = self._kp_idx('Left Hip')
        _rh_x, _rh_y, _rh_c = self._kp_idx('Right Hip')

        body_y_pts = []
        if features[_ls_c] > 0.3: body_y_pts.append(features[_ls_y])
        if features[_rs_c] > 0.3: body_y_pts.append(features[_rs_y])
        if features[_lh_c] > 0.3: body_y_pts.append(features[_lh_y])
        if features[_rh_c] > 0.3: body_y_pts.append(features[_rh_y])

        if body_y_pts:
            self._velocity_y_buffer.append(sum(body_y_pts) / len(body_y_pts))

        ls_x, ls_y, ls_c = self._kp_idx('Left Shoulder')
        rs_x, rs_y, rs_c = self._kp_idx('Right Shoulder')
        lh_x, lh_y, lh_c = self._kp_idx('Left Hip')
        rh_x, rh_y, rh_c = self._kp_idx('Right Hip')

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
            sh_y = (features[ls_y] + features[rs_y]) / 2
        elif valid_ls:
            sh_y = features[ls_y]
        elif valid_rs:
            sh_y = features[rs_y]
        else:
            sh_y = None

        torso_h = abs(sh_y - hip_y) if sh_y is not None else None
        can_scale = torso_h is not None and torso_h > 1e-5

        normalized = features.copy()
        for kp_name in self._SORT_KP_NAMES:
            xi, yi, _ = self._kp_idx(kp_name)
            normalized[xi] -= hip_x
            normalized[yi] -= hip_y
            if can_scale:
                normalized[xi] /= torso_h
                normalized[yi] /= torso_h

        return normalized

    def _check_post_fall_inactivity(self):
        elapsed = time.time() - self._verification_start
        if elapsed >= self._cfg.post_fall_wait:
            return "confirmed"
        body_velocity = self._compute_body_velocity()
        if body_velocity < -self._cfg.post_fall_move_threshold:
            return "cancelled"
        return "monitoring"

    def _run_detection(self, rgb_frame):
        if self._interpreter is None:
            return False, 0.0

        features = self._extract_and_normalize_pose(rgb_frame)
        if features is not None:
            self._feature_buffer.append(features)
        elif len(self._feature_buffer) > 0:
            self._feature_buffer.append(self._feature_buffer[-1])
        else:
            self._feature_buffer.append(np.zeros(self._NUM_FEAT, dtype=np.float32))

        if len(self._feature_buffer) < self._cfg.fall_input_timesteps:
            self.status = f"Buffering ({len(self._feature_buffer)}/{self._cfg.fall_input_timesteps})"
            self.confidence = 0.0
            return False, 0.0

        model_input = np.array(self._feature_buffer, dtype=np.float32)[np.newaxis, ...]
        try:
            self._interpreter.set_tensor(self._input_details[0]['index'], model_input)
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(self._output_details[0]['index'])
            fall_prob = float(output[0][0])
        except Exception as e:
            logger.error("Fall inference error: {}", e)
            return False, 0.0

        is_fall = fall_prob >= self._cfg.fall_confidence_threshold
        self.confidence = fall_prob

        if is_fall:
            body_velocity = self._compute_body_velocity()
            if body_velocity < self._cfg.min_fall_velocity:
                self.status = f"Slow motion (v={body_velocity:.4f})"
                logger.debug("Velocity too low ({:.4f} < {}) — likely sitting/bending (prob={:.2%})",
                             body_velocity, self._cfg.min_fall_velocity, fall_prob)
                return False, fall_prob

            self.status = "FALL DETECTED"
            now = time.time()
            if (now - self._last_alert_time) > self._cfg.fall_alert_cooldown:
                self._last_alert_time = now
                logger.warning("🚨 FALL DETECTED! prob={:.2%} velocity={:.4f}", fall_prob, body_velocity)
                return True, fall_prob
        else:
            self.status = "Standing"

        return False, fall_prob

    def _save_screenshot(self, bgr_frame):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int(time.time() * 1000) % 1000
        filename = f"fall_{timestamp}_{millis:03d}.jpg"
        filepath = self._cfg.screenshot_dir / filename
        try:
            cv2.imwrite(str(filepath), bgr_frame)
            logger.info("Screenshot saved: {}", filepath)
            return str(filepath)
        except Exception as e:
            logger.error("Screenshot save error: {}", e)
            return None

    def process_frame(self, rgb_frame, bgr_frame, current_time):
        """Process one frame through the fall detection FSM.

        Returns (confidence, screenshot_path) when a fall is confirmed, None otherwise.
        """
        if current_time - self._last_process_time < self._cfg.fall_frame_time:
            return None
        self._last_process_time = current_time

        if self._verification_state == self._STATE_IDLE:
            is_fall, fall_prob = self._run_detection(rgb_frame)
            if is_fall:
                self._verification_state = self._STATE_MONITORING
                self._verification_start = time.time()
                self._verification_prob = fall_prob
                self.status = f"⏳ Verifying fall (0.0/{self._cfg.post_fall_wait}s)..."
                logger.info("Fall candidate (prob={:.2%}) — starting {:.0f}s inactivity check",
                            fall_prob, self._cfg.post_fall_wait)

        elif self._verification_state == self._STATE_MONITORING:
            self._extract_and_normalize_pose(rgb_frame)
            elapsed = time.time() - self._verification_start
            result = self._check_post_fall_inactivity()

            if result == "confirmed":
                self._verification_state = self._STATE_IDLE
                self.status = "FALL CONFIRMED"
                self.confidence = self._verification_prob
                logger.critical("🚨 FALL CONFIRMED after {:.0f}s inactivity! prob={:.2%}",
                                self._cfg.post_fall_wait, self._verification_prob)
                screenshot_path = self._save_screenshot(bgr_frame)
                return self._verification_prob, screenshot_path
            elif result == "cancelled":
                self._verification_state = self._STATE_IDLE
                self.status = "Stumble (recovered)"
                self.confidence = 0.0
                logger.info("Person recovered after {:.1f}s — fall cancelled", elapsed)
            else:
                self.status = f"⏳ Verifying fall ({elapsed:.1f}/{self._cfg.post_fall_wait}s)..."

        return None
