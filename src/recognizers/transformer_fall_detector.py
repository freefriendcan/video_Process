"""Transformer-based fall detection using MediaPipe Pose + TFLite model.

Uses a pre-trained Transformer model from punpayut/Fall-Detection that
classifies 30-frame pose keypoint sequences as 'fall' or 'no_fall'.
Achieves 94.9% F1-score vs our legacy geometric heuristic approach.

Architecture:
    Frame → MediaPipe Pose → 17 keypoints (normalized x, y, visibility)
    → Skeleton normalization (hip-centered, torso-scaled)
    → 30-frame sliding window
    → TFLite Transformer inference
    → Fall probability [0.0–1.0]
"""

import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from ..config.settings import Config
from ..detectors.mediapipe_pose import (
    KEYPOINT_NAME_TO_IDX,
    NUM_FEATURES,
    SORTED_KEYPOINT_NAMES,
    MediaPipePoseDetector,
)


def _get_kpt_feature_indices(keypoint_name: str) -> tuple[int, int, int]:
    """Get (x_idx, y_idx, vis_idx) in the sorted feature vector."""
    idx = KEYPOINT_NAME_TO_IDX[keypoint_name]
    return idx * 3, idx * 3 + 1, idx * 3 + 2


def normalize_skeleton_frame(
    features: np.ndarray, min_confidence: float = 0.3
) -> np.ndarray:
    """Normalize a single frame's keypoint features.

    1. Translate so mid-hip is at origin
    2. Scale by torso height (mid-shoulder to mid-hip distance)
    Confidence values are preserved unchanged.

    Exactly matches the normalization in punpayut/Fall-Detection.
    """
    normalized = features.copy()

    # Reference keypoints
    ref_names = {
        'ls': 'Left Shoulder', 'rs': 'Right Shoulder',
        'lh': 'Left Hip', 'rh': 'Right Hip',
    }
    for name in ref_names.values():
        if name not in KEYPOINT_NAME_TO_IDX:
            return features

    ls_x, ls_y, ls_c = _get_kpt_feature_indices(ref_names['ls'])
    rs_x, rs_y, rs_c = _get_kpt_feature_indices(ref_names['rs'])
    lh_x, lh_y, lh_c = _get_kpt_feature_indices(ref_names['lh'])
    rh_x, rh_y, rh_c = _get_kpt_feature_indices(ref_names['rh'])

    # Mid-shoulder
    valid_ls = features[ls_c] > min_confidence
    valid_rs = features[rs_c] > min_confidence
    if valid_ls and valid_rs:
        mid_sh_x = (features[ls_x] + features[rs_x]) / 2
        mid_sh_y = (features[ls_y] + features[rs_y]) / 2
    elif valid_ls:
        mid_sh_x, mid_sh_y = features[ls_x], features[ls_y]
    elif valid_rs:
        mid_sh_x, mid_sh_y = features[rs_x], features[rs_y]
    else:
        mid_sh_x, mid_sh_y = np.nan, np.nan

    # Mid-hip (origin reference)
    valid_lh = features[lh_c] > min_confidence
    valid_rh = features[rh_c] > min_confidence
    if valid_lh and valid_rh:
        mid_hip_x = (features[lh_x] + features[rh_x]) / 2
        mid_hip_y = (features[lh_y] + features[rh_y]) / 2
    elif valid_lh:
        mid_hip_x, mid_hip_y = features[lh_x], features[lh_y]
    elif valid_rh:
        mid_hip_x, mid_hip_y = features[rh_x], features[rh_y]
    else:
        return features  # Can't normalize without hip reference

    # Torso height for scaling
    ref_height = np.nan
    if not np.isnan(mid_sh_y):
        ref_height = abs(mid_sh_y - mid_hip_y)
    can_scale = not (np.isnan(ref_height) or ref_height < 1e-5)

    # Translate and scale each keypoint
    for kp_name in SORTED_KEYPOINT_NAMES:
        x_i, y_i, _ = _get_kpt_feature_indices(kp_name)
        normalized[x_i] -= mid_hip_x
        normalized[y_i] -= mid_hip_y
        if can_scale:
            normalized[x_i] /= ref_height
            normalized[y_i] /= ref_height

    return normalized


class TransformerFallDetector:
    """Transformer-based fall detector using MediaPipe Pose + TFLite.

    Maintains a per-person 30-frame sliding window of normalized
    keypoint features and runs TFLite inference for classification.
    """

    def __init__(self, config: Config):
        self.config = config
        fall_cfg = config.fall_detection

        # Model path
        self.model_path = Path(
            getattr(fall_cfg, 'transformer_model_path', None)
            or 'data/models/fall_detection_transformer.tflite'
        )
        self.input_timesteps = getattr(fall_cfg, 'input_timesteps', 30)
        self.confidence_threshold = getattr(
            fall_cfg, 'transformer_confidence', 0.90
        )
        self.alert_cooldown = fall_cfg.alert_cooldown
        mediapipe_complexity = getattr(fall_cfg, 'mediapipe_complexity', 1)

        # MediaPipe Pose (unified backbone)
        self.pose_detector = MediaPipePoseDetector(
            model_complexity=mediapipe_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Temporal buffer
        self._feature_buffer: deque = deque(maxlen=self.input_timesteps)
        self._last_alert_time: float = 0.0

        # Load TFLite model
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._load_model()

    def _load_model(self):
        """Load TFLite Transformer model."""
        if not self.model_path.exists():
            logger.warning(
                f"TFLite model not found at {self.model_path}. "
                "Fall detection will be disabled."
            )
            return

        try:
            from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
            except ImportError:
                try:
                    from tensorflow.lite import Interpreter as TFLiteInterpreter
                except ImportError:
                    logger.error(
                        "Cannot load TFLite runtime. Install: "
                        "pip install ai-edge-litert"
                    )
                    return

        self._interpreter = TFLiteInterpreter(
            model_path=str(self.model_path)
        )
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        expected = tuple(self._input_details[0]['shape'])
        required = (1, self.input_timesteps, NUM_FEATURES)
        if expected != required:
            logger.error(
                f"Model input shape {expected} != expected {required}. "
                "Check input_timesteps and keypoint config."
            )
            self._interpreter = None
            return

        logger.info(
            f"Transformer fall model loaded: {self.model_path} "
            f"(input: {expected})"
        )

    def detect(self, frame: np.ndarray) -> dict:
        """Detect fall on a single frame.

        Extracts pose, normalizes, buffers, and runs inference
        when the buffer is full (30 frames).

        Returns dict compatible with existing FallDetector interface:
        - fall_detected: bool
        - state: str ('standing', 'falling', 'fallen')
        - confidence: float
        - metrics: dict
        """
        if self._interpreter is None:
            return self._empty_result()

        # Extract sorted features (normalized 0-1 coords)
        features = self.pose_detector.extract_sorted_features(frame)

        if features is not None:
            normalized = normalize_skeleton_frame(features)
            self._feature_buffer.append(normalized)
        else:
            # No person detected — append zeros to keep temporal continuity
            self._feature_buffer.append(
                np.zeros(NUM_FEATURES, dtype=np.float32)
            )

        # Need full buffer for prediction
        if len(self._feature_buffer) < self.input_timesteps:
            return {
                "fall_detected": False,
                "state": "collecting",
                "confidence": 0.0,
                "metrics": {
                    "buffer_fill": len(self._feature_buffer),
                    "buffer_required": self.input_timesteps,
                },
            }

        # Run TFLite inference
        model_input = np.array(
            self._feature_buffer, dtype=np.float32
        )[np.newaxis, ...]  # (1, 30, 51)

        try:
            self._interpreter.set_tensor(
                self._input_details[0]['index'], model_input
            )
            self._interpreter.invoke()
            output = self._interpreter.get_tensor(
                self._output_details[0]['index']
            )
            fall_prob = float(output[0][0])
        except Exception as e:
            logger.error(f"TFLite inference error: {e}")
            return self._empty_result()

        # Classification
        is_fall = fall_prob >= self.confidence_threshold
        state = "fallen" if is_fall else "standing"
        display_confidence = fall_prob if is_fall else (1.0 - fall_prob)

        result = {
            "fall_detected": False,
            "state": state,
            "confidence": display_confidence,
            "metrics": {
                "fall_probability": fall_prob,
                "threshold": self.confidence_threshold,
            },
        }

        if is_fall:
            now = time.time()
            if (now - self._last_alert_time) > self.alert_cooldown:
                self._last_alert_time = now
                result["fall_detected"] = True
                logger.warning(
                    f"🚨 FALL DETECTED! Probability: {fall_prob:.2%}"
                )

        return result

    def get_pose_detector(self) -> MediaPipePoseDetector:
        """Expose the internal pose detector for shared use
        (e.g., gesture detection can reuse the same instance).
        """
        return self.pose_detector

    def reset(self):
        """Clear temporal buffer and cooldown."""
        self._feature_buffer.clear()
        self._last_alert_time = 0.0

    def _empty_result(self) -> dict:
        return {
            "fall_detected": False,
            "state": "standing",
            "confidence": 0.0,
            "metrics": {},
        }
