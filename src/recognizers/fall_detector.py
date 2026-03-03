"""Fall detection using pose-based geometric analysis."""

import time
from collections import deque
from enum import Enum
from typing import Optional

import numpy as np

from ..config.settings import Config
from ..detectors.pose import PoseDetector


class FallState(Enum):
    """States in fall detection state machine."""

    STANDING = "standing"
    FALLING = "falling"
    FALLEN = "fallen"


class FallDetector:
    """
    Pose-based fall detection system.

    Uses geometric analysis of body keypoints to detect falls:
    - Aspect ratio: Person becomes wider when fallen
    - Keypoint analysis: Hip-to-ankle distance changes
    - Temporal consistency: Requires consecutive frames
    """

    # Key point indices from COCO 17-keypoint format
    HIP_LEFT = 11
    HIP_RIGHT = 12
    KNEE_LEFT = 13
    KNEE_RIGHT = 14
    ANKLE_LEFT = 15
    ANKLE_RIGHT = 16
    SHOULDER_LEFT = 5
    SHOULDER_RIGHT = 6

    def __init__(self, config: Config):
        """
        Initialize fall detector.

        Args:
            config: Configuration object
        """
        self.config = config
        self.fall_config = config.fall_detection

        # Initialize pose detector
        self.pose_detector = PoseDetector(config, model_name=self.fall_config.pose_model)

        # Detection parameters
        self.aspect_ratio_threshold = self.fall_config.aspect_ratio_threshold
        self.min_frames_for_fall = self.fall_config.min_frames_for_fall
        self.alert_cooldown = self.fall_config.alert_cooldown

        # State tracking
        self._state = FallState.STANDING
        self._fall_counter = 0
        self._last_alert_time = 0
        self._pose_history: deque = deque(maxlen=10)

    def detect(self, frame: np.ndarray) -> dict:
        """
        Detect if a fall has occurred.

        Args:
            frame: Input frame

        Returns:
            Dictionary with:
            - fall_detected: bool
            - state: current FallState
            - confidence: float (0-1)
            - metrics: dict of calculated metrics
        """
        if not self.fall_config.enabled:
            return self._empty_result()

        # Get pose keypoints
        keypoints = self.pose_detector.get_keypoints(frame, person_idx=0)

        if keypoints is None:
            self._reset_state()
            return self._empty_result()

        # Extract relevant keypoints
        metrics = self._calculate_metrics(keypoints)
        self._pose_history.append(metrics)

        # State machine
        new_state = self._determine_state(metrics)

        # Update state based on transitions
        if new_state == FallState.FALLEN:
            if self._state == FallState.FALLING:
                self._fall_counter += 1
            else:
                self._fall_counter = 1

            if self._fall_counter >= self.min_frames_for_fall:
                # Check cooldown
                current_time = time.time()
                if current_time - self._last_alert_time >= self.alert_cooldown:
                    self._last_alert_time = current_time
                    self._state = new_state
                    return {
                        "fall_detected": True,
                        "state": new_state.value,
                        "confidence": min(1.0, self._fall_counter / self.min_frames_for_fall),
                        "metrics": metrics,
                        "fall_duration": self._fall_counter,
                    }
        elif new_state == FallState.FALLING:
            if self._state == FallState.STANDING:
                self._state = new_state
            self._fall_counter = 0
        else:  # STANDING
            self._reset_state()

        self._state = new_state

        return {
            "fall_detected": False,
            "state": new_state.value,
            "confidence": 0.0,
            "metrics": metrics,
            "fall_counter": self._fall_counter,
        }

    def _calculate_metrics(self, keypoints: np.ndarray) -> dict:
        """
        Calculate geometric metrics from keypoints.

        Args:
            keypoints: (17, 3) array with x, y, confidence

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "aspect_ratio": 0.0,
            "hip_height": 0.0,
            "body_orientation": 0.0,  # Horizontal (0) to Vertical (1)
            "keypoints_visible": 0,
        }

        # Filter valid keypoints
        valid_mask = keypoints[:, 2] > 0.5
        valid_keypoints = keypoints[valid_mask]
        metrics["keypoints_visible"] = len(valid_keypoints)

        if len(valid_keypoints) < 5:
            return metrics

        # Get bounding box of person
        all_points = keypoints[:, :2]
        min_coords = np.min(all_points[valid_mask], axis=0)
        max_coords = np.max(all_points[valid_mask], axis=0)

        width = max_coords[0] - min_coords[0]
        height = max_coords[1] - min_coords[1]

        # Aspect ratio (width/height)
        if height > 0:
            metrics["aspect_ratio"] = width / height

        # Hip height (normalized by frame height)
        hips = keypoints[[self.HIP_LEFT, self.HIP_RIGHT], :2]
        valid_hips = hips[valid_mask[[self.HIP_LEFT, self.HIP_RIGHT]]]

        if len(valid_hips) > 0:
            avg_hip_y = np.mean(valid_hips[:, 1])
            frame_height = np.max(all_points[:, 1])
            if frame_height > 0:
                metrics["hip_height"] = 1.0 - (avg_hip_y / frame_height)

        # Body orientation using shoulder-hip vector
        shoulders = keypoints[[self.SHOULDER_LEFT, self.SHOULDER_RIGHT], :2]
        if valid_mask[self.SHOULDER_LEFT] and valid_mask[self.SHOULDER_RIGHT]:
            shoulder_center = np.mean(shoulders, axis=0)
        elif valid_mask[self.SHOULDER_LEFT]:
            shoulder_center = shoulders[0]
        elif valid_mask[self.SHOULDER_RIGHT]:
            shoulder_center = shoulders[1]
        else:
            shoulder_center = None

        if shoulder_center is not None and len(valid_hips) > 0:
            hip_center = np.mean(valid_hips, axis=0)
            torso_vector = hip_center - shoulder_center

            # Vertical component dominates when standing
            dy = abs(torso_vector[1])
            dx = abs(torso_vector[0])

            if dx + dy > 0:
                metrics["body_orientation"] = dy / (dx + dy)

        return metrics

    def _determine_state(self, metrics: dict) -> FallState:
        """
        Determine fall state from metrics.

        Args:
            metrics: Calculated geometric metrics

        Returns:
            Current FallState
        """
        aspect_ratio = metrics.get("aspect_ratio", 0)
        body_orientation = metrics.get("body_orientation", 0)
        hip_height = metrics.get("hip_height", 0)

        # Check for fallen state (high aspect ratio = person is horizontal)
        if aspect_ratio >= self.aspect_ratio_threshold:
            # Additional check: low body orientation confirms horizontal pose
            if body_orientation < 0.5:
                return FallState.FALLEN

        # Check for falling state (transitioning)
        if aspect_ratio > self.aspect_ratio_threshold * 0.7 or hip_height < 0.3:
            return FallState.FALLING

        return FallState.STANDING

    def _reset_state(self) -> None:
        """Reset fall detection state."""
        self._state = FallState.STANDING
        self._fall_counter = 0

    def reset_cooldown(self) -> None:
        """Reset the alert cooldown timer."""
        self._last_alert_time = 0

    def is_in_cooldown(self) -> bool:
        """Check if currently in alert cooldown period."""
        return time.time() - self._last_alert_time < self.alert_cooldown

    def _empty_result(self) -> dict:
        """Return empty result structure."""
        return {
            "fall_detected": False,
            "state": FallState.STANDING.value,
            "confidence": 0.0,
            "metrics": {},
        }
