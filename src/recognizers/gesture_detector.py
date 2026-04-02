"""Gesture recognition using pose keypoints."""

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from ..config.settings import Config
from ..detectors.pose import PoseDetector


class BaseGesture(ABC):
    """Base class for gesture detection."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Gesture name."""
        pass

    @abstractmethod
    def detect(self, keypoints: np.ndarray) -> bool:
        """
        Detect if gesture is present in current frame.

        Args:
            keypoints: (17, 3) array with x, y, confidence

        Returns:
            True if gesture detected
        """
        pass


class WaveGesture(BaseGesture):
    """Detect waving motion (hand moving side to side)."""

    name = "wave"

    def __init__(self, window_size: int = 10):
        """Initialize with history window."""
        self.wrist_history: deque = deque(maxlen=window_size)
        self.window_size = window_size

    def detect(self, keypoints: np.ndarray) -> bool:
        """Detect wave based on wrist oscillation."""
        # Get right wrist position
        WRIST_IDX = 10
        if keypoints[WRIST_IDX, 2] < 0.5:  # Low confidence
            return False

        wrist_pos = keypoints[WRIST_IDX, :2]
        self.wrist_history.append(wrist_pos)

        if len(self.wrist_history) < self.window_size:
            return False

        # Check for oscillation (left-right movement)
        positions = np.array(self.wrist_history)
        x_positions = positions[:, 0]

        # Count direction changes
        direction_changes = 0
        for i in range(2, len(x_positions)):
            if (x_positions[i] - x_positions[i - 1]) * (x_positions[i - 1] - x_positions[i - 2]) < 0:
                direction_changes += 1

        return direction_changes >= 3


class HandsUpGesture(BaseGesture):
    """Detect both hands raised above shoulders."""

    name = "hands_up"

    def detect(self, keypoints: np.ndarray) -> bool:
        """Detect both wrists above shoulders."""
        # Keypoint indices
        LEFT_WRIST, RIGHT_WRIST = 9, 10
        LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

        # Check confidence
        if any(keypoints[i, 2] < 0.5 for i in [LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER]):
            return False

        # Both wrists should be above (lower y value) than shoulders
        left_wrist_above = keypoints[LEFT_WRIST, 1] < keypoints[LEFT_SHOULDER, 1]
        right_wrist_above = keypoints[RIGHT_WRIST, 1] < keypoints[RIGHT_SHOULDER, 1]

        # Add tolerance for "above"
        tolerance = 20
        left_wrist_above = keypoints[LEFT_WRIST, 1] < keypoints[LEFT_SHOULDER, 1] - tolerance
        right_wrist_above = keypoints[RIGHT_WRIST, 1] < keypoints[RIGHT_SHOULDER, 1] - tolerance

        return left_wrist_above and right_wrist_above


class PointingGesture(BaseGesture):
    """Detect pointing gesture (one arm extended, other down)."""

    name = "pointing"

    def detect(self, keypoints: np.ndarray) -> bool:
        """Detect pointing pose."""
        # Indices
        LEFT_WRIST, RIGHT_WRIST = 9, 10
        LEFT_ELBOW, RIGHT_ELBOW = 7, 8
        LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

        # Check confidence for all needed points
        points = [LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW, LEFT_SHOULDER, RIGHT_SHOULDER]
        if any(keypoints[i, 2] < 0.5 for i in points):
            return False

        # Check if one arm is extended (wrist far from shoulder)
        # and other arm is down (wrist below shoulder)

        left_extended = self._is_arm_extended(keypoints, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        right_extended = self._is_arm_extended(keypoints, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
        left_down = self._is_arm_down(keypoints, LEFT_SHOULDER, LEFT_WRIST)
        right_down = self._is_arm_down(keypoints, RIGHT_SHOULDER, RIGHT_WRIST)

        return (left_extended and right_down) or (right_extended and left_down)

    def _is_arm_extended(self, kpts: np.ndarray, shoulder: int, elbow: int, wrist: int) -> bool:
        """Check if arm is extended (straight)."""
        shoulder_pos = kpts[shoulder, :2]
        elbow_pos = kpts[elbow, :2]
        wrist_pos = kpts[wrist, :2]

        # Check if elbow is roughly between shoulder and wrist
        shoulder_elbow = np.linalg.norm(elbow_pos - shoulder_pos)
        elbow_wrist = np.linalg.norm(wrist_pos - elbow_pos)
        shoulder_wrist = np.linalg.norm(wrist_pos - shoulder_pos)

        # Arm is extended if total length roughly equals shoulder-wrist distance
        return abs(shoulder_elbow + elbow_wrist - shoulder_wrist) < 30

    def _is_arm_down(self, kpts: np.ndarray, shoulder: int, wrist: int) -> bool:
        """Check if arm is hanging down."""
        return kpts[wrist, 1] > kpts[shoulder, 1]  # Wrist below shoulder


class CrouchingGesture(BaseGesture):
    """Detect crouching pose (knees below hip level)."""

    name = "crouching"

    def detect(self, keypoints: np.ndarray) -> bool:
        """Detect crouching based on knee positions."""
        # Indices
        LEFT_HIP, RIGHT_HIP = 11, 12
        LEFT_KNEE, RIGHT_KNEE = 13, 14

        # Check confidence
        if any(keypoints[i, 2] < 0.5 for i in [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE]):
            return False

        # Knees should be above (lower y value) hips in standing,
        # but below (higher y value) hips when crouching
        left_knee_below_hip = keypoints[LEFT_KNEE, 1] > keypoints[LEFT_HIP, 1]
        right_knee_below_hip = keypoints[RIGHT_KNEE, 1] > keypoints[RIGHT_HIP, 1]

        return left_knee_below_hip and right_knee_below_hip


class GestureDetector:
    """
    Main gesture detector with multiple gesture classes.

    Uses temporal smoothing to reduce false positives from pose jitter.
    """

    # Built-in gesture classes
    DEFAULT_GESTURES = [
        WaveGesture,
        HandsUpGesture,
        PointingGesture,
        CrouchingGesture,
    ]

    def __init__(self, config: Config, pose_detector=None):
        """
        Initialize gesture detector.

        Args:
            config: Configuration object
            pose_detector: Optional shared pose detector instance
                           (MediaPipePoseDetector or PoseDetector).
                           If None, creates a new YOLOv8 PoseDetector.
        """
        self.config = config
        self.gesture_config = config.gesture_detection

        # Use shared pose detector or create a new YOLO one
        if pose_detector is not None:
            self.pose_detector = pose_detector
        else:
            self.pose_detector = PoseDetector(config, model_name=self.gesture_config.pose_model)

        # Gesture instances
        self.gestures = [cls() for cls in self.DEFAULT_GESTURES]

        # Smoothing windows
        self.smoothing_window = self.gesture_config.smoothing_window
        self.detection_history: Dict[str, Deque[bool]] = {
            gesture.name: deque(maxlen=self.smoothing_window) for gesture in self.gestures
        }

    def add_gesture(self, gesture: BaseGesture) -> None:
        """
        Add a custom gesture class.

        Args:
            gesture: BaseGesture instance
        """
        self.gestures.append(gesture)
        self.detection_history[gesture.name] = deque(maxlen=self.smoothing_window)

    def detect(self, frame: np.ndarray) -> dict:
        """
        Detect all gestures in frame.

        Args:
            frame: Input frame

        Returns:
            Dictionary with:
            - gestures_detected: list of detected gesture names
            - all_gestures: dict of gesture_name -> confidence
        """
        if not self.gesture_config.enabled:
            return {"gestures_detected": [], "all_gestures": {}}

        # Get pose keypoints
        keypoints = self.pose_detector.get_keypoints(frame, person_idx=0)

        if keypoints is None:
            return {"gestures_detected": [], "all_gestures": {}}

        # Detect each gesture
        detected = []
        all_results = {}

        for gesture in self.gestures:
            is_detected = gesture.detect(keypoints)

            # Update history
            self.detection_history[gesture.name].append(is_detected)

            # Calculate confidence based on history
            history = list(self.detection_history[gesture.name])
            confidence = sum(history) / len(history) if history else 0

            all_results[gesture.name] = confidence

            # Consider detected if confidence above threshold
            if confidence >= self.gesture_config.min_confidence:
                detected.append(gesture.name)

        return {
            "gestures_detected": detected,
            "all_gestures": all_results,
        }

    def detect_single(self, frame: np.ndarray, gesture_name: str) -> dict:
        """
        Detect a specific gesture.

        Args:
            frame: Input frame
            gesture_name: Name of gesture to detect

        Returns:
            Dictionary with detected and confidence
        """
        result = self.detect(frame)
        return {
            "detected": gesture_name in result["gestures_detected"],
            "confidence": result["all_gestures"].get(gesture_name, 0.0),
        }

    def reset_history(self) -> None:
        """Reset detection history for all gestures."""
        for gesture_name in self.detection_history:
            self.detection_history[gesture_name].clear()

    @staticmethod
    def get_gesture_names() -> list[str]:
        """Get list of available gesture names."""
        return [g.name for g in GestureDetector.DEFAULT_GESTURES]
