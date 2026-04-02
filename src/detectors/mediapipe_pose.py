"""Pose estimation using MediaPipe Pose.

Replaces YOLOv8-pose as the unified pose backbone for both
gesture recognition and Transformer-based fall detection.
Outputs COCO-17 keypoint format for compatibility with existing
gesture detector code.
"""

from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np


# MediaPipe landmark indices → COCO-17 keypoint mapping
_MP_TO_COCO = {
    mp.solutions.pose.PoseLandmark.NOSE: 0,
    mp.solutions.pose.PoseLandmark.LEFT_EYE: 1,
    mp.solutions.pose.PoseLandmark.RIGHT_EYE: 2,
    mp.solutions.pose.PoseLandmark.LEFT_EAR: 3,
    mp.solutions.pose.PoseLandmark.RIGHT_EAR: 4,
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER: 5,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER: 6,
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW: 7,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW: 8,
    mp.solutions.pose.PoseLandmark.LEFT_WRIST: 9,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST: 10,
    mp.solutions.pose.PoseLandmark.LEFT_HIP: 11,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP: 12,
    mp.solutions.pose.PoseLandmark.LEFT_KNEE: 13,
    mp.solutions.pose.PoseLandmark.RIGHT_KNEE: 14,
    mp.solutions.pose.PoseLandmark.LEFT_ANKLE: 15,
    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE: 16,
}

# Sorted keypoint names (alphabetical) — matches the Fall-Detection
# repo's training feature order for the Transformer model.
SORTED_KEYPOINT_NAMES = sorted([
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
])
KEYPOINT_NAME_TO_IDX = {name: i for i, name in enumerate(SORTED_KEYPOINT_NAMES)}
NUM_SORTED_KEYPOINTS = len(SORTED_KEYPOINT_NAMES)
NUM_FEATURES = NUM_SORTED_KEYPOINTS * 3  # x, y, visibility per keypoint

# Map from MediaPipe landmark enum → sorted keypoint name
_MP_TO_SORTED_NAME = {
    mp.solutions.pose.PoseLandmark.NOSE: 'Nose',
    mp.solutions.pose.PoseLandmark.LEFT_EYE: 'Left Eye',
    mp.solutions.pose.PoseLandmark.RIGHT_EYE: 'Right Eye',
    mp.solutions.pose.PoseLandmark.LEFT_EAR: 'Left Ear',
    mp.solutions.pose.PoseLandmark.RIGHT_EAR: 'Right Ear',
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER: 'Left Shoulder',
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER: 'Right Shoulder',
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW: 'Left Elbow',
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW: 'Right Elbow',
    mp.solutions.pose.PoseLandmark.LEFT_WRIST: 'Left Wrist',
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST: 'Right Wrist',
    mp.solutions.pose.PoseLandmark.LEFT_HIP: 'Left Hip',
    mp.solutions.pose.PoseLandmark.RIGHT_HIP: 'Right Hip',
    mp.solutions.pose.PoseLandmark.LEFT_KNEE: 'Left Knee',
    mp.solutions.pose.PoseLandmark.RIGHT_KNEE: 'Right Knee',
    mp.solutions.pose.PoseLandmark.LEFT_ANKLE: 'Left Ankle',
    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE: 'Right Ankle',
}

# COCO skeleton pairs for drawing
SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),   # Head
    (5, 6),                             # Shoulders
    (5, 7), (7, 9),                     # Left arm
    (6, 8), (8, 10),                    # Right arm
    (5, 11), (6, 12),                   # Torso
    (11, 12),                           # Hips
    (11, 13), (13, 15),                 # Left leg
    (12, 14), (14, 16),                 # Right leg
]


class MediaPipePoseDetector:
    """MediaPipe Pose detector with dual output: COCO-17 pixel coords
    for gesture detection and raw normalized coords for fall detection.

    Single-person detection (MediaPipe Pose limitation).
    For multi-person, use person detection first then crop.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._last_raw_results = None

    def detect(self, frame: np.ndarray) -> dict:
        """Run pose estimation on a BGR frame.

        Returns dict matching the existing PoseDetector interface:
        - persons_found: bool
        - count: int (0 or 1 for MediaPipe)
        - poses: list of pose dicts with pixel-coord keypoints
        """
        if frame is None or frame.size == 0:
            self._last_raw_results = None
            return self._empty_result()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        results = self.pose.process(rgb)
        self._last_raw_results = results

        if not results.pose_landmarks:
            return self._empty_result()

        landmarks = results.pose_landmarks.landmark

        # Build COCO-17 keypoints in pixel coords
        keypoints_px = []
        confidences = []

        for mp_enum, coco_idx in sorted(_MP_TO_COCO.items(), key=lambda x: x[1]):
            lm = landmarks[mp_enum.value]
            keypoints_px.append([lm.x * w, lm.y * h])
            confidences.append(lm.visibility)

        # Bounding box from visible keypoints
        kpts_arr = np.array(keypoints_px)
        confs_arr = np.array(confidences)
        valid = confs_arr > 0.3
        if valid.any():
            mins = kpts_arr[valid].min(axis=0)
            maxs = kpts_arr[valid].max(axis=0)
            bbox = [int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])]
        else:
            bbox = [0, 0, w, h]

        pose = {
            "keypoints": kpts_arr.tolist(),
            "confidence": confs_arr.tolist(),
            "avg_confidence": float(confs_arr.mean()),
            "bbox": bbox,
        }

        return {
            "persons_found": True,
            "count": 1,
            "poses": [pose],
        }

    def get_keypoints(
        self, frame: np.ndarray, person_idx: int = 0
    ) -> Optional[np.ndarray]:
        """Get COCO-17 keypoints as (17, 3) array [x_px, y_px, confidence].

        Compatible with existing GestureDetector interface.
        """
        result = self.detect(frame)
        if not result["persons_found"] or person_idx >= len(result["poses"]):
            return None

        pose = result["poses"][person_idx]
        kpts = np.array(pose["keypoints"])   # (17, 2)
        conf = np.array(pose["confidence"])  # (17,)
        return np.column_stack([kpts, conf])

    def get_raw_landmarks(self):
        """Get the raw MediaPipe landmark results from the last detect() call.

        Returns the mediapipe results object (has .pose_landmarks).
        Used by TransformerFallDetector for normalized coordinates.
        """
        return self._last_raw_results

    def extract_sorted_features(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract 51-dim feature vector (17 keypoints × 3) in alphabetically
        sorted keypoint order with normalized (0-1) coordinates.

        This is the exact format expected by the Transformer fall model.
        Returns None if no pose detected.
        """
        result = self.detect(frame)
        if not result["persons_found"]:
            return None

        raw_results = self._last_raw_results
        if raw_results is None or not raw_results.pose_landmarks:
            return None

        landmarks = raw_results.pose_landmarks.landmark
        features = np.zeros(NUM_FEATURES, dtype=np.float32)

        for mp_enum, kp_name in _MP_TO_SORTED_NAME.items():
            idx = KEYPOINT_NAME_TO_IDX[kp_name]
            lm = landmarks[mp_enum.value]
            features[idx * 3] = lm.x
            features[idx * 3 + 1] = lm.y
            features[idx * 3 + 2] = lm.visibility

        return features

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()

    @staticmethod
    def _empty_result() -> dict:
        return {"persons_found": False, "count": 0, "poses": []}
