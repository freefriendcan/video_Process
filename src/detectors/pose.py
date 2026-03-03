"""Pose estimation using YOLOv8-pose."""

from typing import Any

import numpy as np

from ..config.settings import Config
from .yolo_base import BaseDetector


class PoseDetector(BaseDetector):
    """
    Pose estimation using YOLOv8-pose.

    Detects 17 body keypoints following COCO format:
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """

    # COCO 17-keypoint format
    KEYPOINT_NAMES = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    # Keypoint pairs for drawing skeleton
    SKELETON_PAIRS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6),  # Shoulders
        (5, 7), (7, 9),  # Left arm
        (6, 8), (8, 10),  # Right arm
        (5, 11), (6, 12),  # Torso
        (11, 12),  # Hips
        (11, 13), (13, 15),  # Left leg
        (12, 14), (14, 16),  # Right leg
    ]

    def __init__(self, config: Config, model_name: str = "yolov8n-pose.pt"):
        """
        Initialize pose detector.

        Args:
            config: Configuration object
            model_name: Name of the YOLO-pose model file
        """
        super().__init__(config, model_name)
        self.min_confidence = config.detection.confidence

    def detect(self, frame: np.ndarray) -> dict:
        """
        Detect poses in a frame.

        Args:
            frame: Input frame as numpy array (BGR)

        Returns:
            Dictionary with:
            - persons_found: bool
            - count: int
            - poses: list of pose data with keypoints and bounding boxes
        """
        if frame is None or frame.size == 0:
            return self._empty_result()

        try:
            # Run inference
            results = self.model(frame, **self.get_inference_kwargs())

            # Extract pose detections
            poses = []

            for result in results:
                if result.keypoints is not None:
                    keypoints_data = result.keypoints.xy.cpu().numpy()  # (N, 17, 2)
                    conf_data = result.keypoints.conf.cpu().numpy()  # (N, 17)
                    boxes_data = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None

                    for i, (kpts, confs) in enumerate(zip(keypoints_data, conf_data)):
                        # Filter by average confidence
                        avg_conf = float(np.mean(confs))
                        if avg_conf < self.min_confidence:
                            continue

                        pose = {
                            "keypoints": kpts.tolist(),  # List of [x, y] for 17 keypoints
                            "confidence": confs.tolist(),  # Confidence for each keypoint
                            "avg_confidence": avg_conf,
                        }

                        # Add bounding box if available
                        if boxes_data is not None and i < len(boxes_data):
                            x1, y1, x2, y2 = boxes_data[i]
                            pose["bbox"] = [int(x1), int(y1), int(x2), int(y2)]

                        poses.append(pose)

            return {
                "persons_found": len(poses) > 0,
                "count": len(poses),
                "poses": poses,
            }

        except Exception as e:
            print(f"Error in pose detection: {e}")
            return self._empty_result()

    def _empty_result(self) -> dict:
        """Return empty result structure."""
        return {
            "persons_found": False,
            "count": 0,
            "poses": [],
        }

    def get_keypoints(self, frame: np.ndarray, person_idx: int = 0) -> np.ndarray | None:
        """
        Get keypoints for a specific person.

        Args:
            frame: Input frame
            person_idx: Index of person (0 = first detected)

        Returns:
            Keypoints array (17, 3) with x, y, confidence or None
        """
        result = self.detect(frame)

        if not result["persons_found"] or person_idx >= len(result["poses"]):
            return None

        pose = result["poses"][person_idx]
        kpts = np.array(pose["keypoints"])  # (17, 2)
        conf = np.array(pose["confidence"])  # (17,)

        # Combine into (17, 3) array
        return np.column_stack([kpts, conf])

    def get_visible_keypoints(self, frame: np.ndarray, min_confidence: float = 0.5) -> list[dict]:
        """
        Get list of visible keypoints above confidence threshold.

        Args:
            frame: Input frame
            min_confidence: Minimum confidence for keypoint visibility

        Returns:
            List of {"name": str, "x": float, "y": float, "confidence": float}
        """
        result = self.detect(frame)
        visible = []

        if result["persons_found"]:
            pose = result["poses"][0]  # Use first person
            for i, (kpt, conf) in enumerate(zip(pose["keypoints"], pose["confidence"])):
                if conf >= min_confidence:
                    visible.append({
                        "name": self.KEYPOINT_NAMES[i],
                        "index": i,
                        "x": float(kpt[0]),
                        "y": float(kpt[1]),
                        "confidence": float(conf),
                    })

        return visible
