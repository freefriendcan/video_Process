"""Person detection using YOLOv10."""

from typing import Any

import numpy as np

from ..config.settings import Config
from .yolo_base import BaseDetector


class PersonDetector(BaseDetector):
    """
    Person detection using YOLOv10n (NMS-free efficient detector).

    Detects people in frames with bounding boxes and confidence scores.
    """

    def __init__(self, config: Config, model_name: str = "yolov10n.pt"):
        """
        Initialize person detector.

        Args:
            config: Configuration object
            model_name: Name of the YOLO model file
        """
        super().__init__(config, model_name)
        self.min_confidence = config.person_detection.min_confidence

    def detect(self, frame: np.ndarray) -> dict:
        """
        Detect people in a frame.

        Args:
            frame: Input frame as numpy array (BGR)

        Returns:
            Dictionary with:
            - persons_found: bool
            - count: int
            - boxes: list of [x1, y1, x2, y2] bounding boxes
            - confidences: list of confidence scores
            - classes: list of class indices (0 = person)
        """
        if frame is None or frame.size == 0:
            return self._empty_result()

        try:
            # Run inference
            results = self.model(frame, **self.get_inference_kwargs())

            # Extract person detections
            persons = []
            confidences = []
            boxes = []

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Filter for person class (usually class 0 in COCO)
                        if int(box.cls[0]) == 0 and float(box.conf[0]) >= self.min_confidence:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            boxes.append([int(x1), int(y1), int(x2), int(y2)])
                            confidences.append(float(box.conf[0]))
                            persons.append({"box": [int(x1), int(y1), int(x2), int(y2)], "confidence": float(box.conf[0])})

            return {
                "persons_found": len(persons) > 0,
                "count": len(persons),
                "boxes": boxes,
                "confidences": confidences,
                "persons": persons,
            }

        except Exception as e:
            print(f"Error in person detection: {e}")
            return self._empty_result()

    def _empty_result(self) -> dict:
        """Return empty result structure."""
        return {
            "persons_found": False,
            "count": 0,
            "boxes": [],
            "confidences": [],
            "persons": [],
        }

    def get_largest_person(self, frame: np.ndarray) -> dict:
        """
        Get the largest (closest) person in the frame.

        Args:
            frame: Input frame

        Returns:
            Dictionary with bounding box and confidence, or empty if no person found
        """
        result = self.detect(frame)

        if not result["persons_found"]:
            return {}

        # Find largest by area
        largest = max(result["persons"], key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))
        return largest

    def count_persons(self, frame: np.ndarray) -> int:
        """
        Count number of persons in frame.

        Args:
            frame: Input frame

        Returns:
            Number of persons detected
        """
        result = self.detect(frame)
        return result["count"]
