"""Visualization utilities for drawing on frames."""

from typing import Dict, Optional

import cv2
import numpy as np

from ..detectors.pose import PoseDetector


class Drawer:
    """Utility class for drawing detections and annotations on frames."""

    # Colors (BGR format for OpenCV)
    COLOR_PERSON = (0, 255, 0)  # Green
    COLOR_FACE = (255, 0, 0)  # Blue
    COLOR_FACE_RECOGNIZED = (0, 255, 0)  # Green
    COLOR_FACE_UNKNOWN = (0, 165, 255)  # Orange
    COLOR_POSE = (255, 0, 255)  # Magenta
    COLOR_FALL = (0, 0, 255)  # Red
    COLOR_GESTURE = (255, 255, 0)  # Cyan
    COLOR_TEXT = (255, 255, 255)  # White
    COLOR_TEXT_BG = (0, 0, 0)  # Black

    def __init__(self, font_scale: float = 0.6, thickness: int = 2):
        """
        Initialize drawer.

        Args:
            font_scale: Scale for text
            thickness: Line thickness
        """
        self.font_scale = font_scale
        self.thickness = thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_person_box(
        self,
        frame: np.ndarray,
        box: list[int],
        label: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """
        Draw person bounding box.

        Args:
            frame: Frame to draw on
            box: [x1, y1, x2, y2] bounding box
            label: Optional label text
            confidence: Optional confidence score
        """
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_PERSON, self.thickness)

        if label or confidence is not None:
            text = f"{label}" if label else ""
            if confidence is not None:
                text += f" {confidence:.2f}"
            self._draw_label(frame, text, (x1, y1 - 10), self.COLOR_PERSON)

    def draw_face_box(
        self,
        frame: np.ndarray,
        box: list[int],
        name: Optional[str] = None,
        confidence: Optional[float] = None,
        recognized: bool = False,
    ) -> None:
        """
        Draw face bounding box with name.

        Args:
            frame: Frame to draw on
            box: [x1, y1, x2, y2] bounding box
            name: Person name if recognized
            confidence: Detection confidence
            recognized: Whether face was recognized
        """
        x1, y1, x2, y2 = box

        color = self.COLOR_FACE_RECOGNIZED if recognized else self.COLOR_FACE_UNKNOWN
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

        # Draw label background
        label = name if recognized else "Unknown"
        if confidence is not None:
            label += f" {confidence:.2f}"

        self._draw_label(frame, label, (x1, y1 - 10), color)

    def draw_pose(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        draw_skeleton: bool = True,
        draw_keypoints: bool = True,
    ) -> None:
        """
        Draw pose keypoints and skeleton.

        Args:
            frame: Frame to draw on
            keypoints: (17, 3) array with x, y, confidence
            draw_skeleton: Whether to draw skeleton connections
            draw_keypoints: Whether to draw individual keypoints
        """
        if draw_skeleton:
            self._draw_skeleton(frame, keypoints)

        if draw_keypoints:
            self._draw_keypoints(frame, keypoints)

    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray) -> None:
        """Draw skeleton connections."""
        for pair in PoseDetector.SKELETON_PAIRS:
            i, j = pair

            # Check if both keypoints are visible
            if keypoints[i, 2] < 0.5 or keypoints[j, 2] < 0.5:
                continue

            pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
            pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))

            cv2.line(frame, pt1, pt2, self.COLOR_POSE, 2)

    def _draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray) -> None:
        """Draw individual keypoints."""
        for i, kpt in enumerate(keypoints):
            if kpt[2] < 0.5:  # Low confidence
                continue

            center = (int(kpt[0]), int(kpt[1]))
            cv2.circle(frame, center, 4, self.COLOR_POSE, -1)

    def draw_fall_alert(
        self,
        frame: np.ndarray,
        state: str,
        confidence: float,
    ) -> None:
        """
        Draw fall detection alert.

        Args:
            frame: Frame to draw on
            state: Current state (standing, falling, fallen)
            confidence: Detection confidence
        """
        h, w = frame.shape[:2]

        # Draw alert box at top
        alert_text = f"FALL DETECTED! State: {state.upper()}"
        text_size = cv2.getTextSize(alert_text, self.font, 1.2, self.thickness)[0]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (w, text_size[1] + 20),
            self.COLOR_FALL,
            -1,
        )
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Draw text
        text_x = (w - text_size[0]) // 2
        cv2.putText(
            frame,
            alert_text,
            (text_x, text_size[1] + 10),
            self.font,
            1.2,
            self.COLOR_TEXT,
            self.thickness + 1,
        )

        # Draw confidence meter
        meter_width = 200
        meter_height = 10
        meter_x = (w - meter_width) // 2
        meter_y = text_size[1] + 25

        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), (100, 100, 100), -1)
        cv2.rectangle(
            frame,
            (meter_x, meter_y),
            (meter_x + int(meter_width * confidence), meter_y + meter_height),
            self.COLOR_FALL,
            -1,
        )

    def draw_gesture(
        self,
        frame: np.ndarray,
        gesture_name: str,
        confidence: float,
    ) -> None:
        """
        Draw gesture detection.

        Args:
            frame: Frame to draw on
            gesture_name: Name of detected gesture
            confidence: Detection confidence
        """
        text = f"Gesture: {gesture_name} ({confidence:.2f})"
        self._draw_label(frame, text, (10, frame.shape[0] - 20), self.COLOR_GESTURE)

    def draw_info(
        self,
        frame: np.ndarray,
        info: Dict[str, str],
    ) -> None:
        """
        Draw info box with multiple lines.

        Args:
            frame: Frame to draw on
            info: Dictionary of label -> value pairs
        """
        y = 30
        line_height = 25

        for label, value in info.items():
            text = f"{label}: {value}"
            self._draw_label(frame, text, (10, y), self.COLOR_TEXT)
            y += line_height

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        """
        Draw text label with background.

        Args:
            frame: Frame to draw on
            text: Text to draw
            position: (x, y) position
            color: Text color
        """
        text_size = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)[0]
        x, y = position

        # Draw background
        cv2.rectangle(
            frame,
            (x, y - text_size[1] - 5),
            (x + text_size[0] + 10, y + 5),
            self.COLOR_TEXT_BG,
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            text,
            (x + 5, y),
            self.font,
            self.font_scale,
            color,
            self.thickness,
        )

    def draw_timestamp(
        self,
        frame: np.ndarray,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Draw timestamp on frame.

        Args:
            frame: Frame to draw on
            timestamp: Optional timestamp string (auto-generated if None)
        """
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(timestamp, self.font, self.font_scale, 1)[0]

        # Draw at bottom right
        x = w - text_size[0] - 10
        y = h - 10

        cv2.putText(frame, timestamp, (x, y), self.font, self.font_scale, self.COLOR_TEXT, 1)

    @staticmethod
    def resize_with_aspect_ratio(
        frame: np.ndarray,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
    ) -> np.ndarray:
        """
        Resize frame while maintaining aspect ratio.

        Args:
            frame: Input frame
            max_width: Maximum width (None for no limit)
            max_height: Maximum height (None for no limit)

        Returns:
            Resized frame
        """
        h, w = frame.shape[:2]

        if max_width and w > max_width:
            scale = max_width / w
            h = int(h * scale)
            w = max_width

        if max_height and h > max_height:
            scale = max_height / h
            w = int(w * scale)
            h = max_height

        if w != frame.shape[1] or h != frame.shape[0]:
            return cv2.resize(frame, (w, h))

        return frame
