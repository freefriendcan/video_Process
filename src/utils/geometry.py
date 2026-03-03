"""Geometric calculations for pose analysis."""

import math
from typing import Optional

import numpy as np


def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        p1: First point [x, y]
        p2: Second point [x, y]

    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(p1 - p2))


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate angle between three points (p2 is vertex).

    Args:
        p1: First point [x, y]
        p2: Vertex point [x, y]
        p3: Third point [x, y]

    Returns:
        Angle in degrees
    """
    v1 = p1 - p2
    v2 = p3 - p2

    # Calculate angle using dot product
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    cos_angle = dot_product / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def calculate_slope(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate slope of line between two points.

    Args:
        p1: First point [x, y]
        p2: Second point [x, y]

    Returns:
        Slope value
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if abs(dx) < 1e-6:
        return float("inf") if dy > 0 else float("-inf")

    return dy / dx


def is_point_above_line(point: np.ndarray, line_p1: np.ndarray, line_p2: np.ndarray) -> bool:
    """
    Check if point is above a line.

    Args:
        point: Point to check [x, y]
        line_p1: First point on line [x, y]
        line_p2: Second point on line [x, y]

    Returns:
        True if point is above the line (smaller y value)
    """
    # Calculate line equation: y = mx + b
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]

    if abs(dx) < 1e-6:
        # Vertical line
        return point[0] < line_p1[0]

    m = dy / dx
    b = line_p1[1] - m * line_p1[0]

    line_y = m * point[0] + b
    return point[1] < line_y


def calculate_aspect_ratio(box: list[int]) -> float:
    """
    Calculate aspect ratio of bounding box.

    Args:
        box: [x1, y1, x2, y2] bounding box

    Returns:
        Width/height ratio
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    if height == 0:
        return float("inf")

    return width / height


def calculate_center(box: list[int]) -> np.ndarray:
    """
    Calculate center point of bounding box.

    Args:
        box: [x1, y1, x2, y2] bounding box

    Returns:
        Center point [x, y]
    """
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def calculate_iou(box1: list[int], box2: list[int]) -> float:
    """
    Calculate Intersection over Union (IoU) of two boxes.

    Args:
        box1: [x1, y1, x2, y2] bounding box
        box2: [x1, y1, x2, y2] bounding box

    Returns:
        IoU value (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def filter_keypoints_by_confidence(
    keypoints: np.ndarray,
    min_confidence: float = 0.5,
) -> np.ndarray:
    """
    Filter keypoints by confidence threshold.

    Args:
        keypoints: (N, 3) array with x, y, confidence
        min_confidence: Minimum confidence threshold

    Returns:
        Boolean mask of valid keypoints
    """
    return keypoints[:, 2] >= min_confidence


def get_body_center(keypoints: np.ndarray) -> Optional[np.ndarray]:
    """
    Calculate body center from keypoints.

    Args:
        keypoints: (17, 3) pose keypoints

    Returns:
        Center point [x, y] or None
    """
    valid_mask = keypoints[:, 2] > 0.5
    valid_points = keypoints[valid_mask, :2]

    if len(valid_points) == 0:
        return None

    return np.mean(valid_points, axis=0)


def calculate_body_orientation(
    keypoints: np.ndarray,
    shoulder_idx: tuple[int, int] = (5, 6),
    hip_idx: tuple[int, int] = (11, 12),
) -> float:
    """
    Calculate body orientation (0 = horizontal, 1 = vertical).

    Args:
        keypoints: (17, 3) pose keypoints
        shoulder_idx: Indices of shoulders
        hip_idx: Indices of hips

    Returns:
        Orientation value (0-1)
    """
    # Get shoulder center
    s1, s2 = shoulder_idx
    if keypoints[s1, 2] > 0.5 and keypoints[s2, 2] > 0.5:
        shoulder_center = (keypoints[s1, :2] + keypoints[s2, :2]) / 2
    elif keypoints[s1, 2] > 0.5:
        shoulder_center = keypoints[s1, :2]
    elif keypoints[s2, 2] > 0.5:
        shoulder_center = keypoints[s2, :2]
    else:
        return 0.5

    # Get hip center
    h1, h2 = hip_idx
    if keypoints[h1, 2] > 0.5 and keypoints[h2, 2] > 0.5:
        hip_center = (keypoints[h1, :2] + keypoints[h2, :2]) / 2
    elif keypoints[h1, 2] > 0.5:
        hip_center = keypoints[h1, :2]
    elif keypoints[h2, 2] > 0.5:
        hip_center = keypoints[h2, :2]
    else:
        return 0.5

    # Calculate torso vector
    torso = hip_center - shoulder_center

    # Calculate orientation (0 = horizontal, 1 = vertical)
    dx = abs(torso[0])
    dy = abs(torso[1])

    if dx + dy == 0:
        return 0.5

    return dy / (dx + dy)
