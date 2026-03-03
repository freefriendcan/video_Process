"""Detection module."""

from .face import FaceDetector
from .person import PersonDetector
from .pose import PoseDetector
from .yolo_base import BaseDetector

__all__ = [
    "BaseDetector",
    "FaceDetector",
    "PersonDetector",
    "PoseDetector",
]
