"""Event data classes for smart home events."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict


class EventType(Enum):
    """Types of events that can occur."""

    FACE_DETECTED = "face_detected"
    FACE_RECOGNIZED = "face_recognized"
    FACE_UNKNOWN = "face_unknown"
    FALL_DETECTED = "fall_detected"
    GESTURE_DETECTED = "gesture_detected"
    PERSON_DETECTED = "person_detected"


@dataclass
class BaseEvent:
    """Base class for all events."""

    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    frame_number: int = 0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "frame_number": self.frame_number,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class FaceEvent(BaseEvent):
    """Event for face detection/recognition."""

    name: str | None = None
    location: list[int] | None = None  # [x1, y1, x2, y2]
    face_count: int = 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "name": self.name,
            "location": self.location,
            "face_count": self.face_count,
        })
        return data


@dataclass
class FallEvent(BaseEvent):
    """Event for fall detection."""

    state: str = "fallen"  # standing, falling, fallen
    fall_duration: int = 0  # frames
    hip_height: float = 0.0
    aspect_ratio: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "state": self.state,
            "fall_duration": self.fall_duration,
            "hip_height": self.hip_height,
            "aspect_ratio": self.aspect_ratio,
        })
        return data


@dataclass
class GestureEvent(BaseEvent):
    """Event for gesture detection."""

    gesture_name: str = ""
    gesture_confidence: float = 0.0
    all_gestures: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "gesture_name": self.gesture_name,
            "gesture_confidence": self.gesture_confidence,
            "all_gestures": self.all_gestures,
        })
        return data


@dataclass
class PersonEvent(BaseEvent):
    """Event for person detection."""

    person_count: int = 0
    locations: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = super().to_dict()
        data.update({
            "person_count": self.person_count,
            "locations": self.locations,
        })
        return data


def create_face_detected_event(location: list[int], confidence: float, **kwargs) -> FaceEvent:
    """Create a face detected event."""
    return FaceEvent(
        event_type=EventType.FACE_DETECTED,
        location=location,
        confidence=confidence,
        **kwargs,
    )


def create_face_recognized_event(name: str, location: list[int], confidence: float, **kwargs) -> FaceEvent:
    """Create a face recognized event."""
    return FaceEvent(
        event_type=EventType.FACE_RECOGNIZED,
        name=name,
        location=location,
        confidence=confidence,
        **kwargs,
    )


def create_face_unknown_event(location: list[int], confidence: float, **kwargs) -> FaceEvent:
    """Create an unknown face event."""
    return FaceEvent(
        event_type=EventType.FACE_UNKNOWN,
        name=None,
        location=location,
        confidence=confidence,
        **kwargs,
    )


def create_fall_event(state: str, confidence: float, metrics: dict, **kwargs) -> FallEvent:
    """Create a fall detection event."""
    return FallEvent(
        event_type=EventType.FALL_DETECTED,
        state=state,
        confidence=confidence,
        hip_height=metrics.get("hip_height", 0.0),
        aspect_ratio=metrics.get("aspect_ratio", 0.0),
        fall_duration=metrics.get("fall_duration", 0),
        **kwargs,
    )


def create_gesture_event(gesture_name: str, confidence: float, all_gestures: dict, **kwargs) -> GestureEvent:
    """Create a gesture detection event."""
    return GestureEvent(
        event_type=EventType.GESTURE_DETECTED,
        gesture_name=gesture_name,
        gesture_confidence=confidence,
        all_gestures=all_gestures,
        confidence=confidence,
        **kwargs,
    )


def create_person_event(count: int, locations: list, **kwargs) -> PersonEvent:
    """Create a person detection event."""
    return PersonEvent(
        event_type=EventType.PERSON_DETECTED,
        person_count=count,
        locations=locations,
        **kwargs,
    )
