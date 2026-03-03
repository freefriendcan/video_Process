"""Event handler for aggregating and dispatching smart home events."""

import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import numpy as np

from .event import BaseEvent, EventType
from .logger import EventLogger


class EventHandler:
    """
    Central event handler for aggregating and dispatching events.

    Features:
    - Event deduplication (suppress duplicate events within cooldown)
    - Multi-listener dispatch (alerters, loggers, etc.)
    - Screenshot capture on events
    """

    def __init__(
        self,
        config,
        logger: Optional[EventLogger] = None,
        screenshot_path: Optional[Union[str, Path]] = None,
        enable_screenshots: bool = True,
    ):
        """
        Initialize event handler.

        Args:
            config: Configuration object
            logger: EventLogger instance
            screenshot_path: Path to save screenshots
            enable_screenshots: Whether to capture screenshots on events
        """
        self.config = config
        self.logger = logger
        self.screenshot_path = Path(screenshot_path) if screenshot_path else Path("data/logs/screenshots")
        self.enable_screenshots = enable_screenshots

        # Create screenshot directory
        if self.enable_screenshots:
            self.screenshot_path.mkdir(parents=True, exist_ok=True)

        # Event listeners
        self.listeners: dict[EventType, list[Callable]] = defaultdict(list)
        self.global_listeners: list[Callable] = []

        # Deduplication tracking
        self._last_events: dict[EventType, tuple[datetime, dict]] = {}
        self._cooldowns: dict[EventType, float] = {
            EventType.FACE_DETECTED: 5.0,
            EventType.FACE_RECOGNIZED: 10.0,
            EventType.FALL_DETECTED: 30.0,
            EventType.GESTURE_DETECTED: 3.0,
            EventType.PERSON_DETECTED: 10.0,
        }

        # Frame counter
        self._frame_count = 0

    def add_listener(self, event_type: EventType, callback: Callable) -> None:
        """
        Add a listener for specific event type.

        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
        """
        self.listeners[event_type].append(callback)

    def add_global_listener(self, callback: Callable) -> None:
        """
        Add a global listener for all events.

        Args:
            callback: Function to call for any event
        """
        self.global_listeners.append(callback)

    def handle_event(self, event: BaseEvent, frame: Optional[np.ndarray] = None) -> bool:
        """
        Process an event: check deduplication, log, and dispatch.

        Args:
            event: Event to handle
            frame: Optional frame for screenshot capture

        Returns:
            True if event was dispatched (not suppressed by cooldown)
        """
        event.frame_number = self._frame_count
        current_time = datetime.now()

        # Check cooldown for deduplication
        last_time, last_data = self._last_events.get(event.event_type, (None, {}))
        cooldown = self._cooldowns.get(event.event_type, 0.0)

        if last_time:
            elapsed = (current_time - last_time).total_seconds()
            if elapsed < cooldown:
                # Check if it's essentially the same event
                if self._is_duplicate(event, last_data):
                    return False

        # Update last event tracking
        self._last_events[event.event_type] = (current_time, event.to_dict())

        # Log event
        if self.logger:
            self.logger.log_event(event)

        # Capture screenshot if enabled
        if self.enable_screenshots and frame is not None:
            self._capture_screenshot(event, frame)

        # Dispatch to type-specific listeners
        for callback in self.listeners.get(event.event_type, []):
            try:
                callback(event)
            except Exception as e:
                print(f"Error in event listener: {e}")

        # Dispatch to global listeners
        for callback in self.global_listeners:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in global listener: {e}")

        return True

    def _is_duplicate(self, event: BaseEvent, last_data: dict) -> bool:
        """
        Check if event is a duplicate of the last one.

        Args:
            event: Current event
            last_data: Last event data

        Returns:
            True if likely a duplicate
        """
        # For face events, check if same person at similar location
        if event.event_type in (EventType.FACE_RECOGNIZED, EventType.FACE_UNKNOWN):
            event_name = getattr(event, "name", None)
            last_name = last_data.get("name")
            return event_name == last_name

        # For fall events, always allow through (important)
        if event.event_type == EventType.FALL_DETECTED:
            return False

        # For gestures, check if same gesture
        if event.event_type == EventType.GESTURE_DETECTED:
            event_gesture = getattr(event, "gesture_name", None)
            last_gesture = last_data.get("gesture_name")
            return event_gesture == last_gesture

        return False

    def _capture_screenshot(self, event: BaseEvent, frame: np.ndarray) -> None:
        """
        Capture and save screenshot on event.

        Args:
            event: Event that triggered screenshot
            frame: Frame to capture
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        event_type = event.event_type.value
        filename = f"{event_type}_{timestamp}.jpg"
        filepath = self.screenshot_path / filename

        try:
            # Draw event info on frame
            annotated = frame.copy()
            self._annotate_frame(annotated, event)

            cv2.imwrite(str(filepath), annotated)
        except Exception as e:
            print(f"Error saving screenshot: {e}")

    def _annotate_frame(self, frame: np.ndarray, event: BaseEvent) -> None:
        """
        Draw event information on frame.

        Args:
            frame: Frame to annotate
            event: Event information
        """
        h, w = frame.shape[:2]
        y_pos = 30

        # Draw event type
        text = f"Event: {event.event_type.value}"
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30

        # Draw timestamp
        text = f"Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30

        # Event-specific info
        if event.event_type == EventType.FACE_RECOGNIZED:
            text = f"Person: {getattr(event, 'name', 'Unknown')}"
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif event.event_type == EventType.FALL_DETECTED:
            text = f"State: {getattr(event, 'state', 'Unknown')}"
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif event.event_type == EventType.GESTURE_DETECTED:
            text = f"Gesture: {getattr(event, 'gesture_name', 'Unknown')}"
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def set_cooldown(self, event_type: EventType, seconds: float) -> None:
        """
        Set cooldown period for event type.

        Args:
            event_type: Type of event
            seconds: Cooldown in seconds
        """
        self._cooldowns[event_type] = seconds

    def increment_frame(self) -> None:
        """Increment frame counter."""
        self._frame_count += 1

    def reset(self) -> None:
        """Reset event tracking."""
        self._last_events.clear()
        self._frame_count = 0
