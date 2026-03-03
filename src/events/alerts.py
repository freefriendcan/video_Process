"""Alert generation for smart home events."""

import json
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from .event import BaseEvent, EventType


class Alerter:
    """
    Base class for alert generators.

    Subclasses implement specific alert mechanisms (console, notification, etc.)
    """

    def send_alert(self, event: BaseEvent) -> None:
        """
        Send an alert for an event.

        Args:
            event: Event to alert about
        """
        raise NotImplementedError


class ConsoleAlerter(Alerter):
    """Console-based alerting."""

    def __init__(self, enable_colors: bool = True):
        """
        Initialize console alerter.

        Args:
            enable_colors: Whether to use colored output
        """
        self.enable_colors = enable_colors

        # Color codes
        self.colors = {
            EventType.FALL_DETECTED: "\033[91m",  # Red
            EventType.FACE_RECOGNIZED: "\033[92m",  # Green
            EventType.GESTURE_DETECTED: "\033[93m",  # Yellow
            EventType.FACE_UNKNOWN: "\033[94m",  # Blue
        }
        self.reset = "\033[0m" if enable_colors else ""

    def send_alert(self, event: BaseEvent) -> None:
        """Send alert to console."""
        color = self.colors.get(event.event_type, "")
        reset = self.reset if self.enable_colors else ""

        message = self._format_message(event)
        prefix = f"{color}[ALERT]{reset}" if self.enable_colors else "[ALERT]"

        print(f"{prefix} {message}")
        logger.info(f"Alert: {event.event_type.value} - {message}")

    def _format_message(self, event: BaseEvent) -> str:
        """Format event as readable message."""
        timestamp = event.timestamp.strftime("%H:%M:%S")

        if event.event_type == EventType.FALL_DETECTED:
            state = getattr(event, "state", "unknown")
            return f"{timestamp} - FALL DETECTED! State: {state}"
        elif event.event_type == EventType.FACE_RECOGNIZED:
            name = getattr(event, "name", "Unknown")
            return f"{timestamp} - Face recognized: {name}"
        elif event.event_type == EventType.FACE_UNKNOWN:
            return f"{timestamp} - Unknown face detected"
        elif event.event_type == EventType.GESTURE_DETECTED:
            gesture = getattr(event, "gesture_name", "unknown")
            return f"{timestamp} - Gesture detected: {gesture}"
        else:
            return f"{timestamp} - {event.event_type.value}"


class FileAlerter(Alerter):
    """File-based alert logging."""

    def __init__(self, log_file: Union[str, Path]):
        """
        Initialize file alerter.

        Args:
            log_file: Path to alert log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def send_alert(self, event: BaseEvent) -> None:
        """Write alert to file."""
        alert_data = {
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "confidence": event.confidence,
            **event.metadata,
        }

        # Add event-specific fields
        if hasattr(event, "name"):
            alert_data["name"] = event.name
        if hasattr(event, "gesture_name"):
            alert_data["gesture"] = event.gesture_name

        with open(self.log_file, "a") as f:
            f.write(json.dumps(alert_data) + "\n")


class CallbackAlerter(Alerter):
    """Alert via callback function."""

    def __init__(self, callback):
        """
        Initialize callback alerter.

        Args:
            callback: Function to call with event
        """
        self.callback = callback

    def send_alert(self, event: BaseEvent) -> None:
        """Call callback with event."""
        try:
            self.callback(event)
        except Exception as e:
            logger.error(f"Error in callback alerter: {e}")


class CompositeAlerter(Alerter):
    """Composite alerter that delegates to multiple alerters."""

    def __init__(self, alerters: Optional[list[Alerter]] = None):
        """
        Initialize composite alerter.

        Args:
            alerters: List of alerters to delegate to
        """
        self.alerters = alerters or []

    def add_alerter(self, alerter: Alerter) -> None:
        """Add an alerter to the composite."""
        self.alerters.append(alerter)

    def send_alert(self, event: BaseEvent) -> None:
        """Send alert via all registered alerters."""
        for alerter in self.alerters:
            try:
                alerter.send_alert(event)
            except Exception as e:
                logger.error(f"Error in alerter {alerter.__class__.__name__}: {e}")


def create_default_alerter(config) -> CompositeAlerter:
    """
    Create default alerter from configuration.

    Args:
        config: Configuration object

    Returns:
        CompositeAlerter with console and file alerters
    """
    alerter = CompositeAlerter()

    # Console alerter
    if config.alerts.console:
        alerter.add_alerter(ConsoleAlerter())

    # File alerter
    log_file = config.alerts.log_file
    if log_file:
        alerter.add_alerter(FileAlerter(log_file))

    return alerter
