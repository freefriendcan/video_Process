"""Event logging system."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from .event import BaseEvent, EventType


class EventLogger:
    """
    Logger for smart home events.

    Logs events to JSONL file for structured logging and analysis.
    """

    def __init__(self, log_file: Optional[Union[str, Path]] = None):
        """
        Initialize event logger.

        Args:
            log_file: Path to log file (defaults to data/logs/events.jsonl)
        """
        if log_file is None:
            log_file = Path("data/logs/events.jsonl")
        else:
            log_file = Path(log_file)

        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure loguru
        logger.add(
            self.log_file.with_suffix(".log"),
            rotation="10 MB",
            retention="7 days",
            level="INFO",
        )

    def log_event(self, event: BaseEvent) -> None:
        """
        Log an event to JSONL file.

        Args:
            event: Event to log
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "event": event.to_dict(),
            }

            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Also log via loguru
            logger.info(f"{event.event_type.value}: {event.metadata}")

        except Exception as e:
            logger.error(f"Error logging event: {e}")

    def log_detection(
        self,
        detection_type: str,
        confidence: float,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Log a generic detection.

        Args:
            detection_type: Type of detection (e.g., "person", "face")
            confidence: Detection confidence
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "detection",
            "detection_type": detection_type,
            "confidence": confidence,
            "metadata": metadata or {},
        }

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Error logging detection: {e}")

    def log_system_event(self, event_type: str, message: str, level: str = "INFO") -> None:
        """
        Log a system event.

        Args:
            event_type: Type of system event
            message: Event message
            level: Log level (INFO, WARNING, ERROR)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "system",
            "event_type": event_type,
            "message": message,
            "level": level,
        }

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Error logging system event: {e}")

        # Also log via loguru
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"{event_type}: {message}")

    def read_events(self, limit: Optional[int] = None) -> list[dict]:
        """
        Read events from log file.

        Args:
            limit: Maximum number of events to read

        Returns:
            List of event dictionaries
        """
        events = []

        if not self.log_file.exists():
            return events

        with open(self.log_file) as f:
            for line in f:
                if limit and len(events) >= limit:
                    break
                try:
                    events.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        return events

    def get_events_by_type(self, event_type: EventType, limit: Optional[int] = None) -> list[dict]:
        """
        Get events of a specific type.

        Args:
            event_type: Type of event to filter
            limit: Maximum number of events

        Returns:
            List of matching events
        """
        all_events = self.read_events(limit)
        return [
            e for e in all_events
            if e.get("event", {}).get("event_type") == event_type.value
        ]

    def clear_logs(self) -> None:
        """Clear all event logs."""
        if self.log_file.exists():
            self.log_file.unlink()

        log_file_log = self.log_file.with_suffix(".log")
        if log_file_log.exists():
            log_file_log.unlink()
