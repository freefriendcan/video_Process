"""Abstract base class for video capture."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseCapture(ABC):
    """Abstract interface for video capture sources."""

    def __init__(self, source: str, fps: int = 30, resolution: Tuple[int, int] = (1280, 720)):
        """
        Initialize the capture source.

        Args:
            source: Source identifier (camera index, RTSP URL, file path)
            fps: Target frames per second
            resolution: Target resolution (width, height)
        """
        self.source = source
        self.fps = fps
        self.resolution = resolution
        self._is_opened = False

    @abstractmethod
    def open(self) -> bool:
        """
        Open the capture source.

        Returns:
            True if successfully opened, False otherwise
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the capture source and release resources."""
        pass

    @abstractmethod
    def read(self):
        """
        Read a single frame.

        Returns:
            Frame object or None if no frame available
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """
        Check if capture source is opened.

        Returns:
            True if opened, False otherwise
        """
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        """Get frame width."""
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        """Get frame height."""
        pass

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self.source})"
