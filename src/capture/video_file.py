"""Video file capture for processing pre-recorded videos."""

from pathlib import Path
from typing import Optional, Union

import cv2

from .base import BaseCapture


class VideoFileCapture(BaseCapture):
    """
    Video capture from file for processing pre-recorded videos.

    Supports any video format that OpenCV can read:
    - MP4, AVI, MOV, MKV, etc.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        loop: bool = False,
        fps_override: Optional[int] = None,
    ):
        """
        Initialize video file capture.

        Args:
            file_path: Path to video file
            loop: Whether to loop the video when it ends
            fps_override: Override the video's native FPS
        """
        self.file_path = Path(file_path)
        self.loop = loop
        self.fps_override = fps_override

        # Will be set when file is opened
        self._cap: Optional[cv2.VideoCapture] = None
        self._total_frames = 0
        self._current_frame = 0

        super().__init__(str(file_path), fps=0, resolution=(0, 0))

    def open(self) -> bool:
        """
        Open the video file.

        Returns:
            True if successfully opened, False otherwise
        """
        if not self.file_path.exists():
            print(f"Video file not found: {self.file_path}")
            return False

        try:
            self._cap = cv2.VideoCapture(str(self.file_path))

            if not self._cap.isOpened():
                return False

            # Get video properties
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self.resolution = (width, height)
            self.fps = int(fps) if self.fps_override is None else self.fps_override
            self._current_frame = 0
            self._is_opened = True

            return True

        except Exception as e:
            print(f"Error opening video file: {e}")
            return False

    def close(self) -> None:
        """Close the video file."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_opened = False

    def read(self):
        """
        Read a single frame from the video file.

        Returns:
            Frame as numpy array or None if:
            - End of file reached (and loop is False)
            - Read failed
        """
        if not self.is_opened():
            return None

        ret, frame = self._cap.read()
        self._current_frame += 1

        if not ret:
            if self.loop and self._total_frames > 0:
                # Reset to beginning
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._current_frame = 0
                ret, frame = self._cap.read()
                return frame if ret else None
            return None

        return frame

    def is_opened(self) -> bool:
        """Check if video file is opened."""
        return self._is_opened and self._cap is not None and self._cap.isOpened()

    @property
    def width(self) -> int:
        """Get frame width."""
        return self.resolution[0]

    @property
    def height(self) -> int:
        """Get frame height."""
        return self.resolution[1]

    @property
    def total_frames(self) -> int:
        """Get total number of frames in the video."""
        return self._total_frames

    @property
    def current_frame(self) -> int:
        """Get current frame position."""
        return self._current_frame

    @property
    def progress(self) -> float:
        """Get video playback progress (0.0 to 1.0)."""
        if self._total_frames > 0:
            return self._current_frame / self._total_frames
        return 0.0

    @property
    def duration(self) -> float:
        """Get video duration in seconds."""
        if self.fps > 0:
            return self._total_frames / self.fps
        return 0.0

    def seek(self, frame_number: int) -> bool:
        """
        Seek to a specific frame.

        Args:
            frame_number: Frame number to seek to

        Returns:
            True if successful
        """
        if self._cap and 0 <= frame_number < self._total_frames:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self._current_frame = frame_number
            return True
        return False

    def seek_time(self, seconds: float) -> bool:
        """
        Seek to a specific time in seconds.

        Args:
            seconds: Time position in seconds

        Returns:
            True if successful
        """
        if self._cap and self.fps > 0:
            frame_number = int(seconds * self.fps)
            return self.seek(frame_number)
        return False

    def __repr__(self) -> str:
        return f"VideoFileCapture(file={self.file_path.name}, frames={self._total_frames})"
