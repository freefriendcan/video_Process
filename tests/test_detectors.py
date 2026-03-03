"""Tests for detection modules."""

import numpy as np
import pytest

from src.config.settings import Config
from src.detectors.person import PersonDetector
from src.detectors.face import FaceDetector
from src.detectors.pose import PoseDetector


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def sample_frame():
    """Create a sample test frame."""
    # Create a blank frame
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_with_person():
    """Create a frame with a person-like shape."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a rough person shape (rectangle)
    cv2 = pytest.importorskip("cv2")
    cv2.rectangle(frame, (200, 100), (440, 450), (255, 255, 255), -1)
    return frame


class TestPersonDetector:
    """Tests for PersonDetector."""

    def test_init(self, config):
        """Test detector initialization."""
        detector = PersonDetector(config)
        assert detector is not None
        assert detector.model_name == "yolov10n.pt"

    def test_detect_empty_frame(self, config, sample_frame):
        """Test detection on empty frame."""
        detector = PersonDetector(config)
        result = detector.detect(sample_frame)

        assert "persons_found" in result
        assert "count" in result
        assert result["count"] == 0
        assert result["persons_found"] is False

    def test_detect_none_frame(self, config):
        """Test detection with None frame."""
        detector = PersonDetector(config)
        result = detector.detect(None)

        assert result["persons_found"] is False
        assert result["count"] == 0

    def test_count_persons(self, config, sample_frame):
        """Test person counting."""
        detector = PersonDetector(config)
        count = detector.count_persons(sample_frame)

        assert isinstance(count, int)
        assert count >= 0


class TestFaceDetector:
    """Tests for FaceDetector."""

    def test_init(self, config):
        """Test detector initialization."""
        detector = FaceDetector(config)
        assert detector is not None
        assert detector.model_name == "yolov8n-face.pt"

    def test_detect_empty_frame(self, config, sample_frame):
        """Test detection on empty frame."""
        detector = FaceDetector(config)
        result = detector.detect(sample_frame)

        assert "faces_found" in result
        assert "count" in result
        assert result["count"] == 0

    def test_extract_face_roi_empty(self, config, sample_frame):
        """Test face ROI extraction on empty frame."""
        detector = FaceDetector(config)
        rois = detector.extract_face_roi(sample_frame)

        assert isinstance(rois, list)
        assert len(rois) == 0


class TestPoseDetector:
    """Tests for PoseDetector."""

    def test_init(self, config):
        """Test detector initialization."""
        detector = PoseDetector(config)
        assert detector is not None
        assert detector.model_name == "yolov8n-pose.pt"

    def test_detect_empty_frame(self, config, sample_frame):
        """Test detection on empty frame."""
        detector = PoseDetector(config)
        result = detector.detect(sample_frame)

        assert "persons_found" in result
        assert "poses" in result

    def test_keypoint_names(self):
        """Test keypoint name definitions."""
        assert len(PoseDetector.KEYPOINT_NAMES) == 17
        assert PoseDetector.KEYPOINT_NAMES[0] == "nose"
        assert PoseDetector.KEYPOINT_NAMES[-1] == "right_ankle"

    def test_skeleton_pairs(self):
        """Test skeleton pair definitions."""
        assert len(PoseDetector.SKELETON_PAIRS) > 0
        # Each pair should be a tuple of 2 indices
        for pair in PoseDetector.SKELETON_PAIRS:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert all(isinstance(i, int) for i in pair)


class TestDeviceDetection:
    """Tests for device detection utilities."""

    def test_get_device(self, config):
        """Test device selection."""
        from src.config.settings import get_device

        device = get_device(config)
        assert device in ["mps", "cpu", "cuda"]

    def test_device_info(self):
        """Test device info retrieval."""
        from src.detectors.yolo_base import BaseDetector

        info = BaseDetector.get_device_info()
        assert isinstance(info, dict)
        assert "mps_available" in info
        assert "cuda_available" in info
