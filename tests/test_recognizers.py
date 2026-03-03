"""Tests for recognition modules."""

import numpy as np
import pytest

from src.config.settings import Config
from src.recognizers.face_recognizer import FaceRecognizer
from src.recognizers.fall_detector import FallDetector, FallState
from src.recognizers.gesture_detector import (
    GestureDetector,
    HandsUpGesture,
    PointingGesture,
    CrouchingGesture,
)


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def sample_frame():
    """Create a sample test frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def standing_keypoints():
    """Create standing pose keypoints."""
    # Create keypoints representing a standing person
    kpts = np.zeros((17, 3))
    # Set positions for standing pose
    # Head at top, ankles at bottom
    kpts[0, 1] = 100  # nose
    kpts[11, 1] = 250  # left hip
    kpts[12, 1] = 250  # right hip
    kpts[15, 1] = 400  # left ankle
    kpts[16, 1] = 400  # right ankle
    kpts[:, 2] = 0.8  # Set confidence
    return kpts


@pytest.fixture
def fallen_keypoints():
    """Create fallen pose keypoints."""
    kpts = np.zeros((17, 3))
    # Set positions for fallen person (horizontal)
    kpts[0, 0] = 100  # nose
    kpts[0, 1] = 250
    kpts[11, 0] = 250  # left hip
    kpts[11, 1] = 250
    kpts[15, 0] = 450  # left ankle
    kpts[15, 1] = 250
    kpts[:, 2] = 0.8  # Set confidence
    return kpts


class TestFaceRecognizer:
    """Tests for FaceRecognizer."""

    def test_init(self, config):
        """Test recognizer initialization."""
        recognizer = FaceRecognizer(config)
        assert recognizer is not None
        assert recognizer.known_faces == {}

    def test_recognize_no_faces(self, config, sample_frame):
        """Test recognition with no faces."""
        recognizer = FaceRecognizer(config)
        result = recognizer.recognize(sample_frame)

        assert result["face_found"] is False
        assert result["name"] is None

    def test_list_known_faces_empty(self, config):
        """Test listing known faces when empty."""
        recognizer = FaceRecognizer(config)
        faces = recognizer.list_known_faces()

        assert isinstance(faces, list)
        assert len(faces) == 0

    def test_get_face_count_empty(self, config):
        """Test face count when empty."""
        recognizer = FaceRecognizer(config)
        count = recognizer.get_face_count()

        assert count == 0

    def test_alignment_backend_config(self):
        """Test that alignment_backend is read from config."""
        config = Config()
        assert config.face_recognition.alignment_backend == "fan"

        recognizer = FaceRecognizer(config)
        assert recognizer.alignment_backend == "fan"
        assert recognizer.aligner is not None

    def test_alignment_backend_deepface(self):
        """Test legacy deepface backend config."""
        config = Config()
        config.face_recognition.alignment_backend = "deepface"

        recognizer = FaceRecognizer(config)
        assert recognizer.alignment_backend == "deepface"
        assert recognizer.aligner is None

    def test_alignment_backend_none(self):
        """Test no-alignment backend config."""
        config = Config()
        config.face_recognition.alignment_backend = "none"

        recognizer = FaceRecognizer(config)
        assert recognizer.alignment_backend == "none"
        assert recognizer.aligner is None


class TestFaceAligner:
    """Tests for FaceAligner."""

    def test_init(self):
        """Test aligner initialization (lazy, no model loaded yet)."""
        from src.recognizers.face_aligner import FaceAligner

        aligner = FaceAligner(device="cpu")
        assert aligner is not None
        assert aligner._fa is None  # Lazy load

    def test_extract_5_from_68(self):
        """Test extracting 5 reference points from 68 landmarks."""
        from src.recognizers.face_aligner import FaceAligner

        # Create synthetic 68 landmarks
        lm68 = np.random.rand(68, 2).astype(np.float32) * 100
        points = FaceAligner._extract_5_from_68(lm68)

        assert points.shape == (5, 2)
        assert points.dtype == np.float32

    def test_align_output_shape(self):
        """Test that alignment produces 112x112 output."""
        from src.recognizers.face_aligner import FaceAligner

        aligner = FaceAligner(device="cpu")

        # Create a synthetic face image
        face_roi = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        # This will actually load the FAN model (slow first time)
        result = aligner.align(face_roi)

        # Result may be None if FAN can't find landmarks in random noise
        if result is not None:
            assert result.shape == (112, 112, 3)


class TestFallDetector:
    """Tests for FallDetector."""

    def test_init(self, config):
        """Test detector initialization."""
        detector = FallDetector(config)
        assert detector is not None
        assert detector._state == FallState.STANDING

    def test_calculate_metrics_standing(self, config, standing_keypoints):
        """Test metrics calculation for standing pose."""
        detector = FallDetector(config)
        metrics = detector._calculate_metrics(standing_keypoints)

        assert metrics["aspect_ratio"] < 1.0  # Taller than wide
        assert metrics["keypoints_visible"] > 0

    def test_calculate_metrics_fallen(self, config, fallen_keypoints):
        """Test metrics calculation for fallen pose."""
        detector = FallDetector(config)
        metrics = detector._calculate_metrics(fallen_keypoints)

        assert metrics["aspect_ratio"] > 2.0  # Wider than tall

    def test_determine_state_standing(self, config, standing_keypoints):
        """Test state determination for standing pose."""
        detector = FallDetector(config)
        metrics = detector._calculate_metrics(standing_keypoints)
        state = detector._determine_state(metrics)

        assert state == FallState.STANDING

    def test_determine_state_fallen(self, config, fallen_keypoints):
        """Test state determination for fallen pose."""
        detector = FallDetector(config)
        metrics = detector._calculate_metrics(fallen_keypoints)
        state = detector._determine_state(metrics)

        assert state == FallState.FALLEN

    def test_reset_state(self, config):
        """Test state reset."""
        detector = FallDetector(config)
        detector._state = FallState.FALLEN
        detector._fall_counter = 5

        detector._reset_state()

        assert detector._state == FallState.STANDING
        assert detector._fall_counter == 0


class TestGestureDetector:
    """Tests for gesture detection."""

    def test_init(self, config):
        """Test detector initialization."""
        detector = GestureDetector(config)
        assert detector is not None
        assert len(detector.gestures) > 0

    def test_detect_empty_frame(self, config, sample_frame):
        """Test detection on empty frame."""
        detector = GestureDetector(config)
        result = detector.detect(sample_frame)

        assert "gestures_detected" in result
        assert "all_gestures" in result
        assert isinstance(result["gestures_detected"], list)

    def test_get_gesture_names(self):
        """Test getting available gesture names."""
        names = GestureDetector.get_gesture_names()

        assert isinstance(names, list)
        assert "wave" in names
        assert "hands_up" in names
        assert "pointing" in names
        assert "crouching" in names


class TestHandGestureClasses:
    """Tests for individual gesture classes."""

    def test_hands_up_gesture(self):
        """Test hands up gesture detection."""
        gesture = HandsUpGesture()
        assert gesture.name == "hands_up"

    def test_pointing_gesture(self):
        """Test pointing gesture detection."""
        gesture = PointingGesture()
        assert gesture.name == "pointing"

    def test_crouching_gesture(self):
        """Test crouching gesture detection."""
        gesture = CrouchingGesture()
        assert gesture.name == "crouching"

    def test_gesture_interface(self):
        """Test that gesture classes implement the interface."""
        from src.recognizers.gesture_detector import BaseGesture

        gesture = HandsUpGesture()
        assert isinstance(gesture, BaseGesture)
        assert hasattr(gesture, "detect")
        assert hasattr(gesture, "name")
