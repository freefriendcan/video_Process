"""Integration tests for the complete pipeline."""

import numpy as np
import pytest

from src.config.settings import Config, load_config


class TestConfig:
    """Tests for configuration system."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_config()
        assert config is not None

    def test_config_has_all_sections(self):
        """Test that config has all required sections."""
        config = Config()

        assert hasattr(config, "video")
        assert hasattr(config, "detection")
        assert hasattr(config, "face_recognition")
        assert hasattr(config, "fall_detection")
        assert hasattr(config, "gesture_detection")
        assert hasattr(config, "alerts")
        assert hasattr(config, "logging")

    def test_video_config(self):
        """Test video configuration."""
        config = Config()

        assert config.video.source in ["camera", "rtsp", "file"]
        assert config.video.fps > 0
        assert len(config.video.resolution) == 2

    def test_detection_config(self):
        """Test detection configuration."""
        config = Config()

        assert 0 <= config.detection.confidence <= 1
        assert 0 <= config.detection.iou <= 1
        assert config.detection.max_detections > 0

    def test_face_recognition_config(self):
        """Test face recognition configuration."""
        config = Config()

        assert config.face_recognition.model in ["VGG-Face", "Facenet", "ArcFace"]
        assert 0 <= config.face_recognition.threshold <= 1

    def test_fall_detection_config(self):
        """Test fall detection configuration."""
        config = Config()

        assert config.fall_detection.aspect_ratio_threshold > 1.0
        assert config.fall_detection.min_frames_for_fall > 0
        assert config.fall_detection.alert_cooldown >= 0

    def test_gesture_detection_config(self):
        """Test gesture detection configuration."""
        config = Config()

        assert config.gesture_detection.smoothing_window > 0
        assert 0 <= config.gesture_detection.min_confidence <= 1


class TestEventSystem:
    """Tests for event system."""

    def test_create_face_event(self):
        """Test creating face events."""
        from src.events.event import create_face_recognized_event, create_face_unknown_event

        event1 = create_face_recognized_event("John", [10, 20, 50, 60], 0.85)
        assert event1.name == "John"
        assert event1.location == [10, 20, 50, 60]
        assert event1.confidence == 0.85

        event2 = create_face_unknown_event([10, 20, 50, 60], 0.75)
        assert event2.name is None

    def test_create_fall_event(self):
        """Test creating fall events."""
        from src.events.event import create_fall_event

        metrics = {"hip_height": 0.2, "aspect_ratio": 3.5, "fall_duration": 5}
        event = create_fall_event("fallen", 0.9, metrics)

        assert event.state == "fallen"
        assert event.hip_height == 0.2
        assert event.aspect_ratio == 3.5

    def test_create_gesture_event(self):
        """Test creating gesture events."""
        from src.events.event import create_gesture_event

        event = create_gesture_event("wave", 0.8, {"wave": 0.8, "pointing": 0.2})

        assert event.gesture_name == "wave"
        assert event.gesture_confidence == 0.8

    def test_event_to_dict(self):
        """Test event serialization."""
        from src.events.event import create_face_recognized_event

        event = create_face_recognized_event("Jane", [10, 20, 50, 60], 0.9)
        data = event.to_dict()

        assert "event_type" in data
        assert "timestamp" in data
        assert "name" in data
        assert data["name"] == "Jane"


class TestUtilities:
    """Tests for utility functions."""

    def test_geometry_distance(self):
        """Test distance calculation."""
        from src.utils.geometry import calculate_distance
        import numpy as np

        p1 = np.array([0, 0])
        p2 = np.array([3, 4])

        distance = calculate_distance(p1, p2)
        assert distance == 5.0

    def test_geometry_aspect_ratio(self):
        """Test aspect ratio calculation."""
        from src.utils.geometry import calculate_aspect_ratio

        box = [0, 0, 100, 50]  # width=100, height=50
        ratio = calculate_aspect_ratio(box)

        assert ratio == 2.0

    def test_geometry_iou(self):
        """Test IoU calculation."""
        from src.utils.geometry import calculate_iou

        box1 = [0, 0, 100, 100]
        box2 = [0, 0, 100, 100]  # Identical box

        iou = calculate_iou(box1, box2)
        assert iou == 1.0

        box3 = [200, 200, 300, 300]  # No overlap
        iou = calculate_iou(box1, box3)
        assert iou == 0.0

    def test_geometry_center(self):
        """Test center calculation."""
        from src.utils.geometry import calculate_center
        import numpy as np

        box = [0, 0, 100, 100]
        center = calculate_center(box)

        assert center[0] == 50
        assert center[1] == 50

    def test_mps_utils(self):
        """Test MPS utility functions."""
        from src.utils.mps_utils import is_apple_silicon, get_optimal_device

        # These should not crash
        is_apple = is_apple_silicon()
        assert isinstance(is_apple, bool)

        device = get_optimal_device()
        assert device in ["mps", "cpu", "cuda"]


class TestModelLoader:
    """Tests for model loading."""

    def test_model_loader_init(self, tmp_path):
        """Test model loader initialization."""
        from src.models.model_loader import ModelLoader

        loader = ModelLoader(tmp_path)
        assert loader.models_dir == tmp_path

    def test_get_model_path(self, tmp_path):
        """Test getting model path."""
        from src.models.model_loader import ModelLoader

        loader = ModelLoader(tmp_path)
        path = loader.get_model_path("yolov8n.pt")

        assert "yolov8n.pt" in str(path)
        assert tmp_path in str(path)

    def test_list_available_models(self):
        """Test listing available models."""
        from src.models.model_loader import list_available_models

        models = list_available_models()

        assert isinstance(models, list)
        assert "yolov8n.pt" in models
        assert "yolov8n-pose.pt" in models
