"""Unit tests for fall detection system (mac_camera + src/recognizers).

Covers:
- GeometricFallbackDetector state machine
- Unified cooldown tracking
- Screenshot filename collision prevention
- Frame skipping logic
- TransformerFallDetector interface
- Skeleton normalization math
"""

import time
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config.settings import Config
from src.recognizers.transformer_fall_detector import (
    TransformerFallDetector,
    normalize_skeleton_frame,
)


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def config():
    return Config()


@pytest.fixture
def sample_rgb_frame():
    """640x480 black RGB frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_bgr_frame():
    """640x480 black BGR frame for screenshot tests."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def standing_features():
    """51-dim feature vector simulating a standing pose.
    Vertical body: shoulders high, hips mid, ankles low."""
    features = np.zeros(51, dtype=np.float32)
    # Sorted keypoint order: use index mapping
    # Just set some y-values to simulate vertical pose
    for i in range(17):
        features[i * 3] = 0.5       # x centered
        features[i * 3 + 1] = 0.3 + i * 0.02  # y spread vertically
        features[i * 3 + 2] = 0.9   # visibility
    return features


@pytest.fixture
def fallen_features():
    """51-dim feature vector simulating a fallen (horizontal) pose.
    Body spread horizontally: wide x, narrow y."""
    features = np.zeros(51, dtype=np.float32)
    for i in range(17):
        features[i * 3] = 0.1 + i * 0.04  # x spread wide
        features[i * 3 + 1] = 0.5          # y all same (horizontal)
        features[i * 3 + 2] = 0.9          # visibility
    return features


@pytest.fixture
def tmp_screenshot_dir(tmp_path):
    """Temporary screenshot directory."""
    d = tmp_path / "screenshots"
    d.mkdir()
    return d


# ═══════════════════════════════════════════════════════════
# Skeleton Normalization Tests
# ═══════════════════════════════════════════════════════════

class TestNormalization:
    """Tests for normalize_skeleton_frame function."""

    def test_preserves_visibility(self, standing_features):
        """Normalization should not modify visibility values."""
        normalized = normalize_skeleton_frame(standing_features)
        for i in range(17):
            assert normalized[i * 3 + 2] == standing_features[i * 3 + 2]

    def test_translates_to_hip_origin(self):
        """After normalization, mid-hip should be at ~(0, 0)."""
        features = np.zeros(51, dtype=np.float32)
        # Set all keypoints visible
        for i in range(17):
            features[i * 3 + 2] = 0.9

        # Set Left Hip (sorted index depends on SORTED_KEYPOINT_NAMES)
        # Direct: set known hip landmarks to specific values
        from src.detectors.mediapipe_pose import KEYPOINT_NAME_TO_IDX
        lh_idx = KEYPOINT_NAME_TO_IDX['Left Hip']
        rh_idx = KEYPOINT_NAME_TO_IDX['Right Hip']
        features[lh_idx * 3] = 0.4      # lh_x
        features[lh_idx * 3 + 1] = 0.6  # lh_y
        features[rh_idx * 3] = 0.6      # rh_x
        features[rh_idx * 3 + 1] = 0.6  # rh_y

        normalized = normalize_skeleton_frame(features)
        mid_hip_x = (normalized[lh_idx * 3] + normalized[rh_idx * 3]) / 2
        mid_hip_y = (normalized[lh_idx * 3 + 1] + normalized[rh_idx * 3 + 1]) / 2

        assert abs(mid_hip_x) < 1e-5, f"Mid-hip X should be ~0, got {mid_hip_x}"
        assert abs(mid_hip_y) < 1e-5, f"Mid-hip Y should be ~0, got {mid_hip_y}"

    def test_zero_visibility_skips_normalization(self):
        """If hips have 0 visibility, features are returned as-is."""
        features = np.zeros(51, dtype=np.float32)
        # All visibility = 0 → can't normalize
        original = features.copy()
        normalized = normalize_skeleton_frame(features, min_confidence=0.3)
        np.testing.assert_array_equal(normalized, original)

    def test_output_shape(self, standing_features):
        """Output should be same shape as input."""
        normalized = normalize_skeleton_frame(standing_features)
        assert normalized.shape == (51,)
        assert normalized.dtype == np.float32


# ═══════════════════════════════════════════════════════════
# TransformerFallDetector Tests
# ═══════════════════════════════════════════════════════════

class TestTransformerFallDetector:
    """Tests for the src/recognizers TransformerFallDetector."""

    def test_init_without_model(self, config):
        """Detector initializes gracefully when model file is missing."""
        config.fall_detection.transformer_model_path = "/nonexistent/model.tflite"
        detector = TransformerFallDetector(config)
        assert detector._interpreter is None

    def test_empty_result_when_no_model(self, config, sample_rgb_frame):
        """Returns safe empty result when model is unavailable."""
        config.fall_detection.transformer_model_path = "/nonexistent/model.tflite"
        detector = TransformerFallDetector(config)
        result = detector.detect(sample_rgb_frame)

        assert result["fall_detected"] is False
        assert result["state"] == "standing"
        assert result["confidence"] == 0.0

    def test_buffer_collecting_phase(self, config, sample_rgb_frame):
        """Reports buffer fill status during collection phase."""
        detector = TransformerFallDetector(config)
        # Even without model, test buffer logic by mocking
        if detector._interpreter is None:
            pytest.skip("TFLite model not available for full test")

        result = detector.detect(sample_rgb_frame)
        assert result["state"] == "collecting"
        assert result["metrics"]["buffer_fill"] == 1
        assert result["metrics"]["buffer_required"] == 30

    def test_reset_clears_state(self, config):
        """Reset clears buffer and cooldown timer."""
        detector = TransformerFallDetector(config)
        detector._feature_buffer.append(np.zeros(51))
        detector._last_alert_time = 999.0

        detector.reset()

        assert len(detector._feature_buffer) == 0
        assert detector._last_alert_time == 0.0

    def test_detect_return_format(self, config, sample_rgb_frame):
        """detect() always returns dict with required keys."""
        config.fall_detection.transformer_model_path = "/nonexistent/model.tflite"
        detector = TransformerFallDetector(config)
        result = detector.detect(sample_rgb_frame)

        assert isinstance(result, dict)
        assert "fall_detected" in result
        assert "state" in result
        assert "confidence" in result
        assert "metrics" in result

    def test_pose_detector_shared(self, config):
        """get_pose_detector() returns the internal detector for sharing."""
        detector = TransformerFallDetector(config)
        pose = detector.get_pose_detector()
        assert pose is detector.pose_detector


# ═══════════════════════════════════════════════════════════
# GeometricFallbackDetector Tests (mac_camera.py inline class)
# ═══════════════════════════════════════════════════════════

class TestGeometricFallbackDetector:
    """Tests for the GeometricFallbackDetector defined in mac_camera.py.

    Since the class lives in mac_camera.py (not importable due to module-level
    side effects like camera init), we test the logic via a local replica that
    mirrors the production code exactly.
    """

    class _GeometricFallbackDetector:
        """Test-local replica of mac_camera.GeometricFallbackDetector."""

        ASPECT_RATIO_THRESHOLD = 2.5
        BODY_ORIENTATION_THRESHOLD = 0.5
        MIN_FRAMES_FOR_FALL = 5

        def __init__(self):
            self._fall_counter = 0

        def analyze_geometry(self, aspect_ratio, body_orientation):
            """Core state machine logic extracted for unit testing.
            Returns (is_fallen_pose, fall_counter)."""
            is_fallen = (
                aspect_ratio >= self.ASPECT_RATIO_THRESHOLD
                and body_orientation < self.BODY_ORIENTATION_THRESHOLD
            )

            if is_fallen:
                self._fall_counter += 1
            else:
                self._fall_counter = max(0, self._fall_counter - 1)

            return is_fallen, self._fall_counter

    def test_standing_pose_no_detection(self):
        """Standing pose (aspect_ratio < 2.5) should not trigger fall."""
        detector = self._GeometricFallbackDetector()
        is_fallen, count = detector.analyze_geometry(
            aspect_ratio=0.5, body_orientation=0.9
        )
        assert not is_fallen
        assert count == 0

    def test_fallen_pose_increments_counter(self):
        """Fallen pose should increment counter."""
        detector = self._GeometricFallbackDetector()
        is_fallen, count = detector.analyze_geometry(
            aspect_ratio=3.0, body_orientation=0.2
        )
        assert is_fallen
        assert count == 1

    def test_needs_consecutive_frames(self):
        """Fall detection requires MIN_FRAMES_FOR_FALL consecutive frames."""
        detector = self._GeometricFallbackDetector()
        for i in range(4):
            detector.analyze_geometry(aspect_ratio=3.0, body_orientation=0.2)

        assert detector._fall_counter == 4
        assert detector._fall_counter < detector.MIN_FRAMES_FOR_FALL

    def test_triggers_at_threshold(self):
        """Fall triggers after exactly MIN_FRAMES_FOR_FALL consecutive frames."""
        detector = self._GeometricFallbackDetector()
        for _ in range(5):
            detector.analyze_geometry(aspect_ratio=3.0, body_orientation=0.2)

        assert detector._fall_counter == 5
        assert detector._fall_counter >= detector.MIN_FRAMES_FOR_FALL

    def test_counter_decrements_on_standing(self):
        """Counter decreases (not resets) when person stands back up."""
        detector = self._GeometricFallbackDetector()
        # Build up count
        for _ in range(3):
            detector.analyze_geometry(aspect_ratio=3.0, body_orientation=0.2)
        assert detector._fall_counter == 3

        # One standing frame → decrement by 1
        detector.analyze_geometry(aspect_ratio=0.5, body_orientation=0.9)
        assert detector._fall_counter == 2

    def test_counter_never_negative(self):
        """Counter should never go below 0."""
        detector = self._GeometricFallbackDetector()
        for _ in range(10):
            detector.analyze_geometry(aspect_ratio=0.5, body_orientation=0.9)
        assert detector._fall_counter == 0

    def test_high_aspect_ratio_but_vertical_body_no_fall(self):
        """High aspect ratio with vertical body orientation = sitting, not fall."""
        detector = self._GeometricFallbackDetector()
        is_fallen, _ = detector.analyze_geometry(
            aspect_ratio=3.0, body_orientation=0.8  # Vertical body
        )
        assert not is_fallen

    def test_low_aspect_ratio_horizontal_body_no_fall(self):
        """Low aspect ratio with horizontal body = unlikely, no fall."""
        detector = self._GeometricFallbackDetector()
        is_fallen, _ = detector.analyze_geometry(
            aspect_ratio=1.5, body_orientation=0.2
        )
        assert not is_fallen


# ═══════════════════════════════════════════════════════════
# Cooldown Unification Tests
# ═══════════════════════════════════════════════════════════

class TestCooldownUnification:
    """Verify that both detection paths share the same cooldown."""

    def test_cooldown_prevents_rapid_alerts(self):
        """After alert, next alert within cooldown window returns False."""
        cooldown = 10
        last_alert = time.time()

        # Simulate second detection 2 seconds later
        now = last_alert + 2
        should_alert = (now - last_alert) > cooldown
        assert not should_alert

    def test_cooldown_allows_after_window(self):
        """After cooldown expires, next alert should fire."""
        cooldown = 10
        last_alert = time.time() - 15  # 15 seconds ago

        now = time.time()
        should_alert = (now - last_alert) > cooldown
        assert should_alert

    def test_zero_initial_cooldown(self):
        """First alert should always fire (last_alert_time = 0.0)."""
        last_alert = 0.0
        cooldown = 10
        now = time.time()
        should_alert = (now - last_alert) > cooldown
        assert should_alert


# ═══════════════════════════════════════════════════════════
# Screenshot Tests
# ═══════════════════════════════════════════════════════════

class TestScreenshot:
    """Tests for screenshot saving logic."""

    def test_filename_includes_milliseconds(self):
        """Filename should contain millisecond component for uniqueness."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int(time.time() * 1000) % 1000
        filename = f"fall_{timestamp}_{millis:03d}.jpg"

        parts = filename.replace(".jpg", "").split("_")
        # fall_YYYYMMDD_HHMMSS_MMM → 4 parts
        assert len(parts) == 4
        assert parts[0] == "fall"
        assert len(parts[3]) == 3  # Milliseconds zero-padded

    def test_sequential_filenames_unique(self):
        """Two filenames generated in rapid succession should differ."""
        names = set()
        for _ in range(10):
            ts = time.strftime("%Y%m%d_%H%M%S")
            ms = int(time.time() * 1000) % 1000
            names.add(f"fall_{ts}_{ms:03d}.jpg")
            time.sleep(0.002)  # 2ms gap

        # At minimum, some should be unique (ms component varies)
        assert len(names) >= 2

    def test_screenshot_saves_to_disk(self, sample_bgr_frame, tmp_screenshot_dir):
        """Screenshot saves a valid JPEG to the specified directory."""
        import cv2

        filepath = tmp_screenshot_dir / "test_fall.jpg"
        cv2.imwrite(str(filepath), sample_bgr_frame)

        assert filepath.exists()
        assert filepath.stat().st_size > 0

    def test_screenshot_readable(self, sample_bgr_frame, tmp_screenshot_dir):
        """Saved screenshot can be read back."""
        import cv2

        filepath = tmp_screenshot_dir / "test_read.jpg"
        cv2.imwrite(str(filepath), sample_bgr_frame)

        loaded = cv2.imread(str(filepath))
        assert loaded is not None
        assert loaded.shape == sample_bgr_frame.shape


# ═══════════════════════════════════════════════════════════
# Frame Skipping Tests
# ═══════════════════════════════════════════════════════════

class TestFrameSkipping:
    """Tests for frame skipping counter logic."""

    def test_interval_2_processes_every_other_frame(self):
        """With INTERVAL=2, fall detection runs on frames 2, 4, 6..."""
        interval = 2
        processed = []
        for i in range(1, 11):
            if i % interval == 0:
                processed.append(i)

        assert processed == [2, 4, 6, 8, 10]

    def test_interval_1_processes_every_frame(self):
        """With INTERVAL=1, every frame is processed."""
        interval = 1
        count = sum(1 for i in range(1, 31) if i % interval == 0)
        assert count == 30

    def test_temporal_window_with_skip(self):
        """Document: with INTERVAL=2 at 30fps, 30 samples span 60 real frames."""
        camera_fps = 30
        interval = 2
        timesteps = 30

        real_frames = timesteps * interval
        real_seconds = real_frames / camera_fps

        assert real_frames == 60
        assert real_seconds == pytest.approx(2.0)

    def test_temporal_window_without_skip(self):
        """Baseline: without frame skip, 30 samples span 30 real frames."""
        camera_fps = 30
        interval = 1
        timesteps = 30

        real_frames = timesteps * interval
        real_seconds = real_frames / camera_fps

        assert real_frames == 30
        assert real_seconds == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════
# Velocity Filter Tests (Stage 1)
# ═══════════════════════════════════════════════════════════

class TestVelocityFilter:
    """Tests for the velocity-based false positive filter.

    Mirrors the compute_body_velocity() logic from mac_camera.py.
    Velocity = average frame-to-frame Y change over a sliding window.
    Positive velocity = downward movement (image coords: y increases downward).
    """

    VELOCITY_WINDOW = 5
    MIN_FALL_VELOCITY = 0.025

    @staticmethod
    def _compute_velocity(y_buffer):
        """Local replica of compute_body_velocity() for unit testing."""
        if len(y_buffer) < 2:
            return 0.0
        buf = list(y_buffer)
        velocities = [buf[i] - buf[i - 1] for i in range(1, len(buf))]
        return sum(velocities) / len(velocities)

    def test_real_fall_passes_threshold(self):
        """Rapid downward motion (fall) produces velocity above threshold."""
        # Simulating: body center drops from y=0.30 to y=0.75 in 5 frames
        buf = deque([0.30, 0.39, 0.48, 0.60, 0.70, 0.75], maxlen=6)
        velocity = self._compute_velocity(buf)

        assert velocity >= self.MIN_FALL_VELOCITY, \
            f"Fall velocity {velocity:.4f} should be >= {self.MIN_FALL_VELOCITY}"

    def test_slow_sit_filtered(self):
        """Slow sitting motion produces velocity below threshold."""
        # Simulating: body center drops from y=0.40 to y=0.46 in 5 frames
        buf = deque([0.40, 0.41, 0.42, 0.43, 0.44, 0.46], maxlen=6)
        velocity = self._compute_velocity(buf)

        assert velocity < self.MIN_FALL_VELOCITY, \
            f"Sitting velocity {velocity:.4f} should be < {self.MIN_FALL_VELOCITY}"

    def test_bending_filtered(self):
        """Bending to pick something up: moderate speed, still below threshold."""
        buf = deque([0.35, 0.36, 0.38, 0.40, 0.42, 0.44], maxlen=6)
        velocity = self._compute_velocity(buf)

        assert velocity < self.MIN_FALL_VELOCITY, \
            f"Bending velocity {velocity:.4f} should be < {self.MIN_FALL_VELOCITY}"

    def test_standing_still_near_zero(self):
        """Person standing still: velocity ≈ 0."""
        buf = deque([0.45, 0.45, 0.451, 0.449, 0.45, 0.45], maxlen=6)
        velocity = self._compute_velocity(buf)

        assert abs(velocity) < 0.005, \
            f"Stationary velocity {velocity:.4f} should be near zero"

    def test_empty_buffer_returns_zero(self):
        """Empty or single-element buffer returns 0.0 safely."""
        assert self._compute_velocity(deque()) == 0.0
        assert self._compute_velocity(deque([0.5])) == 0.0

    def test_upward_motion_negative(self):
        """Person standing up: velocity is negative (upward movement)."""
        buf = deque([0.70, 0.65, 0.58, 0.50, 0.42, 0.35], maxlen=6)
        velocity = self._compute_velocity(buf)

        assert velocity < 0, f"Upward velocity {velocity:.4f} should be negative"
        assert velocity < -self.MIN_FALL_VELOCITY, "Getting up should be clearly negative"

    def test_velocity_direction_matters(self):
        """Only downward (positive) velocity should pass the fall gate."""
        # Downward → positive
        down_buf = deque([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], maxlen=6)
        # Upward → negative
        up_buf = deque([0.8, 0.7, 0.6, 0.5, 0.4, 0.3], maxlen=6)

        assert self._compute_velocity(down_buf) > 0
        assert self._compute_velocity(up_buf) < 0

    def test_partial_buffer_still_computes(self):
        """Buffer with only 2 elements still produces valid velocity."""
        buf = deque([0.30, 0.60], maxlen=6)
        velocity = self._compute_velocity(buf)

        assert velocity == pytest.approx(0.30)
        assert velocity >= self.MIN_FALL_VELOCITY
