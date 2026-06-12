import numpy as np

from capture.ir_detector import IRDetector


class TestIRDetector:
    def test_color_frame_not_ir(self):
        """A frame with distinct channel means should NOT be detected as IR."""
        detector = IRDetector(threshold=3.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 30   # B
        frame[:, :, 1] = 120  # G
        frame[:, :, 2] = 200  # R
        assert detector.is_ir_mode(frame) is False

    def test_greyscale_frame_is_ir(self):
        """A frame where all channels are nearly equal → IR mode."""
        detector = IRDetector(threshold=3.0)
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        assert detector.is_ir_mode(frame) is True

    def test_near_greyscale_within_threshold(self):
        """Channels differ by a small amount but still within threshold."""
        detector = IRDetector(threshold=3.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 98   # B
        frame[:, :, 1] = 100  # G
        frame[:, :, 2] = 102  # R
        # pixel-wise std of [98, 100, 102] ≈ 1.63 < 3.0 → IR
        assert detector.is_ir_mode(frame) is True

    def test_threshold_boundary(self):
        """Strict inequality checks: std >= threshold → NOT IR."""
        # pixel-wise std([80, 90, 100]) ≈ 8.165
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 80
        frame[:, :, 1] = 90
        frame[:, :, 2] = 100
        
        # threshold = 8.16 < 8.165 → NOT IR
        detector1 = IRDetector(threshold=8.16)
        assert detector1.is_ir_mode(frame) is False
        
        # threshold = 8.17 > 8.165 → IR
        detector2 = IRDetector(threshold=8.17)
        assert detector2.is_ir_mode(frame) is True

    def test_custom_threshold(self):
        """Lower threshold makes detection stricter."""
        detector = IRDetector(threshold=1.0)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 98
        frame[:, :, 1] = 100
        frame[:, :, 2] = 102
        # std([98, 100, 102]) ≈ 1.63 > 1.0 → NOT IR
        assert detector.is_ir_mode(frame) is False
