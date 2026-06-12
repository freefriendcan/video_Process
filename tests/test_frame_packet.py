import numpy as np
import pytest

from capture.frame_packet import FramePacket


class TestFramePacket:
    def _make_packet(self, w=640, h=480, ir=False):
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        return FramePacket(bgr=bgr, rgb=rgb, ir_mode=ir, timestamp=1.0, width=w, height=h)

    def test_field_access(self):
        pkt = self._make_packet(320, 240, ir=True)
        assert pkt.width == 320
        assert pkt.height == 240
        assert pkt.ir_mode is True
        assert pkt.timestamp == 1.0
        assert pkt.bgr.shape == (240, 320, 3)
        assert pkt.rgb.shape == (240, 320, 3)

    def test_frozen_prevents_reassignment(self):
        pkt = self._make_packet()
        with pytest.raises(AttributeError):
            pkt.ir_mode = True
        with pytest.raises(AttributeError):
            pkt.bgr = np.zeros((1, 1, 3), dtype=np.uint8)

    def test_numpy_content_mutable(self):
        """frozen=True blocks field reassignment but not in-place numpy ops."""
        pkt = self._make_packet()
        pkt.bgr[0, 0] = [255, 0, 0]
        assert pkt.bgr[0, 0, 0] == 255
