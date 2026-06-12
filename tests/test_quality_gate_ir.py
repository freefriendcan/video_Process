import numpy as np

from config import PipelineConfig
from quality.face_quality_gate import FaceQualityGate


class TestQualityGateIR:
    def _make_face(self, brightness=100, size=80):
        """Create a synthetic face ROI with texture (passes blur gate).

        A uniform flat image has zero Laplacian variance, which triggers
        the blur rejection.  Adding random noise creates enough texture.
        """
        rng = np.random.RandomState(42)
        noise = rng.randint(-20, 20, (size, size, 3), dtype=np.int16)
        roi = np.clip(brightness + noise, 0, 255).astype(np.uint8)
        return roi

    def test_normal_mode_rejects_dim_face(self):
        """Brightness 20 < min_brightness 40 → rejected in normal mode."""
        cfg = PipelineConfig()
        gate = FaceQualityGate(cfg)
        roi = self._make_face(brightness=20, size=80)
        passed, _ = gate.check(roi, ir_mode=False)
        assert passed is False

    def test_ir_mode_accepts_dim_face(self):
        """Brightness 20 > IR min 15 → accepted in IR mode."""
        cfg = PipelineConfig()
        gate = FaceQualityGate(cfg)
        roi = self._make_face(brightness=20, size=80)
        passed, _ = gate.check(roi, ir_mode=True)
        assert passed is True

    def test_ir_mode_rejects_too_dark(self):
        """Brightness 10 < IR min 15 → rejected even in IR mode."""
        cfg = PipelineConfig()
        gate = FaceQualityGate(cfg)
        roi = self._make_face(brightness=10, size=80)
        passed, _ = gate.check(roi, ir_mode=True)
        assert passed is False

    def test_ir_mode_rejects_overexposed(self):
        """Brightness 210 > IR max 200 → rejected in IR mode."""
        cfg = PipelineConfig()
        gate = FaceQualityGate(cfg)
        roi = self._make_face(brightness=210, size=80)
        passed, _ = gate.check(roi, ir_mode=True)
        assert passed is False

    def test_normal_mode_accepts_good_brightness(self):
        """Brightness 120 within [40, 230] → accepted in normal mode."""
        cfg = PipelineConfig()
        gate = FaceQualityGate(cfg)
        roi = self._make_face(brightness=120, size=80)
        passed, _ = gate.check(roi, ir_mode=False)
        assert passed is True

    def test_default_ir_mode_false(self):
        """ir_mode defaults to False when not specified."""
        cfg = PipelineConfig()
        gate = FaceQualityGate(cfg)
        roi = self._make_face(brightness=20, size=80)
        passed, _ = gate.check(roi)
        assert passed is False
