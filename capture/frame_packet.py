from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """Immutable frame bundle produced by FrameProducer.

    Carries the original BGR frame (for annotation), a pre-computed RGB
    copy (for ML models — single conversion per frame), an IR-mode flag,
    and capture metadata.

    ``frozen=True`` prevents field reassignment while still allowing
    in-place numpy mutations on ``bgr`` (e.g. ``cv2.rectangle``).
    """

    bgr: np.ndarray
    rgb: np.ndarray
    ir_mode: bool
    timestamp: float
    width: int
    height: int
