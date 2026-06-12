import numpy as np


class IRDetector:
    """Detect infrared (night-vision) mode from channel statistics.

    Tapo C225 switches to greyscale IR illumination at night. In IR mode,
    individual pixels have very little color variation, so the average
    pixel-wise standard deviation across B, G, R channels drops below *threshold*.

    Cost: ~0.1-0.5 ms per frame using 8x8 subsampling.
    """

    def __init__(self, threshold: float = 3.0):
        self._threshold = threshold

    def is_ir_mode(self, bgr_frame: np.ndarray) -> bool:
        sub = bgr_frame[::8, ::8]
        pixel_std = np.std(sub, axis=2)
        return float(np.mean(pixel_std)) < self._threshold
