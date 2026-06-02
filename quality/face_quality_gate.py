import cv2
import numpy as np

from config import PipelineConfig


class FaceQualityGate:
    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg

    def estimate_frontality(self, keypoints, img_w, img_h, face_bbox) -> bool:
        """Use BlazeFace keypoints to check if face is frontal enough."""
        if not keypoints or len(keypoints) < 2:
            return True

        right_eye_x = keypoints[0].x * img_w
        left_eye_x = keypoints[1].x * img_w

        eye_distance = abs(left_eye_x - right_eye_x)
        face_w = face_bbox[2]

        if face_w > 0 and (eye_distance / face_w) < self._cfg.min_eye_distance_ratio:
            return False

        return True

    def check(self, face_roi, keypoints=None, img_w=640, img_h=480, face_bbox=None):
        """4-layer quality gate: Size/Ratio -> Brightness -> Blur -> Pose.

        Returns (passed: bool, quality_score: float).
        """
        h, w = face_roi.shape[:2]
        cfg = self._cfg

        # Layer 1: Size & Aspect Ratio (instant)
        if w < cfg.min_face_roi_size or h < cfg.min_face_roi_size:
            print(f"[QualityGate] BLOCKED: too small ({w}x{h})")
            return False, 0.0

        aspect_ratio = w / float(h)
        if aspect_ratio < 0.5 or aspect_ratio > 1.5:
            print(f"[QualityGate] BLOCKED: bad aspect ratio ({aspect_ratio:.2f})")
            return False, 0.0

        # Layer 2: Brightness (cheap — single np.mean)
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < cfg.min_brightness or mean_brightness > cfg.max_brightness:
            print(f"[QualityGate] BLOCKED: bad brightness ({mean_brightness:.0f})")
            return False, 0.0

        # Layer 3: Blur (moderate — Laplacian convolution)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        if variance < cfg.min_laplacian_variance:
            print(f"[QualityGate] BLOCKED: too blurry (laplacian={variance:.0f})")
            return False, 0.0

        # Layer 4: Pose via BlazeFace keypoints (zero extra cost)
        if keypoints is not None and face_bbox is not None:
            if not self.estimate_frontality(keypoints, img_w, img_h, face_bbox):
                print(f"[QualityGate] BLOCKED: profile/angled face")
                return False, 0.0

        return True, variance
