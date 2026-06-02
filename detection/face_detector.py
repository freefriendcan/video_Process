import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import PipelineConfig


class FaceDetector:
    def __init__(self, cfg: PipelineConfig):
        if not os.path.exists(cfg.blaze_face_model):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
                cfg.blaze_face_model,
            )
        base_options = python.BaseOptions(model_asset_path=cfg.blaze_face_model)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=cfg.min_detection_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, rgb_frame):
        """Detect faces, return list of (x, y, w, h, keypoints) with padding applied."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._detector.detect(mp_image)

        faces = []
        if result.detections:
            for detection in result.detections:
                bbox = detection.bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

                pad_w = int(w * 0.25)
                pad_h = int(h * 0.35)
                x = max(0, x - pad_w)
                y = max(0, y - pad_h)
                w += pad_w * 2
                h += pad_h * 2

                keypoints = detection.keypoints if hasattr(detection, 'keypoints') else None
                faces.append((x, y, w, h, keypoints))

        return faces
