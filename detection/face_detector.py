from __future__ import annotations

import numpy as np
from loguru import logger

from config import PipelineConfig
from detection.onnx_runtime import BBox, NormalizedKeypoint, OnnxModel, YoloDetection

FaceDetection = tuple[int, int, int, int, tuple[NormalizedKeypoint, ...]]


class FaceDetector:
    """YOLOv11-face detector on the 720p CV frame."""

    _LANDMARK_COUNT = 5

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        self._model = OnnxModel(cfg.face_det_model, cfg.onnx_providers)
        logger.info("FaceDetector initialized with providers: {}", self._model.providers)

    def detect(self, rgb_frame: np.ndarray) -> list[FaceDetection]:
        """Detect faces, returning the legacy tracker contract with keypoints."""
        output, meta = self._model.infer_rgb(rgb_frame)
        detections = self._model.postprocess_yolo(
            output=output,
            meta=meta,
            conf_threshold=self._cfg.face_conf_threshold,
            nms_iou=self._cfg.det_nms_iou,
            keypoint_count=self._LANDMARK_COUNT,
        )

        faces: list[FaceDetection] = []
        frame_h, frame_w = rgb_frame.shape[:2]
        for detection in detections:
            if len(detection.keypoints) < self._LANDMARK_COUNT:
                raise ValueError("YOLOv11-face detection did not include the required 5 landmarks")

            x, y, w, h = self._pad_bbox(detection.bbox, frame_w, frame_h)
            keypoints = self._quality_gate_keypoints(detection, frame_w, frame_h)
            faces.append((x, y, w, h, keypoints))

        return faces

    @staticmethod
    def _pad_bbox(bbox: BBox, frame_w: int, frame_h: int) -> BBox:
        x, y, w, h = bbox
        pad_w = int(w * 0.25)
        pad_h = int(h * 0.35)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame_w, x + w + pad_w)
        y2 = min(frame_h, y + h + pad_h)
        return x1, y1, max(0, x2 - x1), max(0, y2 - y1)

    @staticmethod
    def _quality_gate_keypoints(
        detection: YoloDetection,
        frame_w: int,
        frame_h: int,
    ) -> tuple[NormalizedKeypoint, ...]:
        # Verified empirically against yolov11n-face.onnx (see Phase A landmark
        # validation): the raw YOLO landmark order is spatial, i.e.
        #   kp[0]=image-left eye, kp[1]=image-right eye, kp[2]=nose,
        #   kp[3]=image-left mouth, kp[4]=image-right mouth.
        # The subject's RIGHT eye sits on the image-LEFT, so kp[0] is the
        # anatomical right eye and kp[1] the anatomical left eye. This already
        # matches the quality-gate invariant (kp[0]=right_eye, kp[1]=left_eye)
        # AND the ArcFace template (row0=image-left, row1=image-right, ...),
        # so the landmarks are passed through in native order.
        ordered = detection.keypoints[:5]
        return tuple(
            NormalizedKeypoint(
                x=max(0.0, min(1.0, point[0] / frame_w)),
                y=max(0.0, min(1.0, point[1] / frame_h)),
            )
            for point in ordered
        )
