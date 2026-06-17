from __future__ import annotations

import numpy as np
from loguru import logger

from config import PipelineConfig
from detection.onnx_runtime import BBox, OnnxModel

PersonDetection = tuple[BBox, float]


class PersonDetector:
    """YOLOv11 COCO person detector running on 720p RGB frames."""

    _COCO_PERSON_CLASS_ID = 0

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        self._model = OnnxModel(cfg.person_model, cfg.onnx_providers)
        logger.info("PersonDetector initialized with providers: {}", self._model.providers)

    def detect(self, rgb_frame: np.ndarray) -> list[PersonDetection]:
        output, meta = self._model.infer_rgb(rgb_frame)
        detections = self._model.postprocess_yolo(
            output=output,
            meta=meta,
            conf_threshold=self._cfg.person_conf_threshold,
            nms_iou=self._cfg.det_nms_iou,
            class_id=self._COCO_PERSON_CLASS_ID,
        )
        return [(detection.bbox, detection.score) for detection in detections]
