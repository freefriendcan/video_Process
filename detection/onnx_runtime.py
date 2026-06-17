from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

BBox = tuple[int, int, int, int]
Keypoint = tuple[float, float, float]


@dataclass(frozen=True)
class LetterboxMeta:
    original_width: int
    original_height: int
    input_size: int
    scale: float
    pad_x: float
    pad_y: float


@dataclass(frozen=True)
class YoloDetection:
    bbox: BBox
    score: float
    class_id: int
    keypoints: tuple[Keypoint, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class NormalizedKeypoint:
    x: float
    y: float


class OnnxModel:
    """Small ONNX Runtime wrapper for YOLO-style detectors."""

    def __init__(
        self,
        model_path: str,
        providers: Sequence[str],
        input_size: int = 640,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        import onnxruntime as ort

        self._input_size = input_size
        self._session = ort.InferenceSession(str(path), providers=list(providers))
        self._input_name = self._session.get_inputs()[0].name

    @property
    def providers(self) -> list[str]:
        return list(self._session.get_providers())

    @property
    def input_size(self) -> int:
        return self._input_size

    def preprocess_rgb(self, rgb_frame: np.ndarray) -> tuple[np.ndarray, LetterboxMeta]:
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError("Expected RGB frame with shape HxWx3")

        height, width = rgb_frame.shape[:2]
        scale = min(self._input_size / width, self._input_size / height)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        resized = cv2.resize(
            rgb_frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.full((self._input_size, self._input_size, 3), 114, dtype=np.uint8)
        pad_x = (self._input_size - resized_width) / 2.0
        pad_y = (self._input_size - resized_height) / 2.0
        x0 = int(round(pad_x - 0.1))
        y0 = int(round(pad_y - 0.1))
        canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized

        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
        meta = LetterboxMeta(
            original_width=width,
            original_height=height,
            input_size=self._input_size,
            scale=scale,
            pad_x=float(x0),
            pad_y=float(y0),
        )
        return tensor, meta

    def infer_rgb(self, rgb_frame: np.ndarray) -> tuple[np.ndarray, LetterboxMeta]:
        tensor, meta = self.preprocess_rgb(rgb_frame)
        outputs = self._session.run(None, {self._input_name: tensor})
        return np.asarray(outputs[0]), meta

    @staticmethod
    def postprocess_yolo(
        output: np.ndarray,
        meta: LetterboxMeta,
        conf_threshold: float,
        nms_iou: float,
        class_id: int | None = None,
        keypoint_count: int = 0,
    ) -> list[YoloDetection]:
        predictions = _as_prediction_rows(output)
        detections: list[YoloDetection] = []

        for row in predictions:
            parsed = _parse_prediction_row(row, class_id=class_id, keypoint_count=keypoint_count)
            if parsed is None:
                continue

            score, detected_class_id, keypoint_values = parsed
            if score < conf_threshold:
                continue

            bbox = _xywh_to_original_bbox(row[:4], meta)
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue

            keypoints = _keypoints_to_original(keypoint_values, meta) if keypoint_values else ()
            detections.append(
                YoloDetection(
                    bbox=bbox,
                    score=score,
                    class_id=detected_class_id,
                    keypoints=keypoints,
                )
            )

        return _nms(detections, nms_iou)


def _as_prediction_rows(output: np.ndarray) -> np.ndarray:
    predictions = np.asarray(output, dtype=np.float32).squeeze()
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    if predictions.ndim != 2:
        raise ValueError(f"Unsupported YOLO output shape: {output.shape}")

    if _looks_like_attribute_count(predictions.shape[0]) and not _looks_like_attribute_count(
        predictions.shape[1]
    ):
        predictions = predictions.T

    return predictions


def _looks_like_attribute_count(value: int) -> bool:
    return 5 <= value <= 256


def _parse_prediction_row(
    row: np.ndarray,
    class_id: int | None,
    keypoint_count: int,
) -> tuple[float, int, tuple[float, ...]] | None:
    if row.shape[0] < 5:
        return None

    if keypoint_count > 0:
        expected_values = 5 + keypoint_count * 3
        if row.shape[0] < expected_values:
            raise ValueError(
                f"YOLO face output has {row.shape[0]} values; expected at least {expected_values} "
                "for bbox, score, and 5 landmark triplets"
            )
        return float(row[4]), 0, tuple(float(v) for v in row[5:expected_values])

    if row.shape[0] == 5:
        return float(row[4]), 0, ()

    if row.shape[0] == 6:
        detected_class_id = int(round(float(row[5])))
        if class_id is not None and detected_class_id != class_id:
            return None
        return float(row[4]), detected_class_id, ()

    class_scores = row[4:]
    if class_id is not None:
        if class_id >= class_scores.shape[0]:
            return None
        score = float(class_scores[class_id])
        detected_class_id = class_id
    else:
        detected_class_id = int(np.argmax(class_scores))
        score = float(class_scores[detected_class_id])

    return score, detected_class_id, ()


def _xywh_to_original_bbox(raw_xywh: np.ndarray, meta: LetterboxMeta) -> BBox:
    cx, cy, width, height = [float(value) for value in raw_xywh]
    x1 = (cx - width / 2.0 - meta.pad_x) / meta.scale
    y1 = (cy - height / 2.0 - meta.pad_y) / meta.scale
    x2 = (cx + width / 2.0 - meta.pad_x) / meta.scale
    y2 = (cy + height / 2.0 - meta.pad_y) / meta.scale

    x1 = max(0.0, min(float(meta.original_width), x1))
    y1 = max(0.0, min(float(meta.original_height), y1))
    x2 = max(0.0, min(float(meta.original_width), x2))
    y2 = max(0.0, min(float(meta.original_height), y2))

    x = int(round(x1))
    y = int(round(y1))
    w = int(round(x2 - x1))
    h = int(round(y2 - y1))
    return x, y, w, h


def _keypoints_to_original(values: tuple[float, ...], meta: LetterboxMeta) -> tuple[Keypoint, ...]:
    keypoints: list[Keypoint] = []
    for index in range(0, len(values), 3):
        x = (values[index] - meta.pad_x) / meta.scale
        y = (values[index + 1] - meta.pad_y) / meta.scale
        confidence = values[index + 2]
        keypoints.append(
            (
                max(0.0, min(float(meta.original_width), x)),
                max(0.0, min(float(meta.original_height), y)),
                float(confidence),
            )
        )
    return tuple(keypoints)


def _nms(detections: list[YoloDetection], iou_threshold: float) -> list[YoloDetection]:
    if not detections:
        return []

    ordered = sorted(detections, key=lambda det: det.score, reverse=True)
    kept: list[YoloDetection] = []

    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        ordered = [
            candidate
            for candidate in ordered
            if _iou_xywh(current.bbox, candidate.bbox) <= iou_threshold
        ]

    return kept


def _iou_xywh(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / float(denom)
