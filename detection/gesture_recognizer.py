import os
import time
import urllib.request
from collections.abc import Callable, Sequence
from typing import Protocol

import mediapipe as mp
import numpy as np
from loguru import logger
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import PipelineConfig
from events.dispatcher import EventDispatcher


class WristPose(Protocol):
    track_id: int
    bbox: tuple[int, int, int, int]
    left_wrist: tuple[float, float] | None
    right_wrist: tuple[float, float] | None


HandBBox = tuple[int, int, int, int]
TrackedHandBBox = tuple[HandBBox, int]


class GestureRecognizer:
    """Async MediaPipe gesture recognizer with gaze-lock and sustained-gesture logic."""

    def __init__(
        self,
        cfg: PipelineConfig,
        dispatcher: EventDispatcher,
        get_user_for_track: Callable[[int | None], str],
    ) -> None:
        """
        Args:
            get_user_for_track: thread-safe identity resolver for a person
                track id, or full-frame fallback when the track id is None.
        """
        self._cfg = cfg
        self._dispatcher = dispatcher
        self._get_user_for_track = get_user_for_track

        self._latest_gesture = ""
        self._latest_gesture_time = 0.0
        self._pending_track_id: int | None = None
        self._current_sustained = ""
        self._gesture_start_time = 0.0
        self._last_sent = ""
        self._last_sent_time = 0.0
        self.is_frontal = False
        self._smoothed_hand_bbox: tuple[float, float, float, float] | None = None

        # Prevents Python GC from collecting mp.Image objects while
        # MediaPipe's C++ backend still references them.
        self._async_image_buffer: list[mp.Image] = []
        self._last_timestamp_ms = 0

        if not os.path.exists(cfg.gesture_model):
            logger.info("Downloading gesture_recognizer.task...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
                cfg.gesture_model,
            )

        base_options = python.BaseOptions(
            model_asset_path=cfg.gesture_model,
            delegate=python.BaseOptions.Delegate.CPU,
        )
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            result_callback=self._on_result,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

    @property
    def latest_gesture(self):
        return self._latest_gesture

    @property
    def latest_gesture_time(self):
        return self._latest_gesture_time

    def clear_stale(self, current_time, timeout=1.5):
        if current_time - self._latest_gesture_time > timeout:
            self._latest_gesture = ""

    def _on_result(self, result, output_image: mp.Image, timestamp_ms: int) -> None:
        if result.gestures:
            for hand_gestures in result.gestures:
                top_gesture = hand_gestures[0].category_name

                if top_gesture in ["Thumb_Up", "Victory"] and not self.is_frontal:
                    logger.debug(
                        "Gaze Lock: ignoring '{}' - user not looking at camera",
                        top_gesture,
                    )
                    self._current_sustained = ""
                    continue

                if top_gesture and top_gesture != "None":
                    self._latest_gesture = top_gesture
                    self._latest_gesture_time = time.time()

                    if top_gesture == self._current_sustained:
                        duration = time.time() - self._gesture_start_time
                    else:
                        self._current_sustained = top_gesture
                        self._gesture_start_time = time.time()
                        duration = 0.0

                    if duration > 1.0:
                        if (
                            top_gesture != self._last_sent
                            or (time.time() - self._last_sent_time) > self._cfg.gesture_cooldown
                        ):
                            logger.info(
                                "Sustained gesture: {} (duration: {:.1f}s)",
                                top_gesture,
                                duration,
                            )
                            detected_user = self._get_user_for_track(self._pending_track_id)
                            self._dispatcher.submit(
                                self._dispatcher.send_gesture_event,
                                top_gesture, duration, detected_user,
                            )
                            self._last_sent = top_gesture
                            self._last_sent_time = time.time()
        else:
            self._current_sustained = ""

    def process(
        self,
        rgb_frame: np.ndarray,
        current_time: float,
        poses: Sequence[WristPose] | None = None,
    ) -> None:
        current_ms = int(current_time * 1000)
        if current_ms <= self._last_timestamp_ms:
            current_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = current_ms

        try:
            roi, track_id = self._hand_roi(rgb_frame, poses)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi.copy())
            self._async_image_buffer.append(mp_image)
            if len(self._async_image_buffer) > 30:
                self._async_image_buffer.pop(0)

            self._pending_track_id = track_id
            self._recognizer.recognize_async(mp_image, current_ms)
        except Exception:
            pass

    def _hand_roi(
        self,
        rgb_frame: np.ndarray,
        poses: Sequence[WristPose] | None,
    ) -> tuple[np.ndarray, int | None]:
        raw = self._raw_hand_bbox(rgb_frame, poses)
        if raw is None:
            self._smoothed_hand_bbox = None
            return rgb_frame, None

        bbox, track_id = raw

        if self._smoothed_hand_bbox is None:
            smoothed = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        else:
            alpha = self._cfg.hand_crop_ema_alpha
            smoothed_values = tuple(
                alpha * float(new_value) + (1.0 - alpha) * old_value
                for new_value, old_value in zip(bbox, self._smoothed_hand_bbox)
            )
            smoothed = (
                smoothed_values[0],
                smoothed_values[1],
                smoothed_values[2],
                smoothed_values[3],
            )
        self._smoothed_hand_bbox = smoothed

        x, y, w, h = [int(round(value)) for value in smoothed]
        frame_h, frame_w = rgb_frame.shape[:2]
        x1 = max(0, min(frame_w, x))
        y1 = max(0, min(frame_h, y))
        x2 = max(0, min(frame_w, x + max(1, w)))
        y2 = max(0, min(frame_h, y + max(1, h)))
        if x2 <= x1 or y2 <= y1:
            return rgb_frame, track_id
        return rgb_frame[y1:y2, x1:x2], track_id

    def _raw_hand_bbox(
        self,
        rgb_frame: np.ndarray,
        poses: Sequence[WristPose] | None,
    ) -> TrackedHandBBox | None:
        if not poses:
            return None

        frame_h, frame_w = rgb_frame.shape[:2]
        for pose in poses:
            wrist_points = [
                point
                for point in (pose.left_wrist, pose.right_wrist)
                if point is not None
            ]
            if not wrist_points:
                continue

            person_x, person_y, person_w, person_h = pose.bbox
            pad = int(round(max(person_w, person_h) * self._cfg.hand_crop_pad / 2.0))
            xs = [point[0] for point in wrist_points]
            ys = [point[1] for point in wrist_points]
            x1 = max(0, int(round(min(xs) - pad)))
            y1 = max(0, int(round(min(ys) - pad)))
            x2 = min(frame_w, int(round(max(xs) + pad)))
            y2 = min(frame_h, int(round(max(ys) + pad)))
            if x2 > x1 and y2 > y1:
                return (x1, y1, x2 - x1, y2 - y1), pose.track_id

            fallback_w = max(1, int(round(person_w * self._cfg.hand_crop_pad)))
            fallback_h = max(1, int(round(person_h * self._cfg.hand_crop_pad)))
            return (
                (
                    max(0, min(frame_w, person_x)),
                    max(0, min(frame_h, person_y)),
                    min(frame_w - person_x, fallback_w),
                    min(frame_h - person_y, fallback_h),
                ),
                pose.track_id,
            )

        return None
