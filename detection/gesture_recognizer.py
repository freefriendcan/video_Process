import os
import time
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from loguru import logger

from config import PipelineConfig
from events.dispatcher import EventDispatcher


class GestureRecognizer:
    """Async MediaPipe gesture recognizer with gaze-lock and sustained-gesture logic."""

    def __init__(self, cfg: PipelineConfig, dispatcher: EventDispatcher, get_active_user):
        """
        Args:
            get_active_user: callable returning the current identified user name (thread-safe)
        """
        self._cfg = cfg
        self._dispatcher = dispatcher
        self._get_active_user = get_active_user

        self._latest_gesture = ""
        self._latest_gesture_time = 0.0
        self._is_processing = False
        self._current_sustained = ""
        self._gesture_start_time = 0.0
        self._last_sent = ""
        self._last_sent_time = 0.0
        self.is_frontal = False

        # Prevents Python GC from collecting mp.Image objects while
        # MediaPipe's C++ backend still references them.
        self._async_image_buffer = []
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

    def _on_result(self, result, output_image: mp.Image, timestamp_ms: int):
        if result.gestures:
            for hand_gestures in result.gestures:
                top_gesture = hand_gestures[0].category_name

                if top_gesture in ["Thumb_Up", "Victory"] and not self.is_frontal:
                    logger.debug("Gaze Lock: ignoring '{}' — user not looking at camera", top_gesture)
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
                        if (top_gesture != self._last_sent or
                                (time.time() - self._last_sent_time) > self._cfg.gesture_cooldown):
                            logger.info("Sustained gesture: {} (duration: {:.1f}s)", top_gesture, duration)
                            detected_user = self._get_active_user()
                            self._dispatcher.submit(
                                self._dispatcher.send_gesture_event,
                                top_gesture, duration, detected_user,
                            )
                            self._last_sent = top_gesture
                            self._last_sent_time = time.time()
        else:
            self._current_sustained = ""

        self._is_processing = False

    def process(self, rgb_frame, current_time):
        current_ms = int(current_time * 1000)
        if current_ms <= self._last_timestamp_ms:
            current_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = current_ms

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame.copy())
            self._async_image_buffer.append(mp_image)
            if len(self._async_image_buffer) > 30:
                self._async_image_buffer.pop(0)

            if not self._is_processing:
                self._is_processing = True
                self._recognizer.recognize_async(mp_image, current_ms)
        except Exception:
            self._is_processing = False
