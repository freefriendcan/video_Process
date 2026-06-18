import json
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor

import requests  # type: ignore[import-untyped]
from loguru import logger

from config import PipelineConfig


class EventDispatcher:
    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg
        self._executor = ThreadPoolExecutor(max_workers=cfg.network_pool_size)
        self._presence_session = requests.Session()

    def submit(self, fn, *args, **kwargs):
        return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait=False):
        self._executor.shutdown(wait=wait)

    def send_fall_alert(
        self,
        payload: Mapping[str, object],
        screenshot_bytes: bytes | None = None,
    ) -> None:
        try:
            if screenshot_bytes:
                resp = requests.post(
                    self._cfg.fall_alert_url,
                    data={"payload": json.dumps(dict(payload))},
                    files={"screenshot": ("fall.jpg", screenshot_bytes, "image/jpeg")},
                    timeout=5.0,
                )
            else:
                resp = requests.post(self._cfg.fall_alert_url, json=dict(payload), timeout=3.0)
            logger.info("Fall alert sent to backend: {}", resp.status_code)
        except Exception as e:
            logger.error("Fall alert send error: {}", e)

    def send_identity_event(self, payload: Mapping[str, object]) -> None:
        if not self._cfg.identity_events_enabled:
            return

        try:
            resp = requests.post(self._cfg.identity_event_url, json=dict(payload), timeout=1.0)
            logger.info(
                "Identity event sent: {} -> {}",
                payload.get("event_type"),
                resp.status_code,
            )
        except Exception as e:
            logger.warning("Identity event send error: {}", e)

    def send_gesture_event(self, gesture_name, duration, user_name="Unknown"):
        try:
            payload = {
                "gesture": gesture_name,
                "user": user_name,
                "location": "living_room",
                "timestamp": time.time(),
                "duration": duration,
            }
            self._presence_session.post(self._cfg.gesture_url, json=payload, timeout=0.5)
        except Exception:
            pass

    def send_presence(self, user_name):
        try:
            payload = {"user": user_name, "status": "PRESENT", "location": "living_room"}
            self._presence_session.post(self._cfg.presence_url, json=payload, timeout=0.5)
        except Exception:
            pass

    def send_offline_signal(self):
        try:
            payload = {"user": "System", "status": "camera_offline", "location": "living_room"}
            requests.post(self._cfg.presence_url, json=payload, timeout=0.5)
            logger.info("camera_offline signal sent to backend")
        except Exception as e:
            logger.warning("Failed to send offline signal: {}", e)
