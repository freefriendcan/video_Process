import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

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

    def send_fall_alert(self, confidence, screenshot_path=None):
        try:
            payload = {
                "event": "fall_detected",
                "confidence": round(confidence, 4),
                "timestamp": time.time(),
                "source": "mac_camera",
                "method": "transformer",
            }

            if screenshot_path and Path(screenshot_path).exists():
                with open(screenshot_path, 'rb') as f:
                    files = {'screenshot': (Path(screenshot_path).name, f, 'image/jpeg')}
                    data = {k: str(v) for k, v in payload.items()}
                    resp = requests.post(self._cfg.fall_alert_url, data=data, files=files, timeout=5.0)
            else:
                resp = requests.post(self._cfg.fall_alert_url, json=payload, timeout=3.0)

            print(f"[FALL] Alert sent to backend: {resp.status_code}")
        except Exception as e:
            print(f"[FALL] Alert send error: {e}")

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
            requests.post(self._cfg.presence_url, json=payload, timeout=2.0)
            print("The 'camera_offline' signal was successfully sent to the backend.")
        except Exception as e:
            print(f"An error occurred while sending a signal to the backend: {e}")
