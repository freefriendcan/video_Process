import cv2
import requests

from config import PipelineConfig
from tracking.tracker_manager import TrackerManager


class FaceIdentifier:
    """Encodes a face ROI as JPEG and POSTs it to the Pi backend for identification."""

    def __init__(self, cfg: PipelineConfig, tracker_mgr: TrackerManager):
        self._cfg = cfg
        self._tracker_mgr = tracker_mgr

    def identify(self, face_roi_raw, tracker_id):
        try:
            ret, buffer = cv2.imencode('.jpg', face_roi_raw)
            if not ret:
                print(f"[{tracker_id}] JPEG encode failed in worker thread")
                return

            frame_bytes = buffer.tobytes()
            print(f"[{tracker_id}] The face is being sent to Pi 5, awaiting identification...")

            files = {'image_file': ('face.jpg', frame_bytes, 'image/jpeg')}

            response = requests.post(self._cfg.identify_url, files=files, timeout=8.0)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "authorized":
                    result = self._tracker_mgr.set_user(tracker_id, data["user"], retry_count=0)
                    if result is None:
                        print(f"[{tracker_id}] Tracker already removed, discarding result.")
                    else:
                        print(f"Identity Verified [{tracker_id}]: {data['user']}")
                else:
                    count = self._tracker_mgr.set_user(tracker_id, "Unknown", increment_retry=True)
                    if count is not None:
                        print(f"Stranger or Unknown [{tracker_id}]. (Failed Attempt: {count}/{self._cfg.max_retries})")
            else:
                print(f"[{tracker_id}] Backend error returned.")
                self._tracker_mgr.set_user(tracker_id, "Unknown")

        except Exception as e:
            print(f"[{tracker_id}] Pi could not be reached: {e}")
            self._tracker_mgr.set_user(tracker_id, "Unknown", increment_retry=True)
