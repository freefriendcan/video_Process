import cv2
import requests
from loguru import logger

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
                logger.warning("[{}] JPEG encode failed in worker thread", tracker_id)
                return

            frame_bytes = buffer.tobytes()
            logger.debug("[{}] Sending face to Pi 5 for identification...", tracker_id)

            files = {'image_file': ('face.jpg', frame_bytes, 'image/jpeg')}

            response = requests.post(self._cfg.identify_url, files=files, timeout=8.0)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "authorized":
                    result = self._tracker_mgr.set_user(tracker_id, data["user"], retry_count=0)
                    if result is None:
                        logger.debug("[{}] Tracker already removed, discarding result", tracker_id)
                    else:
                        logger.info("Identity verified [{}]: {}", tracker_id, data['user'])
                else:
                    count = self._tracker_mgr.set_user(tracker_id, "Unknown", increment_retry=True)
                    if count is not None:
                        logger.info("Unknown face [{}] (attempt {}/{})", tracker_id, count, self._cfg.max_retries)
            else:
                logger.warning("[{}] Backend returned error", tracker_id)
                self._tracker_mgr.set_user(tracker_id, "Unknown")

        except Exception as e:
            logger.error("[{}] Pi unreachable: {}", tracker_id, e)
            self._tracker_mgr.set_user(tracker_id, "Unknown", increment_retry=True)
