import threading

import cv2

from config import PipelineConfig


class TrackerManager:
    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg
        self._trackers = {}
        self._lock = threading.Lock()
        self._next_id = 0

    @staticmethod
    def get_center(bbox):
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)

    @staticmethod
    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        if boxAArea + boxBArea - interArea == 0:
            return 0.0
        return interArea / float(boxAArea + boxBArea - interArea)

    def update_all(self, frame, current_time):
        """Update all KCF trackers. Evicts TTL-expired and lost trackers.

        Returns list of dicts for active trackers with keys:
            id, bbox, user, detection_keypoints, last_json_time,
            retry_count, last_identify_time, best_quality_score
        """
        with self._lock:
            snapshot = list(self._trackers.items())

        to_delete = []
        active = []

        for t_id, t_data in snapshot:
            success, bbox = t_data["tracker"].update(frame)

            if success:
                x, y, w, h = [int(v) for v in bbox]
                with self._lock:
                    if t_id not in self._trackers:
                        continue

                    if current_time - t_data.get("last_blazeface_update", current_time) > 1.0:
                        print(f"[{t_id}] Tracker TTL expired (Ghost box). Force deleting.")
                        to_delete.append(t_id)
                        continue

                    t_data["bbox"] = (x, y, w, h)
                    info = {
                        "id": t_id,
                        "bbox": (x, y, w, h),
                        "user": t_data["user"],
                        "detection_keypoints": t_data.get("detection_keypoints"),
                        "last_json_time": t_data.get("last_json_time", 0),
                        "retry_count": t_data["retry_count"],
                        "last_identify_time": t_data["last_identify_time"],
                        "best_quality_score": t_data.get("best_quality_score", 0),
                    }

                active.append(info)
            else:
                to_delete.append(t_id)

        with self._lock:
            for t_id in to_delete:
                if t_id in self._trackers:
                    print(f"Box Lost [{t_id}]. Being deleted from the dictionary.")
                    del self._trackers[t_id]

        return active

    def match_and_update(self, faces, frame, current_time):
        """Match face detections to existing trackers via IoU.
        Reinits drifted KCF trackers for matches, creates new trackers for unmatched faces.

        Returns list of dicts for NEW faces: tracker_id, bbox, keypoints
        """
        new_faces = []

        for face_data in faces:
            x, y, w, h, keypoints = face_data
            matched_id = None

            with self._lock:
                best_iou = 0.0
                for t_id, t_data in self._trackers.items():
                    iou = self.calculate_iou((x, y, w, h), t_data["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        if iou > 0.3:
                            matched_id = t_id

            if matched_id is not None:
                new_tracker = cv2.TrackerKCF_create()
                new_tracker.init(frame, (x, y, w, h))
                with self._lock:
                    if matched_id in self._trackers:
                        self._trackers[matched_id]["tracker"] = new_tracker
                        self._trackers[matched_id]["bbox"] = (x, y, w, h)
                        self._trackers[matched_id]["detection_keypoints"] = keypoints
                        self._trackers[matched_id]["last_blazeface_update"] = current_time
            else:
                print(f"New Face Found! Tracking Initiates (ID: {self._next_id})")
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, (x, y, w, h))

                tracker_id = self._next_id
                self._next_id += 1

                with self._lock:
                    self._trackers[tracker_id] = {
                        "tracker": tracker,
                        "user": "Identifying...",
                        "bbox": (x, y, w, h),
                        "retry_count": 0,
                        "last_identify_time": current_time,
                        "last_json_time": 0,
                        "best_quality_score": 0,
                        "detection_keypoints": keypoints,
                        "last_blazeface_update": current_time,
                    }

                new_faces.append({
                    "tracker_id": tracker_id,
                    "bbox": (x, y, w, h),
                    "keypoints": keypoints,
                })

        return new_faces

    def set_user(self, tracker_id, user, retry_count=None, increment_retry=False):
        """Thread-safe user update. Returns current retry_count, or None if tracker gone."""
        with self._lock:
            if tracker_id not in self._trackers:
                return None
            self._trackers[tracker_id]["user"] = user
            if retry_count is not None:
                self._trackers[tracker_id]["retry_count"] = retry_count
            elif increment_retry:
                self._trackers[tracker_id]["retry_count"] += 1
            return self._trackers[tracker_id]["retry_count"]

    def update_field(self, tracker_id, **fields):
        """Thread-safe field update. No-op if tracker was deleted."""
        with self._lock:
            if tracker_id in self._trackers:
                self._trackers[tracker_id].update(fields)

    def get_active_user(self):
        """Returns the first identified user name, or 'Unknown'."""
        with self._lock:
            for t_data in self._trackers.values():
                if t_data["user"] not in ["Unknown", "Identifying..."]:
                    return t_data["user"]
        return "Unknown"

    def exists(self, tracker_id):
        with self._lock:
            return tracker_id in self._trackers

    @property
    def snapshot(self):
        """Thread-safe copy of all tracker data (excluding cv2 tracker objects)."""
        with self._lock:
            return {
                t_id: {k: v for k, v in t_data.items() if k != "tracker"}
                for t_id, t_data in self._trackers.items()
            }
