from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, cast

import numpy as np
from loguru import logger

from config import PipelineConfig
from detection.face_detector import FaceDetection
from detection.onnx_runtime import NormalizedKeypoint

BBox = tuple[int, int, int, int]
PersonDetection = tuple[BBox, float]
IdentityEventPayload = dict[str, object]


@dataclass
class PersonTrackRecord:
    track_id: int
    bbox: BBox
    user: str
    reid_ok: bool
    last_seen: float
    confidence: float
    first_identified_at: float | None = None
    emitted_user: str | None = None


class _MotTracker(Protocol):
    def update(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray | None = None,
    ) -> np.ndarray:
        ...


class TrackerManager:
    def __init__(
        self,
        cfg: PipelineConfig,
        on_identity_event: Callable[[IdentityEventPayload], None] | None = None,
    ):
        self._cfg = cfg
        self._trackers: dict[int, dict[str, Any]] = {}
        self._face_mot = self._build_face_tracker(cfg)
        self._person_tracks: dict[int, PersonTrackRecord] = {}
        self._person_reid_enabled = cfg.tracker_type.lower() == "botsort"
        self._person_mot = self._build_person_tracker(cfg)
        self._on_identity_event = on_identity_event
        self._lock = threading.Lock()

    @staticmethod
    def _build_face_tracker(cfg: PipelineConfig) -> _MotTracker:
        tracker_type = cfg.face_tracker_type.strip().lower()
        if tracker_type != "bytetrack":
            raise ValueError("face_tracker_type must be 'bytetrack'")

        from boxmot.trackers import ByteTrack

        tracker = ByteTrack(
            min_conf=cfg.face_track_low_conf,
            track_thresh=cfg.face_track_thresh,
            match_thresh=cfg.face_match_thresh,
            track_buffer=cfg.face_track_buffer,
            frame_rate=cfg.face_tracker_frame_rate,
            per_class=False,
            min_hits=1,
        )
        logger.info("Face tracker initialized: ByteTrack")
        return cast(_MotTracker, tracker)

    @staticmethod
    def _build_person_tracker(cfg: PipelineConfig) -> _MotTracker:
        tracker_type = cfg.tracker_type.strip().lower()
        if tracker_type == "botsort":
            from boxmot.reid.core import ReID
            from boxmot.trackers import BotSort

            reid_model = ReID(device=cfg.reid_device, half=cfg.reid_half).model
            tracker = BotSort(
                reid_model=reid_model,
                track_high_thresh=cfg.person_conf_threshold,
                track_low_thresh=cfg.tracker_low_conf,
                new_track_thresh=cfg.tracker_new_track_thresh,
                track_buffer=cfg.tracker_track_buffer,
                match_thresh=cfg.tracker_match_thresh,
                proximity_thresh=cfg.tracker_proximity_thresh,
                appearance_thresh=cfg.tracker_appearance_thresh,
                cmc_method=cfg.tracker_cmc_method,
                frame_rate=cfg.tracker_frame_rate,
                with_reid=True,
                per_class=False,
                min_hits=1,
            )
            logger.info("Person tracker initialized: BoT-SORT + ReID ({})", cfg.reid_device)
            return cast(_MotTracker, tracker)

        if tracker_type == "bytetrack":
            from boxmot.trackers import ByteTrack

            tracker = ByteTrack(
                min_conf=cfg.tracker_low_conf,
                track_thresh=cfg.person_conf_threshold,
                match_thresh=cfg.tracker_match_thresh,
                track_buffer=cfg.tracker_track_buffer,
                frame_rate=cfg.tracker_frame_rate,
                per_class=False,
                min_hits=1,
            )
            logger.warning("Person tracker initialized: ByteTrack fallback (no ReID)")
            return cast(_MotTracker, tracker)

        raise ValueError("tracker_type must be 'botsort' or 'bytetrack'")

    @staticmethod
    def get_center(bbox):
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)

    @staticmethod
    def calculate_iou(box_a, box_b):
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
        y_b = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

        inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
        box_a_area = box_a[2] * box_a[3]
        box_b_area = box_b[2] * box_b[3]

        if box_a_area + box_b_area - inter_area == 0:
            return 0.0
        return inter_area / float(box_a_area + box_b_area - inter_area)

    def update_face_tracks(
        self,
        faces: Sequence[FaceDetection],
        frame: np.ndarray,
        current_time: float,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Update ByteTrack-backed face tracks and return ``(new_faces, active)``."""
        det_array = self._face_detections_to_boxmot(faces)
        results = np.asarray(self._face_mot.update(det_array, frame), dtype=np.float32)
        self._apply_face_results(results, faces, current_time)
        self._evict_stale_face_tracks(current_time)
        return self._new_face_tracks(results), self.active_face_tracks

    @staticmethod
    def _face_detections_to_boxmot(faces: Sequence[FaceDetection]) -> np.ndarray:
        if not faces:
            return np.empty((0, 6), dtype=np.float32)

        rows: list[tuple[float, float, float, float, float, float]] = []
        for x, y, w, h, score, _keypoints in faces:
            rows.append(
                (
                    float(x),
                    float(y),
                    float(x + w),
                    float(y + h),
                    float(score),
                    0.0,
                )
            )
        return np.asarray(rows, dtype=np.float32)

    def _apply_face_results(
        self,
        results: np.ndarray,
        faces: Sequence[FaceDetection],
        current_time: float,
    ) -> None:
        if results.size == 0:
            return
        if results.ndim == 1:
            results = results.reshape(1, -1)

        for row in results:
            if row.shape[0] < 8:
                logger.warning("Unexpected BoxMOT face track row shape: {}", row.shape)
                continue

            track_id = int(row[4])
            bbox = self._xyxy_to_bbox(row[:4])
            keypoints = self._keypoints_for_face_track(row, bbox, faces)

            with self._lock:
                existing = self._trackers.get(track_id, {})
                is_new = track_id not in self._trackers
                self._trackers[track_id] = {
                    "user": existing.get("user", "Identifying..."),
                    "bbox": bbox,
                    "retry_count": existing.get("retry_count", 0),
                    "last_identify_time": existing.get("last_identify_time", current_time),
                    "last_json_time": existing.get("last_json_time", 0),
                    "best_quality_score": existing.get("best_quality_score", 0),
                    "detection_keypoints": keypoints
                    if keypoints is not None
                    else existing.get("detection_keypoints"),
                    "last_seen": current_time,
                    "is_new": is_new,
                }

            if is_new:
                logger.info("New face detected — ByteTrack ID: {}", track_id)

    def _keypoints_for_face_track(
        self,
        row: np.ndarray,
        bbox: BBox,
        faces: Sequence[FaceDetection],
    ) -> tuple[NormalizedKeypoint, ...] | None:
        if not faces:
            return None

        det_index = int(row[7]) if row.shape[0] > 7 else -1
        if 0 <= det_index < len(faces):
            return faces[det_index][5]

        best_iou = 0.0
        best_keypoints: tuple[NormalizedKeypoint, ...] | None = None
        for face in faces:
            face_bbox = (face[0], face[1], face[2], face[3])
            iou = self.calculate_iou(bbox, face_bbox)
            if iou > best_iou:
                best_iou = iou
                best_keypoints = face[5]
        return best_keypoints

    def _new_face_tracks(self, results: np.ndarray) -> list[dict[str, object]]:
        if results.size == 0:
            return []
        if results.ndim == 1:
            results = results.reshape(1, -1)

        new_faces: list[dict[str, object]] = []
        with self._lock:
            for row in results:
                if row.shape[0] < 8:
                    continue
                track_id = int(row[4])
                data = self._trackers.get(track_id)
                if data is None or not bool(data.get("is_new", False)):
                    continue
                new_faces.append(
                    {
                        "tracker_id": track_id,
                        "bbox": data["bbox"],
                        "keypoints": data.get("detection_keypoints"),
                    }
                )
                data["is_new"] = False
        return new_faces

    def update_person_tracks(
        self,
        person_dets: Sequence[PersonDetection],
        frame: np.ndarray,
        current_time: float,
    ) -> list[dict[str, object]]:
        """Update MOT-backed person tracks from detector outputs.

        ``person_dets`` may be empty on non-detection frames. BoxMOT still
        advances its Kalman state; this manager retains the last emitted bbox
        for the configured track buffer so fall/identity state is not pruned
        between detector passes.
        """
        det_array = self._person_detections_to_boxmot(person_dets)
        results = np.asarray(self._person_mot.update(det_array, frame), dtype=np.float32)
        self._apply_person_results(results, current_time)
        self._evict_stale_person_tracks(current_time)
        return self.active_person_tracks

    @staticmethod
    def _person_detections_to_boxmot(person_dets: Sequence[PersonDetection]) -> np.ndarray:
        if not person_dets:
            return np.empty((0, 6), dtype=np.float32)

        rows: list[tuple[float, float, float, float, float, float]] = []
        for bbox, confidence in person_dets:
            x, y, w, h = bbox
            rows.append(
                (
                    float(x),
                    float(y),
                    float(x + w),
                    float(y + h),
                    float(confidence),
                    0.0,
                )
            )
        return np.asarray(rows, dtype=np.float32)

    def _apply_person_results(self, results: np.ndarray, current_time: float) -> None:
        if results.size == 0:
            return
        if results.ndim == 1:
            results = results.reshape(1, -1)

        for row in results:
            if row.shape[0] < 8:
                logger.warning("Unexpected BoxMOT person track row shape: {}", row.shape)
                continue

            track_id = int(row[4])
            confidence = float(row[5])
            bbox = self._xyxy_to_bbox(row[:4])
            with self._lock:
                existing = self._person_tracks.get(track_id)
                user = existing.user if existing is not None else "Unknown"
                first_identified_at = (
                    existing.first_identified_at if existing is not None else None
                )
                emitted_user = existing.emitted_user if existing is not None else None
                self._person_tracks[track_id] = PersonTrackRecord(
                    track_id=track_id,
                    bbox=bbox,
                    user=user,
                    reid_ok=self._person_reid_enabled,
                    last_seen=current_time,
                    confidence=confidence,
                    first_identified_at=first_identified_at,
                    emitted_user=emitted_user,
                )

    @staticmethod
    def _xyxy_to_bbox(values: np.ndarray) -> BBox:
        x1, y1, x2, y2 = [float(value) for value in values[:4]]
        x = int(round(min(x1, x2)))
        y = int(round(min(y1, y2)))
        w = int(round(abs(x2 - x1)))
        h = int(round(abs(y2 - y1)))
        return x, y, w, h

    def _evict_stale_person_tracks(self, current_time: float) -> None:
        base_ttl = max(
            self._cfg.person_detection_interval * 2.0,
            self._cfg.tracker_track_buffer / max(1.0, float(self._cfg.tracker_frame_rate)),
        )
        identity_ttl = max(base_ttl, self._cfg.person_identity_eviction_s)
        pending: list[IdentityEventPayload] = []
        with self._lock:
            stale_track_ids = [
                track_id
                for track_id, record in self._person_tracks.items()
                if current_time - record.last_seen
                > (identity_ttl if record.emitted_user is not None else base_ttl)
            ]
            for track_id in stale_track_ids:
                record = self._person_tracks[track_id]
                if record.emitted_user is not None:
                    dwell_s = max(
                        0.0,
                        record.last_seen - (record.first_identified_at or record.last_seen),
                    )
                    pending.append(
                        self._identity_payload("person_left", record, dwell_s=dwell_s)
                    )
                logger.debug("[person:{}] MOT track expired — removing identity state", track_id)
                del self._person_tracks[track_id]

        for event in pending:
            self._emit_identity(event)

    def _evict_stale_face_tracks(self, current_time: float) -> None:
        ttl = max(
            self._cfg.face_detection_interval * 2.0,
            self._cfg.face_track_buffer / max(1.0, float(self._cfg.face_tracker_frame_rate)),
        )
        with self._lock:
            stale_track_ids = [
                track_id
                for track_id, data in self._trackers.items()
                if current_time - float(data.get("last_seen", current_time)) > ttl
            ]
            for track_id in stale_track_ids:
                logger.debug("[face:{}] MOT track expired — removing identity state", track_id)
                del self._trackers[track_id]

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

    def propagate_identity(self) -> None:
        """Stamp recognized face identities onto containing person tracks."""
        pending: list[IdentityEventPayload] = []
        with self._lock:
            face_snapshot = {
                track_id: dict(data)
                for track_id, data in self._trackers.items()
            }
            person_snapshot = dict(self._person_tracks)

        for face_data in face_snapshot.values():
            user = str(face_data.get("user", "Unknown"))
            if user in {"Unknown", "Identifying..."}:
                continue

            face_bbox = cast(BBox, face_data["bbox"])
            face_center = self.get_center(face_bbox)
            matched_person_id: int | None = None
            matched_ratio = -1.0

            for person_id, person_record in person_snapshot.items():
                if not self._bbox_contains_point(person_record.bbox, face_center):
                    continue
                ratio = self._containment_ratio(face_bbox, person_record.bbox)
                if ratio > matched_ratio:
                    matched_person_id = person_id
                    matched_ratio = ratio

            if matched_person_id is None:
                continue

            with self._lock:
                record = self._person_tracks.get(matched_person_id)
                if record is not None and record.user != user:
                    record.user = user
                    if record.emitted_user != user:
                        if record.first_identified_at is None:
                            record.first_identified_at = time.time()
                        record.emitted_user = user
                        pending.append(self._identity_payload("person_identified", record))
                    logger.info(
                        "[person:{}] Identity propagated from face track: {}",
                        matched_person_id,
                        user,
                    )

        for event in pending:
            self._emit_identity(event)

    def _identity_payload(
        self,
        event_type: str,
        record: PersonTrackRecord,
        dwell_s: float | None = None,
    ) -> IdentityEventPayload:
        payload: IdentityEventPayload = {
            "schema_version": "1.0",
            "event_type": event_type,
            "track_id": record.track_id,
            "user": record.emitted_user or record.user,
            "zone": self._cfg.identity_zone,
            "source": self._cfg.identity_source,
            "ts_wall": datetime.now(timezone.utc).isoformat(),
        }
        if dwell_s is not None:
            payload["dwell_s"] = round(dwell_s, 1)
        return payload

    def _emit_identity(self, event: IdentityEventPayload) -> None:
        if self._on_identity_event is not None:
            self._on_identity_event(event)
            return
        logger.info("[identity-event] {}", event)

    @staticmethod
    def _bbox_contains_point(bbox: BBox, point: tuple[float, float]) -> bool:
        x, y, w, h = bbox
        px, py = point
        return x <= px <= x + w and y <= py <= y + h

    @staticmethod
    def _containment_ratio(inner: BBox, outer: BBox) -> float:
        ix, iy, iw, ih = inner
        ox, oy, ow, oh = outer
        inter_x1 = max(ix, ox)
        inter_y1 = max(iy, oy)
        inter_x2 = min(ix + iw, ox + ow)
        inter_y2 = min(iy + ih, oy + oh)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        inner_area = max(1, iw * ih)
        return inter_area / float(inner_area)

    def get_active_user(self):
        """Returns the first identified user name, or 'Unknown'."""
        with self._lock:
            for t_data in self._trackers.values():
                if t_data["user"] not in ["Unknown", "Identifying..."]:
                    return t_data["user"]
        return "Unknown"

    def user_for_person_track(self, track_id: int | None) -> str:
        """Resolve a person track identity for gesture attribution.

        If a track id is supplied, attribution is strict to that person track.
        The fallback to any identified person is used only when the gesture ROI
        came from the full frame and no pose track was available.
        """
        with self._lock:
            if track_id is not None:
                record = self._person_tracks.get(track_id)
                if record is None:
                    return "Unknown"
                user = record.emitted_user or record.user
                return user if user not in ("Unknown", "Identifying...") else "Unknown"

            for record in self._person_tracks.values():
                user = record.emitted_user or record.user
                if user not in ("Unknown", "Identifying..."):
                    return user
        return "Unknown"

    def exists(self, tracker_id):
        with self._lock:
            return tracker_id in self._trackers

    @property
    def active_face_tracks(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "id": track_id,
                    "bbox": data["bbox"],
                    "user": data["user"],
                    "detection_keypoints": data.get("detection_keypoints"),
                    "last_json_time": data.get("last_json_time", 0),
                    "retry_count": data.get("retry_count", 0),
                    "last_identify_time": data.get("last_identify_time", 0),
                    "best_quality_score": data.get("best_quality_score", 0),
                }
                for track_id, data in sorted(self._trackers.items())
            ]

    @property
    def active_person_tracks(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "id": record.track_id,
                    "bbox": record.bbox,
                    "user": record.user,
                    "reid_ok": record.reid_ok,
                    "last_seen": record.last_seen,
                    "confidence": record.confidence,
                }
                for record in sorted(self._person_tracks.values(), key=lambda item: item.track_id)
            ]

    @property
    def person_snapshot(self) -> dict[int, dict[str, object]]:
        with self._lock:
            return {
                record.track_id: {
                    "id": record.track_id,
                    "bbox": record.bbox,
                    "user": record.user,
                    "reid_ok": record.reid_ok,
                    "last_seen": record.last_seen,
                    "confidence": record.confidence,
                }
                for record in self._person_tracks.values()
            }

    @property
    def snapshot(self):
        """Thread-safe copy of face-track metadata."""
        with self._lock:
            return {t_id: dict(t_data) for t_id, t_data in self._trackers.items()}
