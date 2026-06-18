import signal
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from loguru import logger

from capture.frame_producer import FrameProducer
from config import PipelineConfig
from detection.face_detector import FaceDetection, FaceDetector
from detection.fall_detector import FallDetector, PoseTrackData
from detection.fall_geometry import fall_region_px
from detection.gesture_recognizer import GestureRecognizer
from detection.onnx_runtime import BBox, NormalizedKeypoint
from detection.person_detector import PersonDetection, PersonDetector
from events.dispatcher import EventDispatcher
from identification.face_identifier import FaceIdentifier
from quality.face_quality_gate import FaceQualityGate
from streaming.vision_state import FallRegion, TrackedFace, TrackedPerson, VisionState
from streaming.vision_ws_server import VisionWSServer
from tracking.tracker_manager import TrackerManager

# Configure loguru: remove default handler, add custom sinks
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    ),
    level="DEBUG",
    colorize=True,
)
logger.add(
    "data/logs/pipeline_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="gz",
)


class VisionPipeline:
    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg
        self._quality_gate = FaceQualityGate(cfg)
        self._dispatcher = EventDispatcher(cfg)
        self._tracker_mgr = TrackerManager(
            cfg,
            on_identity_event=lambda event: self._dispatcher.submit(
                self._dispatcher.send_identity_event,
                event,
            ),
        )
        self._producer = FrameProducer(cfg)
        self._ws_server = VisionWSServer(cfg)
        self._person_det = PersonDetector(cfg)
        self._face_det = FaceDetector(cfg)
        self._gesture_rec = GestureRecognizer(
            cfg,
            self._dispatcher,
            self._tracker_mgr.get_active_user,
        )
        self._fall_det = FallDetector(cfg)
        self._identifier = FaceIdentifier(cfg, self._tracker_mgr)
        self._frame_counter = 0
        self._last_state_time = 0.0
        self._fps = 0.0
        self._running = False
        self._shutdown_lock = threading.Lock()
        self._shutdown_started = False

    def run(self):
        self._cfg.preflight()
        self._producer.open()
        self._ws_server.start()
        self._running = True
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        self._camera_loop()

    def shutdown(self):
        with self._shutdown_lock:
            if self._shutdown_started:
                logger.info("Shutdown already in progress; duplicate request ignored")
                return False
            self._shutdown_started = True

        logger.info("CTRL+C caught — shutting down camera safely...")
        self._running = False
        try:
            self._dispatcher.send_offline_signal()
        finally:
            cleanup_steps = (
                ("dispatcher", self._dispatcher.shutdown),
                ("frame producer", self._producer.release),
                ("vision websocket", self._ws_server.stop),
            )
            for name, shutdown_fn in cleanup_steps:
                try:
                    shutdown_fn()
                except Exception as exc:
                    logger.exception("Error during {} shutdown: {}", name, exc)
        return True

    def _handle_signal(self, signum, frame):
        if self._shutdown_started:
            logger.warning("Second interrupt received; forcing exit")
            sys.exit(130)

        self.shutdown()
        sys.exit(0)

    def _camera_loop(self):
        quality_gate = self._quality_gate
        dispatcher = self._dispatcher
        tracker_mgr = self._tracker_mgr
        gesture_rec = self._gesture_rec
        fall_det = self._fall_det
        person_det = self._person_det
        face_det = self._face_det
        producer = self._producer
        ws_server = self._ws_server

        last_detection_time = 0
        last_person_detection_time = 0.0

        while self._running:
            pkt = producer.read_latest()
            if pkt is None:
                time.sleep(0.005)
                continue

            frame = pkt.bgr
            frame_h, frame_w = pkt.height, pkt.width
            current_time = time.time()
            self._frame_counter += 1

            rgb_frame = pkt.rgb

            face_detections: list[FaceDetection] = []
            if current_time - last_detection_time > self._cfg.face_detection_interval:
                face_detections = face_det.detect(rgb_frame)
                last_detection_time = current_time
            new_faces, active = tracker_mgr.update_face_tracks(
                face_detections,
                frame,
                current_time,
            )

            for t in active:
                t_id = t["id"]
                x, y, w, h = t["bbox"]
                user = t["user"]
                if user not in ["Unknown", "Identifying..."]:
                    is_frontal = quality_gate.estimate_frontality(
                        t["detection_keypoints"],
                        frame_w,
                        frame_h,
                        (x, y, w, h),
                    )
                    gesture_rec.is_frontal = is_frontal

                    if current_time - t["last_json_time"] > 3.0:
                        dispatcher.submit(dispatcher.send_presence, user)
                        tracker_mgr.update_field(t_id, last_json_time=current_time)

            self._identify_new_faces(
                new_faces=new_faces,
                frame=frame,
                frame_size=(frame_w, frame_h),
                ir_mode=pkt.ir_mode,
            )

            person_dets: list[PersonDetection] = []
            if current_time - last_person_detection_time > self._cfg.person_detection_interval:
                person_dets = person_det.detect(rgb_frame)
                last_person_detection_time = current_time
            person_tracks = tracker_mgr.update_person_tracks(person_dets, frame, current_time)

            active_person_ids = {int(person_track["id"]) for person_track in person_tracks}
            fall_det.sync_tracks(active_person_ids)

            pose_tracks: list[PoseTrackData] = []
            for person_track in person_tracks:
                person_track_id = int(person_track["id"])
                person_bbox = cast(BBox, person_track["bbox"])
                fall_result = fall_det.process_track(
                    track_id=person_track_id,
                    rgb_frame=rgb_frame,
                    bgr_frame=frame,
                    bbox=person_bbox,
                    frame_size=(frame_w, frame_h),
                    current_time=current_time,
                )
                if fall_result.pose is not None:
                    pose_tracks.append(fall_result.pose)
                if fall_result.alert is not None:
                    dispatcher.submit(dispatcher.send_fall_alert, fall_result.alert)

            gesture_rec.process(rgb_frame, current_time, pose_tracks)
            gesture_rec.clear_stale(current_time)

            tracker_mgr.propagate_identity()
            faces = self._build_faces(tracker_mgr.snapshot, frame_w, frame_h)
            persons = self._build_persons(tracker_mgr.person_snapshot, frame_w, frame_h)
            fall_region = self._build_fall_region(frame_w, frame_h)
            poses = self._build_poses(pose_tracks)
            fall_state = {
                "status": fall_det.status,
                "confidence": fall_det.confidence,
                "verifying": fall_det.is_verifying,
                "elapsed": fall_det.verification_elapsed,
                "tracks": fall_det.track_statuses,
            }
            state = VisionState(
                ts=current_time,
                fid=self._frame_counter,
                fps=self._estimate_fps(current_time),
                pw=frame_w,
                ph=frame_h,
                faces=faces,
                persons=persons,
                gesture=gesture_rec.latest_gesture,
                fall=fall_state,
                fall_region=fall_region,
                poses=poses,
                ir=pkt.ir_mode,
            )
            ws_server.update_state(state)

    def _identify_new_faces(
        self,
        new_faces: Sequence[Mapping[str, object]],
        frame: np.ndarray,
        frame_size: tuple[int, int],
        ir_mode: bool,
    ) -> None:
        frame_w, frame_h = frame_size
        for new_face in new_faces:
            tracker_id = cast(int, new_face["tracker_id"])
            x, y, w, h = cast(BBox, new_face["bbox"])
            if ir_mode:
                logger.info("[{}] IR mode active; local ArcFace identify skipped", tracker_id)
                self._tracker_mgr.set_user(tracker_id, "Unknown")
                continue

            face_roi = frame[max(0, y) : y + h, max(0, x) : x + w]
            if face_roi.size <= 0:
                self._tracker_mgr.set_user(tracker_id, "Unknown")
                continue

            keypoints = cast(Sequence[NormalizedKeypoint], new_face["keypoints"])
            passed, _score = self._quality_gate.check(
                face_roi,
                keypoints,
                frame_w,
                frame_h,
                (x, y, w, h),
                ir_mode=False,
            )
            if not passed:
                self._tracker_mgr.set_user(tracker_id, "Unknown")
                continue

            roi_copy = face_roi.copy()
            crop_keypoints = self._identifier.crop_keypoints(
                keypoints,
                (x, y, w, h),
                (frame_w, frame_h),
            )
            self._dispatcher.submit(self._identifier.identify, roi_copy, crop_keypoints, tracker_id)

    @staticmethod
    def _build_faces(
        snapshot: Mapping[int, Mapping[str, object]],
        frame_w: int,
        frame_h: int,
    ) -> list[TrackedFace]:
        faces: list[TrackedFace] = []
        if frame_w <= 0 or frame_h <= 0:
            return faces

        for tracker_id, data in snapshot.items():
            x, y, w, h = cast(BBox, data["bbox"])
            x1 = max(0.0, min(float(frame_w), float(x)))
            y1 = max(0.0, min(float(frame_h), float(y)))
            x2 = max(0.0, min(float(frame_w), float(x + w)))
            y2 = max(0.0, min(float(frame_h), float(y + h)))
            faces.append(TrackedFace(
                id=tracker_id,
                user=str(data["user"]),
                nx=x1 / frame_w,
                ny=y1 / frame_h,
                nw=(x2 - x1) / frame_w,
                nh=(y2 - y1) / frame_h,
            ))
        return faces

    @staticmethod
    def _build_persons(
        snapshot: Mapping[int, Mapping[str, object]],
        frame_w: int,
        frame_h: int,
    ) -> list[TrackedPerson]:
        persons: list[TrackedPerson] = []
        if frame_w <= 0 or frame_h <= 0:
            return persons

        for track_id, data in snapshot.items():
            x, y, w, h = cast(BBox, data["bbox"])
            x1 = max(0.0, min(float(frame_w), float(x)))
            y1 = max(0.0, min(float(frame_h), float(y)))
            x2 = max(0.0, min(float(frame_w), float(x + w)))
            y2 = max(0.0, min(float(frame_h), float(y + h)))
            persons.append(TrackedPerson(
                id=track_id,
                user=str(data.get("user", "Unknown")),
                nx=x1 / frame_w,
                ny=y1 / frame_h,
                nw=(x2 - x1) / frame_w,
                nh=(y2 - y1) / frame_h,
            ))
        return persons

    def _build_fall_region(self, frame_w: int, frame_h: int) -> FallRegion | None:
        if frame_w <= 0 or frame_h <= 0:
            return None

        rx, ry, rw, rh = fall_region_px(
            frame_w,
            frame_h,
            aspect=self._cfg.fall_pose_aspect,
        )
        if rw <= 0 or rh <= 0:
            return None

        return FallRegion(
            nx=rx / frame_w,
            ny=ry / frame_h,
            nw=rw / frame_w,
            nh=rh / frame_h,
        )

    @staticmethod
    def _build_poses(pose_tracks: Sequence[PoseTrackData]) -> list[dict[str, object]]:
        poses: list[dict[str, object]] = []
        for pose in pose_tracks:
            points: dict[str, list[float]] = {}
            for name, point in (pose.points or {}).items():
                nx, ny, visibility = point
                points[name] = [nx, ny, visibility]
            poses.append({"id": pose.track_id, "points": points})
        return poses

    def _estimate_fps(self, current_time):
        if self._last_state_time <= 0:
            self._last_state_time = current_time
            return 0.0

        delta = current_time - self._last_state_time
        self._last_state_time = current_time
        if delta <= 0:
            return self._fps

        instant_fps = 1.0 / delta
        self._fps = instant_fps if self._fps <= 0 else (self._fps * 0.8 + instant_fps * 0.2)
        return self._fps


def main():
    cfg = PipelineConfig()
    pipeline = VisionPipeline(cfg)
    pipeline.run()


if __name__ == '__main__':
    main()
