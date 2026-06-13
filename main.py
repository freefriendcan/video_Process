import signal
import sys
import threading
import time

from loguru import logger

# Configure loguru: remove default handler, add custom sinks
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
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


from config import PipelineConfig
from quality.face_quality_gate import FaceQualityGate
from events.dispatcher import EventDispatcher
from detection.face_detector import FaceDetector
from detection.gesture_recognizer import GestureRecognizer
from detection.fall_detector import FallDetector
from tracking.tracker_manager import TrackerManager
from identification.face_identifier import FaceIdentifier
from capture.frame_producer import FrameProducer
from streaming.vision_state import TrackedFace, VisionState
from streaming.vision_ws_server import VisionWSServer


class VisionPipeline:
    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg
        self._quality_gate = FaceQualityGate(cfg)
        self._dispatcher = EventDispatcher(cfg)
        self._tracker_mgr = TrackerManager(cfg)
        self._producer = FrameProducer(cfg)
        self._ws_server = VisionWSServer(cfg)
        self._face_det = FaceDetector(cfg)
        self._gesture_rec = GestureRecognizer(cfg, self._dispatcher, self._tracker_mgr.get_active_user)
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
        cfg = self._cfg
        quality_gate = self._quality_gate
        dispatcher = self._dispatcher
        tracker_mgr = self._tracker_mgr
        gesture_rec = self._gesture_rec
        fall_det = self._fall_det
        face_det = self._face_det
        identifier = self._identifier
        producer = self._producer
        ws_server = self._ws_server

        last_detection_time = 0

        while self._running:
            pkt = producer.read_latest()
            if pkt is None:
                time.sleep(0.005)
                continue

            frame = pkt.bgr
            frame_h, frame_w = pkt.height, pkt.width
            current_time = time.time()
            self._frame_counter += 1

            active = tracker_mgr.update_all(frame, current_time)

            for t in active:
                t_id = t["id"]
                x, y, w, h = t["bbox"]
                user = t["user"]
                if user not in ["Unknown", "Identifying..."]:
                    is_frontal = quality_gate.estimate_frontality(t["detection_keypoints"], frame_w, frame_h, (x, y, w, h))
                    gesture_rec.is_frontal = is_frontal

                    if current_time - t["last_json_time"] > 3.0:
                        dispatcher.submit(dispatcher.send_presence, user)
                        tracker_mgr.update_field(t_id, last_json_time=current_time)

                if user == "Unknown" and t["retry_count"] < cfg.max_retries:
                    face_roi = frame[max(0, y):y+h, max(0, x):x+w]

                    if face_roi.size > 0:
                        passed, current_quality = quality_gate.check(
                            face_roi, t["detection_keypoints"], frame_w, frame_h,
                            (x, y, w, h), ir_mode=pkt.ir_mode
                        )
                        if passed:
                            cooldown = 2.0 * (2 ** t["retry_count"])
                            time_ok = (current_time - t["last_identify_time"]) > cooldown
                            quality_ok = current_quality > t["best_quality_score"] * 0.8

                            if time_ok and quality_ok:
                                logger.info("[{}] Quality OK, retrying identification (attempt {}/{})", t_id, t['retry_count']+1, cfg.max_retries)
                                tracker_mgr.update_field(
                                    t_id,
                                    user="Identifying...",
                                    last_identify_time=current_time,
                                    best_quality_score=max(t["best_quality_score"], current_quality),
                                )

                                roi_copy = face_roi.copy()
                                dispatcher.submit(identifier.identify, roi_copy, t_id)

            rgb_frame = pkt.rgb

            gesture_rec.process(rgb_frame, current_time)

            gesture_rec.clear_stale(current_time)

            fall_result = fall_det.process_frame(rgb_frame, frame, current_time)
            if fall_result:
                conf, screenshot_path = fall_result
                dispatcher.submit(dispatcher.send_fall_alert, conf, screenshot_path)


            if current_time - last_detection_time > 0.15:
                faces = face_det.detect(rgb_frame)
                new_faces = tracker_mgr.match_and_update(faces, frame, current_time)

                for nf in new_faces:
                    t_id = nf["tracker_id"]
                    x, y, w, h = nf["bbox"]
                    face_roi = frame[max(0, y):y+h, max(0, x):x+w]
                    if face_roi.size > 0:
                        passed, _ = quality_gate.check(
                            face_roi, nf["keypoints"], frame_w, frame_h,
                            (x, y, w, h), ir_mode=pkt.ir_mode
                        )
                        if passed:
                            roi_copy = face_roi.copy()
                            dispatcher.submit(identifier.identify, roi_copy, t_id)
                        else:
                            tracker_mgr.set_user(t_id, "Unknown")

                last_detection_time = current_time

            faces = self._build_faces(tracker_mgr.snapshot, frame_w, frame_h)
            fall_state = {
                "status": fall_det.status,
                "confidence": fall_det.confidence,
                "verifying": fall_det.is_verifying,
                "elapsed": fall_det.verification_elapsed,
            }
            state = VisionState(
                ts=current_time,
                fid=self._frame_counter,
                fps=self._estimate_fps(current_time),
                pw=frame_w,
                ph=frame_h,
                faces=faces,
                gesture=gesture_rec.latest_gesture,
                fall=fall_state,
                ir=pkt.ir_mode,
            )
            ws_server.update_state(state)

    @staticmethod
    def _build_faces(snapshot, frame_w, frame_h):
        faces = []
        if frame_w <= 0 or frame_h <= 0:
            return faces

        for tracker_id, data in snapshot.items():
            x, y, w, h = data["bbox"]
            x1 = max(0.0, min(float(frame_w), float(x)))
            y1 = max(0.0, min(float(frame_h), float(y)))
            x2 = max(0.0, min(float(frame_w), float(x + w)))
            y2 = max(0.0, min(float(frame_h), float(y + h)))
            faces.append(TrackedFace(
                id=tracker_id,
                user=data["user"],
                nx=x1 / frame_w,
                ny=y1 / frame_h,
                nw=(x2 - x1) / frame_w,
                nh=(y2 - y1) / frame_h,
            ))
        return faces

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
