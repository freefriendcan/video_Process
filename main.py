import signal
import sys
import threading
import time

import cv2

from config import PipelineConfig
from quality.face_quality_gate import FaceQualityGate
from events.dispatcher import EventDispatcher
from detection.face_detector import FaceDetector
from detection.gesture_recognizer import GestureRecognizer
from detection.fall_detector import FallDetector
from tracking.tracker_manager import TrackerManager
from identification.face_identifier import FaceIdentifier
from capture.frame_producer import FrameProducer
from streaming.mjpeg_server import MjpegServer


class VisionPipeline:
    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg
        self._quality_gate = FaceQualityGate(cfg)
        self._dispatcher = EventDispatcher(cfg)
        self._tracker_mgr = TrackerManager(cfg)
        self._producer = FrameProducer(cfg)
        self._server = MjpegServer(cfg)
        self._face_det = FaceDetector(cfg)
        self._gesture_rec = GestureRecognizer(cfg, self._dispatcher, self._tracker_mgr.get_active_user)
        self._fall_det = FallDetector(cfg)
        self._identifier = FaceIdentifier(cfg, self._tracker_mgr)

    def run(self):
        self._producer.open()
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        threading.Thread(target=self._camera_loop, daemon=True).start()
        self._server.run()

    def shutdown(self):
        print("\nCTRL+C Caught! Camera is being safely turned off...")
        try:
            self._dispatcher.send_offline_signal()
        finally:
            self._dispatcher.shutdown()
            self._producer.release()

    def _handle_signal(self, signum, frame):
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
        server = self._server

        last_detection_time = 0

        while True:
            success, frame = producer.read()
            if not success:
                time.sleep(0.1)
                continue

            frame_h, frame_w = frame.shape[:2]
            current_time = time.time()

            active = tracker_mgr.update_all(frame, current_time)

            for t in active:
                t_id = t["id"]
                x, y, w, h = t["bbox"]
                user = t["user"]

                color = (0, 0, 255) if user == "Unknown" else (0, 255, 0)
                if user == "Identifying...": color = (255, 0, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"ID: {user}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
                            face_roi, t["detection_keypoints"], frame_w, frame_h, (x, y, w, h)
                        )
                        if passed:
                            cooldown = 2.0 * (2 ** t["retry_count"])
                            time_ok = (current_time - t["last_identify_time"]) > cooldown
                            quality_ok = current_quality > t["best_quality_score"] * 0.8

                            if time_ok and quality_ok:
                                print(f"[{t_id}] Quality OK, retrying identification (attempt {t['retry_count']+1}/{cfg.max_retries})...")
                                tracker_mgr.update_field(
                                    t_id,
                                    user="Identifying...",
                                    last_identify_time=current_time,
                                    best_quality_score=max(t["best_quality_score"], current_quality),
                                )

                                roi_copy = face_roi.copy()
                                dispatcher.submit(identifier.identify, roi_copy, t_id)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            gesture_rec.process(rgb_frame, current_time)

            gesture_rec.clear_stale(current_time)
            if gesture_rec.latest_gesture:
                cv2.putText(frame, f"Gesture: {gesture_rec.latest_gesture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            fall_result = fall_det.process_frame(rgb_frame, frame, current_time)
            if fall_result:
                conf, screenshot_path = fall_result
                dispatcher.submit(dispatcher.send_fall_alert, conf, screenshot_path)

            if fall_det.is_verifying:
                elapsed = fall_det.verification_elapsed
                cv2.putText(frame,
                            f"VERIFYING FALL ({elapsed:.1f}s/{cfg.post_fall_wait}s)",
                            (20, frame_h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            elif fall_det.status == "FALL CONFIRMED":
                cv2.putText(frame,
                            f"FALL CONFIRMED ({fall_det.confidence:.0%})",
                            (20, frame_h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            elif fall_det.status and fall_det.status not in ("Standing", ""):
                cv2.putText(frame, fall_det.status, (20, frame_h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if current_time - last_detection_time > 0.15:
                faces = face_det.detect(rgb_frame)
                new_faces = tracker_mgr.match_and_update(faces, frame, current_time)

                for nf in new_faces:
                    t_id = nf["tracker_id"]
                    x, y, w, h = nf["bbox"]
                    face_roi = frame[max(0, y):y+h, max(0, x):x+w]
                    if face_roi.size > 0:
                        passed, _ = quality_gate.check(
                            face_roi, nf["keypoints"], frame_w, frame_h, (x, y, w, h)
                        )
                        if passed:
                            roi_copy = face_roi.copy()
                            dispatcher.submit(identifier.identify, roi_copy, t_id)
                        else:
                            tracker_mgr.set_user(t_id, "Unknown")

                last_detection_time = current_time

            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    server.update_frame(buffer.tobytes())
            except Exception as e:
                print(f"Encode Error: {e}")


def main():
    cfg = PipelineConfig()
    pipeline = VisionPipeline(cfg)
    pipeline.run()


if __name__ == '__main__':
    main()
