"""Main CLI entry point for YOLO Smart Home system."""

import signal
import sys
from pathlib import Path

import cv2
import numpy as np
import click
from loguru import logger

from .config.settings import Config, load_config, get_device
from .capture.base import BaseCapture
from .capture.camera import CameraCapture, RTSPCapture
from .capture.video_file import VideoFileCapture
from .capture.frame_queue import FrameQueue
from .detectors.person import PersonDetector
from .detectors.face import FaceDetector
from .detectors.pose import PoseDetector
from .recognizers.face_recognizer import FaceRecognizer
from .recognizers.fall_detector import FallDetector
from .recognizers.gesture_detector import GestureDetector
from .events.event import (
    create_face_recognized_event,
    create_face_unknown_event,
    create_fall_event,
    create_gesture_event,
    create_person_event,
)
from .events.handler import EventHandler
from .events.alerts import create_default_alerter
from .events.logger import EventLogger
from .utils.draw import Drawer
from .utils.mps_utils import print_device_info, test_mps
from .models.model_loader import ModelLoader


class SmartHomeProcessor:
    """Main processor for smart home video analysis."""

    def __init__(self, config: Config):
        """
        Initialize processor.

        Args:
            config: Configuration object
        """
        self.config = config
        self.running = False

        # Initialize detectors
        self.person_detector = PersonDetector(config) if config.person_detection.enabled else None
        self.face_detector = FaceDetector(config) if config.face_recognition.enabled else None
        self.pose_detector = PoseDetector(config) if (
            config.fall_detection.enabled or config.gesture_detection.enabled
        ) else None

        # Initialize recognizers
        self.face_recognizer = FaceRecognizer(config) if config.face_recognition.enabled else None
        self.fall_detector = FallDetector(config) if config.fall_detection.enabled else None
        self.gesture_detector = GestureDetector(config) if config.gesture_detection.enabled else None

        # Initialize event handling
        event_logger = EventLogger(config.alerts.log_file)
        alerter = create_default_alerter(config)
        self.event_handler = EventHandler(
            config,
            logger=event_logger,
            screenshot_path=config.alerts.screenshot_path,
            enable_screenshots=config.alerts.screenshot_on_event,
        )
        self.event_handler.add_global_listener(alerter.send_alert)

        # Visualization
        self.drawer = Drawer()
        self.show_display = True

    def process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """
        Process a single frame through all detectors.

        Args:
            frame: Input frame
            frame_number: Frame counter

        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        self.event_handler.frame_count = frame_number

        # Person detection
        if self.person_detector:
            person_result = self.person_detector.detect(frame)
            if person_result["persons_found"]:
                for person in person_result["persons"]:
                    self.drawer.draw_person_box(
                        annotated,
                        person["box"],
                        confidence=person["confidence"],
                    )

                # Create person event
                event = create_person_event(
                    count=person_result["count"],
                    locations=person_result["boxes"],
                )
                self.event_handler.handle_event(event, frame)

        # Face recognition
        if self.face_recognizer:
            face_result = self.face_recognizer.recognize(frame)
            if face_result["face_found"]:
                box = face_result["location"]
                name = face_result["name"]
                conf = face_result["confidence"]

                self.drawer.draw_face_box(
                    annotated,
                    box,
                    name=name or "Unknown",
                    confidence=conf,
                    recognized=name is not None,
                )

                # Create face event
                if name:
                    event = create_face_recognized_event(name, box, conf)
                else:
                    event = create_face_unknown_event(box, conf)
                self.event_handler.handle_event(event, frame)

        # Fall detection
        if self.fall_detector:
            fall_result = self.fall_detector.detect(frame)
            if fall_result["fall_detected"]:
                self.drawer.draw_fall_alert(
                    annotated,
                    fall_result["state"],
                    fall_result["confidence"],
                )

                event = create_fall_event(
                    fall_result["state"],
                    fall_result["confidence"],
                    fall_result["metrics"],
                )
                self.event_handler.handle_event(event, frame)

        # Gesture detection
        if self.gesture_detector:
            gesture_result = self.gesture_detector.detect(frame)
            for gesture_name in gesture_result["gestures_detected"]:
                confidence = gesture_result["all_gestures"][gesture_name]
                self.drawer.draw_gesture(annotated, gesture_name, confidence)

                event = create_gesture_event(
                    gesture_name,
                    confidence,
                    gesture_result["all_gestures"],
                )
                self.event_handler.handle_event(event, frame)

        # Draw timestamp
        self.drawer.draw_timestamp(annotated)

        # Draw info overlay
        info = {
            "Frame": str(frame_number),
            "Device": get_device(self.config),
        }
        self.drawer.draw_info(annotated, info)

        return annotated

    def run(self, capture) -> None:
        """
        Main processing loop.

        Args:
            capture: BaseCapture instance
        """
        self.running = True
        frame_count = 0

        logger.info("Starting processing loop...")

        with FrameQueue(capture, queue_size=self.config.video.queue_size) as queue:
            while self.running:
                frame = queue.read(timeout=1.0)

                if frame is None:
                    if isinstance(capture, VideoFileCapture):
                        logger.info("End of video file reached")
                        break
                    continue

                frame_count += 1

                # Process frame
                annotated = self.process_frame(frame, frame_count)

                # Display
                if self.show_display:
                    resized = self.drawer.resize_with_aspect_ratio(annotated, max_width=1280)
                    cv2.imshow("YOLO Smart Home", resized)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("Quit requested by user")
                        break
                    elif key == ord("s"):
                        # Toggle display
                        self.show_display = not self.show_display

        self.running = False
        cv2.destroyAllWindows()

    def stop(self) -> None:
        """Stop processing."""
        self.running = False


def create_capture(config: Config) -> BaseCapture:
    """
    Create appropriate capture based on config.

    Args:
        config: Configuration object

    Returns:
        BaseCapture instance
    """
    video_config = config.video

    if video_config.source == "camera":
        # Check if path looks like RTSP URL
        if isinstance(video_config.path, str) and video_config.path.startswith("rtsp://"):
            return RTSPCapture(
                video_config.path,
                fps=video_config.fps,
                resolution=video_config.resolution,
            )
        else:
            return CameraCapture(
                video_config.path,
                fps=video_config.fps,
                resolution=video_config.resolution,
            )

    elif video_config.source == "rtsp":
        return RTSPCapture(
            video_config.path,
            fps=video_config.fps,
            resolution=video_config.resolution,
        )

    elif video_config.source == "file":
        return VideoFileCapture(
            video_config.path,
            loop=False,
        )

    else:
        raise ValueError(f"Unknown video source: {video_config.source}")


# CLI Commands

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """YOLO Smart Home - Face recognition, fall detection, and gesture recognition."""
    pass


@cli.command()
@click.option("--source", type=str, default="camera", help="Video source: camera, rtsp, file")
@click.option("--path", type=str, default="0", help="Camera index, RTSP URL, or file path")
@click.option("--config", type=str, default=None, help="Path to config file")
@click.option("--no-display", is_flag=True, help="Disable display window")
def run(source, path, config, no_display):
    """Run smart home video processing."""
    # Load configuration
    cfg = load_config(config)

    # Override with command line args
    cfg.video.source = source
    cfg.video.path = path

    logger.info("Starting YOLO Smart Home System")
    logger.info(f"Video source: {source} ({path})")

    # Print device info
    print_device_info()

    # Create processor
    processor = SmartHomeProcessor(cfg)
    processor.show_display = not no_display

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        processor.stop()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Create capture
        capture = create_capture(cfg)

        # Run processing
        processor.run(capture)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

    logger.info("Processing complete")


@cli.command()
@click.option("--name", type=str, required=True, help="Name of the person")
@click.option("--source", type=str, default="camera", help="Video source")
@click.option("--path", type=str, default="0", help="Camera index or file path")
@click.option("--config", type=str, default=None, help="Path to config file")
@click.option("--faces", type=int, default=5, help="Number of face images to capture")
def register_face(name, source, path, config, faces):
    """Register a face for recognition."""
    cfg = load_config(config)

    logger.info(f"Registering face for: {name}")
    print(f"\n=== Face Registration: {name} ===")
    print("Position your face in front of the camera.")
    print(f"We will capture {faces} images.")
    print("Press SPACE to capture, 'q' to quit.\n")

    # Create recognizer
    recognizer = FaceRecognizer(cfg)

    # Create capture
    if source == "camera":
        from .capture.camera import CameraCapture
        capture = CameraCapture(path)
    else:
        from .capture.video_file import VideoFileCapture
        capture = VideoFileCapture(path)

    if not capture.open():
        logger.error("Failed to open capture source")
        return

    cv2.namedWindow("Face Registration")

    # Create detector once, outside the loop
    detector = FaceDetector(cfg)

    captured_count = 0
    frame_count = 0

    while captured_count < faces:
        frame = capture.read()
        if frame is None:
            continue

        frame_count += 1

        # Detect face
        result = detector.detect(frame)

        # Draw face box
        display = frame.copy()
        if result["faces_found"]:
            for face in result["faces"]:
                x1, y1, x2, y2 = face["box"]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw instructions
        cv2.putText(
            display,
            f"Captured: {captured_count}/{faces}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display,
            "SPACE: Capture | Q: Quit",
            (10, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Face Registration", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" ") and result["faces_found"]:
            # Get largest face from current detection (no re-detection)
            face_data = max(
                result["faces"],
                key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]),
            )
            x1, y1, x2, y2 = face_data["box"]

            # Use padded ROI for better alignment during embedding extraction
            from .recognizers.face_recognizer import FaceRecognizer as FR
            face_roi = FR._extract_padded_roi(frame, x1, y1, x2, y2)

            if recognizer.register_face(face_roi, name):
                captured_count += 1
                print(f"Captured {captured_count}/{faces}")

    capture.close()
    cv2.destroyAllWindows()

    print(f"\nRegistration complete! Registered {captured_count} images for {name}")
    print(f"Total faces in database: {recognizer.get_face_count()}")


@cli.command()
@click.option("--config", type=str, default=None, help="Path to config file")
def list_models(config):
    """List available YOLO models."""
    cfg = load_config(config)
    loader = ModelLoader(cfg.models_dir)

    print("\n=== Available Models ===\n")

    models = loader.list_models()

    for model in models:
        status = "✓" if model["exists"] else "✗"
        size = f" ({model['size_mb']:.1f} MB)" if model.get("size_mb") else " (not downloaded)"
        print(f"{status} {model['name']}{size}")

    print()


@cli.command()
@click.option("--config", type=str, default=None, help="Path to config file")
def device_info(config):
    """Show device and acceleration information."""
    print("\n=== Device Information ===\n")
    print_device_info()

    print("\n=== MPS Test ===\n")
    results = test_mps()

    for key, value in results.items():
        if key != "error":
            print(f"{key}: {value}")

    if results.get("error"):
        print(f"\nError: {results['error']}")

    print()


@cli.command()
@click.option("--config", type=str, default=None, help="Path to config file")
@click.option("--name", type=str, help="Name to remove from face database")
def forget_face(config, name):
    """Remove a face from the recognition database."""
    cfg = load_config(config)
    recognizer = FaceRecognizer(cfg)

    if not name:
        # List all faces
        faces = recognizer.list_known_faces()
        if not faces:
            print("No faces registered.")
            return

        print("\n=== Registered Faces ===\n")
        for face in faces:
            print(f"  - {face}")
        print()
        return

    # Remove specific face
    if recognizer.forget_face(name):
        print(f"Removed face: {name}")
    else:
        print(f"Face not found: {name}")


def main():
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
