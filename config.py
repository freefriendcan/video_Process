import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class PipelineConfig:
    # RTSP (Tapo C225)
    rtsp_url: str = ""
    rtsp_hires_url: str = ""
    rtsp_reconnect_delay: float = 2.0
    rtsp_max_reconnect_delay: float = 30.0

    # IR Detection
    ir_std_threshold: float = 3.0

    # Frame
    frame_width: int = 640
    frame_height: int = 480

    # Backend
    pi_ip: str = "100.105.136.5"
    pi_port: int = 8000

    # Model paths
    blaze_face_model: str = "blaze_face_short_range.tflite"
    gesture_model: str = "gesture_recognizer.task"
    fall_model: str = "data/models/fall_detection_transformer.tflite"

    # Face detection
    min_detection_confidence: float = 0.5

    # Gesture
    gesture_cooldown: float = 1.0

    # Fall detection
    fall_input_timesteps: int = 30
    fall_confidence_threshold: float = 0.90
    fall_alert_cooldown: int = 10
    target_fall_fps: int = 15
    screenshot_dir: Path = field(default_factory=lambda: Path("data/logs/screenshots"))

    # Velocity filter
    velocity_window: int = 5
    min_fall_velocity: float = 0.025

    # Post-fall verification
    post_fall_wait: float = 3.0
    post_fall_move_threshold: float = 0.015

    # Quality gate
    min_face_roi_size: int = 60
    min_laplacian_variance: int = 50
    min_brightness: int = 40
    max_brightness: int = 230
    min_eye_distance_ratio: float = 0.15

    # Network
    max_retries: int = 3
    network_pool_size: int = 4

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 5002

    @property
    def fall_frame_time(self) -> float:
        return 1.0 / self.target_fall_fps

    @property
    def identify_url(self) -> str:
        return f"http://{self.pi_ip}:{self.pi_port}/vision/identify"

    @property
    def presence_url(self) -> str:
        return f"http://{self.pi_ip}:{self.pi_port}/vision/update_presence"

    @property
    def fall_alert_url(self) -> str:
        return f"http://{self.pi_ip}:{self.pi_port}/vision/fall_alert"

    @property
    def gesture_url(self) -> str:
        return f"http://{self.pi_ip}:{self.pi_port}/vision/gesture"

    def __post_init__(self):
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Build RTSP URLs from environment when not set explicitly
        if not self.rtsp_url:
            ip = os.environ.get("TAPO_IP", "")
            user = os.environ.get("TAPO_USER", "")
            pw = os.environ.get("TAPO_PASS", "")
            if ip and user and pw:
                self.rtsp_url = f"rtsp://{user}:{pw}@{ip}:554/stream2"
                self.rtsp_hires_url = f"rtsp://{user}:{pw}@{ip}:554/stream1"
