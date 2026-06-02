from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    # Camera
    camera_index: int = 0
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
