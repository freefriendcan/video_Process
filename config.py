import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


@dataclass
class PipelineConfig:
    # RTSP (Tapo C225)
    rtsp_url: str = ""
    rtsp_hires_url: str = ""
    rtsp_reconnect_delay: float = 2.0
    rtsp_max_reconnect_delay: float = 30.0

    # go2rtc proxy
    go2rtc_enabled: bool = False
    go2rtc_host: str = "localhost"
    go2rtc_rtsp_port: int = 8554
    go2rtc_webrtc_port: int = 8555
    go2rtc_api_port: int = 1984
    go2rtc_stream_name: str = "living_room"
    go2rtc_camera_stream_name: str = ""
    go2rtc_hires_stream_name: str = ""
    go2rtc_dashboard_stream_name: str = ""
    go2rtc_dashboard_mode: str = "webrtc"
    go2rtc_connect_timeout: float = 0.75

    # Vision overlay WebSocket
    ws_vision_host: str = "0.0.0.0"
    ws_vision_port: int = 5003
    ws_vision_fps: float = 15.0

    # IR Detection
    ir_std_threshold: float = 3.0

    # Frame
    frame_width: int = 1280
    frame_height: int = 720

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

        self.go2rtc_enabled = _env_bool("GO2RTC_ENABLED", self.go2rtc_enabled)
        self.go2rtc_host = os.environ.get("GO2RTC_HOST", self.go2rtc_host)
        self.go2rtc_rtsp_port = _env_int("GO2RTC_RTSP_PORT", self.go2rtc_rtsp_port)
        self.go2rtc_webrtc_port = _env_int("GO2RTC_WEBRTC_PORT", self.go2rtc_webrtc_port)
        self.go2rtc_api_port = _env_int("GO2RTC_API_PORT", self.go2rtc_api_port)
        self.go2rtc_stream_name = os.environ.get("GO2RTC_STREAM_NAME", self.go2rtc_stream_name)
        self.go2rtc_camera_stream_name = os.environ.get(
            "GO2RTC_CAMERA_STREAM_NAME",
            self.go2rtc_camera_stream_name or f"{self.go2rtc_stream_name}_sd",
        )
        self.go2rtc_hires_stream_name = os.environ.get(
            "GO2RTC_HIRES_STREAM_NAME",
            self.go2rtc_hires_stream_name or f"{self.go2rtc_stream_name}_hd",
        )
        self.go2rtc_dashboard_stream_name = os.environ.get(
            "GO2RTC_DASHBOARD_STREAM_NAME",
            self.go2rtc_dashboard_stream_name or self.go2rtc_hires_stream_name,
        )
        self.go2rtc_dashboard_mode = os.environ.get(
            "GO2RTC_DASHBOARD_MODE",
            self.go2rtc_dashboard_mode,
        )
        self.go2rtc_connect_timeout = _env_float(
            "GO2RTC_CONNECT_TIMEOUT",
            self.go2rtc_connect_timeout,
        )

        self.ws_vision_host = os.environ.get("WS_VISION_HOST", self.ws_vision_host)
        self.ws_vision_port = _env_int("WS_VISION_PORT", self.ws_vision_port)
        self.ws_vision_fps = _env_float("WS_VISION_FPS", self.ws_vision_fps)

        if self.go2rtc_enabled:
            self.rtsp_url = (
                f"rtsp://{self.go2rtc_host}:{self.go2rtc_rtsp_port}/"
                f"{self.go2rtc_camera_stream_name}"
            )
            self.rtsp_hires_url = (
                f"rtsp://{self.go2rtc_host}:{self.go2rtc_rtsp_port}/"
                f"{self.go2rtc_hires_stream_name}"
            )
        elif not self.rtsp_url:
            # Build direct RTSP URLs from environment when not set explicitly.
            ip = os.environ.get("TAPO_IP", "")
            user = os.environ.get("TAPO_USER", "")
            pw = os.environ.get("TAPO_PASS", "")
            if ip and user and pw:
                self.rtsp_url = f"rtsp://{user}:{pw}@{ip}:554/stream2"
                self.rtsp_hires_url = f"rtsp://{user}:{pw}@{ip}:554/stream1"

    def preflight(self):
        """Validate external services needed by the configured capture source."""
        if not self.go2rtc_enabled:
            return

        self._check_go2rtc_port(
            service="RTSP",
            port=self.go2rtc_rtsp_port,
            guidance="Start it with: cd go2rtc && docker compose up -d",
        )
        self._check_go2rtc_port(
            service="API",
            port=self.go2rtc_api_port,
            guidance=(
                f"Check it with: curl http://{self.go2rtc_host}:"
                f"{self.go2rtc_api_port}/api/streams"
            ),
        )

    def _check_go2rtc_port(self, service: str, port: int, guidance: str):
        try:
            sock = socket.create_connection(
                (self.go2rtc_host, port),
                timeout=self.go2rtc_connect_timeout,
            )
        except OSError as exc:
            raise RuntimeError(
                f"go2rtc {service} is not reachable at "
                f"{self.go2rtc_host}:{port}. {guidance}"
            ) from exc

        sock.close()

    @property
    def go2rtc_dashboard_url(self) -> str:
        query = urlencode(
            {
                "src": self.go2rtc_dashboard_stream_name,
                "mode": self.go2rtc_dashboard_mode,
            }
        )
        return f"http://{self.go2rtc_host}:{self.go2rtc_api_port}/stream.html?{query}"
