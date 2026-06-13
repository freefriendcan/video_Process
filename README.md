# Vision Pipeline — Edge-AI Smart Home

Real-time computer vision pipeline for smart home environments. Runs face tracking with identification, gesture recognition, and transformer-based fall detection on a live camera stream. Analyzed insights are transmitted to a downstream [proactive-home-agent](https://github.com/yourusername/proactive-home-agent) for agentic decision-making.

Optimized for macOS with Apple Silicon (M1/M2/M3/M4).

## Features

- **Face Detection & Tracking**: BlazeFace short-range detection + KCF tracker with IoU-based re-identification
- **Face Identification**: Quality-gated face crops sent to remote Pi 5 for DeepFace-based identification
- **Gesture Recognition**: MediaPipe async gesture recognizer with sustained-gesture detection and gaze-lock filtering
- **Fall Detection**: Transformer-based fall classifier (TFLite) with 3-stage pipeline:
  1. MediaPipe Pose keypoint extraction + hip-centered normalization
  2. Velocity gate (reject slow movements like sitting/bending)
  3. Post-fall inactivity verification (3s confirmation window)
- **Live Streaming**: go2rtc WebRTC HD video for dashboards plus a lightweight vision-state WebSocket for overlays
- **Graceful Shutdown**: SIGINT/SIGTERM handler sends `camera_offline` signal to backend

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start go2rtc
docker compose -f go2rtc/docker-compose.yaml up -d

# Run the pipeline
uv run python main.py
```

Open the dashboard video stream with WebRTC:

```text
http://localhost:1984/stream.html?src=living_room_hd&mode=webrtc
```

For local Docker Desktop usage, go2rtc advertises `127.0.0.1:8555` as the WebRTC ICE candidate. Close any older go2rtc tabs opened without `mode=webrtc`; those tabs will show up as `mse/fmp4` consumers.

The pipeline publishes overlay metadata separately on `ws://localhost:5003`.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  mac_camera.py                       │
│                                                      │
│  ┌──────────┐   ┌────────────┐   ┌───────────────┐  │
│  │ RTSP SD   │──▶│ Processing │──▶│ Vision State  │  │
│  │ Capture   │   │ Loop       │   │ WebSocket     │  │
│  └──────────┘   └─────┬──────┘   └───────────────┘  │
│                       │                              │
│         ┌─────────────┼─────────────┐                │
│         ▼             ▼             ▼                │
│  ┌────────────┐ ┌──────────┐ ┌──────────────┐       │
│  │ BlazeFace  │ │ Gesture  │ │ Fall         │       │
│  │ + KCF      │ │ (async)  │ │ Detection    │       │
│  │ Tracking   │ │          │ │ (Transformer)│       │
│  └─────┬──────┘ └────┬─────┘ └──────┬───────┘       │
│        │              │              │               │
│        ▼              ▼              ▼               │
│  ┌──────────────────────────────────────────────┐    │
│  │         ThreadPoolExecutor (HTTP)             │    │
│  │   → Pi:8000/vision/identify                   │    │
│  │   → Pi:8000/vision/update_presence            │    │
│  │   → Pi:8000/vision/gesture                    │    │
│  │   → Pi:8000/vision/fall_alert                 │    │
│  └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ proactive-home-agent │
              │ (Pi 5 Backend)       │
              └─────────────────────┘
```

## Models

| Model | File | Purpose |
|-------|------|---------|
| BlazeFace Short Range | `blaze_face_short_range.tflite` | Face detection (auto-downloaded) |
| MediaPipe Gesture Recognizer | `gesture_recognizer.task` | Hand gesture recognition (auto-downloaded) |
| Fall Detection Transformer | `data/models/fall_detection_transformer.tflite` | Pose-sequence fall classification |
| YOLOv8n Face | `data/models/yolov8n-face.pt` | Reserved for future on-device face detection |

## Streaming Layout

| Purpose | go2rtc stream | Transport | Source |
|---------|---------------|-----------|--------|
| Computer vision input | `living_room_sd` | RTSP | Tapo `/stream2` |
| Dashboard video | `living_room_hd` | WebRTC | Tapo `/stream1` |
| Overlay metadata | `ws://localhost:5003` | WebSocket JSON | Pipeline state |

## Configuration

Key constants are defined at the top of `mac_camera.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `PI_IP` | `100.105.136.5` | Backend agent IP address |
| `FALL_CONFIDENCE_THRESHOLD` | `0.90` | Minimum probability for fall alert |
| `FALL_ALERT_COOLDOWN` | `10s` | Seconds between fall alerts |
| `POST_FALL_WAIT` | `3.0s` | Inactivity verification window |
| `GESTURE_COOLDOWN` | `1.0s` | Minimum interval between gesture events |
| `MIN_FACE_ROI_SIZE` | `60px` | Minimum face size for identification |

## Requirements

- Python 3.10+
- macOS (tested on Apple Silicon)
- Webcam access
- Network access to Pi 5 backend (for face identification and event dispatch)

## License

MIT License
