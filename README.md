# YOLO Smart Home Video Processing System

An AI-powered smart home video processing system using YOLO-based computer vision. Performs real-time face recognition, fall detection, and gesture recognition with support for live camera streams (RTSP/IP cameras, USB webcams) and pre-recorded video files.

Optimized for Apple Silicon Mac (M1/M2/M3) using MPS (Metal Performance Shaders) acceleration.

## Features

- **Face Recognition**: Detect and identify known persons using DeepFace embeddings
- **Fall Detection**: Pose-based fall detection for elderly care and safety monitoring
- **Gesture Recognition**: Recognize gestures like waving, hands up, pointing, and crouching
- **Multiple Input Sources**: USB cameras, RTSP/IP cameras, and video files
- **Real-time Processing**: Threaded capture for non-blocking inference
- **Event System**: Alert generation with configurable cooldowns and logging
- **MPS Acceleration**: Optimized for Apple Silicon with automatic device selection

## Installation

### Requirements

- Python 3.10 or higher
- macOS (for MPS acceleration) or Linux/Windows
- 4GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd video_Process
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Models will be automatically downloaded on first run to `data/models/`.

## Quick Start

### Run with default camera

```bash
python -m src.main run
```

### Run with video file

```bash
python -m src.main run --source file --path path/to/video.mp4
```

### Run with RTSP camera

```bash
python -m src.main run --source rtsp --path rtsp://192.168.1.100:554/stream
```

## Commands

### `run` - Start video processing

```bash
python -m src.main run [OPTIONS]
```

Options:
- `--source`: Video source (camera, rtsp, file)
- `--path`: Camera index, RTSP URL, or file path
- `--config`: Path to config file
- `--no-display`: Disable display window

### `register-face` - Register a face for recognition

```bash
python -m src.main register-face --name "John Doe"
```

Options:
- `--name`: Name of the person (required)
- `--source`: Video source
- `--faces`: Number of face images to capture (default: 5)

### `list-models` - List available YOLO models

```bash
python -m src.main list-models
```

### `device-info` - Show device and acceleration information

```bash
python -m src.main device-info
```

### `forget-face` - Remove a face from database

```bash
python -m src.main forget-face --name "John Doe"
```

## Configuration

Configuration is managed via `config/default_config.yaml`:

```yaml
video:
  source: "camera"
  path: "0"
  fps: 30
  resolution: [1280, 720]
  queue_size: 10

detection:
  device: "auto"    # auto, mps, cpu, cuda
  confidence: 0.5
  iou: 0.45

face_recognition:
  enabled: true
  model: "VGG-Face"
  threshold: 0.4
  database_path: "data/embeddings/faces.pkl"

fall_detection:
  enabled: true
  aspect_ratio_threshold: 2.5
  min_frames_for_fall: 5
  alert_cooldown: 30

gesture_detection:
  enabled: true
  smoothing_window: 5
  min_confidence: 0.7

alerts:
  console: true
  log_file: "data/logs/events.jsonl"
  screenshot_on_event: true
  screenshot_path: "data/logs/screenshots"
```

You can also override settings via environment variables:
```bash
export VIDEO_DETECTION_CONFIDENCE=0.7
export VIDEO_FACE_RECOGNITION_THRESHOLD=0.3
python -m src.main run
```

## Project Structure

```
video_Process/
├── src/
│   ├── config/          # Configuration management
│   ├── capture/         # Video capture (camera, file, RTSP)
│   ├── detectors/       # YOLO-based detectors
│   ├── recognizers/     # High-level recognition (face, fall, gesture)
│   ├── events/          # Event handling and alerts
│   ├── utils/           # Visualization and geometry
│   └── models/          # Model management
├── data/
│   ├── models/          # Downloaded YOLO models
│   ├── embeddings/      # Face embeddings database
│   ├── logs/            # Event logs
│   └── recordings/      # Optional video recordings
├── config/
│   └── default_config.yaml
└── tests/               # Test suite
```

## Supported Gestures

- **wave**: Hand waving motion detection
- **hands_up**: Both hands raised above shoulders
- **pointing**: One arm extended, other arm down
- **crouching**: Knees below hip level

## Adding Custom Gestures

Extend `BaseGesture` in `src/recognizers/gesture_detector.py`:

```python
class MyGesture(BaseGesture):
    name = "my_gesture"

    def detect(self, keypoints: np.ndarray) -> bool:
        # Implement gesture detection logic
        # keypoints: (17, 3) array with x, y, confidence
        return True  # Return True if gesture detected
```

Register in `GestureDetector.__init__()` or use `add_gesture()`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| MPS not available | Ensure PyTorch >= 2.0 with MPS support |
| Camera not opening | Check permissions; try different camera index |
| Poor face recognition | Lower threshold or register more angles |
| Fall detection too sensitive | Increase `min_frames_for_fall` |
| High CPU usage | Reduce resolution or FPS |

## License

MIT License
