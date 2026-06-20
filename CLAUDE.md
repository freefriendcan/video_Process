# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time edge-AI vision pipeline for smart home environments, running on macOS with Apple Silicon. Captures a live webcam feed and runs three parallel CV pipelines: face tracking with remote identification, gesture recognition, and transformer-based fall detection. Events are dispatched over HTTP to a Raspberry Pi 5 backend ([proactive-home-agent](https://github.com/yourusername/proactive-home-agent)) for agentic decision-making.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
# or with uv:
uv pip install -r requirements.txt

# Run the pipeline (starts camera + Flask MJPEG server on :5001)
python mac_camera.py

# Clean up logs and fall screenshots
python scripts/cleanup_logs.py

# Lint / format (dev dependencies)
ruff check .
black --check .
mypy .
pytest
```

## Architecture

The entire pipeline lives in a single file: `mac_camera.py` (~955 lines). There is no package structure — it's a monolithic script with module-level state.

### Threading Model

- **Main thread**: Flask MJPEG server (`app.run`)
- **Camera thread**: `camera_processing_loop()` runs in a daemon thread, drives all CV processing in a tight `while True` loop
- **Network pool**: `ThreadPoolExecutor(max_workers=4)` handles all HTTP calls to the Pi backend (identification, presence, gestures, fall alerts)
- **Async gesture**: MediaPipe's `LIVE_STREAM` mode runs gesture inference on its own internal thread; results arrive via `gesture_callback`

Shared state (`active_trackers`, gesture/fall globals) is guarded by `trackers_lock` or the GIL. The `latest_jpeg_frame` global is the bridge between the camera thread and Flask's MJPEG generator.

### Three CV Pipelines (all in camera thread)

1. **Face Detection & Tracking**: BlazeFace detection every 150ms → KCF tracker between detections → IoU matching for re-identification → quality-gated face crops sent to Pi for DeepFace identification (with exponential backoff retries)
2. **Gesture Recognition**: Async MediaPipe gesture recognizer with a gaze-lock filter (ignores gestures when user isn't facing camera) and sustained-gesture detection (>1s hold required before dispatch)
3. **Fall Detection**: 3-stage pipeline — (a) MediaPipe Pose → hip-centered normalized 51-dim feature extraction, (b) TFLite transformer inference on 30-frame sliding window, (c) velocity gate rejects slow movements, (d) 3-second post-fall inactivity verification before confirming

### Backend Communication

All HTTP calls go to `PI_IP` (Tailscale IP, default `100.105.136.5`) on port 8000:
- `POST /vision/identify` — face crop → DeepFace identification
- `POST /vision/update_presence` — periodic presence heartbeats
- `POST /vision/gesture` — sustained gesture events
- `POST /vision/fall_alert` — fall alerts with optional screenshot attachment

### Data Layout

- `data/models/` — TFLite models (fall detection transformer, YOLOv8n face)
- `data/embeddings/` — face embedding pickle files
- `data/logs/` — event logs and fall detection screenshots
- `data/recordings/` — video recordings (gitignored)
- Root `.tflite`/`.task` files — BlazeFace and gesture recognizer models (auto-downloaded on first run)

## Key Configuration

All tunable constants are at the top of `mac_camera.py` (lines 29-97). The most critical:
- `PI_IP` — backend agent IP
- `FALL_CONFIDENCE_THRESHOLD` (0.90), `MIN_FALL_VELOCITY` (0.025), `POST_FALL_WAIT` (3.0s) — fall detection sensitivity
- `TARGET_FALL_FPS` (15) — fall processing rate limit
- `GESTURE_COOLDOWN` (1.0s) — minimum interval between gesture dispatches

## Caveats

- The codebase has Turkish comments in several places (developer's native language).
- Camera opens at module import time (`cv2.VideoCapture(CAMERA_INDEX)` at line 30), so importing `mac_camera` has side effects.
- Model files are large binaries, gitignored, and auto-downloaded or expected to exist at startup.
- The `async_image_buffer` list (line 647) is intentional — it prevents Python's GC from collecting `mp.Image` objects while MediaPipe's C++ backend still references them. Do not remove it.
- SSL verification is globally disabled at startup (lines 20-25) as a macOS workaround.
