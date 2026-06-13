# Handoff - video_Process

Last verified: 2026-06-13

## Current Goal

Continue from the new two-stream architecture:

- Computer vision uses the SD RTSP stream.
- Dashboard video uses the HD stream over WebRTC.
- Overlay/state metadata is sent separately over the pipeline WebSocket.

## Streaming Architecture

| Purpose | Stream | Transport | Source |
| --- | --- | --- | --- |
| Computer vision input | `living_room_sd` | RTSP via go2rtc `:8554` | Tapo `/stream2`, lowest profile, typically 720p |
| Dashboard video | `living_room_hd` | WebRTC via go2rtc `:8555` | Tapo `/stream1`, high-resolution profile |
| Overlay metadata | `ws://localhost:5003` | WebSocket JSON | Pipeline `VisionState` |

Important: `living_room_cam` was intentionally removed to avoid ambiguity. Do not reintroduce it unless the user explicitly asks.

## Key Files

- `go2rtc/go2rtc.yaml`
  - Defines only:
    - `living_room_sd -> rtsp://${TAPO_USER}:${TAPO_PASS}@${TAPO_IP}:554/stream2`
    - `living_room_hd -> rtsp://${TAPO_USER}:${TAPO_PASS}@${TAPO_IP}:554/stream1`
  - WebRTC config:
    - `listen: ":8555"`
    - `candidates: ["${GO2RTC_WEBRTC_CANDIDATE}"]`

- `.env`
  - Contains local camera credentials and go2rtc settings.
  - Current WebRTC candidate:
    - `GO2RTC_WEBRTC_CANDIDATE=100.90.235.67:8555`
  - Do not expose or copy the camera password into docs or logs.

- `config.py`
  - `PipelineConfig.rtsp_url` resolves to:
    - `rtsp://localhost:8554/living_room_sd`
  - `PipelineConfig.go2rtc_dashboard_url` resolves to:
    - `http://localhost:1984/stream.html?src=living_room_hd&mode=webrtc`
  - `preflight()` checks go2rtc RTSP `8554` and API `1984` before starting the capture loop.

- `capture/frame_producer.py`
  - OpenCV/FFmpeg reads `cfg.rtsp_url`.
  - Reconnect backoff is interruptible through a `threading.Event`, so shutdown no longer waits for a 30s sleep.

- `main.py`
  - Starts `FrameProducer` and `VisionWSServer`.
  - Shutdown is idempotent.
  - `VisionWSServer` sends JSON overlay metadata only; it does not send video.

- `streaming/vision_ws_server.py`
  - Broadcasts `VisionState` JSON at `WS_VISION_FPS`.

- `tests/test_go2rtc_config.py`
  - Enforces that go2rtc only has `living_room_sd` and `living_room_hd`.
  - Enforces WebRTC candidate config uses `${GO2RTC_WEBRTC_CANDIDATE}`.

## Start Commands

If already inside `/Users/ogulcanozdemir/video_Process/go2rtc`:

```bash
docker compose up -d
docker compose ps
docker compose logs -f
```

In another terminal, start the pipeline:

```bash
cd /Users/ogulcanozdemir/video_Process
uv run python main.py
```

Dashboard HD WebRTC URL:

```text
http://localhost:1984/stream.html?src=living_room_hd&mode=webrtc
```

Close older go2rtc tabs opened without `mode=webrtc`; those can leave `mse/fmp4` consumers in the API output.

## Verification Commands

From repo root:

```bash
docker compose -f go2rtc/docker-compose.yaml ps
curl http://localhost:1984/api/streams
lsof -nP -iTCP:8554 -sTCP:LISTEN
```

Redacted stream API from inside the container:

```bash
docker compose -f go2rtc/docker-compose.yaml exec -T go2rtc sh -c \
  "wget -qO- http://127.0.0.1:1984/api/streams | sed -E 's#(rtsp://)[^:@/]+:[^@/]+@#\1***:***@#g'"
```

Expected when pipeline and dashboard are running:

- `living_room_hd` consumer:
  - `format_name: webrtc`
  - `protocol: ws+udp` or `ws+tcp`
- `living_room_sd` consumer:
  - `format_name: rtsp`
  - `protocol: rtsp+tcp`
  - `user_agent: Lavf...` from OpenCV/FFmpeg

If `living_room_hd` shows `format_name: mse/fmp4`, the dashboard is not using the required WebRTC path. Close the tab and reopen exactly:

```text
http://localhost:1984/stream.html?src=living_room_hd&mode=webrtc
```

## Test Status

Last full suite run:

```bash
uv run pytest
```

Result:

```text
25 passed
```

Known warnings:

- `websockets.server.WebSocketServerProtocol` deprecation warnings.
- MediaPipe `inference_feedback_manager` warnings at runtime. These are benign and not related to RTSP/WebRTC.

## Known Failure Modes

- `DESCRIBE failed: 401 Unauthorized`
  - Camera rejected credentials.
  - Use the Tapo local RTSP/ONVIF camera account, not the cloud account.

- `DESCRIBE failed: 404 Not Found`
  - Usually wrong go2rtc stream name or upstream auth failure surfaced through go2rtc.
  - Current valid names are only `living_room_sd` and `living_room_hd`.

- `Connection refused` on `localhost:8554` or `localhost:1984`
  - go2rtc is not running or ports are not published.
  - Start with `docker compose -f go2rtc/docker-compose.yaml up -d`.

- HD dashboard shows `mse/fmp4`
  - Wrong dashboard URL or WebRTC ICE candidate issue.
  - Current candidate is `100.90.235.67:8555`.
  - Confirm port `8555/tcp` and `8555/udp` are published.

## Worktree Note

The worktree is intentionally dirty from the ongoing implementation. Do not reset or revert unrelated files unless the user explicitly asks. Current notable areas include:

- go2rtc integration under `go2rtc/`
- WebSocket overlay state under `streaming/vision_state.py` and `streaming/vision_ws_server.py`
- go2rtc/config tests under `tests/`
- removal of old `streaming/mjpeg_server.py`
- dependency updates in `pyproject.toml`, `requirements.txt`, and `uv.lock`

## Suggested Next Work

- Build or connect the real dashboard UI so it embeds `living_room_hd` via WebRTC and overlays `ws://localhost:5003` metadata.
- Add a small dashboard health panel that reports:
  - HD transport is WebRTC, not MSE.
  - pipeline is connected to `living_room_sd`.
  - latest overlay timestamp/fps.
- Consider using `FrameProducer.capture_hires()` for fall screenshots. It exists, but confirmed fall screenshots currently come from the processed frame path.
