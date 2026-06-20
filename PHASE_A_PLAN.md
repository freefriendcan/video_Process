# Phase A — Production Foundation & Gated Baselines (Implementation Handoff)

> **Audience:** the implementing agent. This document is self-contained: it resolves every
> blocker raised in the previous handoff (missing models, gallery schema, tracking contract,
> yolov8n-face.pt fallback question) with concrete, locked decisions. Do not re-ask these.
> **Repo:** `/Users/ogulcanozdemir/video_Process`. **Compute:** Mac Studio M1 Ultra, 128 GB.
> Implement incrementally; after each step run `uv run pytest` and `uv run python main.py`.

---

## 0. Locked Decisions (these resolve the prior blockers)

| Concern | Decision |
|---|---|
| **Face detection** | Replace BlazeFace with **YOLOv11-face**. Acquire weights → export to **ONNX** → run via **ONNX Runtime + CoreML EP** on the 720p frame. Output: bbox + 5 face keypoints. |
| **Person detection** | **ultralytics `yolo11n.pt`** (COCO, class 0 = person) → export to **ONNX** → ONNX-RT + CoreML EP. |
| **Face identification** | Local **insightface `buffalo_l` ArcFace** (`w600k_r50`, 512-d) via ONNX-RT + CoreML EP. **Remove** the `/vision/identify` HTTP round-trip entirely. |
| **Enrollment gallery** | Use existing `data/embeddings/faces.pkl`. Confirmed schema: `dict[str_label -> np.ndarray (N, 512) float64]` (multiple embeddings per person). Cosine match. |
| **Tracking** | **Extend `tracking/tracker_manager.py`** to manage BOTH face tracks (existing) and person tracks (new) in one manager. |
| **`yolov8n-face.pt`** | **Not used.** Leave the file in place, dormant. YOLOv11-face ONNX is the Phase-A detector. (No silent fallback.) |
| **New dependency** | Add **`insightface`**. `onnxruntime` is already a dep (CoreML EP ships on Apple Silicon). **`coremltools` is NOT required** — we use the ONNX route, not `.mlpackage`. |
| **Runtime** | All NEW models (YOLOv11-face, yolo11n-person, ArcFace) → ONNX-RT with provider list `["CoreMLExecutionProvider", "CPUExecutionProvider"]`. |

---

## 1. Binding Global Constraints (do not violate)

- **C1 — 720p only.** All CV runs on the 720p sub-stream `/stream2` (`living_room_sd`). No 2K ingestion/inference. No full-res 2K crops.
- **C2 — go2rtc unchanged.** `go2rtc/go2rtc.yaml`, ports, stream names stay as-is. 720p = CV, 2K = dashboard only.
- **C3 — MediaPipe stays `Delegate.CPU`.** Do NOT switch MediaPipe Pose/Gesture/Hands to GPU. (GPU delegate degrades/breaks on this arch.)
- **C4 — Fall model stack frozen.** Keep MediaPipe Pose + the existing TFLite fall transformer and its 51-dim hip-centered feature format. No pose-model migration. *(The transformer is trained strictly on MediaPipe-Pose features; swapping pose mandates dataset conversion + retraining + recalibration — that is Phase C.)*
- **C5 — Gesture telemetry frozen.** The `/vision/gesture` payload stays byte-for-byte baseline: `{gesture, user, location:"living_room", timestamp, duration}`. No `track_id`, no coordinates, no new fields. `/vision/update_presence` also unchanged.

---

## 2. Artifact Acquisition (do this first; the previously-missing models)

Place all model files under `data/models/`. Add their paths to `config.py` (see §3).

1. **Person detector — `data/models/yolo11n-person.onnx`**
   ```bash
   uv run python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', imgsz=640, dynamic=False, simplify=True)"
   # ultralytics auto-downloads yolo11n.pt (COCO). Move the produced yolo11n.onnx -> data/models/yolo11n-person.onnx
   ```
   At inference, filter to class 0 (person).

2. **Face detector — `data/models/yolov11n-face.onnx`**
   - Acquire a **YOLOv11-face** checkpoint (community `yolo-face` project provides YOLOv11 face weights with 5 landmarks — **confirm the source/commit before downloading; do not invent a URL**).
   - Export: `YOLO('<yolov11-face>.pt').export(format='onnx', imgsz=640, simplify=True)` → move to `data/models/yolov11n-face.onnx`.
   - **Verify** the ONNX output tensor includes 5 keypoints per face (needed for ArcFace alignment + the quality-gate frontality check). If the chosen checkpoint lacks landmarks, stop and report — alignment depends on them.

3. **ArcFace — insightface `buffalo_l`**
   ```bash
   uv add insightface          # add to deps
   uv run python -c "import insightface; insightface.app.FaceAnalysis(name='buffalo_l').prepare(ctx_id=0)"
   # downloads buffalo_l models into ~/.insightface/models/buffalo_l (incl. recognition w600k_r50, 512-d)
   ```
   We use only the **recognition** model (`w600k_r50`, 512-d) for embeddings; detection comes from YOLOv11-face. Confirm embedding dim == 512 to match `faces.pkl`.

**Acceptance for §2:** all three artifacts load; a smoke test prints YOLO person boxes, YOLO face boxes+kps, and a 512-d ArcFace embedding on a sample 720p frame.

---

## 3. Config Additions (`config.py`)

Add to `PipelineConfig` (keep existing fields):
```python
# New detector models (ONNX, CoreML EP)
person_model: str = "data/models/yolo11n-person.onnx"
face_det_model: str = "data/models/yolov11n-face.onnx"
onnx_providers: tuple = ("CoreMLExecutionProvider", "CPUExecutionProvider")

# Detection thresholds
person_conf_threshold: float = 0.5
face_conf_threshold: float = 0.5
det_nms_iou: float = 0.45

# Local ArcFace identification
arcface_model_name: str = "buffalo_l"
gallery_path: str = "data/embeddings/faces.pkl"
face_match_cosine_threshold: float = 0.35   # TUNABLE — calibrate against faces.pkl in §8

# Person crop padding for pose (fraction of bbox)
person_crop_pad: float = 0.15

# Gesture hand-ROI crop
hand_crop_pad: float = 0.6      # generous pad around wrist
hand_crop_ema_alpha: float = 0.5  # EMA smoothing on crop coords
```
Keep ALL existing fall constants unchanged (`fall_input_timesteps=30`, `min_fall_velocity=0.025`, `fall_confidence_threshold=0.90`, `post_fall_wait=3.0`, `post_fall_move_threshold=0.015`, `fall_alert_cooldown=10`). Note: `identify_url` becomes unused — leave the property but stop calling it.

---

## 4. Component A1 — Shared ONNX inference helper

New module `detection/onnx_runtime.py`:
- Thin wrapper: `class OnnxModel` loading a `.onnx` with `onnxruntime.InferenceSession(path, providers=cfg.onnx_providers)`.
- Methods to preprocess a 720p RGB frame (letterbox→640, normalize) and postprocess YOLO outputs (NMS, scale boxes back to 720p coords).
- Reused by both the person detector and the face detector.

**Verify:** unit test that a session is created with CoreML EP listed in `session.get_providers()`.

---

## 5. Component A2 — Person detection → gated, cropped, per-track fall

**Goal:** *Person → (crop) → MediaPipe Pose → TFLite fall*, never on empty rooms; per-track FSM.

1. **`detection/person_detector.py`** — YOLOv11 person (ONNX) on the 720p frame; returns list of person bboxes (class 0, conf ≥ `person_conf_threshold`).
2. **Extend `tracking/tracker_manager.py`** (see §9 contract): add a `person` track namespace producing stable `person_track_id`s (KCF + IoU like the face path, or ByteTrack-lite — KCF reuse is acceptable for Phase A).
3. **Refactor `detection/fall_detector.py` to be PER-TRACK:**
   - Move the single global FSM state into a `dict[person_track_id -> FallTrackState]`.
   - For each active person track: crop the 720p frame to its bbox (padded by `person_crop_pad`), run **MediaPipe Pose (CPU)** on the crop, extract the **same 51-dim hip-centered features** (unchanged), feed the **same 30-frame window** + **same TFLite transformer**.
   - **Preserve exactly:** 30-frame window, `min_fall_velocity=0.025` velocity gate, in-classifier confidence threshold (`0.90`), 3 s post-fall inactivity FSM, 10 s alert cooldown — per track.
   - `fall_state ∈ {idle, falling, on_floor, recovered}` per track.
4. **`main.py` change (`main.py:169`):** replace the unconditional `fall_det.process_frame(...)` with: detect persons → update person tracks → for each person track run the gated per-track fall step. Emit a fall event (per §7) when a track transitions to a confirmed fall.

**Verify:** empty room → MediaPipe Pose NOT invoked (assert via log/counter). One person → Pose runs on the crop; scripted fall clip → exactly one `fall` event with the right `fall_state` transitions. Two simulated person tracks maintain independent FSMs.

---

## 6. Component A3 — Face detection swap (BlazeFace → YOLOv11-face ONNX)

Rewrite `detection/face_detector.py` internals, **preserving the output contract** that `tracker_manager.match_and_update` consumes: a list of `(x, y, w, h, keypoints)` where `keypoints` carries the eye points used by `quality.face_quality_gate.estimate_frontality`.
- Run YOLOv11-face (ONNX) on `pkt.rgb` (720p); map the 5 landmarks into the existing `keypoints` shape the quality gate expects (right eye / left eye at minimum).
- Keep the existing 0.15 s detection cadence and the IoU re-id in `tracker_manager`.

**Verify:** faces detected on 720p; frontality gate still functions; FPS ≥ baseline.

---

## 7. Component A4 — Local ArcFace identification (remove Pi round-trip)

Rewrite `identification/face_identifier.py`:
- On init: load `insightface` recognition model (`buffalo_l`) and the gallery from `cfg.gallery_path` into `dict[label -> (N,512) np.float32]` (cast from float64; L2-normalize each row once).
- `identify(face_roi, keypoints, tracker_id)`:
  1. Align the crop using the 5 keypoints (similarity transform to 112×112) — standard ArcFace alignment.
  2. Compute 512-d embedding; L2-normalize.
  3. Cosine-match vs every gallery embedding; take the best label. If best cosine ≥ `face_match_cosine_threshold` → `set_user(tracker_id, label, retry_count=0)`; else `set_user(tracker_id, "Unknown")`.
- **No HTTP.** Delete the `requests.post(self._cfg.identify_url, ...)` path.
- **Identify-once-then-track (constraint):** identify ONCE on face-track birth (quality-gated, as today) and bind identity to the track ID. **Re-identify only on track loss / new track**, not on the backoff timer. Update `main.py:138-161`: remove the `2·2^retry` re-fire loop; keep the initial gated identify on new tracks (`main.py:179-191`).
- Keep `FaceQualityGate.check(...)` gating before identify (size/blur/brightness/eye-distance, IR-aware) unchanged.

**Verify:** known face → correct label with NO Pi call (assert no outbound to `/vision/identify`); unknown face → "Unknown"; identity stays stable across a 60 s walk-around (no re-fire spam in logs); end-to-end identify latency < ~50 ms.

---

## 8. Component A5 — Action-oriented fall event schema (no top-level confidence)

Update `events/dispatcher.py` `send_fall_alert(...)` to emit (POST `/vision/fall_alert`, keep HTTP transport):
```jsonc
{
  "schema_version": "1.0",
  "event_type": "fall",
  "track_id": 7,                       // person track id
  "fall_state": "on_floor",            // falling | on_floor | recovered
  "bbox": [x, y, w, h],                // 720p pixel coords
  "centroid": [cx, cy],
  "frame_size": [1280, 720],
  "diagnostics": {                     // internal posture/velocity telemetry — NOT a top-level score
    "body_velocity": 0.041,
    "torso_angle_deg": 78.0,
    "inactivity_elapsed_s": 3.0
  },
  "ts_wall": "2026-06-17T...Z",
  "ts_monotonic": 173...,
  "source": "mac_studio_living_room",
  "zone": "living_room",
  "media": { "snapshot_path": "data/logs/screenshots/fall_7.jpg" }  // if captured
}
```
- **CRITICAL:** NO top-level `confidence`. Keep the model probability only in internal logs / `diagnostics` if needed.
- Snapshot capture path/behavior unchanged (still from the processed frame; do not switch to `capture_hires` in Phase A).
- **Do NOT touch** `send_gesture_event` or `send_presence` (C5).

**Verify:** the agent parses `track_id`/`fall_state`/spatial fields; confidence is absent at top level; gesture/presence payloads diff-clean vs baseline.

---

## 9. Component A6 — Gesture attention crop (CPU, frame-level, telemetry frozen)

Update `detection/gesture_recognizer.py`:
- Source a **hand ROI from the person pose wrists** (now available from §5's gated pose) with `hand_crop_pad` padding; apply **EMA smoothing** (`hand_crop_ema_alpha`) on the crop coordinates to kill jitter.
- Run the **MediaPipe Gesture Recognizer on the cropped ROI only**, not the whole 720p frame.
- **C3:** keep `Delegate.CPU`. **Hard:** keep **frame-level** inference + the existing **sustain (1 s) / cooldown (1 s)** logic. **No** temporal voting / sequence accumulation.
- **C5:** the dispatched gesture event stays exactly baseline.
- Fallback: if no pose/wrist is available for a frame, retain current whole-frame behavior so gesture is never lost.

**Verify:** gestures recognized from the cropped ROI; gesture payload byte-identical to baseline; CPU load lower than whole-frame.

---

## 10. Tracking Contract (extended `TrackerManager`)

One manager, two track types:
- **Face tracks** (existing): keys unchanged (`tracker, user, bbox, retry_count, last_identify_time, detection_keypoints, ...`). Identity lives here.
- **Person tracks** (new): `{tracker, bbox, person_track_id}` in a separate dict / id-space; `fall_state` FSM keyed by `person_track_id`.
- **Linking identity ↔ person:** when a face bbox center is contained in a person bbox, propagate the face's identity label onto that person track (optional, for richer fall context). The fall event's `track_id` is the **person** track id. Gesture does **not** carry identity changes beyond the existing `get_active_user` behavior (C5).
- Keep `snapshot`/`update_all`/`match_and_update` working for faces; add parallel methods for persons (mirror the KCF+IoU pattern). Preserve thread-safety (`self._lock`).

---

## 11. Dependencies

- `requirements.txt` + `pyproject.toml`: **add `insightface`**.
- `onnxruntime` already pinned `>=1.16,<1.24` — CoreML EP available on Apple Silicon; no change.
- `ultralytics`, `torch` already present (used only for one-time ONNX export of YOLO weights).
- **Do NOT add `coremltools`** (ONNX route). `deepface` remains declared but is now unused by the live path (leave as-is; do not remove in Phase A).

---

## 12. Out of Scope (defer)

- Any 2K CV ingestion / 2K crops (C1).
- MediaPipe → GPU delegate (C3).
- Pose-model migration (RTMPose/RTMW/ViTPose), fall-transformer retraining (C4 → Phase C).
- Temporal voting for gestures; any change to gesture/presence telemetry (C5).
- MQTT/gRPC transport, unified envelope across event types, offline spooling/retry hardening (→ Phase B).
- Appearance-aware tracker (BoT-SORT/OC-SORT) — KCF+IoU is sufficient for Phase A (→ Phase C optional).

---

## 13. Verification (end-to-end)

```bash
cd go2rtc && docker compose up -d          # streams unchanged: living_room_sd (720p) + living_room_hd (2K)
cd /Users/ogulcanozdemir/video_Process
uv run pytest                               # baseline 25 pass + new tests (onnx helper, person gate, per-track FSM, fall schema, local id)
uv run python main.py
```
Then confirm:
- CV binds **`living_room_sd` (720p)** only; dashboard still uses `living_room_hd` (2K WebRTC) — two-stream design untouched.
- Empty room: no MediaPipe Pose calls; person present: pose runs on the crop.
- Known face identified locally (no `/vision/identify` traffic), identity stable, identify-once.
- Fall clip → action-oriented event with `track_id`/`fall_state`, **no top-level confidence**.
- Gesture event diff-clean vs baseline; MediaPipe still on CPU.
- `powermetrics` / Activity Monitor: YOLO + ArcFace on ANE/GPU (CoreML EP), MediaPipe on CPU.

**Calibration note:** `face_match_cosine_threshold` (default 0.35) must be tuned against `faces.pkl` — verify `ogulcan`/`OG` classify correctly and an out-of-gallery face returns "Unknown".
