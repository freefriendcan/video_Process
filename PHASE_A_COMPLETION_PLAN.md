# Phase A — Completion Implementation Plan (for the Codex agent)

> Repo: `/Users/ogulcanozdemir/video_Process`. macOS / Apple Silicon, ONNX Runtime + CoreML EP, `uv run` for everything.
> The Phase A core is already implemented and reviewed as plan-faithful. This document lists the **remaining changes**, in order, each with concrete files + acceptance criteria. Be surgical — every edit must trace to a step. After each step run `uv run pytest`, `uv run ruff check <touched>`, `uv run mypy <touched>` and keep them green (baseline 35 tests).

## Binding constraints (do not violate)
- **C1** CV stays on 720p `/stream2`; no 2K. **C2** don't touch `go2rtc/`. **C3** MediaPipe (Pose/Gesture) stays `Delegate.CPU`. **C4** keep MediaPipe-Pose + TFLite fall stack and all fall constants exactly. **C5** `/vision/gesture` and `/vision/update_presence` HTTP payloads stay byte-for-byte unchanged — located identity goes over the `VisionState` WebSocket, NOT these.

## Already applied (verify only)
- **Person detector = YOLO11m**: `data/models/yolo11m-person.onnx` `(1,84,8400)`; `config.py` `person_model` points to it. `data/models/yolov11n-face.onnx` `(1,20,8400)` is the face model. *Verify:* both load under `onnxruntime` with CoreML EP.

---

## STEP 1 — Mac webcam enrollment-capture script
**Goal:** capture a user's face photos from the **Mac webcam** into `data/enroll/<name>/`, ready for `scripts/enroll_faces.py`. (Dashboard/Pi capture is a later cross-repo step; this local script comes first.)

**New file `scripts/capture_enrollment.py`:**
- Open the Mac webcam: `cv2.VideoCapture(0)` (AVFoundation). Add `--camera-index` (default 0).
- **Reuse** `FaceDetector` (`detection/face_detector.py`) and `FaceQualityGate` (`quality/face_quality_gate.py`) so only frames with a single, good-quality, roughly-frontal face are saved (mirrors the live identify gate; prevents enrolling blurry/no-face shots).
- Capture **N good frames** (`--count`, default 10) with a short spacing (`--interval`, default ~0.5 s) or on keypress; encourage slight pose/expression variation between shots (print prompts). Save JPEGs to `data/enroll/<name>/<name>_NN.jpg` (`--name` required).
- Optional `--enroll` flag: after capture, invoke the enrollment build (call `scripts/enroll_faces.py`'s `main()` or `subprocess`), rebuilding `faces.pkl`.
- Headless-friendly: if no display, skip the preview window (`--no-preview`); still save frames.
- Repo-root import shim like `scripts/enroll_faces.py` (`sys.path.insert(0, parent.parent)`).

**Acceptance:** `uv run python scripts/capture_enrollment.py --name testuser --count 5` saves 5 face-containing JPEGs under `data/enroll/testuser/`; frames with no/low-quality face are skipped with a logged reason; `--enroll` chains into a rebuilt `faces.pkl`.

---

## STEP 2 — Skip identification in IR / night mode
**Goal:** never compute an RGB ArcFace embedding on IR frames (buffalo_l is RGB-trained; day/color-only per decision).

**`main.py`** (new-face identify block, ~`main.py:193-211`): wrap the identify dispatch in `if not pkt.ir_mode:`. In IR, set the track to `"Unknown"` (or leave `"Identifying…"`) and do **not** submit `identifier.identify`. Do **not** loosen the quality gate for this — it already *relaxes* brightness in IR (`face_quality_gate.py:51-52`), which is the opposite of what recognition needs.

**Acceptance:** with `pkt.ir_mode=True` no `identifier.identify` is dispatched (assert via a unit test that drives the identify-gating with a stubbed IR packet, or a logged counter); daytime path unchanged.

---

## STEP 3 — Appearance-aware person tracker (Phase-C tracker, now in Phase A)
**Goal:** replace the KCF + IoU person tracking with a proper **multi-object tracker with ReID** so person `track_id`s stay stable across distance, occlusion, and crossings — the foundation for identity-follows-body (Step 4).

**Approach:** keep the ONNX/CoreML **person detector**; feed its detections into a standalone MOT. Use **`boxmot`** (BoT-SORT with ReID; OC-SORT/ByteTrack also available). New dependency: add `boxmot` to `pyproject.toml` + `requirements.txt`. *(boxmot pulls a small OSNet ReID checkpoint on first use; runs on CPU/MPS on Apple Silicon.)*

**Changes:**
- `detection/person_detector.py`: `detect()` returns detections **with confidence** (e.g., `list[tuple[BBox, float]]` or an `(N,5)` xywh+conf array) instead of bbox-only, so the tracker gets scores.
- `detection/onnx_runtime.py`: no change (already yields `YoloDetection.score`).
- `tracking/tracker_manager.py`: replace the `_person_trackers` KCF machinery (`_create/_reinit/_update_existing/_evict_*`, `advance_person_tracks`) with a single `boxmot` tracker instance:
  - `update_person_tracks(person_dets, frame, current_time)`: convert dets to the tracker's `xyxy,conf,cls` format, call `tracker.update(dets, frame)`, map returned tracks → per-track records `{id: <stable track_id>, bbox: xywh, user: "Unknown", reid_ok: bool, last_seen}`. Keep a `user` field for Step 4.
  - Detection cadence: keep `person_detection_interval`; on **non-detection** frames call `tracker.update(np.empty((0,6)), frame)` so the Kalman motion model coasts the tracks (replaces KCF carry / `advance_person_tracks`).
  - `active_person_tracks` / `person_snapshot`: include `user`.
  - Config: add `tracker_type` (default `"botsort"`), `reid_device` (`"mps"`/`"cpu"`), `reid_half` (False), and any track-buffer params, in the existing config style.
- `main.py`: pass detections (with conf) into `update_person_tracks`; remove the `advance_person_tracks` branch (the MOT coasts internally — still gate the **detector** by `person_detection_interval`, but call the tracker every frame).
- `fall_detector.process_track(track_id=...)` is unaffected (still keyed by the now-stable int track IDs); `fall_det.sync_tracks(active_person_ids)` keeps pruning to live IDs.

**Notes for the agent:** pin the `boxmot` version and confirm the exact class name/`update()` signature (it has varied across releases). If ReID setup is problematic, ship **ByteTrack** (motion-only, no ReID weights) as an interim and leave a TODO to enable BoT-SORT+ReID — but the target is appearance-aware.

**Acceptance:** a person walking toward/away and briefly occluded keeps the **same** `track_id`; two people crossing recover their IDs better than the old KCF (observe/log); `uv run python main.py` reaches the loop with the new tracker; tests green.

---

## STEP 4 — Identity-follows-body (recognize close → track at distance)
**Goal:** bind the close-range face identity to the persistent person/body track so identity survives beyond face-recognition range (~1.5 m).

**Changes:**
- `tracking/tracker_manager.py`:
  - person track records already carry `user` (Step 3, default `"Unknown"`).
  - `propagate_identity()`: for each **identified** face track (`_trackers[*]["user"]` not in `{"Unknown","Identifying…"}`), find the person track whose bbox **contains the face-bbox center** (tie-break by max containment ratio) and stamp `user` onto it. **Latest close recognition wins** (refresh on re-approach).
  - clear a person track's `user` when its track ends (sync to live IDs, like `fall_det.sync_tracks`).
- `main.py`: after `update_person_tracks` + face identify each loop, call `tracker_mgr.propagate_identity(...)`; build a `persons` list for `VisionState`.
- `streaming/vision_state.py`: add a frozen `TrackedPerson{id:int, user:str, nx,ny,nw,nh:float}` dataclass and a `persons: Sequence[TrackedPerson]` field on `VisionState`, serialized in `to_dict()` (additive — overlay/agent gains located identity; HTTP contracts untouched per C5).

**Acceptance:** approach camera (<1.5 m) → face identified → the **person track** now reports that `user`; walk to 5–8 m (face undetectable) → identity **persists** on the body track in `VisionState.persons`; leave + return → re-identified; ID-switch on crossing is mitigated by Step 3's ReID.

---

## STEP 5 — Threshold calibration helper (Tapo domain)
**Goal:** calibrate `face_match_cosine_threshold` (`config.py`, currently 0.35) on **real Tapo crops**, since webcam→Tapo genuine cosines run lower than webcam→webcam.

**New file `scripts/calibrate_threshold.py`:** reuse `FaceIdentifier.for_enrollment(cfg)` + `embed_face` + the gallery loader. Input: a folder of labeled **Tapo** face crops (enrolled people) + some strangers. Compute each crop's best-gallery cosine; print genuine vs impostor distributions (min/mean/percentiles) and recommend a threshold between the modes (expect ~0.3–0.45). Do not auto-edit config — print the suggested value for the user to set with rationale.

**Acceptance:** running it on a labeled Tapo set prints genuine/impostor cosine stats and a recommended threshold; genuine clearly separable from impostor.

---

## Out of scope / coordination (do NOT do now)
- Dashboard→Mac enrollment hand-off over the network (Pi cross-repo) — Step 1 local capture comes first.
- Pushing located identity to `proactive-home-agent` over HTTP — stays on the `VisionState` WebSocket (C5).
- Fall pose-model migration (C4 frozen); any presence/gesture payload change (C5).

## End-to-end verification
1. `uv run python scripts/capture_enrollment.py --name <you> --count 10 --enroll` → `faces.pkl` built from webcam.
2. Capture Tapo daylight crops → `uv run python scripts/calibrate_threshold.py` → set `face_match_cosine_threshold`.
3. `cd go2rtc && docker compose up -d` then `uv run python main.py`:
   - daylight, approach <1.5 m → `Identity verified locally [..]: <name>`; person track gains the name.
   - walk to distance → identity persists in `VisionState.persons`; person `track_id` stable.
   - IR/night → no identify attempts.
   - stranger → `Unknown`; fall/gesture paths unchanged; MediaPipe on CPU, YOLO/ArcFace on CoreML EP.
4. `uv run pytest` green (add tests: IR-skip gate, `propagate_identity` containment, `VisionState.persons` serialization, person-detector conf output).
