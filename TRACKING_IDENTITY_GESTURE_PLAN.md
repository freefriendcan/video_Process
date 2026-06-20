# Post-Phase-A Upgrades — Face ByteTrack + Pi Identity Events + Temporal Gesture (Handoff)

> Repo: `/Users/ogulcanozdemir/video_Process`. macOS / Apple Silicon, `uv run` for everything.
> Phase A is complete and green. This handoff covers **three independent workstreams** (WS1/WS2/WS3); each can ship
> on its own. Be surgical — every edit must trace to a step. After each step run `uv run pytest`,
> `uv run ruff check <touched>`, `uv run mypy <touched>` and keep them green (baseline 38 tests).

## Binding constraints (do not violate)
- **C1** CV stays on 720p `/stream2`; no 2K. **C2** don't touch `go2rtc/`. **C3** MediaPipe (Pose/Gesture/Hands) stays
  `Delegate.CPU`. **C4** fall stack (MediaPipe Pose + TFLite transformer + all fall constants) frozen.
- **C5** `/vision/gesture` and `/vision/update_presence` HTTP bodies stay byte-for-byte. Located identity does **NOT**
  reuse those endpoints — WS2 introduces a **separate** event channel, default-off (no Pi dependency until coordinated).
- No new heavy deps for WS1/WS2 (`boxmot` already present). WS3 Stage 2 (ST-GCN) is the only place that may add a model.

---

# WS1 — Face tracking: KCF → ByteTrack

## Why
Faces currently use **per-face `cv2.TrackerKCF` + IoU re-id** (`tracking/tracker_manager.py`: `update_all`,
`match_and_update`, `_new_kcf_tracker`). KCF drifts, throws "ghost boxes" (the 1 s TTL hack at `tracker_manager.py:141`),
and re-assigns IDs unstably across turn-away/occlusion — which churns identity binding and the identity→person
propagation. **ByteTrack** (Kalman motion + two-stage IoU association) gives stable IDs through brief occlusion at
lower CPU for multiple faces. **Use ByteTrack, not BoT-SORT, for faces:** BoT-SORT's OSNet ReID is trained on
full-body crops and is meaningless/harmful on faces, and **ArcFace already provides face appearance/identity**.
`boxmot.ByteTrack` is already wired for the person path (`tracker_manager.py:82-95`) — reuse the same idiom with a
**second tracker instance** for the face namespace.

## Design (mirror the existing person path)
1. **Face detector exposes confidence.** `detection/face_detector.py` currently returns `(x, y, w, h, keypoints)` and
   discards the ONNX `YoloDetection.score`. Change `detect()` to return the score (e.g. `(x, y, w, h, score, keypoints)`
   or `(bbox, score, keypoints)`) — ByteTrack needs per-detection scores for its high/low-conf split. **Update every
   unpack site** (see impact list). *(This mirrors the Phase-A person-detector change that added conf.)*
2. **`tracking/tracker_manager.py`:**
   - Add `self._face_mot = ByteTrack(...)` built from a new face-tracker config block (below). Keep `_trackers:
     dict[int, dict]` for per-track metadata (`user`, `retry_count`, `last_identify_time`, `last_json_time`,
     `best_quality_score`, `detection_keypoints`, `bbox`) **but key it by the ByteTrack track id** (drop `_next_id`).
   - **Remove** KCF machinery: `_new_kcf_tracker`, the `_CvTracker` Protocol, the KCF loop in `update_all`, and the
     KCF reinit/new-tracker logic in `match_and_update`. (Remove only this; leave unrelated code.)
   - **New method `update_face_tracks(faces, frame, current_time) -> (new_faces, active)`** (analogous to
     `update_person_tracks`):
     - Convert `faces` to boxmot dets `(x1,y1,x2,y2,conf,cls=0)`; empty `(0,6)` array on non-detection frames so the
       Kalman state **coasts** (replaces the per-frame KCF visual update).
     - `results = self._face_mot.update(dets, frame)`; for each returned track row map → `_trackers[track_id]`,
       **preserving** existing `user`/`retry_count`/`last_identify_time` across updates (sticky, like persons).
     - **Keypoint attachment (critical):** ByteTrack returns boxes only. Use boxmot's source-detection-index output
       column (confirm it — the person path reads `row[:8]`; the det index is typically `row[7]`) to copy the matched
       detection's `keypoints` onto the track record; **fall back to highest-IoU** detection if the index column is
       absent. Keypoints stay attached so the quality gate / ArcFace alignment / gesture frontality keep working.
       Between detections keypoints are stale — same as KCF today.
     - A track id not previously in `_trackers` → init (`user="Identifying..."`, `retry_count=0`) and append to
       `new_faces` (`{tracker_id, bbox, keypoints}` — **unchanged shape** for `_identify_new_faces`).
     - `active` = list with the same keys `update_all` returned (`id, bbox, user, detection_keypoints,
       last_json_time, retry_count, last_identify_time, best_quality_score`) — so the main-loop frontality/presence
       block is untouched.
   - **Eviction:** mirror `_evict_stale_person_tracks` for faces (TTL from `face_track_buffer / face_tracker_frame_rate`),
     removing `_trackers` entries whose track id stopped being returned. Track loss → identity re-fires on the next new
     track (preserves Phase-A "identify-once-then-track").
   - `set_user`, `update_field`, `propagate_identity`, `get_active_user`, `snapshot`, `exists` operate on `_trackers`
     by id — **unchanged**.
3. **`main.py` camera loop:** promote the hardcoded `0.15` (main.py:186) to `cfg.face_detection_interval`. Each frame:
   detect at cadence → `new_faces, active = tracker_mgr.update_face_tracks(faces_or_empty, frame, current_time)` →
   feed `active` to the existing frontality/presence loop (138-153) and `new_faces` to `_identify_new_faces`. Remove
   the separate `update_all` call. (ByteTrack runs every frame; the **detector** stays interval-gated.)

## Config (`config.py`, mirror the person block)
```python
face_detection_interval: float = 0.15     # was hardcoded in main.py
face_tracker_type: str = "bytetrack"
face_track_thresh: float = 0.5            # reuse face_conf_threshold semantics (high-conf)
face_track_low_conf: float = 0.1
face_match_thresh: float = 0.8
face_track_buffer: int = 30
face_tracker_frame_rate: int = 30
```

## Impact analysis — systems affected by the face KCF→ByteTrack swap
1. **`tracking/tracker_manager.py`** — core rewrite of the face path (above). Highest blast radius.
2. **`detection/face_detector.py`** — `detect()` now returns score; **all unpack sites** must update:
   `update_face_tracks` (new), `scripts/calibrate_threshold.py:_largest_face` + its `x,y,w,h,keypoints = face` unpack
   (calibrate_threshold.py:146-151), and `main.py:_identify_new_faces` (`x,y,w,h,keypoints = ...`). Trace every
   `FaceDetection` consumer.
3. **`main.py` camera loop ordering** — face MOT moves to run every frame (cadence-gated detector). The
   frontality/presence loop and `_identify_new_faces` contracts stay the same (shapes unchanged).
4. **Identity binding / identify-once (Phase-A A4)** — depends on stable ids + "new track" detection. ByteTrack ids are
   **more** stable → fewer re-identify re-fires; an id switch on crossing correctly triggers re-identification.
   `retry_count`/`last_identify_time` preserved per track.
5. **Keypoints → quality gate + ArcFace alignment** — must stay attached to tracks via the det-index/IoU mapping. If
   this breaks, frontality (gesture gaze-lock) and ArcFace alignment degrade. **Add a test.**
6. **Gesture gaze-lock** (`gesture_rec.is_frontal`, main.py:149) — fed from face-track frontality; works as long as
   keypoints stay attached.
7. **Presence heartbeat** (per-track `last_json_time`, main.py:151) — keyed per face track; ids differ from KCF
   (cosmetic).
8. **`propagate_identity`** (face→person) — operates on `_trackers` `user`+`bbox`; structurally unaffected, and
   **improved** by steadier face boxes. Feeds WS2.
9. **VisionState `faces` / overlay** — `TrackedFace.id` becomes the ByteTrack id; `_build_faces` uses `snapshot`
   (bbox+user). Overlay `F{id}` id-space changes only.
10. **Thread-safety** — `_face_mot` is mutated only on the camera thread; `_trackers` reads/writes stay under
    `self._lock`. Identify runs on the pool but only via locked `set_user`/`exists`. Keep it that way.
11. **Performance** — drops per-face KCF visual updates (O(faces) CPU) for one Kalman update/frame, but ByteTrack
    quality scales with detection frequency. **Verify** at the current 0.15 s cadence; if association is loose, lower
    `face_detection_interval` (YOLOv11n-face on CoreML EP is cheap).
12. **Birth latency** — ByteTrack may need ≥1 confirmation hit before a track is "active," so identification can fire
    ~1 detection later. Upside: transient 1-frame face noise no longer spawns wasted ArcFace identify calls.
13. **Tests** — `tests/test_phase_a_contracts.py` asserts KCF/`match_and_update`/`update_all` behavior; replace with
    `update_face_tracks` contract tests (new_faces shape, active shape, sticky identity across updates, keypoint
    attachment, new-id-on-birth, stale eviction). Remove KCF-specific assertions.
14. **Dead code cleanup** — only the KCF symbols listed above; nothing else.

## WS1 acceptance
- A face turning away briefly / walking through partial occlusion keeps the **same** track id and identity (vs KCF
  flicker). No "ghost box" TTL needed. Frontality + identify-once still work; `uv run python main.py` reaches the loop;
  tests green.

---

# WS2 — Located identity → Pi events (session start + end; never Unknown)

## Why / rule
Located identity (the person track's `user`) currently goes **only over the VisionState WebSocket** (:5003), never to
the Pi (C5). Per decision, when Pi integration lands we want, **without identity flapping**:
- **Session start:** the first time a person track is confidently identified (Unknown → known name) → emit
  `person_identified`.
- **Session end:** when that person track is **reset** (evicted/lost in `_evict_stale_person_tracks`) → emit
  `person_left` carrying the last known identity + dwell time.
- **Never** emit on `name → Unknown` (face out of range / turned away). Identity is a property of the **body track**,
  which already stays sticky (`_apply_person_results` preserves `user`; `propagate_identity` only stamps known users).

## Design (additive, default-off — no Pi dependency yet)
- `tracking/tracker_manager.py`: track per-person session state (the last **emitted** identity per track id). In
  `propagate_identity`, when a track's user transitions `Unknown/None → <known>` (and not yet emitted), enqueue a
  **session-start** event. In `_evict_stale_person_tracks`, if an evicted track had an emitted identity, enqueue a
  **session-end** event with `dwell_s = last_seen - first_identified_at`. Emit each `(track_id, start)` once;
  `name A → name B` on the same track (rare with ByteTrack/ReID) emits a fresh `person_identified` for B.
- **Transport is a callback, not hardwired HTTP.** Give `TrackerManager` an optional `on_identity_event(event: dict)`
  hook; `main.py` wires it to `EventDispatcher`. Add `EventDispatcher.send_identity_event(payload)` posting to a
  **new** `POST /vision/identity_event` endpoint — **gated by `cfg.identity_events_enabled` (default `False`)**. While
  off: log only (and it already shows over VisionState). This keeps WS2 entirely inside this repo with **zero behavior
  change** until the Pi side adds the endpoint; when Phase B lands, route it through the durable outbox.
- **C5:** this is a brand-new endpoint/payload — it does **not** touch `/vision/gesture` or `/vision/update_presence`.

## Event payload (new contract for `proactive-home-agent`)
```jsonc
{ "schema_version": "1.0", "event_type": "person_identified" | "person_left",
  "track_id": 7, "user": "ogulcan", "zone": "living_room", "source": "mac_studio_living_room",
  "ts_wall": "2026-...Z", "dwell_s": 42.0 /* person_left only */ }
```

## Config
```python
identity_events_enabled: bool = False
identity_event_url property -> f"http://{pi_ip}:{pi_port}/vision/identity_event"
```

## WS2 acceptance
- Approach < 1.5 m → identified → exactly one `person_identified` (when enabled); turn away / walk to distance → **no**
  event (identity sticks, no Unknown emitted); leave frame (track evicted) → exactly one `person_left` with dwell.
  With `identity_events_enabled=False` (default) nothing is POSTed and the pipeline is byte-for-byte unchanged.

---

# WS3 — Gesture roadmap item 8 (attention crop + temporal)

## Stage 0 — Attention crop: ALREADY DONE (verify only)
`GestureRecognizer._hand_roi()` (gesture_recognizer.py:154-184) already crops to a **pose-wrist hand ROI** with
`hand_crop_pad` padding + `hand_crop_ema_alpha` EMA smoothing, and **falls back to the whole frame** when no
pose/wrist exists — so it is inherently non-breaking. **Model attribution:** the crop is geometry from **MediaPipe
Pose** wrists (supplied by the fall detector via `pose_tracks`); recognition is still the **MediaPipe Gesture
Recognizer** (`gesture_recognizer.task`). No work here beyond confirming wrists flow in (`main.py:183`).

## Stage 1 — Temporal voting (low-risk, do this)
Add a sliding-window vote over the last `N` recognizer results to harden against frame-level flicker (the current
>1 s sustain already suppresses isolated single-frame triggers; voting adds **dropout tolerance** so one bad frame
no longer resets a genuine sustained gesture, and gives a cleaner statistical gate).
- `detection/gesture_recognizer.py` `_on_result`: push each frame's top gesture into a fixed-size deque; a gesture is
  "active" only if it holds **≥ `gesture_vote_min` of the last `gesture_vote_window`** results. Feed that smoothed
  result into the existing `_current_sustained` / `>1 s` / `gesture_cooldown` logic — **do not replace** sustain/cooldown.
- Config: `gesture_vote_window: int = 7`, `gesture_vote_min: int = 4`.
- **C3** CPU unchanged; **C5** dispatched payload byte-for-byte unchanged. Fully additive.

## Stage 2 — ST-GCN dynamic gestures (PLANNED, optional, heavier — can be dropped)
Static per-frame classification can't recognize **dynamic** gestures (wave/swipe). That needs a sequence model:
- Run **MediaPipe HandLandmarker** (CPU, C3) on the same attention-cropped ROI to get 21-landmark sequences; buffer
  `T` frames; classify with a small **ST-GCN** over the landmark graph → dynamic gesture classes; apply the Stage-1
  temporal voting on top. Keep the existing static MediaPipe Gesture path **in parallel** (additive — nothing removed).
- Requires a model (pretrained landmark-gesture checkpoint or a trained one) + a dynamic-gesture dataset, and a
  decision on the gesture vocabulary. **This is Phase-C/D scope** — out of this handoff unless explicitly requested;
  listed here so the staging is clear.

## WS3 acceptance
- Stage 1: legitimate sustained gestures survive occasional dropped frames (fewer misses); isolated spurious frames
  never fire; gesture payload diff-clean vs baseline; MediaPipe still on CPU.

---

## Suggested order & out of scope
- **Order:** WS1 (foundation — steadier ids improve WS2 propagation) → WS2 → WS3 Stage 1. WS3 Stage 2 deferred.
- **Out of scope:** Phase B durable transport (on hold); fall model/threshold changes (C4); any `/vision/gesture` or
  `/vision/update_presence` payload change (C5); `go2rtc/` (C2); MediaPipe GPU (C3).

## End-to-end verification
```bash
cd go2rtc && docker compose up -d
cd /Users/ogulcanozdemir/video_Process
uv run pytest
uv run ruff check tracking/ detection/ events/ config.py main.py && uv run mypy tracking/ detection/ events/ config.py main.py
uv run python main.py
```
Then confirm WS1 id stability (turn-away/occlusion keeps id+identity), WS2 session-start/end semantics with
`identity_events_enabled` on a stub listener (no Unknown emits), and WS3 Stage-1 dropout tolerance + diff-clean
gesture payload.
