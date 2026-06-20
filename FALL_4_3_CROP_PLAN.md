# Fall Detection — Restore 4:3 Pose Geometry (Strategy A center-crop) — Handoff Plan

> **Deliverable:** feed MediaPipe Pose a **4:3 center-crop** of the 1280×720 frame so the frozen
> TFLite transformer sees the body proportions it was trained/deployed on (legacy `mac_camera.py`
> ran 640×480 = 4:3). This corrects the input-distribution shift introduced by the Tapo C225's
> 16:9 stream — the cause of the near-constant 90–99% `fall_prob`. **No model, threshold, or
> velocity-gate change.**

## Why this is the fix (root cause, verified in code)

- The transformer is fed `landmark.x, landmark.y` from MediaPipe Pose, each normalized to **[0,1] of
  the input image** (`fall_detector.py:408-409`). Hip-centering + torso-scaling remove translation
  and uniform scale **but NOT aspect-ratio** (documented at `fall_detector.py:280-288`).
- Legacy input was **4:3** (640×480); the Tapo CV stream is **16:9** (1280×720, SPS-decoded). On 16:9
  every normalized **X** is squeezed relative to training → out-of-distribution proportions →
  constant high `fall_prob`. The velocity gate is currently the only thing suppressing false alarms.
- **Fix:** crop the frame to 4:3 **before** Pose, restoring the training-time normalization. We do
  **center-crop (Strategy A)**: `1280×720 → central 960×720`.

### Two properties that make Strategy A the least-invasive correct change
1. **Velocity gate is undisturbed.** Strategy A crops **width only** (`ry=0, rh=720` = full height).
   MediaPipe `landmark.y` stays normalized to the same 720 px height, so `body_y`, `_compute_body_velocity`,
   and the `min_fall_velocity` gate operate on the **identical Y scale** as before. Only **X**
   normalization changes (the part that was wrong). → **No velocity recalibration needed by this change.**
2. **Gesture handoff is preserved for free.** Wrist points are mapped back through `crop_bbox` in
   `_landmark_point` (`fall_detector.py:533-541`): `x = crop_x + landmark.x*crop_w`. Setting
   `crop_bbox = (160, 0, 960, 720)` maps wrists back into **full-frame pixel coords**, exactly what
   `gesture_recognizer.py` expects. No gesture code changes.

---

## Sequencing

**This plan is the prerequisite for `OVERLAY_FALL_REGION_PLAN.md`.** It introduces the single shared
geometry helper `detection/fall_geometry.py::fall_region_px`, which the overlay plan later reads to
broadcast `fall_region`. Implement **this** first.

---

## C4 re-scope (explicit amendment — operator-approved)

Phase-A constraint **C4** froze the fall stack (MediaPipe Pose + TFLite transformer + all fall
constants). This plan **amends C4** to permit exactly ONE change: **the frame geometry fed to Pose**
(full 16:9 → 4:3 center-crop). Justification: it **restores** the model's original deployment
condition rather than retuning it. **Everything else in C4 stays frozen:**

- TFLite transformer weights, `fall_input_timesteps`, `fall_confidence_threshold` — unchanged.
- Hip-centering / torso-scaling (`_hip_centered_features`) — unchanged.
- Velocity gate (`_compute_body_velocity`, `min_fall_velocity`, `velocity_window`) — unchanged.
- Post-fall verification (`post_fall_wait`, `post_fall_move_threshold`) — unchanged.
- MediaPipe Pose `Delegate.CPU` and its detection/tracking confidences — unchanged.

## Binding constraints (do not violate)

- **F1** — Only the Pose-input frame geometry changes. Do not touch the transformer, normalization,
  thresholds, or the Y-based velocity path.
- **F2** — Face, person, and gesture detectors keep receiving the full 1280×720 frame (`main.py`
  passes the same `rgb_frame` to them; **do not** route the crop to anything but Pose).
- **F3** — Person bbox in the alert payload stays full-frame (`bbox` arg to `process_track` is
  unchanged; it is not the Pose input).
- **F4** — The cropped array passed to `self._pose_detector.process(...)` **must be C-contiguous**
  (a numpy column-slice is a non-contiguous view; MediaPipe needs `np.ascontiguousarray`).
- After each step: `uv run pytest`, `uv run ruff check <touched>`, `uv run mypy <touched>` stay green.

---

## STEP 1 — `detection/fall_geometry.py` (new, pure)

```python
def fall_region_px(
    frame_w: int, frame_h: int, *, aspect: float = 4.0 / 3.0
) -> tuple[int, int, int, int]:
    """Central rect of the given aspect inside frame_w×frame_h (Strategy A center-crop).

    Returns (x, y, w, h) in pixels. With aspect <= 0, or a degenerate frame, returns the
    full frame (no-op) so the crop can be disabled cleanly.
    """
    if aspect <= 0 or frame_w <= 0 or frame_h <= 0:
        return 0, 0, frame_w, frame_h
    target_w = round(frame_h * aspect)
    if target_w <= frame_w:
        x_off = (frame_w - target_w) // 2
        return x_off, 0, target_w, frame_h
    target_h = round(frame_w / aspect)
    y_off = (frame_h - target_h) // 2
    return 0, y_off, frame_w, target_h
```
- `fall_region_px(1280, 720)` → `(160, 0, 960, 720)`.

## STEP 2 — Wire the crop into `FallDetector.process_track`

In `fall_detector.py:289-294`, replace the full-frame Pose setup:

```python
# BEFORE
frame_w, frame_h = frame_size
pose_rgb = rgb_frame
pose_box: BBox = (0, 0, frame_w, frame_h)
if pose_rgb.size == 0:
    ...
```
```python
# AFTER
frame_w, frame_h = frame_size
rx, ry, rw, rh = fall_region_px(frame_w, frame_h, aspect=self._cfg.fall_pose_aspect)
pose_rgb = np.ascontiguousarray(rgb_frame[ry:ry + rh, rx:rx + rw])   # F4: contiguous
pose_box: BBox = (rx, ry, rw, rh)                                    # maps wrists back to full frame
if pose_rgb.size == 0:
    ...
```
- `pose_rgb` (the 4:3 crop) flows into `_run_detection` / `_extract_and_normalize_pose` as `rgb_crop`
  → MediaPipe now normalizes landmarks to 4:3. **This single substitution is the actual fix.**
- `pose_box` flows in as `crop_bbox` → wrist mapping (`_landmark_point`) returns full-frame pixels →
  gesture handoff unchanged.
- `bbox` (person box) is untouched and still used for the alert payload (F3).
- Import `from detection.fall_geometry import fall_region_px` at the top of `fall_detector.py`.

> Both Pose call paths (`_run_detection` for idle, `_extract_and_normalize_pose` for monitoring)
> already receive `rgb_crop`/`crop_bbox` from these two locals, so this one edit covers both states.

## STEP 3 — Config

`config.py` `PipelineConfig`, near the fall block (lines 123-136):
```python
fall_pose_aspect: float = 4.0 / 3.0   # 4:3 = legacy training geometry. Set <=0 to disable crop.
```
- Optional env override (mirror existing `_env_float` style) if the operator wants to A/B test
  with/without the crop at runtime: `fall_pose_aspect = _env_float("FALL_POSE_ASPECT", self.fall_pose_aspect)`.
  Setting it to `0` makes `fall_region_px` return the full frame → exact pre-change behavior, for a
  clean baseline comparison.

## STEP 4 — Tests (`tests/`, mirror existing contract style)

- **Geometry:** `fall_region_px(1280,720) == (160,0,960,720)`; a portrait frame (e.g. `720×1280`)
  returns a full-width, vertically-centered 4:3 rect; the returned rect's aspect is 4:3 (±1 px);
  `aspect<=0` and a zero-size frame both return the full frame.
- **Crop wiring (inject a fake pose detector):** stub `self._pose_detector.process` to return a known
  landmark set; assert that:
  - the array handed to `process` has shape `(720, 960, 3)` and is C-contiguous (F4);
  - a wrist landmark at crop-local `(0.5, 0.5)` maps to full-frame pixel `(160 + 0.5*960, 0.5*720) =
    (640, 360)` via `PoseTrackData.left/right_wrist` (proves `crop_bbox` offset is correct);
  - the model-input X feature for a landmark differs from the pre-crop value (proves X is now
    4:3-normalized) while the Y feature / `body_y` is **unchanged** (proves velocity scale preserved).
- **Regression:** existing fall contract tests in `tests/test_phase_a_contracts.py` stay green.

---

## Validation procedure (empirical — we have ZERO real Tapo falls on record)

Do this BEFORE trusting the gate; the fix changes the model's input distribution, so re-observe it:

1. **Baseline (optional):** run with `FALL_POSE_ASPECT=0` (crop disabled) and confirm the current
   pathology — `fall_prob` pinned ~90–99% regardless of posture.
2. **With fix (`fall_pose_aspect=4/3`, default):** stand / sit / bend in view and confirm via logs
   (`fall_detector` per-frame status) that **`fall_prob` now tracks posture** — low while standing,
   high only when actually on the floor. This is the primary success signal.
3. **Stage one real fall** inside the 4:3 zone and confirm:
   (a) `fall_prob` rises above 0.90, AND
   (b) the logged `body_velocity` actually **clears `0.025`** (so the gate passes a true fall), AND
   (c) the 3 s inactivity check confirms → `FALL CONFIRMED`.
4. **Only if** a real fall's velocity falls short of `0.025` do we revisit `min_fall_velocity` — and
   the correct change then is **scale-invariant** (divide velocity by torso height), never `abs()` or
   2D magnitude (those reintroduce false positives). That is a **separate, later** decision.
5. **Non-regression:** confirm face boxes, person tracks, and gesture recognition behave exactly as
   before (they never saw the crop).

---

## Out of scope (do NOT do now)

- Any change to `min_fall_velocity`, the transformer, thresholds, or normalization (see C4 re-scope).
- Strategy B (pad) — Strategy A is the approved approach.
- The overlay fall-zone panel — that is `OVERLAY_FALL_REGION_PLAN.md`, implemented after this.
- Routing the crop to face/person/gesture (F2) or changing the alert bbox (F3).

## End-to-end verification

```bash
cd go2rtc && docker compose up -d
cd /Users/ogulcanozdemir/video_Process
uv run pytest && uv run ruff check detection/fall_geometry.py detection/fall_detector.py config.py \
  && uv run mypy detection/fall_geometry.py detection/fall_detector.py config.py
uv run --no-sync python main.py
```
Then run the **Validation procedure** above (steps 2–3 are the ones that prove the fix worked).
