# Tapo Crop Collection — Implementation Plan (for the Codex agent)

> Repo: `/Users/ogulcanozdemir/video_Process`. macOS / Apple Silicon, ONNX Runtime + CoreML EP, `uv run` for everything.
> Adds the missing data-collection step that **precedes** `scripts/calibrate_threshold.py`: a tool that gathers **labeled face crops from the real Tapo stream** so the cosine threshold can be calibrated in the deployment domain. Be surgical — one new file. After the change run `uv run pytest`, `uv run ruff check <touched>`, `uv run mypy <touched>` and keep them green (baseline 38 tests).

## Binding constraints (do not violate)
- **C1** CV stays on the 720p `/stream2` Tapo sub-stream. **C2** do not touch `go2rtc/`. **C3/C4/C5** unaffected (no fall/gesture/HTTP changes).
- **Do NOT modify** `scripts/capture_enrollment.py`, `scripts/enroll_faces.py`, or `scripts/calibrate_threshold.py` — they are reviewed and green. Reuse, don't refactor.

## Motivation / data flow
`calibrate_threshold.py <crop_root>` expects a folder of **labeled Tapo face crops** (`<crop_root>/<label>/*.jpg`): one subfolder per enrolled person plus some `stranger_*` folders. It re-detects each saved image with `FaceDetector`, embeds via ArcFace, and reports genuine/impostor cosine distributions. Today there is **no tool to produce those crops** — the user must assemble them by hand. This plan adds that collector, sourcing frames from the **same Tapo stream the live pipeline uses** (so the crops match the calibration domain).

## STEP 1 — `scripts/capture_tapo_crops.py`
**Goal:** capture good-quality, single-face crops from the live Tapo stream into `<crop-root>/<label>/<label>_NN.jpg`, ready for `calibrate_threshold.py`.

**Source = Tapo stream (NOT the webcam).** Reuse the pipeline capture path so crops are real-domain:
- `cfg = PipelineConfig()`; `cfg.preflight()` (go2rtc reachability check, same as `main.py`).
- `producer = FrameProducer(cfg)`; `producer.open()`; loop on `pkt = producer.read_latest()` (returns `Optional[FramePacket]`; `None` → short `time.sleep` and continue). `FramePacket` fields: `.bgr`, `.rgb`, `.ir_mode`, `.width`, `.height`. Always `producer.release()` in a `finally`.

**Per-frame gate (reuse Phase-A components):**
- **Skip IR frames:** `if pkt.ir_mode: continue` — identification is day/color only (buffalo_l is RGB-trained), so night crops must never enter the calibration set.
- Detect with `FaceDetector(cfg).detect(pkt.rgb)`. Require **exactly one** face in frame (`len(faces) == 1`) to keep each saved crop unambiguously the target label; otherwise skip with a logged reason. (Mirror the single-good-face logic in `scripts/capture_enrollment.py:_single_quality_face`, but do not import from it — keep this script self-contained.)
- Quality-gate that face with `FaceQualityGate(cfg).check(face_roi, keypoints, frame_w, frame_h, (x,y,w,h), ir_mode=False)`; skip on failure. Do **not** loosen the gate.

**Saving — bounds-safe padded crop:**
- Crop a **padded** region around the detected bbox (arg `--pad`, fraction of bbox, default `0.4`) so the saved image keeps enough context for `calibrate_threshold` to re-detect the face, while trimming background/other people.
- **CRITICAL (correctness):** clamp the padded box to the frame before slicing — `x0=max(0,...)`, `y0=max(0,...)`, `x1=min(frame_w,...)`, `y1=min(frame_h,...)`. Negative numpy indices wrap around and would corrupt the crop. Skip if the resulting region is empty.
- Write BGR JPEG to `<crop-root>/<label>/<label>_NN.jpg` (zero-padded index, next free number like `capture_enrollment._next_output_path`).

**CLI (argparse, mirror `capture_enrollment.py` style + repo-root `sys.path` shim):**
- `--label` (required) — output subfolder / calibration label (person name, or e.g. `stranger_01`).
- `--count` (default 10) — number of good crops to save.
- `--interval` (default 0.5 s) — min spacing between auto-saved crops.
- `--pad` (default 0.4) — bbox padding fraction.
- `--crop-root` (default `data/tapo_crops`) — calibration root.
- `--no-preview` — headless (skip the OpenCV preview window).
- `--keypress` — save only the currently-good frame on Space (optional, like `capture_enrollment`).

**Do NOT add a `--calibrate` auto-chain.** Unlike `capture_enrollment --enroll`, calibration needs the **full multi-label set** (all enrolled people + strangers), which is collected across several runs of this script. Calibration stays a separate, explicit step. Print a final hint: `Collected N crops for '<label>'. When all labels (people + stranger_*) are collected, run: uv run python scripts/calibrate_threshold.py <crop-root>`.

**Acceptance:**
- `uv run python scripts/capture_tapo_crops.py --label testuser --count 5` (go2rtc up, daylight) saves 5 single-face JPEGs under `data/tapo_crops/testuser/`; multi-face / no-face / low-quality / IR frames are skipped with a logged reason; padded crops are bounds-safe (no wraparound) and re-detectable by `calibrate_threshold.py`.
- Running `calibrate_threshold.py data/tapo_crops` on a set built this way (≥1 enrolled label + ≥1 `stranger_*`) prints genuine/impostor stats and a recommended threshold.

**Tests (optional but preferred):** a small unit test for the bounds-safe padding helper (e.g. a face bbox near the frame edge yields a clamped, non-empty region with no negative indices). Keep it pure (no stream/model needed), in `tests/test_phase_a_contracts.py` or a new `tests/test_capture_tapo_crops.py`.

**Verification:**
```bash
cd go2rtc && docker compose up -d
uv run python scripts/capture_tapo_crops.py --label <you> --count 10        # daylight, ~1-1.5 m
uv run python scripts/capture_tapo_crops.py --label stranger_01 --count 10  # someone not enrolled
uv run python scripts/calibrate_threshold.py data/tapo_crops
uv run pytest && uv run ruff check scripts/capture_tapo_crops.py && uv run mypy scripts/capture_tapo_crops.py
```

## Out of scope
- No changes to the live pipeline, `go2rtc/`, or the three existing scripts.
- No automatic threshold writing to `config.py` (calibration still only *recommends*).
