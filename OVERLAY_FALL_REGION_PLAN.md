# Overlay — Dedicated Fall-Zone Panel (Handoff Plan)

> **Deliverable:** add a second, dedicated panel to `web/overlay.html` that visualizes ONLY the
> region the fall detector operates on (the 4:3 area MediaPipe Pose receives), fed from a single
> authoritative source in `VisionState`. Optionally render the live pose skeleton inside that panel
> so the operator can confirm not just *where* the zone is but *what the fall detector sees and
> whether it is catching the body*.

## Confirmed decisions

- **Scope = Stage 1 + Stage 2** (region panel **and** live pose skeleton). Both are in scope; Stage 2
  is not optional for this handoff.
- **Region strategy = A (center-crop)** — `fall_region_px` returns the central 4:3 crop
  (`(160,0,960,720)` on 1280×720). The panel and the fall Pose input both use this.

## Sequencing (READ FIRST)

**This plan is implemented AFTER the Fall-Detection 4:3-crop plan is merged.** That plan introduces
the 4:3 geometry MediaPipe Pose consumes (center-crop "Strategy A", recommended; or pad "Strategy B").
This overlay plan **does not invent geometry** — it broadcasts the *same* region the fall stack
already uses, so the panel can never drift from the real Pose input.

- If the fall-crop plan exposes the region as a shared pure helper (see **Single source of truth**),
  this plan just reads it.
- If for any reason this plan lands first, it introduces the helper and the fall-crop plan consumes
  it. **Either order is fine — there must be exactly ONE function that returns the region.**

---

## Binding constraints (do not violate)

- **V1 — visualization only.** No change to fall logic, the TFLite transformer, MediaPipe Pose
  settings, hip-centering, the velocity gate, or any fall constant (**C4 stays frozen**). New backend
  code only *reads* already-computed data and *exposes* it on the WS state.
- **V2 — other detectors untouched.** Face, person, and gesture detectors keep receiving the full
  1280×720 frame exactly as today. Their inputs, ratios, and behavior do not change.
- **V3 — single source of truth.** The 4:3 region is computed in exactly ONE place and consumed by
  both (a) the fall Pose-input crop and (b) the `fall_region` broadcast. The panel follows whatever
  strategy (A/B) the fall plan chose, automatically.
- **V4 — additive WS schema.** New `VisionState` fields are optional and default to absent/None.
  Existing overlay consumers (faces/persons/fall/gesture) are unaffected. This is the VisionState
  WebSocket (:5003) — separate from the Pi event payloads, so the C5 wire-payload freeze is N/A here.
- After each step: `uv run pytest`, `uv run ruff check <touched>`, `uv run mypy <touched>` stay green.

---

## Coordinate conventions (used throughout)

- **`fall_region`** = the rectangle, in **full-frame normalized coords** `[0,1]` relative to `pw×ph`,
  that the fall Pose visually consumes. For Strategy A on 1280×720 it is
  `{nx: 0.125, ny: 0.0, nw: 0.75, nh: 1.0}` (the central 960×720). For Strategy B (pad) it is the
  full frame `{nx:0, ny:0, nw:1, nh:1}` and the panel letterboxes it into 4:3.
- **Pose skeleton points** (Stage 2) = `{name: [nx, ny, vis]}`, also in **full-frame normalized
  coords**, so the panel can map any point into panel-local space via `fall_region`:
  ```
  localX = (nx - region.nx) / region.nw * panelW
  localY = (ny - region.ny) / region.nh * panelH   # draw only if vis > 0.3 and 0<=local<=1
  ```
- Panel `drawImage` source rect (in **video pixel** coords): `sx = region.nx * v.videoWidth`, etc.

---

## STAGE 1 — Region panel (region box only, no skeleton)

### STEP 1.1 — Backend: single-source region helper + `fall_region` on `VisionState`

**Single source of truth.** Add a pure helper that returns the fall region as a pixel BBox:
```python
# detection/fall_geometry.py  (new, pure — no cv2/model imports needed for the math)
def fall_region_px(frame_w: int, frame_h: int, *, aspect: float = 4 / 3) -> tuple[int, int, int, int]:
    """Central rect of the given aspect inside frame_w×frame_h (Strategy A center-crop)."""
    target_w = round(frame_h * aspect)
    if target_w <= frame_w:
        x_off = (frame_w - target_w) // 2
        return x_off, 0, target_w, frame_h
    target_h = round(frame_w / aspect)
    y_off = (frame_h - target_h) // 2
    return 0, y_off, frame_w, target_h
```
- **The fall-crop plan must call this same helper** to crop the Pose input (or, if it already added
  the crop inline, refactor that geometry into this helper and have both call sites use it).
  If Strategy B (pad) is chosen there, this helper returns the full-frame rect and the pad offset is
  the fall stack's internal concern; the panel still renders the returned rect contain-fit.

**`streaming/vision_state.py`:** add an optional region field (mirror the `TrackedFace` dataclass
style):
```python
@dataclass(frozen=True)
class FallRegion:
    nx: float; ny: float; nw: float; nh: float
    def to_dict(self) -> dict[str, Any]: return {"nx": self.nx, "ny": self.ny, "nw": self.nw, "nh": self.nh}
```
Add to `VisionState`: `fall_region: FallRegion | None = None`, and in `to_dict()` emit
`"fall_region": self.fall_region.to_dict() if self.fall_region else None`. (VisionWSServer already
serializes via `to_dict()`, so the field rides along automatically.)

**`main.py` `_camera_loop`:** compute once per frame from the same helper and attach to the state:
```python
rx, ry, rw, rh = fall_region_px(frame_w, frame_h)
fall_region = FallRegion(nx=rx / frame_w, ny=ry / frame_h, nw=rw / frame_w, nh=rh / frame_h) \
    if frame_w > 0 and frame_h > 0 else None
```
Pass `fall_region=fall_region` into the `VisionState(...)` constructor (alongside the existing
`fall=fall_state`). No other call sites change.

### STEP 1.2 — Frontend: dedicated fall-zone panel in `web/overlay.html`

Pure client-side rendering from the same `lastState`. The panel reuses the already-playing
`<video-stream>` — **no second go2rtc stream, no extra decode** — by `drawImage`-cropping the
region from `videoEl()`.

1. **Markup/CSS:** add a fixed 4:3 panel (e.g. bottom-right, ~320×240) with a label strip:
   ```html
   <div id="fallzone"><canvas id="fallcanvas"></canvas><div id="falllabel">FALL ZONE (4:3)</div></div>
   ```
   Style it `position:absolute`, bordered, `z-index` above the main overlay, `pointer-events:none`.
2. **Render (inside the existing `draw()` rAF loop, after the main overlay draw):**
   ```js
   function drawFallZone(st) {
     const c = document.getElementById('fallcanvas'); const v = videoEl();
     const ctx = c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height);
     const reg = st && st.fall_region;
     if (!reg || !v || !v.videoWidth) { /* show “no region” placeholder */ return; }
     const sx = reg.nx*v.videoWidth, sy = reg.ny*v.videoHeight,
           sw = reg.nw*v.videoWidth, sh = reg.nh*v.videoHeight;
     // contain-fit the (possibly 16:9, Strategy B) source into the 4:3 panel
     const s = Math.min(c.width/sw, c.height/sh), dw = sw*s, dh = sh*s,
           ox = (c.width-dw)/2, oy = (c.height-dh)/2;
     ctx.drawImage(v, sx, sy, sw, sh, ox, oy, dw, dh);
     // stash {ox,oy,dw,dh,reg} for Stage 2 skeleton mapping
   }
   ```
   Set `fallcanvas.width/height` to the panel's pixel size once (e.g. 320×240).
3. **Bonus (cheap, recommended):** in the MAIN overlay `draw()`, also outline `st.fall_region` as a
   thin labeled rect (reuse `box(ctx, r, reg.nx, reg.ny, reg.nw, reg.nh, '#fb923c', 'FALL ZONE')`)
   so the operator sees where the zone sits on the full 16:9 scene AND zoomed in the panel — this is
   exactly what validates the Strategy-A coverage trade-off (who falls outside the zone).
4. **Fallback:** if `fall_region` is absent (fall-crop plan not merged, or `pw/ph<=0`), render a dim
   “no fall region” placeholder in the panel and skip the main-view outline.

### STAGE 1 tests
- `tests/`: `fall_region_px(1280,720)` returns `(160,0,960,720)`; a portrait frame
  (e.g. 720×1280) returns a full-width, vertically-centered rect; aspect is exactly 4:3 (±1 px).
- `VisionState(..., fall_region=FallRegion(...)).to_dict()["fall_region"]` has keys `nx,ny,nw,nh`;
  with `fall_region=None` it serializes to `None`.
- (Frontend is manual-verified; no JS test harness exists in repo.)

---

## STAGE 2 — Live pose skeleton in the panel (IN SCOPE — confirmed)

Shows "what the fall detector sees + is it catching the body" by drawing the MediaPipe Pose skeleton
inside the zone panel.

### STEP 2.1 — Backend: expose skeleton points (C4-safe, additive)

The fall detector already computes all 17 landmarks per track in `_extract_and_normalize_pose`
(pre-normalization `features`, in **pose-input** coords) and already returns a `PoseTrackData` per
track via `fall_result.pose`, which `main.py` collects into `pose_tracks` (currently for gesture).

- Extend `PoseTrackData` (dataclass in `detection/fall_detector.py`) with an additive optional field:
  `points: dict[str, tuple[float, float, float]] | None = None` — `{kp_name: (nx, ny, vis)}` in
  **full-frame normalized** coords. Populate it in `_pose_track_data` by mapping each landmark
  through `crop_bbox` (the same offset already used for wrists) and dividing by `frame_w/frame_h`.
  This is read-only exposure — **no change to features, normalization, gating, or thresholds (C4).**
- `streaming/vision_state.py`: add `poses: Sequence[Mapping] = ()` to `VisionState` (each entry
  `{"id": track_id, "points": {name: [nx,ny,vis], ...}}`), emitted in `to_dict()`.
- `main.py`: build `poses` from the `pose_tracks` already gathered in the loop (reuse, don't recompute
  Pose) and pass into `VisionState`.

### STEP 2.2 — Frontend: draw skeleton into the panel

- Using the `{ox,oy,dw,dh,reg}` stashed by `drawFallZone`, map each broadcast point to panel-local
  coords via the formula in **Coordinate conventions**, drawing only points with `vis > 0.3`.
- Draw joints (filled dots) and bones (line segments) using this connection list over the fall
  keypoint names:
  ```
  L/R Shoulder; L/R Hip; L Shoulder–L Hip; R Shoulder–R Hip;
  L Shoulder–L Elbow–L Wrist; R Shoulder–R Elbow–R Wrist;
  L Hip–L Knee–L Ankle; R Hip–R Knee–R Ankle.
  ```
- Color the skeleton by fall state if desired (reuse `st.fall.tracks[id].fall_state`): idle=green,
  falling=amber, on_floor=red.

### STAGE 2 tests
- `PoseTrackData.points` for a synthetic landmark set maps correctly into full-frame normalized
  coords (offset via `crop_bbox`, in `[0,1]`).
- `VisionState(..., poses=[...]).to_dict()["poses"]` round-trips the `{id, points}` shape; default is
  an empty list.

---

## Out of scope (do NOT do now)

- Implementing or altering the 4:3 crop itself — that is the **Fall-Detection plan's** job; this plan
  only consumes its region helper.
- Any change to fall thresholds/model/Pose config (**C4**), or to face/person/gesture detectors (V2).
- A second go2rtc stream or server-side cropped video — the panel crops client-side from the existing
  player.
- Pi event payloads / dispatcher / outbox — unrelated.

---

## End-to-end verification

```bash
cd go2rtc && docker compose up -d
cd /Users/ogulcanozdemir/video_Process
uv run pytest && uv run ruff check streaming/ main.py detection/fall_geometry.py && uv run mypy streaming/ main.py
uv run --no-sync python main.py
python3 -m http.server 8090 --bind 127.0.0.1   # from repo root
# open http://localhost:8090/web/overlay.html
```
Manual:
1. **Region correctness:** the "FALL ZONE (4:3)" panel shows the central 960×720 crop live; the
   main view shows the matching orange outline. The two always agree (single source of truth).
2. **Coverage check:** walk to the left/right edge of the room — confirm when you leave the orange
   outline you also leave the panel (validates the Strategy-A horizontal coverage trade-off BEFORE
   relying on it).
3. **Strategy follow:** if the fall plan switches A↔B, the panel/outline change with zero overlay
   edits (region comes from the shared helper).
4. **(Stage 2)** Stand / sit / lie down in the zone — the skeleton tracks the body in the panel and
   recolors by fall state; confirms Pose is actually locking onto the person inside the zone.
