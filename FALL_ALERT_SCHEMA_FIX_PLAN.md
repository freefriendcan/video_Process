# Fall Alert Schema Mismatch — Pi Backend Fix — Codex Handoff

> Pi repo only: `proactive-home-agent` (`/Users/ogulcanozdemir/proactive-home-agent`), **branch `new-event`**
> (the active branch for all new work; `local-autonomy` is the old branch — do not touch it).
> The Mac (`video_Process`) fall stack was restructured; the Pi `/vision/fall_alert` handler still reads the old
> payload and 500s on every fall. This fix makes the Pi parse the **new structured schema**. No Mac change.
> `new-event` is in sync with `origin/new-event` and already carries the WS2 identity work (commit `3ca5f3c`);
> this fix layers on top. Anchors below verified against `new-event`.

---

## Root cause (verified)

`POST /vision/fall_alert` → `vision_router.py:403` `confidence = body["confidence"]` raises **KeyError → 500**.
The Mac no longer sends top-level `confidence` or `timestamp`.

**Mac sends (JSON, `fall_detector.py:_build_alert_payload`, `dispatcher.send_fall_alert` POSTs `json=`):**
```jsonc
{ "schema_version":"1.0", "event_type":"fall", "track_id":7, "fall_state":"on_floor",
  "bbox":[x,y,w,h], "centroid":[cx,cy], "frame_size":[w,h],
  "diagnostics": { "body_velocity":float, "torso_angle_deg":float, "inactivity_elapsed_s":float },
  "ts_wall":"...Z", "ts_monotonic":float, "source":"mac_studio_living_room", "zone":"living_room",
  "media": { "snapshot_path":"data/logs/screenshots/..." }  // optional; Mac-local path, NOT fetchable by Pi
}
```

**Pi expects (old, `vision_router.py:386-412`):** top-level `confidence` + `timestamp`; screenshot as multipart bytes.

Three breaks: (1) `confidence` removed — the alert itself = confirmed fall (Mac already passed velocity gate +
post-fall inactivity); (2) `timestamp` → `ts_wall`/`ts_monotonic`; (3) screenshot is now a Mac-local **path**, not
multipart bytes, so the Pi gets no image until Phase B serves it.

`confidence` on the Pi is used **only for human-readable text** (`{confidence:.0%}` at
`vision_router.py:330,341,357` and `presence_service.log_fall_event` `:57`). It does not gate any logic.

---

## Decision

Pi adapts to the Mac's new schema (don't revert the Mac — its structured schema is intentional and richer). Keep
the legacy multipart branch as a backward-compat fallback (older/monolith clients may still post it); only fix the
JSON branch. Downstream emergency flow signatures (`ask_and_wait_for_fall(confidence, screenshot_bytes)`,
`execute_fall_emergency_lockdown`) stay **unchanged** — we just feed them a sensible `confidence` and
`screenshot_bytes=None`.

## Code anchors (Pi, verified)

- `backend/api/routers/vision_router.py`: imports `Request`, `BaseModel` already present (`:1-2`);
  `handle_fall_alert` (`:386-412`); JSON branch (`:401-405`); `ask_and_wait_for_fall(confidence, screenshot_bytes)`
  (`:254`); `execute_fall_emergency_lockdown(...)` (`:320`); `{confidence:.0%}` (`:330,341,357`).
- `backend/api/services/presence_service.py`: `log_fall_event(location, confidence)` (`:57`).

---

## Step 1 — Add a Pydantic model (near `IdentityEvent`/`GestureEvent`, `vision_router.py:34-52`)

```python
class FallDiagnostics(BaseModel):
    body_velocity: float = 0.0
    torso_angle_deg: float = 0.0
    inactivity_elapsed_s: float = 0.0

class FallMedia(BaseModel):
    snapshot_path: str | None = None

class FallAlertEvent(BaseModel):
    schema_version: str = "1.0"
    event_type: str = "fall"
    track_id: int | None = None
    fall_state: str | None = None
    bbox: list[float] | None = None
    centroid: list[float] | None = None
    frame_size: list[int] | None = None
    diagnostics: FallDiagnostics = FallDiagnostics()
    ts_wall: str | None = None
    ts_monotonic: float | None = None
    source: str = "mac_camera"
    zone: str = "living_room"
    media: FallMedia | None = None
```
(All fields tolerant/optional so a slightly different Mac build never 500s; bad types → 422, not 500.)

## Step 2 — Rewrite ONLY the JSON branch of `handle_fall_alert` (`:401-405`)

Keep the `if "multipart" in content_type:` branch exactly as-is (legacy). Replace the `else:` (JSON) branch:

```python
    else:
        body = await request.json()
        event = FallAlertEvent.model_validate(body)
        # New schema carries no confidence: the alert *is* a confirmed fall
        # (Mac already passed the velocity gate + post-fall inactivity check).
        confidence = 1.0
        timestamp = event.ts_monotonic if event.ts_monotonic is not None else time.time()
        source = event.source
        # media.snapshot_path is a Mac-local filesystem path the Pi cannot read.
        # Until Phase B serves the image, proceed without a screenshot.
        screenshot_bytes = None
        if event.media and event.media.snapshot_path:
            logger.info(f"Fall snapshot path (not fetchable yet): {event.media.snapshot_path}")
```

Everything below the branch (`logger.critical`, `log_fall_event`, `add_task(ask_and_wait_for_fall, ...)`,
`return`) stays unchanged — `confidence=1.0` keeps `{confidence:.0%}` rendering as `100%`, `screenshot_bytes=None`
is already an accepted value (`Optional[bytes]` everywhere downstream).

> **Optional polish (only if cheap):** pass `fall_state` / `body_velocity` / `inactivity_elapsed_s` into the
> notification text so the emergency message is richer than "100%". Skip if it widens the diff much.

## Step 3 — Confirm `model_validate` import & error mapping

`FallAlertEvent.model_validate` raises `pydantic.ValidationError` on a malformed body. FastAPI does **not**
auto-convert that to 422 when you validate manually inside the handler — it would surface as 500. Wrap it:

```python
        try:
            event = FallAlertEvent.model_validate(body)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=f"Invalid fall payload: {e}")
```
Add `from pydantic import BaseModel, ValidationError` (currently only `BaseModel` is imported, `:2`).

## Step 4 — Verification

```bash
cd /Users/ogulcanozdemir/proactive-home-agent/backend
python3 -c "import ast; ast.parse(open('api/routers/vision_router.py').read()); print('ast ok')"
uvicorn main:app --host 0.0.0.0 --port 8000   # if runnable locally
```
Manual smoke (new schema — must return 200, not 500):
```bash
curl -sS -X POST localhost:8000/vision/fall_alert -H 'Content-Type: application/json' -d '{
  "schema_version":"1.0","event_type":"fall","track_id":7,"fall_state":"on_floor",
  "bbox":[10,20,100,200],"centroid":[60,120],"frame_size":[1280,720],
  "diagnostics":{"body_velocity":0.04,"torso_angle_deg":78.0,"inactivity_elapsed_s":3.0},
  "ts_wall":"2026-06-18T00:00:00Z","ts_monotonic":123.4,"source":"mac_studio_living_room","zone":"living_room",
  "media":{"snapshot_path":"data/logs/screenshots/fall_7.jpg"}
}' -w '\nHTTP %{http_code}\n'
```
Expect `{"status":"fall_alert_received"}` + `HTTP 200`, log line `FALL ALERT RECEIVED — confidence: 100%`. Then
fire a **real** fall from the Mac and confirm 200 + the emergency flow runs with no screenshot.

If the Pi has a test suite later, add: JSON new-schema → 200; missing `confidence`/`timestamp` no longer 500;
malformed body → 422; legacy multipart still works.

---

## Constraints / notes

- Surgical: new model + JSON-branch rewrite + one import. Do **not** touch the multipart branch, the emergency
  lockdown flow, or downstream signatures.
- This endpoint is a **separate channel** from WS2 identity events — unrelated to `IDENTITY_AUTHORITATIVE`.
- `confidence=1.0` is a deliberate placeholder (the alert means "confirmed fall"). If a real probability is wanted
  on the Pi later, the Mac must re-add it to the payload — coordinate as a schema bump (`schema_version`).
- **Phase B follow-up:** make the snapshot reachable (Mac attaches bytes or serves a URL) so `screenshot_bytes` is
  populated again; today the Pi runs the fall flow image-less.
- All edits land on **`new-event`** (in sync with `origin/new-event`). Leave `local-autonomy` untouched.
```