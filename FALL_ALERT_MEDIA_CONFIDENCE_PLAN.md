# Fall Alert — Real Confidence + Screenshot to Telegram (multipart) — Codex Handoff

> Two repos. **Part A** = `video_Process` (Mac edge pipeline). **Part B** = `proactive-home-agent`
> **branch `new-event`** (`local-autonomy` is the old branch — do not touch it).
> Fixes two regressions introduced by the fall-alert schema stopgap:
> (1) the Pi sent a hardcoded `confidence=1.0` instead of the Mac's real fall probability;
> (2) the fall screenshot stopped reaching the Pi (Mac sent only a local `snapshot_path`), so the Telegram
> **photo** stopped firing — only text alerts went out.
> macOS / `uv run` for Part A. After Part A: `uv run pytest`, `uv run ruff check <touched>`, `uv run mypy <touched>` green.

---

## Root cause (verified, both repos)

- **Confidence:** the Mac has the real probability at fall-confirm time — `state.verification_prob`
  (`fall_detector.py:339` `state.confidence = state.verification_prob`) — but `_build_alert_payload` never puts it
  in the payload. The Pi can't read what isn't sent, so the JSON branch hardcodes `confidence = 1.0` → always "100%".
- **Screenshot:** the Mac saves the JPEG to disk and sends only `media.snapshot_path` (a Mac-local path the Pi
  cannot read). The Pi sets `screenshot_bytes = None`, so `execute_fall_emergency_lockdown`
  (`vision_router.py:373-377`) takes the `else` branch → `send_telegram_alert` (text), never `send_telegram_photo`.

## Design (decided: **multipart**, like the old system)

- The probability travels in **`diagnostics.fall_probability`** (0–1 float). Both Pi branches read it (no
  top-level `confidence`, preserving the existing schema shape).
- The image travels as **multipart** when present: Mac POSTs `data={"payload": <json string>}` +
  `files={"screenshot": <jpeg bytes>}`. When there is **no** screenshot, Mac falls back to the existing
  `json=payload` POST (hits the Pi JSON branch). So: **image present → multipart branch; image absent → JSON branch**;
  both read confidence from `diagnostics.fall_probability`.

> Why the fallback split: `requests` with only `data=` (no files) sends `application/x-www-form-urlencoded`, which
> the Pi's `if "multipart" in content_type` check would miss. Sending JSON when there's no file keeps both branches
> well-formed. The screenshot is the normal case, so multipart is the primary path.

---

# PART A — Mac (`video_Process`)

## A1. `detection/fall_detector.py` — add probability to diagnostics

- `FallDiagnostics` TypedDict (`:45-48`): add `fall_probability: float`.
- `_build_alert_payload` (`:659-692`), in the `diagnostics` dict (`:677-681`): add
  `"fall_probability": round(float(state.confidence), 4),`
  (`state.confidence` was just set to `verification_prob` at confirm time — use it; or pass `verification_prob`
  explicitly. Both are equal at this point.)
- The existing Mac test `test_fall_alert_payload_has_action_schema_without_top_level_confidence` still holds
  (no **top-level** confidence). **Add** one assertion: `payload["diagnostics"]["fall_probability"] == <expected>`
  (set `state.confidence`/`verification_prob` in that test's `FallTrackState`).

## A2. `detection/fall_detector.py` — carry the JPEG bytes out

- `FallProcessingResult` (frozen dataclass, `:81-86`): add `snapshot_bytes: bytes | None = None`.
- In the **confirmed** block (`:346-354`), encode the frame once, keep saving to disk, and attach bytes:
```python
ok, buf = cv2.imencode(".jpg", bgr_frame)
snapshot_bytes = buf.tobytes() if ok else None
screenshot_path = self._save_screenshot(bgr_frame)   # unchanged: still logs to data/logs/screenshots
alert = self._build_alert_payload(... screenshot_path=screenshot_path ...)
```
  Then return `FallProcessingResult(track_id=..., bbox=..., pose=..., alert=alert, snapshot_bytes=snapshot_bytes)`.
  (`media.snapshot_path` stays in the payload for Mac-side logs / future Phase B URL serving.)

## A3. `main.py` — pass bytes to the dispatcher (`:202-203`)

```python
if fall_result.alert is not None:
    dispatcher.submit(dispatcher.send_fall_alert, fall_result.alert, fall_result.snapshot_bytes)
```

## A4. `events/dispatcher.py` — multipart when a screenshot exists

Add `import json` (top of file). Update `send_fall_alert` (`:23-29`):
```python
def send_fall_alert(self, payload: Mapping[str, object], screenshot_bytes: bytes | None = None) -> None:
    try:
        if screenshot_bytes:
            resp = requests.post(
                self._cfg.fall_alert_url,
                data={"payload": json.dumps(dict(payload))},
                files={"screenshot": ("fall.jpg", screenshot_bytes, "image/jpeg")},
                timeout=5.0,
            )
        else:
            resp = requests.post(self._cfg.fall_alert_url, json=dict(payload), timeout=3.0)
        logger.info("Fall alert sent to backend: {}", resp.status_code)
    except Exception as e:
        logger.error("Fall alert send error: {}", e)
```

## A5. Tests (Part A)

- A1 assertion (probability in diagnostics) above.
- `send_fall_alert` with `screenshot_bytes=b"..."` → posts **multipart** (`files=` set, `data["payload"]` is the
  JSON-dumped payload); with `screenshot_bytes=None` → posts **JSON** (`json=` set, no `files`). Mock
  `events.dispatcher.requests.post` and assert the call shape for both. (Mirror the existing dispatcher tests in
  `tests/test_phase_a_contracts.py`.)

---

# PART B — Pi (`proactive-home-agent`, branch `new-event`)

## B1. Models (`vision_router.py:56-78`)

- `FallDiagnostics`: add `fall_probability: float | None = None`.
- Add `import json` (top of file — currently not imported).

## B2. JSON branch (`vision_router.py:~424`, the block we already added) — read real confidence

Replace the hardcoded `confidence = 1.0` with:
```python
confidence = event.diagnostics.fall_probability if event.diagnostics.fall_probability is not None else 1.0
```
Keep `screenshot_bytes = None` here (JSON path = no image; that's the no-screenshot fallback). Everything else in
this branch unchanged (still 422 on ValidationError, still logs `media.snapshot_path` if present).

## B3. Multipart branch (`vision_router.py:416-422`) — rewrite to the new schema

Replace the old flat reads (`form["confidence"]`, `form["timestamp"]`, `form["screenshot"]`) with:
```python
    if "multipart" in content_type:
        form = await request.form()
        try:
            event = FallAlertEvent.model_validate(json.loads(form["payload"]))
        except (KeyError, ValueError, ValidationError) as e:
            raise HTTPException(status_code=422, detail=f"Invalid fall multipart payload: {e}")
        confidence = event.diagnostics.fall_probability if event.diagnostics.fall_probability is not None else 1.0
        source = event.source
        screenshot_file = form.get("screenshot")
        if screenshot_file is not None:
            screenshot_bytes = await screenshot_file.read()
```
`timestamp` was never used downstream (dead in both old branches) — drop it; do not reintroduce it.

Everything after the branch is unchanged: `log_fall_event("living_room", confidence)`,
`add_task(ask_and_wait_for_fall, confidence, screenshot_bytes)`. With `screenshot_bytes` now populated,
`execute_fall_emergency_lockdown` (`:374-375`) takes the `send_telegram_photo` path again, and
`{confidence:.0%}` renders the real probability.

> Legacy note: the old monolith that sent flat-field multipart is gone (`mac_camera.py` is a stub). If you want a
> belt-and-suspenders fallback, you may keep the old flat parsing when `"payload"` is absent — optional, not required.

## B4. Verification (Part B)

```bash
cd /Users/ogulcanozdemir/proactive-home-agent/backend
python3 -c "import ast; ast.parse(open('api/routers/vision_router.py').read()); print('ast ok')"
```
TestClient smoke (no DB needed, as before):
- **multipart** with `payload` (diagnostics.fall_probability=0.93) + a `screenshot` file → 200; assert the handler
  computed `confidence==0.93` and `screenshot_bytes` is non-None (log line `FALL ALERT RECEIVED — confidence: 93%`).
- **JSON** with diagnostics.fall_probability=0.88, no file → 200; `confidence==0.88`, screenshot None.
- malformed multipart (`payload` not valid JSON / missing) → 422.

---

## End-to-end acceptance (both flags/repos)

Trigger a real fall from the Mac:
1. Pi receives **multipart**, logs `FALL ALERT RECEIVED — confidence: <real %>` (matches the Mac's
   `verification_prob`, **not** 100%).
2. On no-response, Telegram sends the **photo** (`send_telegram_photo`) with the real confidence in the caption.
3. With a real fall but screenshot encode failure → Mac sends JSON, Pi still posts confidence + text Telegram.

## Constraints / notes

- Surgical: Part A touches `fall_detector.py` (2 fields + confirm block), `main.py` (1 line),
  `dispatcher.py` (multipart + `import json`). Part B touches one model field, the two branches of
  `handle_fall_alert`, and `import json`. Do not touch the emergency-lockdown / TTS / SMS / voice flow.
- Probability is `verification_prob` (0–1); Pi renders `{confidence:.0%}`. If you ever want a different number on
  the Pi, change it on the Mac (single source of truth) and bump `schema_version`.
- Phase B (later): also expose the snapshot via a fetchable URL so non-fall consumers can pull it; not needed here.
- All Part B edits land on **`new-event`**; leave `local-autonomy` untouched.
```