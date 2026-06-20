# Session Handoff — Enrollment Migration + Gesture/Frontality + Dashboard WebRTC (2026-06-20)

> Purpose: seed a **fresh chat** to continue without re-deriving context. My role: **review Codex
> implementations against plans + write handoff plans; Codex implements.** Keep that split.
> This file is gitignored (`*_PLAN.md`).

## Repos, branches, sync state

- **Mac edge pipeline:** `video_Process` (`/Users/ogulcanozdemir/video_Process`), branch **`fall-detection-new`**.
  4 commits ahead of origin (unpushed, carry-over). **⚠️ ALL of this session's Mac work is UNCOMMITTED in
  the working tree** (see below) — committing + pushing it is the #1 next action.
- **Pi backend:** `proactive-home-agent` (`/Users/ogulcanozdemir/proactive-home-agent`), branch **`new-event`**
  (active; `local-autonomy` is OLD — do not touch). **In sync with origin (clean).**

## What landed this session

### Pi `new-event` (committed ✅ + reviewed)
- `9d8dcfa` — modularize vision API + **7-angle face enrollment** (thin-client enrollment support, live presence).
- `8433d67` — **deprecate biometric login** + sync user management (Mac↔Pi): `POST /users/guest` (idempotent,
  JWT, owner-linked), `/users/register` & `/vision/identify` deprecation warnings, removed `list_users`
  `face_embedding != None` filter, gesture parity warning guarded (`user not in (Unknown,Guest,A Stranger)`).
- `76ec1b9` — **speaker threshold 0.65 → 0.55** (P2; Berkay was dropping to "Guest" at 0.61–0.62).

### Mac `fall-detection-new` (UNCOMMITTED working tree — all reviewed ✅, ruff/mypy/pytest green = 75 passed)
- **Enrollment migration backend** (untracked: `api/`, `services/`, `repositories/`, `identification/gallery_store.py`,
  `scripts/import_faces_gallery.py`, `tests/test_enrollment_api.py`): ArcFace enrollment on Mac, 7-angle single
  multipart batch (`POST /enroll`), `FaceRepository` (SQLite `data/embeddings/faces.db`), hot-reload `GalleryStore`,
  REST API (`api/app.py` `create_app` + CORS, default `("*",)`), `main.py` starts `VisionAPIServer` (:8800).
- **Gesture identity fix** (`tracking/tracker_manager.py` `user_for_person_track`, `detection/gesture_recognizer.py`,
  `main.py`): gesture user resolved from the **person track** that produced the hand ROI (not the short-lived face
  track). `_pending_track_id` bound at submit; `track_id` carried through `_raw_hand_bbox`/`_hand_roi`.
- **Gesture reliability Fix 1** (`gesture_recognizer.py`): removed the manual `_is_processing` single-flight gate
  (it wedged permanently when MediaPipe LIVE_STREAM dropped the in-flight callback → total gesture silence under
  load). Now submits every frame; MediaPipe's flow-limiter handles backpressure.
- **Fix 2** (`main.py`): `gesture_rec.is_frontal` reset each frame (no longer freezes after face-track expiry).
- **P4 pose-based frontality** (`config.py`, `detection/fall_detector.py`, `main.py`): gaze-lock frontality now
  derived from **pose landmarks** (nose+eyes+shoulders: `min(vis) ≥ 0.5` and `eye_dx/shoulder_dx ≥ 0.30`), exposed
  as `PoseTrackData.frontal`, set in `_pose_track_data`. `main.py`: `if pose_tracks: gesture_rec.is_frontal =
  any(p.frontal …)` (sticky, ~15Hz). Removed the face-based `estimate_frontality` gaze-lock path (quality-gate use
  untouched). Config: `pose_frontal_min_visibility=0.5`, `pose_frontal_eye_shoulder_ratio=0.30`.
- (Working tree also carries the enrollment-related `face_identifier.py`, `pyproject.toml`, `requirements.txt`,
  `uv.lock`, `tests/test_pipeline_shutdown.py` edits.)

## Runtime validation (live, both repos)

End-to-end **working**: Fix1+Fix2+P4 hold across long sessions; gestures fire continuously even while sitting
(no more silence); identity attribution correct (`ogulcan`/`Berkay`, not Unknown); device-control mappings
(`Thumb_Up→turn_on`, `Thumb_Down→turn_off`, `Pointing_Up→turn_on`); **Closed_Fist SOS** fires (grace→lockdown→
Twilio+Telegram); **fall→Open_Palm cancel** works (`FALL EMERGENCY ABORTED BY USER`).

## Decisions locked (do NOT relitigate)

1. **Gesture identity = PERSON track**, resolved via `user_for_person_track(track_id)`; face track only refines the
   hand ROI. `track_id` present but unidentified → `Unknown` (no fabrication); `None` → fall back to any identified.
2. **No manual single-flight gate** in the gesture recognizer — rely on MediaPipe's flow-limiter (submit every frame).
3. **Frontality from POSE**, not face track; `is_frontal` sticky-updated only on pose frames.
4. **Enrollment lives on Mac** (ArcFace, buffalo_l, 512-d). Pi is a thin client. **Label parity invariant:
   Mac gallery label == Pi `User.username`** (case-sensitive).
5. **Pi auto-create / owner-escalation experiment REJECTED & reverted.** The real cause of "Unknown gestures" was
   Mac-side (wrong user sent); do NOT reintroduce implicit User-row creation or guest→owner gesture resolution on Pi
   (security hole + contradicts the "no auto guest" product decision).
6. **Speaker threshold = 0.55** (Pi).

## Open items / next actions (priority order)

1. **COMMIT + PUSH the Mac working tree** (enrollment backend + gesture identity + Fix1/Fix2/P4). It's all reviewed
   and green but uncommitted; one logical commit (or a few) then push `fall-detection-new`.
2. **P0 (Pi) — owner SOS announced as "Intruder detected".** Berkay/ogulcan (homeowners) holding Closed_Fist SOS
   trigger `execute_emergency_lockdown` whose TTS says *"Intruder detected!"*. Owner-initiated SOS must be a
   help/medical message, not intruder. **No plan written yet** — needs a Pi handoff.
3. **Fix 3 (Mac) — decouple gesture hand ROI from the fall pose.** Plan ready in
   `GESTURE_PIPELINE_RELIABILITY_HANDOFF_PLAN.md` (feed person-track regions every frame; person-bbox crop fallback
   when no wrists). Not implemented.
4. **Dashboard WebRTC camera.** Plan ready: `DASHBOARD_WEBRTC_CAMERA_HANDOFF_PLAN.md`. Current `CameraFeed.tsx`
   iframe points at wrong host/stream/mode (`100.105.136.5` Pi + `living_room_cam` + `mse`). Fix: Mac `.env`
   `GO2RTC_WEBRTC_CANDIDATE=100.90.235.67:8555` + frontend iframe→`http://100.90.235.67:1984/stream.html?src=living_room_hd&mode=webrtc`
   (or native `RTCPeerConnection`), via `NEXT_PUBLIC_GO2RTC_URL`. Not implemented.
5. **P1 (infra) — Mac STT server down.** Pi STT calls `http://100.119.128.11:8000/transcribe` (a separate Mac,
   `MAC_STT_URL`) → `Connection failed` → voice fall-confirmation dead (Open_Palm cancel saved it). Start that STT
   server.
6. **P0 gallery cleanup.** Mac `data/embeddings/faces.db` still has stale labels `OG`, `meto`, `teko` (no Pi User
   rows → parity-broken warnings / mislabeling). Delete via `DELETE /enrolled/<label>` (or re-enroll meto/teko).
7. **Carry-over (still open):** fall e2e Pi runtime smoke; durable outbox for `person_left` (fire-and-forget today);
   serve fall snapshot for non-fall consumers; coordinated identity-flag rollout (both default-off now).

## Known runtime issues observed (logged, not yet code-fixed)
- P0 owner SOS "intruder" wording (item 2). · P1 STT down (item 5). · Tracker/face-track churn (IDs climb fast,
  faces "too small"/"profile") → identity occasionally drops to Unknown. · Fall model baseline high while sitting
  (95%+; velocity gate is the only guard; near-misses at v≈0.026–0.029 correctly cancelled). · TTS splits decimals
  ("25.75°C" → "…25." + "75°C"). · Agent greeting/goodbye collision on entry. · Gesture cooldown sends ~9 dup events
  per held gesture (SOS grace tolerates).

## Plan files (gitignored, in `video_Process/`)
This session: `ENROLLMENT_MIGRATION_PLAN.md`, `CORS_FIX_HANDOFF_PLAN.md`, `USER_TABLE_AGENTIC_PLAN.md`,
`AGENTIC_RECOGNITION_IMPACT_PLAN.md`, `GESTURE_IDENTITY_FIX_HANDOFF_PLAN.md`,
`GESTURE_PIPELINE_RELIABILITY_HANDOFF_PLAN.md` (Fix1 done; Fix2/Fix3 — Fix3 pending),
`GESTURE_FRONTALITY_AND_SPEAKER_HANDOFF_PLAN.md` (P4+P2 done),
`DASHBOARD_WEBRTC_CAMERA_HANDOFF_PLAN.md` (pending).

## Key anchors (verified this session)
- Gesture: `tracking/tracker_manager.py:user_for_person_track`; `detection/gesture_recognizer.py` (`_on_result`,
  `_raw_hand_bbox`/`_hand_roi` carry `track_id`, no `_is_processing`); `main.py` camera loop (~180 presence-only
  face loop, ~224 `gesture_rec.is_frontal = any(pose.frontal …)` before `gesture_rec.process`).
- Frontality: `detection/fall_detector.py:_pose_is_frontal` + `PoseTrackData.frontal` (set in `_pose_track_data`,
  fed by `_extract_and_normalize_pose` ← both IDLE `_run_detection` and MONITORING).
- Enrollment: `services/enrollment_service.py`, `repositories/face_repository.py`, `identification/gallery_store.py`,
  `api/app.py` (`create_app`, CORS), `api/routers/enrollment_router.py`, `tests/test_enrollment_api.py`.
- Streaming: go2rtc on Mac (`go2rtc/go2rtc.yaml`: `living_room_sd`=CV, `living_room_hd`=dashboard; api :1984
  origin "*", webrtc :8555 needs `GO2RTC_WEBRTC_CANDIDATE`). config `go2rtc_dashboard_stream_name/mode` scaffolding.
- Pi gesture/SOS: `backend/api/routers/vision_router.py` (`handle_gesture`, `delayed_emergency_lockdown`,
  `execute_emergency_lockdown` ← "Intruder detected" TTS); `backend/api/services/speaker_service.py:identify_speaker`
  (threshold 0.55).

## Networking
Mac Tailscale `100.90.235.67` (vision API :8800, WS :5003, go2rtc :1984/:8555). Pi `100.105.136.5` (backend :8000,
frontend :3000). STT Mac `100.119.128.11:8000`. Start Pi backend:
`IDENTITY_AUTHORITATIVE=true IDENTITY_PRESENT_BACKSTOP_S=300 uvicorn main:app --host 0.0.0.0 --reload` (from
`backend/`, venv active; `.env` provides DATABASE_URL etc.). Frontend env: `NEXT_PUBLIC_API_URL`,
`NEXT_PUBLIC_VISION_API_URL`, (new) `NEXT_PUBLIC_GO2RTC_URL`.

## Constraints carried over
720p `/stream2` for CV · don't break `go2rtc/` streams · MediaPipe CPU · gesture/`update_presence` payloads stable ·
surgical edits, no heavy deps · camera-thread callbacks never under `self._lock` · both identity flags default-off
until coordinated · Pi work on `new-event` only (leave `local-autonomy`) · no Claude/Anthropic attribution in commits.
