# Architectural Analysis — Enrollment Pipeline Migration

> **Goal:** make `proactive-home-agent`'s onboarding/dashboard a **thin client** that captures raw photos
> and ships them to `video_process`. `video_process` becomes the authoritative service for embedding
> generation, enrollment, persistence, and identity queries.
>
> **Status of this document:** analysis / blueprint only. **No implementation code is produced here.**
> Grounded in the current code (file:line references throughout). Read the *Critical Findings* section
> first — two of them change the shape of the work.

---

## 0. Terminology (the two repos are mislabeled by the task, and it matters)

| Task term | Actual repo | What it is |
|---|---|---|
| "frontend" | `proactive-home-agent` | Has **both** a Next.js UI (`frontend/`) **and** a FastAPI backend (`backend/`). Today the *backend* does the embedding heavy-lifting. |
| "backend" (target heavy-lifter) | `video_process` | The Mac edge vision pipeline. Today it is a **pure client** — no REST server, no DB. |

When the task says "the `proactive-home-agent` frontend generates embeddings locally," that is imprecise:
the **browser** only captures JPEG blobs; the **Pi FastAPI backend** generates embeddings
(`backend/api/services/vision_service.py:62 register_face` → DeepFace). "Locally" = "on the Pi," not "in
the browser." The migration moves that embedding step **from the Pi backend to `video_process`**.

---

## 1. Critical Findings (read before planning)

### F1 — Identification is ALREADY local to `video_process`. ✅
`main.py:280` dispatches `self._identifier.identify(...)` against an in-process ArcFace gallery
(`identification/face_identifier.py`). The `EventDispatcher` has **no** call to the Pi's `/vision/identify`
anymore (`events/dispatcher.py` — only fall/identity/gesture/presence). So the live recognition path is
done. **This migration is about enrollment + persistence + query APIs + frontend rewiring, not
recognition.** Do not re-build identification.

### F2 — Embedding-space mismatch (the single biggest correctness risk). ⚠️
- **Pi** enrolls with **DeepFace `GhostFaceNet`** (`vision_service.py:25`), cosine-matched against
  `User.face_embedding`.
- **`video_process`** enrolls + identifies with **insightface ArcFace `buffalo_l` / `w600k_r50.onnx`**,
  512-dim, L2-normalized, aligned via YOLO-face 5-point landmarks
  (`face_identifier.py:149-194`, `face_match_cosine_threshold = 0.35` at `config.py:101`).

These are **different vector spaces**. Once enrollment moves to `video_process`, the Pi's existing
`face_embedding` values are **dead/incompatible** and the Pi's DeepFace path must not be the source of
truth. **Win:** enrollment and live identification will finally share one embedding space (both ArcFace),
which the current split never guaranteed. The enrollment script already proves the shared path
(`scripts/enroll_faces.py:84` uses `FaceIdentifier.embed_face`, the same call the live path uses).

### F3 — `video_process` has no web server, no DB, no auth, no user model.
Today it exposes only a **WebSocket overlay broadcaster** (`streaming/vision_ws_server.py`, port 5003) and
acts as an HTTP **client** to the Pi. The gallery is a **pickle file** `data/embeddings/faces.pkl` —
`dict[str, np.ndarray(N, 512)]` (`face_identifier.py:122-147`) — built **offline** by
`scripts/enroll_faces.py` from `data/enroll/<label>/*.jpg`. To become the enrollment backend it must gain:
(a) an HTTP API layer, (b) a writable persistence layer, (c) a runtime-reloadable gallery shared with the
camera thread, and (d) a decision on identity/auth (see D2). **This is the bulk of the work.**

### F4 — Runtime gallery is load-once and not thread-safe for writes.
`FaceIdentifier._load_gallery` runs only in `__init__` (`face_identifier.py:32`). The camera thread reads
`self._gallery` with no lock. Enrolling at runtime means appending to the in-memory gallery **and**
persisting it, from a *different* thread (the API request thread) than the reader (camera loop). Needs a
lock + atomic swap (see Backend §2.2).

### F5 — "Tracker Manager" is ambiguous; there are two.
1. `video_process/tracking/tracker_manager.py` — the live MOT/identity tracker (exposes `snapshot`,
   `person_snapshot`, `active_face_tracks`, `get_active_user` — `tracker_manager.py:497-560`).
2. The dashboard's **Identity Management** panel `frontend/src/components/UserManager.tsx` — lists enrolled
   users (`GET /users/list`) and enrolls new ones.

In "frontend dashboard" context the task means **#2 should query `video_process`** for (a) the enrolled
user list and (b) optionally live tracking state. `video_process` already publishes live tracking over WS
5003, so "real-time tracking data" is mostly a wiring change, not new compute.

---

## 2. Decisions Required (surface to user — do not silently pick)

> These flip the implementation. Recommended option first.

- **D1 — Persistence layer in `video_process`.**
  - **(A, recommended) SQLite via SQLModel** — one `faces.db` with `enrolled_user` + `face_embedding`
    tables; embeddings stored as BLOB (float32 bytes) or JSON. Gives real metadata, relations, deletes,
    multi-angle rows, audit. Mirrors the Pi's existing SQLModel idiom → familiar to the implementer.
  - (B) **Keep the pickle gallery** + a sidecar `metadata.json`. Smallest diff, but no relational queries,
    awkward concurrent writes, no per-embedding metadata.
  - **Recommendation: A.** The target explicitly says "database storage," and per-angle/per-user metadata
    needs structure. The pickle gallery is retained only as the **in-memory** runtime representation
    (rebuilt from the DB on load), preserving the existing `FaceIdentifier` read path.

- **D2 — Identity / auth provenance (who is this photo for?).**
  Today the Pi infers the username from a **JWT** (`user_router.py:17 get_current_user`); the browser never
  sends a name on `/users/register`. `video_process` has no auth/user concept.
  - **(A, recommended) Frontend sends an explicit `label`/`username` field** with the photo (the
    `add-guest` flow already does this — `UserManager.tsx:156`). `video_process` stays unauthenticated on
    the trusted LAN/Tailscale (same trust boundary it already uses for the Pi event ingress).
  - (B) **Pi proxies**: browser → Pi (`/users/register`, JWT) → Pi forwards bytes to `video_process`. Keeps
    JWT/ownership intact, but `video_process` is still the enroller. Adds a hop; Pi stays in the path.
  - **Recommendation: A** for the onboarding/dashboard direct-to-edge target; note **B** is the lower-risk
    path if JWT ownership (owner_id / guest roles, `user_router.py:76`) must be preserved. **This choice
    determines whether `video_process` needs any auth at all.**

- **D3 — What remains the Pi's job for identity?**
  After migration the Pi's `vision_service`/`face_recognizer` DeepFace enrollment is redundant (F2).
  - **(recommended)** Treat the Pi's face enrollment + `User.face_embedding` as **deprecated/dead**; the Pi
    keeps **users/rooms/devices/auth/voice** (`speaker_service` still owns voice — `user_router.py:37`).
    `video_process` owns **faces** only. Voice enrollment stays on the Pi (out of scope here).
  - The frontend's `/users/register` call currently sends **both** `image_file` and `audio_file` in one
    request (`Step3Biometrics.tsx:151-159`). Migration must **split** these: face → `video_process`,
    voice → Pi.

---

## 3. Frontend Architecture (`proactive-home-agent` — Next.js UI)

Two components capture faces today, with **identical** capture logic (front/left/right/up/down, 5 JPEG
blobs at quality 0.9):
- Onboarding: `frontend/src/components/onboarding/Step3Biometrics.tsx`
- Dashboard: `frontend/src/components/UserManager.tsx`

> **Target: 7 angles, one batch.** The capture flow expands from 5 → **7 angles**, and all 7 are sent in a
> **single** multipart `POST /enroll` request. Concrete default angle set (define once as a shared config
> constant so prompts/order are tweakable in one place):
> `front, left, right, up, down, upLeft, upRight`.
> The two new diagonals improve yaw×pitch coverage for ArcFace. The browser *mechanism*
> (`getUserMedia` → `capturePhoto` → `canvas.toBlob`) is unchanged; only the **state machine length
> (5→7)**, the **progress UI** (5→7 dots/steps), and the **dispatch** (now one batched request) change.

### 3.1 Onboarding refactor (`Step3Biometrics.tsx`)

**Current** (`executeFinalSave`, lines 127-201): captures 5 angles; sends the **front** image + audio to
`POST /users/register` (Pi), then loops the other 4 angles as separate `POST /users/register` calls (one
HTTP request *per angle* — 5 round trips). Username is implicit via JWT.

**Target — thin client:**
1. **Capture 7 angles, send all 7 in ONE multipart request** to `video_process`'s new enrollment endpoint
   (`POST {VISION_API}/enroll`). Package the 7 blobs (`front, left, right, up, down, upLeft, upRight`) as a
   single `files[]` array (each part named after its angle so the backend can tag `source_angle`) plus a
   `label` field (D2-A). This collapses **7 captures → 1 round trip** and lets the backend batch-embed all
   angles into one user record. Extend the capture state machine + progress UI from 5 to 7 steps.
2. **Split voice out:** keep the audio upload going to the Pi (`/users/register` with audio only), since
   voice stays on the Pi (D3). Two parallel calls: faces → `video_process`, voice → Pi.
3. New env var `NEXT_PUBLIC_VISION_API_URL` (the `video_process` REST base) alongside the existing
   `NEXT_PUBLIC_API_URL` (Pi). Default to the Tailscale/LAN host:port chosen in Backend §1.
4. Error/loading UX: the "saving" animation (lines 131-146, 4 fake stages) stays, but the real success/fail
   gate becomes the single `/enroll` response. On failure surface the backend's reason (e.g. "No face
   detected in 'left.jpg'") — the backend can return per-angle results so the user knows which angle to
   retake.

> The capture *primitives* (`capturePhoto`, `getUserMedia`, `canvas.toBlob`) are **unchanged**. What
> changes: the state machine grows 5→7 angles, the progress UI grows 5→7 steps, and `executeFinalSave`
> becomes a single batched `/enroll` request. Keep it surgical — don't touch the camera plumbing.

### 3.2 Dashboard "Tracker Manager" / Identity panel (`UserManager.tsx`)

This is the "Tracker Manager in the frontend dashboard" (F5 #2). Required changes:

- **Enrollment (`handleSave`, lines 144-203):** same refactor as onboarding — one batched multipart
  `POST {VISION_API}/enroll` with `label = name` + **7 angle files** in a single `files[]` array, instead of
  the per-angle loop against `/users/add-guest`. Extend the panel's capture state machine + progress dots
  5→7. Voice (if any) still to the Pi.
- **User list (`fetchUsers`, lines 44-57):** swap `GET {API_URL}/users/list` (Pi DB) →
  `GET {VISION_API}/enrolled` (the `video_process` enrolled-user list). This is the "fetch enrollment data
  from the backend API rather than local instances" requirement.
- **Delete (`handleDelete`, lines 61-72):** swap `DELETE {API_URL}/users/{name}` →
  `DELETE {VISION_API}/enrolled/{label}` so deletes hit the gallery's source of truth. (If the Pi must also
  forget the user for rooms/voice, issue both — but face truth lives in `video_process`.)
- **Real-time tracking (new, optional but requested):** to show *who is currently in frame*, subscribe to
  the existing WS overlay stream (port 5003) **or** add a lightweight `GET {VISION_API}/tracking/active`
  polling endpoint (Backend §1) that returns the current `snapshot`/`person_snapshot`. Recommend the REST
  poll for the dashboard list view (simpler than a WS client in this panel); reserve WS for live video
  overlay.

---

## 4. Backend Architecture (`video_process`)

Adopt a clean **router → service → repository** split (the task's separation-of-concerns requirement),
layered on top of the existing pipeline without disturbing the camera thread.

```
            HTTP (multipart photos / queries)
                       │
   api/routers/enrollment_router.py   api/routers/tracking_router.py
                       │                         │
   services/enrollment_service.py        (reads TrackerManager snapshot)
        │            │
  FaceIdentifier   repositories/face_repository.py  ──► SQLite (faces.db)
 (embed_face)               │
                     GalleryStore (in-mem, lock-guarded)  ──► FaceIdentifier._gallery
```

### 4.1 API / Routers (the new HTTP surface)

`video_process` needs an HTTP server. **Recommendation: FastAPI** (async multipart, Pydantic, mirrors the
Pi's stack → consistent for the implementer; the WS overlay server stays as-is on 5003). Run it on a new
port (e.g. `8800`) in its own thread/process alongside the camera loop, OR as a sibling ASGI app — see §4.4.

**Endpoints:**

| Method & path | Purpose | Request | Response |
|---|---|---|---|
| `POST /enroll` | Enroll/append a user from raw photos (one batch) | multipart: `label` (str) + `files[]` (**7 JPEGs**, named per angle; accept 1–7 so partial/retake batches still work) | `{label, embeddings_added, per_image:[{name, angle, ok, reason?}], total_embeddings}` |
| `GET /enrolled` | List enrolled users (replaces Pi `/users/list`) | — | `{users:[{label, num_embeddings, updated_at}]}` |
| `DELETE /enrolled/{label}` | Remove a user from gallery + DB | — | `{status, label}` |
| `GET /tracking/active` | Live tracking snapshot for the dashboard | — | `{faces:[{id,user,bbox}], persons:[...], ts}` |
| `GET /healthz` | Liveness/readiness | — | `{status, gallery_users, model_loaded}` |

Routers contain **no business logic** — they validate input, call a service, shape the HTTP response. Auth:
none if D2-A (LAN trust); add a shared-secret header if required.

### 4.2 Services (the workflow / business logic)

**`EnrollmentService`** — the core. Workflow for `POST /enroll`:
1. Decode each uploaded JPEG (`cv2.imdecode`), like `vision_service.register_face` does today
   (`vision_service.py:64-75`), incl. the max-width downscale.
2. **Detect → align → embed** using the **existing live path**: `FaceDetector.detect` → largest face →
   `FaceIdentifier.crop_keypoints` → `FaceIdentifier.embed_face` (512-dim, L2-normalized). This is
   *exactly* `scripts/enroll_faces.py:70-85` — **reuse that logic; do not fork a second embedding path**
   (guards against re-introducing F2). Skip images with no/low-quality face; record a per-image reason.
3. Persist embeddings + metadata via `FaceRepository` (§4.3), keyed by `label`, one row per angle
   (`source_angle` from the part name). A full enrollment yields **up to 7 embeddings/user**. Append for
   re-enrollment (multi-angle); pick a cap ≥ 7 per person (the Pi capped at 5 — `vision_service.py:104`;
   ArcFace can hold more, so 7+ is fine).
4. **Hot-reload the runtime gallery** so the camera thread identifies the new user immediately:
   `GalleryStore.upsert(label, embeddings)` under a lock, then atomically swap `FaceIdentifier._gallery`
   (resolves F4). No process restart.
5. Return per-image results so the frontend can prompt retakes.

**`TrackingService`** (thin) — reads `TrackerManager.snapshot` / `person_snapshot` (`tracker_manager.py:541-560`)
and maps to the API DTO for `GET /tracking/active`. Read-only; must respect the tracker's lock contract
(the manager already guards state; just read its public snapshot properties).

### 4.3 Repositories & Database (storage strategy)

**Recommendation (D1-A): SQLite via SQLModel**, file `data/embeddings/faces.db`. Two tables:

```
enrolled_user
  id            INTEGER PK
  label         TEXT UNIQUE NOT NULL      -- gallery key (== person name)
  role          TEXT DEFAULT 'user'       -- optional, parallels Pi roles
  created_at    DATETIME
  updated_at    DATETIME

face_embedding
  id            INTEGER PK
  user_id       INTEGER FK -> enrolled_user.id (ON DELETE CASCADE, indexed)
  vector        BLOB NOT NULL             -- 512 float32 = 2048 bytes (np.tobytes); JSON also fine
  dim           INTEGER DEFAULT 512
  source_angle  TEXT                      -- 'front'|'left'|... (from upload filename)
  created_at    DATETIME
```

**Why two tables (not the Pi's single JSON column):** the Pi stores all of a user's vectors in one JSON
`face_embedding` column (`models.py:104`, `List[float]` typed but used as list-of-lists). That blocks
per-angle metadata, per-embedding deletes, and clean caps. A child table is the right normalization and the
target explicitly asks how embeddings are "structured, linked, and saved."

**`FaceRepository`** — the only code that touches the DB. Methods (logic, not code):
`upsert_user(label) -> user`, `add_embeddings(user_id, vectors, angles)`, `list_users() -> [(label, count,
updated_at)]`, `get_all_embeddings() -> dict[label, np.ndarray(N,512)]` (used to build the gallery on
startup), `delete_user(label)`.

**Gallery linkage:** on startup `GalleryStore` calls `FaceRepository.get_all_embeddings()` and builds the
same `dict[str, (N,512) float32 L2-normalized]` structure `FaceIdentifier` already expects
(`face_identifier.py:122-147`) — so `FaceIdentifier`'s read path is **unchanged**; only its *source* moves
from pickle → DB-backed store. Keep `data/embeddings/faces.pkl` working as a fallback/export if desired, but
the DB is the source of truth.

**Migration of existing data:** the current `faces.pkl` (ArcFace) can be imported into the DB one-time.
The Pi's `User.face_embedding` (DeepFace/GhostFaceNet) **cannot** be imported (F2 — wrong space); those
users must be **re-enrolled** through the new flow.

### 4.4 Threading / process model (don't break the camera loop)

The camera loop is a tight `while` in a daemon thread (`main.py:118-238`); the WS server runs separately
(`main.py:80`). The FastAPI app must run **without blocking** the camera loop:
- Run uvicorn in its own thread (or separate process) started from `VisionPipeline.run` after
  `_ws_server.start()`.
- Share `FaceIdentifier` (its `GalleryStore`) and `TrackerManager` instances with the API layer by passing
  references at construction.
- **Concurrency contract:** the API thread *writes* the gallery; the camera thread *reads* it. All gallery
  mutation goes through `GalleryStore` lock + atomic swap (F4). Tracking reads use `TrackerManager`'s
  existing snapshot properties (already lock-safe). Mirror the project's rule: callbacks/reads off the
  camera thread never hold `TrackerManager._lock` longer than the snapshot copy.

---

## 5. Step-by-Step Implementation Plan (sequential, layered)

> Built bottom-up (repository → service → router → frontend) so each layer is testable before the next.
> Each step has a verify check.

**Phase 0 — Decisions & scaffolding**
1. Confirm **D1** (SQLite), **D2** (explicit `label`, no auth on LAN), **D3** (Pi face path deprecated,
   voice stays). → *verify:* decisions recorded; no code yet.
2. Add deps to `video_process` (`fastapi`, `uvicorn`, `sqlmodel`, `python-multipart`); create
   `api/`, `services/`, `repositories/` packages. → *verify:* `import` works; `ruff`/`mypy` clean.

**Phase 1 — Repository + DB (data layer)**
3. Define SQLModel `EnrolledUser` + `FaceEmbedding` tables (§4.3) and the SQLite engine at
   `data/embeddings/faces.db`. → *verify:* tables auto-create on first run; unit test inserts + reads back a
   512-float32 vector losslessly.
4. Implement `FaceRepository` (upsert/add/list/get_all/delete). → *verify:* unit tests for each method,
   incl. cascade delete.
5. One-time importer: `faces.pkl` → DB. → *verify:* existing gallery users appear via
   `FaceRepository.list_users()`.

**Phase 2 — Gallery store + identifier wiring (runtime)**
6. Introduce `GalleryStore` (lock + atomic swap) seeded from `FaceRepository.get_all_embeddings()`; have
   `FaceIdentifier` read its gallery through the store instead of the load-once pickle
   (`face_identifier.py:32`). → *verify:* live pipeline still identifies enrolled users exactly as before
   (regression check against current behavior).
7. Add `GalleryStore.upsert/remove` that updates memory **and** calls the repository. → *verify:* unit test:
   upsert → `FaceIdentifier._best_gallery_match` finds the new label without restart.

**Phase 3 — Services (business logic)**
8. `EnrollmentService.enroll(label, images)` reusing the `enroll_faces.py` detect→align→embed path (§4.2),
   persisting via repo + `GalleryStore.upsert`. → *verify:* unit test with sample JPEGs → embeddings added,
   per-image reasons returned, gallery updated.
9. `TrackingService.active()` mapping `TrackerManager.snapshot`/`person_snapshot` → DTO. → *verify:* returns
   current tracks while pipeline runs.

**Phase 4 — Routers + server (HTTP surface)**
10. FastAPI app + routers: `enrollment_router` (`POST /enroll`, `GET /enrolled`, `DELETE /enrolled/{label}`)
    and `tracking_router` (`GET /tracking/active`, `GET /healthz`). Routers call services only. → *verify:*
    `curl`/httpx integration tests for each endpoint.
11. Launch uvicorn in its own thread from `VisionPipeline.run` (§4.4), sharing identifier + tracker. →
    *verify:* API responds while camera loop runs; no FPS regression in `main.py` loop.

**Phase 5 — Frontend rewiring (thin client)**
12. Add `NEXT_PUBLIC_VISION_API_URL`. Refactor `Step3Biometrics.executeFinalSave` → single batched
    `POST /enroll` (faces) + separate voice POST to Pi. → *verify:* onboarding enrolls a user end-to-end;
    user immediately identified by the live pipeline.
13. Refactor `UserManager`: `handleSave` → batched `/enroll`; `fetchUsers` → `GET /enrolled`;
    `handleDelete` → `DELETE /enrolled/{label}`; (optional) live list via `GET /tracking/active`. →
    *verify:* dashboard lists/deletes against `video_process`; enrolled users persist across restart.

**Phase 6 — Cleanup / deprecation**
14. Mark the Pi's face enrollment (`vision_service.register_face`, `face_embedding` column, `/users/register`
    image branch) as deprecated; keep voice + users/rooms/devices/auth. Remove the per-angle loop dead code
    on the frontend. → *verify:* no frontend path still posts face images to the Pi; Pi no longer needed for
    face truth.
15. Docs: update `video_process/CLAUDE.md` (new REST surface, DB, ports) and the Pi README. → *verify:*
    onboarding works with the Pi's DeepFace face path fully bypassed.

---

## 6. Risks & Watch-items

- **R1 (F2):** any code that still reads the Pi's `User.face_embedding` for recognition will silently
  mis-match against ArcFace. Audit and disable. Re-enroll all users.
- **R2 (F4):** gallery write/read race. Gate *all* mutation behind `GalleryStore`'s lock + atomic swap;
  never mutate `FaceIdentifier._gallery` in place from the API thread.
- **R3:** camera-loop latency. Embedding 5 images is CPU/ANE work; do it on the **API thread**, not the
  camera thread. The camera thread only swaps in the finished gallery.
- **R4 (D2):** if D2-A (no auth) is chosen, `video_process`'s `/enroll` is an open write endpoint on the
  LAN/Tailscale. Acceptable only within the existing trust boundary; add a shared secret if exposed wider.
- **R5:** transport size. **Seven** full-frame JPEGs in one multipart request is notably larger than the
  current per-angle posts; confirm the new server's upload limit (FastAPI/uvicorn body size) and the
  frontend timeout. Consider downscaling each blob client-side before send (the backend already downscales
  to 640px wide — `vision_service.py:70-75` — so capturing/sending huge frames is wasted bytes).
- **R6:** voice/face split (D3) — ensure the onboarding "skip voice" / "skip face" branches still resolve
  now that the two go to different services.

---

## 7. Out of Scope (explicitly)

- Voice/speaker enrollment (stays on the Pi).
- Live identification algorithm (already local in `video_process` — F1).
- Rooms/devices/users/auth domains on the Pi (unchanged except the deprecated face path).
- go2rtc / streaming / fall / gesture pipelines (untouched).
```
