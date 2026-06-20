# WS2 — Located Identity → Pi Events (Authoritative Entry/Exit) — Handoff

> Two repos. **Part A** = `video_Process` (Mac edge pipeline, this repo). **Part B** = `proactive-home-agent`
> (Raspberry Pi 5 backend, `/Users/ogulcanozdemir/proactive-home-agent`, branch `local-autonomy`).
> Both sides ship **default-off** and are flipped on **together** — zero behavior change until coordinated.
> macOS / `uv run` for Part A. After each step: `uv run pytest`, `uv run ruff check <touched>`, `uv run mypy <touched>` green.

---

## Context — why this change

Located identity (the **person body track's** `user`) today travels **only** over the VisionState WebSocket
(`:5003`); the Pi never learns *who is in the room as a stable session*. The Pi currently infers occupancy from
`POST /vision/update_presence` heartbeats keyed by **name**, with two structural weaknesses:

1. **Exit is a 20 s timeout guess.** `continuous_presence_check()` (`vision_router.py:623`) polls every 15 s; if
   `now - last_seen > 20 s` (`presence_service.py` `check_timeouts`) it declares an EXIT and fires
   `trigger_agent_proactively(name, "exited")`. Departures are late and imprecise; dwell time is unknown.
2. **Name-keyed, flap-prone.** `active_people` is `Dict[name, {...}]` (`presence_service.py`). When a known user
   turns away and the face drops to `Unknown`, the body track on the Mac stays sticky but the Pi has no concept of
   "the same body" — occupancy logic is built on the volatile face name, not the durable track.

WS2 sends the Pi an **edge-debounced, body-track-keyed session signal**: one `person_identified` when a track is
first confidently named, one `person_left` (with **dwell**) when that track is evicted — and **never** on
`name → Unknown`. Per the chosen integration depth (**Authoritative entry/exit**), these become the *canonical*
entry/exit on the Pi: `person_identified` drives the "entered" greeting, `person_left` drives the "exited" action
with real dwell, and the legacy name+timeout path is **demoted to liveness** for identified users.

**Source of truth for the geometry of this signal already exists** and is correct:
- `TrackerManager.propagate_identity()` (`tracking/tracker_manager.py:374`) stamps a known face identity onto the
  containing person track and **only ever writes known names** (skips `Unknown`/`Identifying...`, line 384-386).
- `TrackerManager._evict_stale_person_tracks()` (`:326`) is the single, centralized point where a body track dies.
- `_apply_person_results()` (`:291`) preserves `user` across MOT updates (sticky identity, line 306-307).

So WS2 is **purely additive instrumentation** on two methods that already do the right thing.

---

# PART A — Mac edge pipeline (`video_Process`)

## A1. `config.py` — new fields + URL (mirror existing `pi_ip`/`pi_port` block at `:67-68`, `:159-172`)

```python
# Located-identity → Pi session events (WS2). Default-off: no Pi dependency until coordinated.
identity_events_enabled: bool = False
identity_zone: str = "living_room"
identity_source: str = "mac_studio_living_room"
```
Add a property next to `gesture_url` (`:171`):
```python
@property
def identity_event_url(self) -> str:
    return f"http://{self.pi_ip}:{self.pi_port}/vision/identity_event"
```
Add an env override in `__post_init__` (alongside `:204-207`) **only if** you want runtime toggling:
```python
self.identity_events_enabled = _env_bool("IDENTITY_EVENTS_ENABLED", self.identity_events_enabled)
```

## A2. `tracking/tracker_manager.py` — emit on identify + on evict

**(a) Extend the record** (`PersonTrackRecord`, `:19-26`) with two fields:
```python
first_identified_at: float | None = None   # wall time of first known-name stamp
emitted_user: str | None = None             # last identity already emitted as a session-start
```

**(b) Add a callback to `__init__`** (`:40`):
```python
def __init__(self, cfg: PipelineConfig, on_identity_event: Callable[[dict], None] | None = None):
    ...
    self._on_identity_event = on_identity_event
```
When `None` → log only (default). The callback **must be non-blocking** (see A3 — it submits to the pool).

**(c) Emit session-start in `propagate_identity()`** (`:374`). Today the only state change is
`record.user = user` (`:407`). Augment: when a track's identity becomes a *new known name* not yet emitted,
stamp `first_identified_at` (if unset), set `emitted_user`, and **collect** an event. A→B re-identification on the
same track (rare) emits a fresh `person_identified` for B.

> **Concurrency rule (critical):** `propagate_identity` and `_evict_stale_person_tracks` run on the **camera thread**
> and mutate `_person_tracks` under `self._lock`. **Build the event dicts inside the lock, append them to a local
> list, then call `self._on_identity_event(ev)` AFTER releasing the lock** — never invoke the callback while holding
> `self._lock` (the callback hops to the dispatcher pool, but keep the lock hold tight regardless).

```python
# inside propagate_identity, where record.user != user today:
if record is not None and record.user != user:
    record.user = user
    if record.emitted_user != user:          # None -> known, or A -> B
        if record.first_identified_at is None:
            record.first_identified_at = time.time()
        record.emitted_user = user
        pending.append(self._identity_payload("person_identified", record))
    logger.info("[person:{}] Identity propagated from face track: {}", matched_person_id, user)
# ... after the loop, lock released:
for ev in pending:
    self._emit_identity(ev)
```

**(d) Emit session-end in `_evict_stale_person_tracks()`** (`:326`). Before `del self._person_tracks[track_id]`
(`:339`), if `record.emitted_user` is set, collect a `person_left` with dwell, then emit after the lock:
```python
dwell_s = max(0.0, record.last_seen - (record.first_identified_at or record.last_seen))
pending.append(self._identity_payload("person_left", record, dwell_s=dwell_s))
```

**(e) Helpers:**
```python
def _identity_payload(self, event_type, record, dwell_s=None) -> dict:
    payload = {
        "schema_version": "1.0",
        "event_type": event_type,
        "track_id": record.track_id,
        "user": record.emitted_user or record.user,
        "zone": self._cfg.identity_zone,
        "source": self._cfg.identity_source,
        "ts_wall": datetime.now(timezone.utc).isoformat(),
    }
    if dwell_s is not None:
        payload["dwell_s"] = round(dwell_s, 1)
    return payload

def _emit_identity(self, event: dict) -> None:
    if self._on_identity_event is not None:
        self._on_identity_event(event)
    else:
        logger.info("[identity-event] {}", event)   # default-off: visible in logs only
```

**Invariant — never emit on `name → Unknown`:** there is no code path that does. `propagate_identity` never writes
`Unknown` (guard at `:384-386`); the only "end" is eviction. Turn-away / out-of-range keeps the sticky track alive,
so no event fires. ✔

## A3. `main.py` — wire the callback (non-blocking, pool-dispatched)

`TrackerManager` is built at `main.py:52`. Pass an adapter that pushes the POST onto the existing network pool so
the camera loop never blocks on HTTP:
```python
self._tracker_mgr = TrackerManager(
    cfg,
    on_identity_event=lambda ev: self._dispatcher.submit(self._dispatcher.send_identity_event, ev),
)
```
No other loop changes — `propagate_identity()` already runs once per loop (`main.py:202`) and eviction runs inside
`update_person_tracks()` (`main.py:177`).

## A4. `events/dispatcher.py` — `send_identity_event` (gated, default-off)

Add alongside the other senders (`:23-56`). Mirror the fall sender's shape; gate on the flag:
```python
def send_identity_event(self, payload: Mapping[str, object]) -> None:
    if not self._cfg.identity_events_enabled:
        return                                   # default-off: pipeline byte-for-byte unchanged
    try:
        resp = requests.post(self._cfg.identity_event_url, json=dict(payload), timeout=1.0)
        logger.info("Identity event sent: {} -> {}", payload.get("event_type"), resp.status_code)
    except Exception as e:
        logger.warning("Identity event send error: {}", e)
```
**C5:** brand-new endpoint/body — does **not** touch `/vision/gesture` or `/vision/update_presence` payloads.
**Phase B:** when the durable outbox lands, route this through it (identity events are durable like falls); for now
fire-and-forget on the pool is fine since the feature is off until coordinated.

## A5. Tests (`tests/`, mirror existing contract style, no real sockets)

- `propagate_identity` first known stamp → exactly one `person_identified` with `track_id`/`user`; second identical
  stamp → **no** new event; A→B → one new event for B.
- Stamp then evict → exactly one `person_left` with `dwell_s ≈ last_seen - first_identified_at`.
- A track that was **never** identified → eviction emits **nothing**.
- `name → Unknown` cannot occur via `propagate_identity` (assert it never emits on Unknown).
- `send_identity_event` with `identity_events_enabled=False` performs **no** POST (assert the session/`requests` mock
  is untouched); with it `True`, posts to `identity_event_url` with the exact payload keys.
- Callback is invoked **outside** the lock (no deadlock): a callback that calls back into a `TrackerManager` locked
  method must not hang.

## A6. Acceptance (Part A, with `identity_events_enabled=True` against a stub listener)

Approach < 1.5 m → identified → **one** `person_identified`. Turn away / step back (face → Unknown) → **no** event.
Leave frame (track evicted) → **one** `person_left` with plausible `dwell_s`. With the flag `False` (default),
**nothing** is POSTed and the pipeline is byte-for-byte unchanged.

---

# PART B — Raspberry Pi backend (`proactive-home-agent`, FastAPI)

> Files: app entry `backend/main.py` (`app.include_router(vision_router.router)`); routes
> `backend/api/routers/vision_router.py` (`APIRouter(prefix="/vision")`, `:32`); occupancy
> `backend/api/services/presence_service.py` (singleton); models `backend/database/models.py` (SQLModel/PostgreSQL);
> tables created in `backend/init_db.py`. No auth on `/vision/*` (trusted edge). Existing greeting/exit trigger:
> `trigger_agent_proactively(person_name, event_type)` (`vision_router.py:57`), anti-spam `ActionState.last_greeting_times`
> (900 s, `:52-56`), exit loop `continuous_presence_check()` (`:623`) → `presence_service.check_timeouts()`.

## B0. Safety flag — flip with the Mac

Add a backend config/env `IDENTITY_AUTHORITATIVE` (default **False**). All behavior changes below are **gated** on it.
While `False` the Pi behaves exactly as today even if the new endpoint receives traffic (it just logs/stores). Flip
`True` **only** together with the Mac's `identity_events_enabled`.

## B1. New model — `IdentityEvent` (Pydantic) + optional `OccupancySession` (SQLModel)

In `vision_router.py` near `PresenceEvent`/`GestureEvent` (`:34-46`):
```python
class IdentityEvent(BaseModel):
    schema_version: str = "1.0"
    event_type: str            # "person_identified" | "person_left"
    track_id: int
    user: str
    zone: str = "living_room"
    source: str = "mac_studio_living_room"
    ts_wall: str
    dwell_s: float | None = None
```
Optional audit table in `backend/database/models.py` (additive; add to `init_db.py`):
```python
class OccupancySession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user: str
    track_id: int
    source: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    dwell_s: Optional[float] = None
```
Persisting sessions gives true dwell analytics (the name+timeout model can't). Keep it optional — log-only is a valid
first cut.

## B2. New endpoint `POST /vision/identity_event` (authoritative entry/exit)

```python
@router.post("/identity_event")
async def handle_identity_event(event: IdentityEvent, background_tasks: BackgroundTasks):
    presence_service.log_identity_session(event)          # history_ledger + optional OccupancySession

    if not IDENTITY_AUTHORITATIVE:                          # gated; behaves as pure audit while off
        return {"status": "logged"}

    if event.event_type == "person_identified":
        # Canonical ENTRY. Seed name-keyed occupancy as identity-sourced so the
        # timeout loop won't double-handle its exit, then greet (respecting anti-spam).
        presence_service.mark_identity_present(event.user, event.zone)
        background_tasks.add_task(trigger_agent_proactively, event.user, "entered")

    elif event.event_type == "person_left":
        presence_service.clear_identity_present(event.user, event.zone)
        # Pass dwell into the agent context (see B4).
        background_tasks.add_task(trigger_agent_proactively, event.user, "exited", event.dwell_s)

    return {"status": "ok"}
```

## B3. Demote the legacy paths for **identified** users (keep them for Unknown/Guest)

Identity events fire **only for known users** (never `Unknown`). Guests/strangers still rely on the existing
`/vision/update_presence` + timeout path, so **do not delete it** — scope the demotion to identified users:

- **`presence_service.handle_detection`** (`:~85`): add a `source`/`identity` flag on `active_people[name]` entries
  created by `mark_identity_present`. When `IDENTITY_AUTHORITATIVE` and the name was entered via identity_event,
  `update_presence` ENTRY must **not** re-greet (it only refreshes `last_seen` → liveness). Unknown/Guest unchanged.
- **`presence_service.check_timeouts`** (`:~123`): for identity-sourced names, **do not use the normal 20 s timeout**
  — their exit is authoritative via `person_left`. **But do NOT fully disable the timeout for them either** (see B5):
  apply a long **safety backstop** instead (`identity_present_backstop_s`, default `300`). Unknown/Guest keep the
  normal 20 s timeout unchanged.

This is the "presence ENTRY → liveness only / timeout check → guarded" half of the chosen design, done **surgically**
so guest handling and the camera-offline path (`vision_router.py:598-605`) are untouched.

## B4. Dwell into the agent prompt (optional polish)

`trigger_agent_proactively` (`:57`) already computes "time away" for `entered` from `history_ledger`. Give it an
optional `dwell_s` param so the `exited` branch can say e.g. *"…was here for 40 minutes"* / drive energy-save logic
from a real number instead of the 20 s-timeout guess.

## B5. Ghost-occupant safety backstop (do NOT fully disable the timeout)

**The risk this guards against.** `send_identity_event` is currently **fire-and-forget** (no Phase B outbox yet,
timeout `1.0 s`). If a `person_left` POST is dropped — a momentary Wi-Fi/Pi blip exactly at eviction — there is **no
retry**. Meanwhile the Pi still has that user in `active_people` flagged identity-sourced, and B3 makes the timeout
loop *skip* them. Net result: the user is gone but the Pi believes they are **present forever** (a "ghost occupant"):
the `exited` action / energy-save never fires. The only things that would clear it are the next `person_identified`
for the same name or a `camera_offline` (clean shutdown clears `active_people`, `vision_router.py:598`). The old
system never had this failure mode because the 20 s timeout swept everything; authoritative mode deliberately removed
that safety net — so we put a weaker one back.

**The backstop.** Instead of *skipping* the timeout for identity-sourced users, give them a **long** one:
```python
# presence_service / config
identity_present_backstop_s: int = 300   # 5 min; normal exit comes from person_left long before this
```
In `check_timeouts`: identity-sourced entries expire at `identity_present_backstop_s` (not the 20 s default); when
one does, treat it as a normal timeout EXIT (log + `trigger_agent_proactively(name, "exited")`, no dwell). Normal
flow never reaches it — `person_left` fires within seconds of eviction — but a dropped `person_left` is reaped within
5 minutes instead of leaking forever.

**Phase B note.** This backstop is a *seatbelt*, not the fix. The real cure is making `person_left` **durable**: when
the Phase B SQLite outbox lands, route `send_identity_event` through it (identity events are durable like falls) so a
dropped `person_left` is replayed on reconnect and the ghost never forms. **Recommendation:** land Phase B before
flipping `IDENTITY_AUTHORITATIVE=True` in production; until then the 5 min backstop bounds the worst case.

## B6. Pi acceptance

With both flags on: a known user entering → **one** greeting from `person_identified` (not two); turning away →
**no** spurious exit/re-entry; leaving → **one** `exited` with correct dwell, fired immediately on track eviction
(not 20 s later). A guest (never identified) still enters/exits via the legacy presence+timeout path. A **dropped
`person_left`** is reaped by the 5 min backstop (B5), not leaked forever. With `IDENTITY_AUTHORITATIVE=False`, the
endpoint only logs and the backend is unchanged.

---

## Contributions / why this matters

1. **Precise departures + real dwell.** Exit fires the instant the body track is evicted, carrying measured dwell —
   replacing the 15 s-poll / 20 s-timeout guess. Faster goodbye / energy-save, and accurate "away for N minutes".
2. **Flap-proof greetings.** Occupancy is anchored to the **sticky body track**, not the volatile face name. Turning
   away (face → Unknown) no longer risks phantom exit→re-enter→re-greet cycles.
3. **Edge owns the truth.** The Mac already knows the exact identify and evict moments; the Pi stops *inferring*
   them. Less guesswork, fewer false agent wake-ups.
4. **Foundation for multi-occupant reasoning.** Per-`track_id` sessions don't collapse identical names the way the
   name-keyed model does — required for "two people in the room" logic and per-session analytics (`OccupancySession`).
5. **Safe, reversible rollout.** Both sides default-off and flip together; until then, byte-for-byte unchanged
   behavior on both repos. C5-clean (new endpoint, no touched payloads). Forward-compatible with Phase B (route the
   Mac sender through the durable outbox later).

## Binding constraints

- **C5** `/vision/gesture` and `/vision/update_presence` bodies stay byte-for-byte; this is a **separate** channel.
- **Default-off both sides**; no new heavy deps (`boxmot`/`requests`/`sqlite`/FastAPI already present).
- **Surgical**: every edit traces to a step above; don't refactor `propagate_identity`/`check_timeouts` beyond the
  additions; don't remove the legacy presence/timeout path (it still serves Unknown/Guest).
- Camera-thread safety: identity callbacks dispatched off-thread, never invoked under `self._lock`.

## End-to-end verification

```bash
# Mac (Part A)
cd /Users/ogulcanozdemir/video_Process
uv run pytest && uv run ruff check tracking/ events/ config.py main.py && uv run mypy tracking/ events/ config.py main.py
IDENTITY_EVENTS_ENABLED=true uv run python main.py     # against a stub listener first

# Pi (Part B)
cd /Users/ogulcanozdemir/proactive-home-agent/backend
# add IdentityEvent route + presence flags; IDENTITY_AUTHORITATIVE=false first (audit only), then true
uvicorn main:app --host 0.0.0.0 --port 8000
```
Manual: enter → one greeting; turn away → silence; leave → one exit + dwell; guest still works via legacy path;
**drop a `person_left` (block the Pi at eviction) → user is reaped by the 5 min backstop, not leaked (B5)**;
flags off → both sides unchanged.
