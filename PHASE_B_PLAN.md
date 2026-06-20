# Phase B — Durable HTTP Event Delivery (Implementation Handoff)

> Repo: `/Users/ogulcanozdemir/video_Process`. macOS / Apple Silicon, `uv run` for everything.
> Phase A is complete and green. Phase B hardens **backend event delivery** only. Be surgical — one new module
> + a `dispatcher` refactor. After each step run `uv run pytest`, `uv run ruff check <touched>`,
> `uv run mypy <touched>` and keep them green (baseline 38 tests).

## Context

**Why.** Every backend event is a **fire-and-forget HTTP POST** on a 4-worker thread pool (`events/dispatcher.py`):
`send_fall_alert` (rich Phase-A schema), `send_gesture_event` and `send_presence` (bare C5 payloads),
`send_offline_signal`. There is **no retry, no delivery guarantee, errors are swallowed**. If the Pi backend
(`proactive-home-agent`) is unreachable for a moment (reboot / Wi-Fi blip), events — **including fall alerts** —
are lost silently.

**What.** `PHASE_A_PLAN.md §12` deferred *"MQTT/gRPC transport, unified envelope across event types, offline
spooling/retry hardening"* to Phase B. Per the latest scoping decision Phase B is **narrowed to HTTP hardening**:

1. A **disk-persistent SQLite outbox** so critical events survive process restarts and Pi downtime.
2. A **single background sender** that drains the outbox with **exponential backoff + reconnect replay**.
3. **Per-event-type policy** (fall = durable + replay; gesture = short TTL; presence = unspooled, latest-wins).

**MQTT / gRPC are OUT of Phase B** — reconsidered only after all gesture + fall-detection testing is complete.
The outbox row introduced here is designed as the reusable internal "envelope" so a future MQTT phase can publish
the same structured event without re-plumbing.

---

## Binding constraints (do not violate)

- **C1** CV stays on 720p `/stream2`; no 2K. **C2** don't touch `go2rtc/`. **C3** MediaPipe stays `Delegate.CPU`.
  **C4** fall stack (MediaPipe Pose + TFLite transformer + all fall constants) frozen.
- **C5 — wire payloads frozen.** The JSON **body** POSTed to `/vision/gesture` and `/vision/update_presence` must
  stay **byte-for-byte** identical to today. The fall body keeps its Phase-A §8 schema. **The outbox stores and
  replays the exact body string** — hardening changes *when/how reliably* we POST, never *what* we POST.
  Idempotency metadata rides in an HTTP **header**, not the JSON body, so C5 holds.
- **No new transport deps.** No `paho-mqtt`, no broker, no gRPC. `sqlite3` is stdlib; `requests` already present.
  One new module + a dispatcher refactor.

---

## Current behavior (what we are hardening) — `events/dispatcher.py`

| Method | Today | Phase B |
|---|---|---|
| `send_fall_alert(payload)` | `requests.post(fall_alert_url, json=payload, timeout=3.0)`, logs, swallows errors | **Durable enqueue** → background sender, retry until delivered or aged out |
| `send_gesture_event(...)` | `Session.post(gesture_url, ..., timeout=0.5)`, silent | **Short-TTL enqueue**; dropped if not delivered before TTL (stale command) |
| `send_presence(user)` | `Session.post(presence_url, ..., timeout=0.5)`, silent | **Unspooled, best-effort** (latest-wins); unchanged |
| `send_offline_signal()` | `requests.post(presence_url, ..., timeout=0.5)` at shutdown | **Unspooled, best-effort** direct post |

Call sites that must keep working unchanged (do **not** edit them):
- `main.py:181` `dispatcher.submit(dispatcher.send_fall_alert, fall_result.alert)`
- `main.py:152` `dispatcher.submit(dispatcher.send_presence, user)`
- `detection/gesture_recognizer.py:119` `dispatcher.submit(dispatcher.send_gesture_event, ...)`
- `main.py:89` `self._dispatcher.send_offline_signal()`; `main.py:92` `("dispatcher", self._dispatcher.shutdown)`

**Keep the public method signatures identical.** `submit()` may stay (enqueue is a fast SQLite insert) — either way
the method contract is unchanged.

---

## STEP 1 — `events/outbox.py` (new): durable SQLite outbox

Thread-safe, append-and-drain. **Pure** (no `requests`, no pipeline imports) so it unit-tests without a network.

**Schema** (`sqlite3`, `check_same_thread=False` + `threading.Lock`; WAL mode):
```
outbox(
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id          TEXT UNIQUE,     -- uuid4 hex; idempotency / dedup key
  event_type        TEXT,            -- "fall" | "gesture" | "presence"
  url               TEXT,            -- exact target URL
  body              TEXT,            -- exact JSON body string (C5: posted verbatim)
  created_wall      REAL,            -- time.time() at enqueue
  expires_wall      REAL,            -- created_wall + ttl, or NULL = never
  attempts          INTEGER DEFAULT 0,
  next_attempt_wall REAL             -- when this row is next eligible to send
)
```
> Use **wall-clock** (`time.time()`) for created/expires/next_attempt — monotonic clocks reset across the process
> restarts this outbox exists to survive.

**Methods:**
- `__init__(db_path)` — open/create DB, `PRAGMA journal_mode=WAL`, create table if absent.
- `enqueue(event_type, url, body, *, ttl=None, dedup_key=None) -> str` — insert one row; `event_id = dedup_key or
  uuid4().hex`; `expires_wall = now+ttl if ttl else None`; `next_attempt_wall = now`. **Dedup:** `INSERT OR IGNORE`
  on the `event_id` UNIQUE constraint (same key twice → one row). Returns `event_id`.
- `claim_due(now, limit=50) -> list[Row]` — rows with `next_attempt_wall <= now` AND (`expires_wall IS NULL` OR
  `expires_wall > now`), ordered by `id` (FIFO → falls replay in order).
- `mark_delivered(id)` — delete the row.
- `mark_failed(id, *, backoff_base, backoff_max)` — `attempts += 1`;
  `next_attempt_wall = now + min(backoff_max, backoff_base * 2**(attempts-1)) * jitter(0.8..1.2)`.
- `purge_expired(now) -> int` — delete rows past `expires_wall`; return count (logging).
- `pending_count() -> int` — diagnostics/tests.

**Tests** (`tests/test_outbox.py`, tmp db, injected fake clock):
- enqueue→claim_due returns it; mark_delivered removes it.
- TTL: expired row not returned by `claim_due`; `purge_expired` deletes it.
- backoff: `next_attempt_wall` grows per attempt, capped at `backoff_max`.
- dedup: same `dedup_key` twice → `pending_count()==1`.
- **restart replay:** enqueue, drop the object, reopen `Outbox(same_path)` → row still pending.

---

## STEP 2 — Refactor `events/dispatcher.py` to route through the outbox

Keep all four public methods + `submit`/`shutdown`. Internals:
- `__init__`: build `self._outbox = Outbox(cfg.outbox_db_path)`; `self._send_session = requests.Session()` (owned by
  the sender thread); start the sender thread (Step 3). Keep `self._executor` and `self._presence_session`.
- `send_fall_alert(payload)`: `self._outbox.enqueue("fall", cfg.fall_alert_url, json.dumps(payload),
  ttl=cfg.fall_event_ttl, dedup_key=payload.get("event_id"))`. **Do not POST here** — the sender owns delivery.
- `send_gesture_event(...)`: build the **exact same legacy dict** as today, then
  `self._outbox.enqueue("gesture", cfg.gesture_url, json.dumps(payload), ttl=cfg.gesture_event_ttl)`.
- `send_presence(user)`: **unchanged** — direct `self._presence_session.post(...)`, swallow errors. **Not** spooled.
- `send_offline_signal()`: **unchanged** — direct best-effort post.

**C5 guard:** the gesture dict must equal today's literal `{"gesture","user","location":"living_room","timestamp",
"duration"}` and presence the bare `{"user","status","location"}`. Add a regression test asserting the enqueued/POSTed
gesture body parses back to exactly the legacy keys/values.

---

## STEP 3 — Background sender (reconnect + replay)

A single daemon thread (not the pool) owning the network, in `events/dispatcher.py` or `events/sender.py`:
- Loop every `cfg.outbox_poll_interval` (e.g. 0.25 s) until a stop `threading.Event`:
  1. `purge_expired(now)` (log dropped count).
  2. `rows = outbox.claim_due(now)`.
  3. Per row: `self._send_session.post(row.url, data=row.body,
     headers={"Content-Type":"application/json","X-Event-Id":row.event_id}, timeout=3.0)`.
     **`data=row.body`** (already-serialized string) — never re-serialize, so the body is byte-for-byte (C5).
     `2xx` → `mark_delivered`; else → `mark_failed(...)`.
  4. Nothing due → sleep the poll interval.
- **Reconnect/replay is emergent:** while the Pi is down attempts fail and back off; rows persist; when the Pi
  returns the next `claim_due` drains the backlog FIFO. No explicit connectivity probe needed.
- `X-Event-Id` is a **header**, not body → C5-safe; lets the Pi de-dupe replays later (harmless until it does).
- **Shutdown** (`shutdown(wait)`): set stop event, join the sender (short timeout), `executor.shutdown(wait)`, close
  the outbox DB. `send_offline_signal()` is still called first by `main.py:89`.

---

## STEP 4 — Config + gitignore

`config.py` `PipelineConfig` additions (existing style):
```python
outbox_db_path: str = "data/logs/outbox.db"   # under already-gitignored data/logs/
outbox_poll_interval: float = 0.25
outbox_backoff_base: float = 1.0              # seconds
outbox_backoff_max: float = 60.0
fall_event_ttl: float = 3600.0                # falls durable up to 1h, then dropped+logged
gesture_event_ttl: float = 5.0                # stale gesture command dropped after 5s
```
- DB lives under `data/logs/` so existing `.gitignore` (`data/logs/*`) covers it. **Verify** the WAL side-files
  (`outbox.db-wal`, `outbox.db-shm`) are also under `data/logs/` (ignored). If a top-level path is used instead,
  add `data/*.db*` to `.gitignore`.
- No dependency changes (`sqlite3` is stdlib).

---

## STEP 5 — Tests

Add to `tests/` (mirror existing contract-test style; mock the network, never hit a real socket):
- Outbox unit tests (Step 1 list).
- **Dispatcher routing:** fall → durable enqueue; gesture → enqueue with `gesture_event_ttl`; presence → **not**
  enqueued (direct session post; `outbox.pending_count()==0`).
- **C5 regression:** capture the body the sender POSTs for gesture and presence; assert it deserializes to the exact
  legacy dicts (`location=="living_room"`, same keys).
- **Sender success/failure:** stub a session returning 200 → row deleted; 503 → row retained, `attempts==1`, future
  `next_attempt_wall`.
- **Restart replay e2e:** enqueue a fall, drop dispatcher, reopen on same DB → row still pending, sender delivers it.

---

## Out of scope (do NOT do now)

- **MQTT / gRPC / any broker** — deferred until all gesture + fall-detection testing is complete. Keep the outbox row
  reusable as the future "envelope".
- **Changing any wire payload** (C5).
- **Pi-side (`proactive-home-agent`) changes** — single-repo phase. `X-Event-Id` is forward-compat; the Pi need not
  honor it yet.
- **Spooling presence** — latest-wins best-effort is correct.
- Fall/gesture/pose model or threshold changes (C4); VisionState WebSocket (:5003) is untouched.

---

## End-to-end verification

```bash
cd go2rtc && docker compose up -d
cd /Users/ogulcanozdemir/video_Process
uv run pytest                         # baseline 38 + new outbox/dispatcher/sender tests
uv run ruff check events/ config.py && uv run mypy events/ config.py
uv run python main.py
```
Manual:
1. **Happy path:** Pi up → trigger gesture / scripted fall → backend receives it; `data/logs/outbox.db` drains empty.
2. **Pi down → replay:** stop the Pi listener, trigger a fall → row persists, sender logs backoff; restart the Pi →
   fall delivered within one backoff window, FIFO.
3. **Restart durability:** trigger a fall with the Pi down, `Ctrl-C`, restart `main.py` → queued fall replayed on
   startup.
4. **Stale gesture dropped:** Pi down, trigger a gesture, wait > `gesture_event_ttl`, bring Pi up → gesture **not**
   delivered (TTL drop, logged).
5. **C5:** capture POST bodies → `/vision/gesture` and `/vision/update_presence` bodies byte-for-byte identical to the
   pre-Phase-B baseline.
