# WS2 Identity Events — Session Handoff

> Purpose: seed a **fresh chat** to continue WS2 without re-deriving context. Read this + the full plan at
> `WS2_IDENTITY_EVENTS_PLAN.md` (same repo, gitignored). Both files are untracked (`*_PLAN.md`).

## What WS2 is (one paragraph)

Send the Pi a **body-track-keyed, edge-debounced session signal**: one `person_identified` when a person body track
is first confidently named, one `person_left` (+ `dwell_s`) when that track is evicted — **never** on `name → Unknown`.
On the Pi these become the **authoritative** entry/exit, replacing the current name-keyed presence heartbeat + 20 s
timeout guess. Two repos, both **default-off**, flipped on **together**.

## Status — where we are

- ✅ **Analysis complete & grounded.** Both repos inspected (Mac via files; Pi `proactive-home-agent` via the
  claude-context index — `/Users/ogulcanozdemir/proactive-home-agent`, branch `local-autonomy`, the live copy; the
  `-1` copy is stale Nov-2025, ignore it).
- ✅ **Full handoff written:** `WS2_IDENTITY_EVENTS_PLAN.md` — Part A (Mac `video_Process`) + Part B (Pi). Includes
  the ghost-occupant **safety backstop** (B5) and Phase-B durability note.
- ⏳ **Not yet implemented.** No code written. Next step is handing the plan to the Codex agent.

## Decisions locked (do not relitigate)

1. **Integration depth = Authoritative entry/exit.** `person_identified` → canonical "entered" greeting;
   `person_left(dwell)` → canonical "exited"; legacy presence ENTRY → **liveness only** for identified users; the
   20 s timeout exit → **guarded** for identified users.
2. **Unknown/Guest keep the legacy presence + 20 s timeout path** (they never get identity events). Don't delete it.
3. **Default-off both sides**, flip together: Mac `identity_events_enabled` (default False), Pi
   `IDENTITY_AUTHORITATIVE` (default False). Until both on → byte-for-byte unchanged behavior.
4. **Ghost-occupant backstop:** for identity-sourced users, don't *disable* the timeout — set a **long** one
   (`identity_present_backstop_s`, default 300 s) so a dropped `person_left` is reaped in ≤5 min, not leaked forever.
   Real cure = route `send_identity_event` through the **Phase B durable outbox**; recommend landing Phase B before
   flipping authoritative on in prod.
5. **C5 honored:** brand-new `/vision/identity_event` endpoint; `/vision/gesture` and `/vision/update_presence`
   bodies untouched.

## Code anchors (verified)

**Mac — `video_Process`:**
- `tracking/tracker_manager.py`: `PersonTrackRecord` (`:19`); `__init__` (`:40`, add `on_identity_event` callback);
  `propagate_identity()` (`:374`, only writes known names — emit `person_identified` here); `_apply_person_results()`
  (`:291`, keeps `user` sticky); `_evict_stale_person_tracks()` (`:326`, single death point — emit `person_left` here).
- `events/dispatcher.py`: add `send_identity_event` (gated). `config.py`: add flag + `identity_event_url` near
  `:159-172`. `main.py:52` wire callback (pool-dispatched, **never** call back under `self._lock`).

**Pi — `proactive-home-agent/backend`:**
- `main.py`: `app.include_router(vision_router.router)`, lifespan (`:16-31`).
- `api/routers/vision_router.py`: `APIRouter(prefix="/vision")` (`:32`); `update_presence` (`:591`); `handle_gesture`
  (`:506`); `fall_alert` (`:358`); `trigger_agent_proactively(name, event_type)` (`:57`); anti-spam
  `ActionState.last_greeting_times` 900 s (`:52`); exit loop `continuous_presence_check()` (`:623`); `camera_offline`
  clears `active_people` (`:598`).
- `api/services/presence_service.py`: singleton; `active_people` name-keyed; `handle_detection` (`:~85`);
  `check_timeouts` (`:~123`, 20 s); `history_ledger`; `_update_db_last_seen`.
- `database/models.py`: SQLModel/PostgreSQL (add optional `OccupancySession`); `init_db.py` creates tables.

## Mac payload contract

```jsonc
{ "schema_version": "1.0", "event_type": "person_identified" | "person_left",
  "track_id": 7, "user": "meto", "zone": "living_room", "source": "mac_studio_living_room",
  "ts_wall": "2026-...Z", "dwell_s": 42.0 /* person_left only */ }
```
`dwell_s` is measured from **first identification** (`first_identified_at`), not track birth.

## Next actions (for the new chat)

1. Hand `WS2_IDENTITY_EVENTS_PLAN.md` **Part A** to Codex → implement on `video_Process`; validate with
   `identity_events_enabled=True` against a stub listener. `uv run pytest / ruff / mypy` green.
2. Then **Part B** on the Pi repo (separate, independent): new endpoint + presence demotion + backstop.
3. Keep both flags **off** until coordinated; consider Phase B outbox before authoritative-on in prod.

## Constraints carried over

C1 720p `/stream2` · C2 don't touch `go2rtc/` · C3 MediaPipe CPU · C4 fall stack frozen · C5 gesture/presence bodies
byte-for-byte · no new heavy deps · surgical edits only · camera-thread callback never under `self._lock`.
