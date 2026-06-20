# WS2 Follow-up — Identified-Track Eviction TTL (occlusion flap fix) — Codex Handoff

> Single repo: `video_Process` (Mac edge pipeline). Surgical, additive fix. Default behavior for
> **unidentified** tracks stays byte-for-byte; only **identified** person tracks get a longer eviction grace.
> macOS / `uv run`. After the change: `uv run pytest`, `uv run ruff check <touched>`, `uv run mypy <touched>` green.

---

## Problem (why this change)

`person_left` fires from `TrackerManager._evict_stale_person_tracks()` when a body track is evicted from
`_person_tracks`. The eviction TTL today is:

```python
ttl = max(person_detection_interval * 2.0, tracker_track_buffer / tracker_frame_rate)
    = max(0.15 * 2, 30 / 30) = 1.0 s
```

This 1.0 s TTL is **shorter than BoTSORT's internal track survival window**. With `tracker_type="botsort"`
(ReID on), BoTSORT keeps a "lost" track alive for `~max_time_lost` update calls — roughly
`tracker_track_buffer * person_detection_interval ≈ 30 * 0.15 ≈ 4.5 s` of wall time — and re-assigns the **same
track_id** when the person reappears.

The mismatch causes a **flap** on brief occlusions (person walks behind furniture / at frame edge for 1–4 s):

1. BoTSORT stops outputting track 7 → our `last_seen` freezes → at 1.0 s we evict + **delete the record** →
   spurious `person_left` (an "exited" / energy-save action fires while the person is still in the room).
2. Person reappears within BoTSORT's buffer → BoTSORT returns the same track_id 7 → `_apply_person_results`
   re-creates a **fresh** record (`emitted_user=None`, `first_identified_at=None`) → next `propagate_identity`
   re-emits **`person_identified`**.

Net: `person_left → person_identified` churn for a person who never left. On the Pi, `"exited"` has **no
anti-spam** (only `"entered"` does), so this is user-visible (lights turned off while present).

**Fix:** keep an **identified** track's record alive long enough to cover BoTSORT's track-survival window, so a
brief occlusion refreshes `last_seen` on reappearance (identity state already preserved by
`_apply_person_results`) instead of evicting + re-identifying. Unidentified tracks keep the current short TTL —
they emit nothing, so evicting them fast is harmless.

> **Rejected alternative — debounce on `person_left`:** would require tracking pending exits and cancelling them
> on reappearance (extra state machine). The TTL approach is simpler and self-cancelling: reappearance refreshes
> `last_seen`, no pending state to manage. Prefer the TTL approach.

---

## Code anchors (verified)

- `config.py:108` `person_detection_interval = 0.15`; `:120-121` `tracker_track_buffer = 30`,
  `tracker_frame_rate = 30`; env-override block at `:213-220` (add the new override here).
- `tracking/tracker_manager.py:342-360` `_evict_stale_person_tracks()` — the only change site for the TTL logic.
- `tracking/tracker_manager.py` `PersonTrackRecord` carries `emitted_user` (set only for identified tracks) and
  `first_identified_at`; `_apply_person_results()` already **preserves** both across MOT updates for the same
  track_id (no change needed there).
- Tests: `tests/test_phase_a_contracts.py` — existing helper `_identity_manager(events)` and
  `test_identity_event_emits_left_with_dwell_on_evict` / `test_identity_event_does_not_emit_left_for_never_identified_track`
  are the patterns to mirror.

---

## Step 1 — `config.py`: new tunable

Add next to the Person MOT block (after `:121`):

```python
# Identified person tracks get a longer eviction grace so brief occlusions don't
# flap person_left → person_identified. Must exceed BoTSORT's effective track survival
# (~tracker_track_buffer * person_detection_interval ≈ 4.5 s). Unidentified tracks
# keep the short detection-cadence TTL.
person_identity_eviction_s: float = 5.0
```

Add the env override in `__post_init__` (next to `:217-220`):

```python
self.person_identity_eviction_s = _env_float(
    "PERSON_IDENTITY_EVICTION_S",
    self.person_identity_eviction_s,
)
```

(`_env_float` already exists and is used in this block.)

## Step 2 — `tracking/tracker_manager.py`: per-record TTL in `_evict_stale_person_tracks()`

Replace the single `ttl` + `stale_track_ids` comprehension (`:343-353`) with a base TTL for unidentified tracks
and a longer TTL for identified ones. Keep the rest (payload build under lock, emit after lock) **unchanged**.

```python
def _evict_stale_person_tracks(self, current_time: float) -> None:
    base_ttl = max(
        self._cfg.person_detection_interval * 2.0,
        self._cfg.tracker_track_buffer / max(1.0, float(self._cfg.tracker_frame_rate)),
    )
    identity_ttl = max(base_ttl, self._cfg.person_identity_eviction_s)
    pending: list[IdentityEventPayload] = []
    with self._lock:
        stale_track_ids = [
            track_id
            for track_id, record in self._person_tracks.items()
            if current_time - record.last_seen
            > (identity_ttl if record.emitted_user is not None else base_ttl)
        ]
        for track_id in stale_track_ids:
            record = self._person_tracks[track_id]
            if record.emitted_user is not None:
                dwell_s = max(
                    0.0,
                    record.last_seen - (record.first_identified_at or record.last_seen),
                )
                pending.append(
                    self._identity_payload("person_left", record, dwell_s=dwell_s)
                )
            logger.debug("[person:{}] MOT track expired — removing identity state", track_id)
            del self._person_tracks[track_id]

    for event in pending:
        self._emit_identity(event)
```

Only the TTL selection changed; eviction, dwell, payload, and emit-after-lock are identical to today.

> **Note on the slot-freeing side effect:** identified tracks now linger in `_person_tracks` up to
> `person_identity_eviction_s` after they truly leave. This is fine — these are real people, the dict is small,
> and `person_left` for a genuine departure is just delayed by ≤ ~5 s (still far faster and more precise than the
> old 20 s name-keyed timeout). Unidentified/ghost detections still clear in 1.0 s.

## Step 3 — Tests (`tests/test_phase_a_contracts.py`, mirror existing style, no sockets)

Add cases using the existing `_identity_manager(events)` helper. Set `manager._cfg = PipelineConfig()` (already
done in the helper) so `person_detection_interval=0.15`, `tracker_track_buffer/frame_rate → base_ttl=1.0`,
`person_identity_eviction_s=5.0`.

1. **Identified track survives the occlusion window:** record with `emitted_user="Ada"`, `last_seen=10.0`,
   `first_identified_at=4.0`; call `_evict_stale_person_tracks(current_time=13.0)` (gap 3.0 s, > base 1.0 s but
   < identity 5.0 s) → **no eviction, no event**, record still present.
2. **Identified track evicted past the long TTL:** same record, `current_time=16.0` (gap 6.0 s > 5.0 s) →
   **exactly one `person_left`** with `dwell_s ≈ last_seen - first_identified_at`, record removed. (Mirrors the
   existing `test_identity_event_emits_left_with_dwell_on_evict`, just with the longer threshold.)
3. **Unidentified track unchanged:** record with `emitted_user=None`, `last_seen=10.0`,
   `current_time=11.5` (gap 1.5 s > base 1.0 s) → **evicted, no event** (confirms short TTL still applies to
   unidentified tracks — guards against regressing the existing
   `test_identity_event_does_not_emit_left_for_never_identified_track` behavior).
4. **No-flap regression (record preserved across re-detection):** create an identified record, simulate an
   occlusion shorter than `identity_ttl` by NOT evicting, then re-run `_apply_person_results` with the same
   `track_id` and assert `emitted_user`/`first_identified_at` are unchanged and `propagate_identity` does **not**
   emit a new `person_identified`. (This mostly re-asserts existing preservation behavior; include it so the
   fix's intent is locked.)

## Step 4 — Verification

```bash
cd /Users/ogulcanozdemir/video_Process
uv run pytest tests/test_phase_a_contracts.py
uv run pytest
uv run ruff check config.py tracking/tracker_manager.py tests/test_phase_a_contracts.py
uv run mypy config.py tracking/tracker_manager.py tests/test_phase_a_contracts.py
```

All green. Then a manual smoke test (optional, flag on against a stub listener): stand in view → one
`person_identified`; step behind an obstacle for ~2–3 s and return → **no** `person_left`/`person_identified`
churn; actually leave the frame → one `person_left` after ≤ ~5 s with plausible `dwell_s`.

---

## Constraints (carry over from WS2)

- Surgical: only the TTL selection in `_evict_stale_person_tracks` + the one config field/env override change.
  Do **not** touch `propagate_identity`, `_apply_person_results`, the payload helpers, or the emit-after-lock
  discipline.
- Default-off identity events still hold: with `identity_events_enabled=False` nothing is POSTed regardless of
  TTL. This change only affects *when* `person_left` would be emitted, not whether it is sent.
- No new deps. Camera-thread/lock discipline unchanged (payloads built under `self._lock`, callback after release).
- `person_identity_eviction_s` is tunable via `PERSON_IDENTITY_EVICTION_S`; if real-world occlusions still flap,
  raise it (and/or revisit `tracker_track_buffer`) rather than reintroducing the short TTL.
```