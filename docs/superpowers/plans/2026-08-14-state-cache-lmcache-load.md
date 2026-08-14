# State-Cache LMCache Offload — wiring the load half

Date: 2026-08-14
Branch: `zejun/state-cache-lmcache-load` (= `ganyi/state_cache_offload` + `main`)
Reads with: [`2026-08-14-state-cache-lmcache-offload-final-design.md`](../specs/2026-08-14-state-cache-lmcache-offload-final-design.md)
— that document describes the spill half as built; this one finishes §11 of it.

## What this buys

Today `OFFLOAD_STATE=1` is write-only: every eviction costs a D2D, a D2H, a
staging row of HBM and LMCache capacity, and nothing reads the bytes back. The
resume it is meant to recover is the one `checkpoints_evicted` counts — a
request whose **KV prefix is still in HBM** but whose **state checkpoint was
spent** to admit somebody else. Without the tier that prefix is recomputed from
token 0.

## Scope, and the one thing deliberately left out

In scope: §11 items 1, 2, 3, 4, 6, 7, 8 of the final design.

Out of scope, deliberately: §11 item 5 and the §10 refusal it gates.
`_decide_load_after_alloc` keeps returning `per_req_cache_state_boundary` for
every `has_per_req_cache` sequence, so a hybrid still never loads paged KV from
LMCache. That is what makes the whole `P > L` hazard class unreachable here
rather than merely clamped: the state boundary comes from `_gated_hit`, which
walks `block_hashes` — and `can_allocate` builds that list out of **HBM
`kv.lookup` hits only**. So `P <= L` holds by construction, with no clamp in the
loop to get wrong. `clamp_state_boundary` and `_JointPark` therefore stay
uncalled; they are the tools for the day someone lifts §10, and lifting §10 is
its own project with its own proof obligation.

Consequence for the reader of §11.4: there is **no joint park** in this plan.
A hybrid has exactly one leg in flight, so parking on the pair would be a
`_JointPark` armed with `needs_kv=False` on every single request — structure
with no second case. The state load reuses the KV load's *lifecycle*
(`WAITING_FOR_REMOTE_KVS` → `finished_loading`/`failed_loading` →
`_is_offload_prefill_resume`), not its metadata.

### Deviation from §11.2, and why

The final design suggests carrying `(state_hash, target_group)` outbound on
`ScheduledBatch`, alongside `state_spill_pairs`. This plan puts it on
`LMCacheOffloadMetadata` instead — the request-keyed channel the KV load
already rides (`build_connector_meta` → `start_load_kv`). Three reasons, in
order:

1. **A batch that is never forwarded strands its payload.** The scheduler
   already guards `state_spill_pairs` with `if scheduled_seqs` for exactly this
   reason. A load is issued *precisely when* the requesting sequence is not in
   the batch — it is parked — so the empty-batch case is not an edge here, it
   is the common case.
2. **A load needs no compute-stream ordering.** The spill rides the batch
   because it needs a D2D copy issued on the forward's stream, ahead of that
   batch's checkpoint copies, and one event recorded after them.
   `StagedTransfer.unpack` synchronizes its own producing stream before
   returning, and the resuming forward is at least one scheduler step later, so
   a load has no fence to place.
3. **The completion is request-keyed.** It has to come back as
   `finished_loading`/`failed_loading` to unpark a sequence; a channel whose
   other half is already request-keyed is the one to send it out on.

The spill direction is untouched and keeps the batch channel.

## Global constraints

- **Default-off stays exact.** With `OFFLOAD_STATE` unset, every site added
  here is either not reached or returns immediately, and no test in the
  existing suite changes behaviour. The final design's §3 table stays true.
- **`indexed` ≠ `reachable`** (final design §6) is still the governing idea.
  `_attach_state_group`'s disown guard is **kept**, and gets a second caller:
  LMCache's LRU can drop bytes under an advertised hash, so a load that finds
  nothing must land somewhere safe.
- Every new env read goes through one function in
  `atom/model_engine/state_offload.py`, same rule as
  `state_offload_staging_groups()`.
- Tests first per task; each fix proven non-vacuous by reverting it and
  watching the expected failure.

---

## Task 0 — Reconcile the merge: two cushions, two meanings

**This is a real bug on the merged branch, not bookkeeping.** `main` gained
`STATE_CKPT_EXTRA_ENTRIES` (PR #1874, "V4 checkpoint capacity headroom"), whose
`state_pool()` does:

```python
if envs.is_set("STATE_CKPT_EXTRA_ENTRIES"):
    extra_entries = int(envs.STATE_CKPT_EXTRA_ENTRIES)   # OVERWRITES
```

and this branch passes the staging ring in through that same parameter
(`extra_entries=state_offload_staging_groups() * span`). Two failures follow:

1. Setting `STATE_CKPT_EXTRA_ENTRIES` **silently deletes the staging ring**.
   The arena loses the rows `state_entry_views(num_groups + slot)` addresses;
   GDN's slice clamps rather than raising, so the symptom is a short segment
   and a crash inside the packer, or a wrong-length entry.
2. The two cushions have **opposite admission semantics** and this branch gave
   them one. Checkpoint headroom is extra groups the state pool is *meant* to
   lease — that is the whole feature. The staging ring must never be leased.
   `admission_entries = entries - extra_entries` neuters #1874 outright:
   `config.pool_entries` is `admission_entries`, so `num_groups` falls back to
   `max_num_seqs` and the headroom disappears.

`tests/test_v4_sub_pool_spec.py::test_checkpoint_extra_entries_override_is_owned_by_state_pool`
fails on the merged branch today and is the alarm for #1.

**Fix.** Split the field:

- `SubPoolSpec.staging_entries` — allocated, never admissible. Owned by
  `state_pool()`, which computes it from `state_offload_staging_groups() *
  entries_per_req` (groups × multiplicity, per final design §5.3).
- `SubPoolSpec.extra_entries` — allocated **and** admissible, unchanged
  meaning, still overridable by `STATE_CKPT_EXTRA_ENTRIES`.
- `plan_pools`: `entries = per_req + extra + staging`,
  `admission_entries = per_req + extra`.
- `deepseek_v4_attn.py` and `gdn_attn.py` drop their `extra_entries=` argument
  and their local `state_offload_staging_groups()` import.

Tests (`tests/test_sub_pool_spec.py`, `tests/test_v4_sub_pool_spec.py`):

- the two cushions **add**, they do not clobber (the regression above);
- headroom is admissible, staging is not;
- the highest staging group's last row is still inside the tensor when both are
  set (the arithmetic `test_staging_groups_are_sized_by_multiplicity` pins,
  now with the ring sourced from the env rather than the call site);
- the existing V4 ownership test passes unmodified.

## Task 1 — The engine learns what a pending load is

`atom/model_engine/state_offload.py`:

- `StateOffloadIndex.request_load(req_id, h) -> bool` — record `req_id → h`,
  `loads_attempted += 1`. Refuses (False) if the hash is not in `hashes`.
- `complete_load(req_id)` / `fail_load(req_id)` — the second does
  `loads_failed += 1` and `forget(h)`, so the next request does not repeat an
  attempt at bytes LMCache has dropped.
- `abandon_load(req_id)` — the request went away before its bytes landed
  (abort, preempt). Neither a success nor a failure: the hash is still believed
  good, so it must **not** be forgotten.
- `stats()` grows `loads_completed`. The final design calls
  `loads_attempted`/`loads_failed` "structurally 0"; that sentence and its
  comment die here.

`atom/model_engine/state_pool.py`:

- `_resumable_from(h, tokens)` — the offload arm additionally requires
  `should_load_state(tokens, floor)`. `resumable_hit` already knows the
  boundary in blocks, so it passes `(i + 1) * hbs`.
  **Why the floor belongs here and nowhere downstream:** the scan stops at the
  first accepted boundary. Accepting a too-short offload rung and declining it
  later re-creates exactly the shadowing bug of §6 — the shorter *resident*
  rung the walk-back would have reached is never tried. Declining inside the
  predicate lets the scan keep walking.
- `STATE_OFFLOAD_LOADS_WIRED = True`, last (Task 7).

Floor: `OFFLOAD_STATE_MIN_LOAD_TOKENS`, read by
`state_offload_min_load_tokens()`. **Default 0** — deliberately unlike KV's
8192. A KV load's cost scales with the hit, so a short one moves little data
for a whole round trip; a state load moves one flat entry (53.6 MiB measured)
whatever the boundary is, and the round trip it replaces is a prefill
recompute that scales with the boundary. The floor is therefore a knob for
"my entry is huge and my hits are tiny", not a default posture. Documented
with that arithmetic so nobody copies the 8192.

## Task 2 — `BlockManager`: attach the group, record the load

`_attach_state_group`, the `src < 0` branch. Today it pops a fresh group and
returns `hit_hash == -1`. Now, before giving up on a positive `hit_hash`:

```
tier has this hash  ->  pop a group, request_load(seq.id, hit_hash),
                        seq.state_load_hash = hit_hash, return True
otherwise           ->  unchanged: return hit_hash == -1  (disown)
```

The target is a **real pool group**, not a staging entry (final design §5.4):
the bytes land where the resuming request will read them. `state_fork_src`
stays -1 — the loaded group *is* the incoming state, and both backends already
read their own group when there is no fork (`prepare_state_indices` maps
`non_spec_state_indices_in` to `base`, V4 copies nothing).

Also:

- `take_state_loads() -> list[(req_id, hash, group)]`, drained by the
  scheduler each pass.
- `deallocate` calls `abandon_load` for a sequence with a pending load, or the
  index leaks an entry per aborted request.

## Task 3 — `Sequence.state_load_hash`

One field, `-1` when nothing is in flight, restored by `deallocate`. It is what
the scheduler tests to decide to park, and what the failure path needs to know
a zeroed `num_cached_tokens` is owed.

## Task 4 — Scheduler: park, unpark, and the failure that must not be silent

- Admission loop, after `_notify_connector_after_prefill_alloc` and the KV
  `_confirm_remote_load_after_alloc`: a sequence with `state_load_hash >= 0`
  goes to `_park_for_remote_load`. Same backpressure accounting
  (`_count_inflight_load`) as a KV load, because it occupies a slot the same
  way.
- Success: the existing `_mark_offload_load_ready` path runs. `offload_loaded_
  tokens` is untouched by a state load, so its KV promotion block is skipped on
  its own condition — no new branch needed there, but `complete_load` and
  clearing `state_load_hash` do need one.
- **Failure is the part that must not be quiet.** `_consume_failed_remote_kv`
  must, for a sequence whose state load failed:
  `fail_load(req_id)`, `seq.num_cached_tokens = 0`, `offload_loaded_tokens = 0`,
  `state_load_hash = -1`. Skipping the zero is silent wrong output: the blocks
  are still claimed, `_is_offload_prefill_resume` sends the sequence straight
  to suffix prefill, and `has_initial_state` reads `num_cached_tokens > 0` over
  a group nobody filled. This is the same disown `allocate` does, one step
  later.
- `state_load_hash` is cleared on `preempt` too — the group goes back and the
  bytes would land nowhere.

## Task 5 — Scheduler-side connector: carry the loads out

- `LMCacheOffloadConnectorScheduler.enqueue_state_loads(pairs)` stashes them;
  `build_connector_meta()` moves them onto
  `LMCacheOffloadMetadata.state_loads` and clears the stash — the same
  drain-once discipline `_reqs_need_recv` has.
- `MultiConnectorScheduler.enqueue_state_loads` forwards to the first sub that
  implements it, matching how it already forwards `should_park_for_load_
  after_alloc`.
- The scheduler calls it through `hasattr`, like every other offload-only hook.

## Task 6 — Worker: submit, and never leave a request parked

- `StateOffloadTier.submit_load(req_id, h, entry_index)` — drop the `index`
  constructor parameter entirely. It was there for `loads_attempted` /
  `loads_failed` / `forget`, all three of which now live in the engine, and a
  worker-side index is not reachable from a spawned process anyway. Reporting
  by req_id is the whole contract.
- `start_load_kv` submits each state load.
- **If there is no tier, report the load failed.** A tier can legitimately
  refuse to build (final design §4: wrong connector, PP, no state backend,
  entry larger than the staging buffer) while the engine's ring exists and
  hands out loads. Silently dropping them parks the request forever. The
  refusal paths must therefore push the req_ids straight onto the failed set —
  which the engine already knows how to handle, because it is the same
  `failed_loading` an LRU miss produces.
- `get_finished` unions the tier's `(done, failed)` into
  `finished_loading` / `failed_loading`. No new `KVConnectorOutput` field: the
  aggregator's existing per-request quorum is exactly what a state load needs
  (every rank holds a shard; a load that only some ranks completed is not a
  load), and a hybrid has no KV load in flight to confuse it with — §10 is what
  makes that statement true, so it is repeated at the union site.

## Task 7 — Flip the gate, delete the write-only warning

- `STATE_OFFLOAD_LOADS_WIRED = True`, with its comment rewritten to say what it
  now guards rather than what is missing.
- Delete `BlockManager.__init__`'s write-only warning and
  `test_turning_the_tier_on_warns_that_it_is_write_only`, per final design §2.
  Keep `test_the_write_only_warning_is_silent_when_the_tier_is_off` in spirit:
  it becomes the default-off assertion.
- The control pair in `tests/test_state_offload_resume.py` inverts. A
  `loads_unwired(monkeypatch)` helper appears next to `loads_wired`, and
  `test_a_spilled_rung_does_not_shadow_a_resident_one` uses it — the property
  it pins (accepting an unreachable boundary shadows a resident one) is still
  the reason the predicate is written the way it is, and losing the test would
  lose the reason.
  `test_the_gate_is_the_only_thing_holding_the_spilled_rung_back` now asserts
  the boundary is **kept** and a load is pending, not that it is disowned.

## Task 8 — Observability and docs

- The periodic `state checkpoints:` line already prints whatever
  `checkpoint_fates()` returns, so the new counters appear for free. What needs
  writing is what they mean: `loads_attempted` vs `loads_completed` is the
  index's false-positive rate (LMCache's LRU dropped bytes the index still
  advertises), and it is the number that says whether the tier is sized right.
- `atom/kv_transfer/offload/README.md`, `docs/environment_variables.md`, and a
  successor to the final-design doc.

## Task 9 — GPU validation (handed to the user)

Not run here. The precondition test comes first and does not need the load
path at all: run the target workload with `OFFLOAD_STATE` **off** and read
`checkpoints_evicted` off the `state checkpoints:` line. Near zero → the pool
is big enough and this tier should stay off.

Then, with it on, the numbers that decide it:

| Read | Means |
|---|---|
| `state_offload_spills_dropped` / `spills_requested` | ring too shallow → raise `OFFLOAD_STATE_STAGING_GROUPS` |
| `state_offload_loads_completed` / `loads_attempted` | index false-positive rate (LRU dropped the bytes) |
| `checkpoints_evicted` before vs after | what the tier was supposed to recover |
| TTFT p50/p99 | the H2D is on the critical path of a parked request |

Project rules apply: `AITER_LOG_LEVEL=WARNING`, `rm -rf /root/.cache/atom/*`
before restart, confirm with `rocm-smi --showmemuse` (VRAM% > 0) rather than
`curl /health`.
