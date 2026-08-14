# State-Cache LMCache Offload — the load half, as built

Date: 2026-08-14
Status: **implemented; spill and load both wired**
Branch: `zejun/state-cache-lmcache-load` (= `ganyi/state_cache_offload` + `main`)
Reads with: [`2026-08-14-state-cache-lmcache-offload-final-design.md`](2026-08-14-state-cache-lmcache-offload-final-design.md)
Plan: [`../plans/2026-08-14-state-cache-lmcache-load.md`](../plans/2026-08-14-state-cache-lmcache-load.md)

The companion document describes the spill half and is still accurate about it.
Its §2 ("write-only, on purpose"), §11 ("what wiring the load path requires")
and the `loads_*`-are-structurally-zero remarks in §7 are **superseded here**.

---

## 1. What changed, in one paragraph

`BlockManager._attach_state_group` used to have one answer when the HBM state
index missed a boundary the scan had accepted: disown it and recompute. It now
asks the offload tier first. On a yes the request keeps the boundary, takes a
pool group, and parks; the worker fetches the entry into that group; the
request wakes and prefills only the suffix. The resume this recovers is exactly
the one `checkpoints_evicted` counts — a request whose KV prefix is still in
HBM but whose state checkpoint was spent to admit somebody else.

## 2. Scope: the §10 refusal stays, and that is load-bearing

`_decide_load_after_alloc` still returns `per_req_cache_state_boundary` for
every `has_per_req_cache` sequence, so a hybrid never loads paged KV from
LMCache. That is not an unfinished edge; it is what makes the `P > L` hazard
class **unreachable** rather than merely clamped. With it in place the state
boundary comes from `_gated_hit` over `block_hashes`, and `can_allocate` builds
that list out of HBM `kv.lookup` hits only — so `P <= L` holds by construction,
with no clamp in the loop to get wrong.

Consequences, all deliberate:

- **There is no joint park.** A hybrid has exactly one leg in flight, so
  `_JointPark` would be armed with `needs_kv=False` on every request:
  structure with no second case. It and `clamp_state_boundary` remain uncalled,
  for whoever lifts §10.
- The scheduler **warns** rather than handles if a request ever has both legs
  pending, and drops the state leg. Two transfers resolved by one report would
  unpark a request while the second is still writing.

## 3. Deviation from the companion's §11.2

That document suggested carrying `(state_hash, target_group)` outbound on
`ScheduledBatch`, next to `state_spill_pairs`. It rides
`LMCacheOffloadMetadata.state_loads` instead — the request-keyed channel the KV
load already uses. Three reasons, in order:

1. **A batch that is never forwarded strands its payload.** The scheduler
   already guards `state_spill_pairs` with `if scheduled_seqs` for this reason.
   A load is issued precisely when its request is *not* in the batch — it is
   parked — so the empty batch is the common case, not an edge. (The engine
   dispatches `connector_meta_output` even for a zero-sequence batch, which is
   what makes this work; `_dispatch_idle_offload_work` was taught to count
   state loads as work for the same reason.)
2. **A load needs no compute-stream ordering.** The spill rides the batch
   because it needs a D2D copy issued on the forward's stream, ahead of that
   batch's checkpoint copies, with one event recorded after them. A load writes
   a group belonging to a parked request that no forward touches, and
   `StagedTransfer.unpack` synchronizes its producing stream before returning.
3. **The completion is request-keyed.** It has to come back as
   `finished_loading` / `failed_loading` to unpark a sequence.

The spill direction is untouched and keeps the batch channel.

## 4. The path, step by step

1. **Accept.** `resumable_hit` scans right to left; `_resumable_from(h, tokens)`
   accepts an offload-only hash when `STATE_OFFLOAD_LOADS_WIRED` is True, the
   tier holds it, and `tokens >= OFFLOAD_STATE_MIN_LOAD_TOKENS`.
2. **Attach.** `_attach_state_group` misses in HBM, pops a group, and calls
   `StateOffloadIndex.request_load(seq.id, h)`. That refuses a hash the index
   does not hold — believing is not delivering, and a load is only ever
   resolved by a report, so offering one for bytes no `get` can produce would
   park the request against a wake-up that never comes. The refusal falls
   through to the pre-existing disown.
3. **Park.** `seq.state_load_hash` is set; the scheduler parks the request in
   `WAITING_FOR_REMOTE_KVS` with `_count_inflight_load`, exactly as for a KV
   load.
4. **Publish.** `_publish_state_loads()` drains
   `BlockManager.take_state_loads()` into the connector, immediately before
   `build_connector_meta()`. A connector that cannot carry them fails them on
   the spot rather than leaving the requests parked.
5. **Fetch.** `LMCacheOffloadConnector._start_state_loads` submits each to
   `StateOffloadTier.submit_load`, on the tier's own executor — separate from
   the KV connector's two, though shared with this tier's own spills; see
   `submit_load` for why that split was not copied. A worker with no tier
   reports them failed.
6. **Report.** `_do_load` puts the request id in `_done` or `_failed`;
   `get_finished` merges those into `finished_loading` / `failed_loading`.
7. **Apply.** `Scheduler._update_from_kv_xfer_finished` calls
   `BlockManager.settle_state_load(req_id, ok)` — a no-op for the many ids that
   are plain KV loads — and queues the wake-up.
8. **Resume, or disown.** On success `_mark_offload_load_ready` clears
   `state_load_hash` and `_is_offload_prefill_resume` runs the suffix prefill
   with no re-allocation. On failure `_consume_failed_remote_kv` zeroes
   `num_cached_tokens`, which is the same disown `allocate` performs, one step
   later.

## 5. The three settlements, and why there are three

`StateOffloadIndex` distinguishes:

| | counter | hash |
|---|---|---|
| `complete_load` | `loads_completed` | stays indexed — a load reads LMCache, it does not consume it |
| `fail_load` | `loads_failed` | **forgotten** — the miss is the only evidence anyone gets that the LRU dropped the bytes |
| `abandon_load` | neither | stays indexed — an abort says nothing about the bytes |

Collapsing `abandon` into `fail` would forget a hash that is still perfectly
loadable and cost the next request over that prefix a full recompute.

**An abandoned load does not get its group back immediately.** A worker thread
is still writing it on its own stream, so `deallocate` parks the group in
`_orphan_load_groups` and `settle_state_load` releases it when the report
lands. Handing it to the next admission would deliver another request's state
after the fact, under a `has_initial_state` that is already true.

## 6. Where the floor lives, and why it is 0

`OFFLOAD_STATE_MIN_LOAD_TOKENS` is consulted inside `_resumable_from`, not at
the load site. A floor has to **decline** a rung so the right-to-left scan
keeps walking; accepting one and then skipping the transfer ends the scan on a
boundary nobody fills, which is the shadowing failure of the companion's §6.
The HBM branch is not gated by it — that hit costs no transfer, so there is
nothing to amortize.

The default is 0, deliberately unlike KV's 8192. A KV load moves bytes
proportional to the hit, so a short one spends a round trip moving very little.
A state load moves one flat entry (53.6 MiB measured) whatever the boundary is,
while the prefill it saves grows *with* the boundary. There is no length below
which the transfer is the expensive half. The knob is for an unusually large
entry, or an index with a bad false-positive rate.

## 7. Sizing: two cushions that must add

`main`'s `STATE_CKPT_EXTRA_ENTRIES` (#1874, V4 checkpoint headroom) and this
branch's staging ring arrived through the same `SubPoolSpec.extra_entries`, and
the env override *assigns*: setting the headroom silently deleted the ring, and
the arena lost the rows `state_entry_views(num_groups + slot)` addresses. They
also want opposite admission semantics — headroom is capacity the pool leases
out, the ring is a buffer the spill path writes into behind a request's back.

Split into two fields, both owned by `state_pool()` so each env var has exactly
one reader:

| field | allocated | admissible |
|---|:---:|:---:|
| `extra_entries` (`STATE_CKPT_EXTRA_ENTRIES`) | yes | **yes** |
| `staging_entries` (`OFFLOAD_STATE_STAGING_GROUPS`, gated by `OFFLOAD_STATE`) | yes | no |

## 8. What did not change

- The disown guard in `_attach_state_group`. LMCache's LRU can drop bytes under
  an advertised hash at any time.
- Spill ordering, the staging ring, the aggregator's hash/slot quorum, the PP
  refusal, the connector requirement, and the `entry_bytes > staging_bytes`
  refusal — all as the companion describes them.
- Default-off. With `OFFLOAD_STATE` unset every site added here is unreached.

## 9. Verification

- Full CPU suite: **1339 passed**, 1 pre-existing failure
  (`test_state_spill_copy_order`, an `aiter` import in this sandbox, fails
  identically on the base commit).
- `black` and `ruff` clean on every changed file.
- Each fix proven non-vacuous by reverting it and observing the expected
  failure — including the floor (two tests) and the sizing split (the merged
  tree's own `test_checkpoint_extra_entries_override_is_owned_by_state_pool`).
- **GPU end-to-end validation has not been run.** See the plan's Task 9 for the
  precondition measurement and the counters to read.
