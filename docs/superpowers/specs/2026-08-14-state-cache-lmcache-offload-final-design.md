# State-Cache LMCache Offload — Final Design (as built)

Date: 2026-08-14
Status: **the spill half, as built. The load half landed afterwards** — see
[`2026-08-14-state-cache-lmcache-load-as-built.md`](2026-08-14-state-cache-lmcache-load-as-built.md),
which supersedes §2 ("write-only, on purpose"), §11 ("what wiring the load path
requires"), and the `loads_*` row of §7. §2 is kept verbatim as the record of
what the spill-only branch was; every other section has been corrected in place
where the load half changed it, and still describes the tree.
Branch: `feat/state-cache-lmcache-offload`, base `805ae015`, 43 commits, 40 files, +5614/−249
Supersedes: [`2026-08-12-state-cache-lmcache-offload-design.md`](2026-08-12-state-cache-lmcache-offload-design.md)
(that document says "not yet planned or implemented" and describes intent, not
the built system — read this one instead)
Scope as built: DeepSeek-V4 and GDN/KDA (Qwen3-Next, Qwen3.5, Kimi-K3)

> **This is a handoff document.** It describes what exists in the tree today,
> what deliberately does not, and what the next agent has to do to finish the
> feature. Everything below was verified against the code on 2026-08-14. Where
> it names a symbol, that symbol exists; line numbers are deliberately avoided
> because they rot.

---

## 1. What this is, in one paragraph

A hybrid model (GDN/KDA recurrent state) or DeepSeek-V4 (compressor ring)
keeps one **per-request state** entry alongside its paged KV. `StateGroupPool`
holds a fixed number of these; when they run out, `pop()` spends the
least-recently-used checkpoint and its bytes were previously **discarded**.
That loss is exactly what the `checkpoints_evicted` counter measures. This
branch adds a tier beneath the pool: on eviction the checkpoint's bytes are
copied into a small staging ring inside the same HBM arena, packed by a worker
thread, and stored in LMCache under the checkpoint's ATOM block hash. The
engine then records the hash in an in-memory index. (This document stops
there; the load half that reads them back is described in the companion.)

The state cannot be rebuilt from cached KV — the KV cache holds the
compressor's *output* while the state is its rolling *input window* — which is
why the state needs a tier of its own rather than riding the existing paged-KV
offload.

## 2. Scope boundary: write-only, on purpose

`atom/model_engine/state_pool.py` declares:

```python
STATE_OFFLOAD_LOADS_WIRED = False
```

This is a statement about the code, not a policy knob. The spill direction is
wired end to end. The load direction has **no caller at all**:

- `StateOffloadTier.submit_load` exists and is never invoked.
- The worker builds its tier with `index=None` (`StateOffloadIndex` lives in
  the engine process, and only the spill path — which merely *reports* — runs
  in the worker).
- Nothing puts a `(state_hash, target_group)` pair on the outbound
  `ScheduledBatch`.
- No request-keyed state completion comes back to unpark a sequence.

Three pure helpers for the load path exist, are unit-tested, and have no
production caller: `clamp_state_boundary`, `_JointPark`, `should_load_state`
(all in `atom/kv_transfer/offload/state_tier.py`).

Consequently `OFFLOAD_STATE=1` today costs a D2D copy, a D2H, a staging row of
HBM and LMCache capacity per eviction, and buys nothing. The engine says so
out loud at startup (`BlockManager.__init__` emits a warning), and
`tests/test_state_offload_resume.py::test_turning_the_tier_on_warns_that_it_is_write_only`
pins it. **Delete that warning together with its test when loads land.**

## 3. The default-off guarantee

The standing requirement from the user was: *"you need to make sure that the
hybrid model works well with only paged_kv offload."* With `OFFLOAD_STATE`
unset the branch is behaviourally identical to before it:

| Site | With `OFFLOAD_STATE` unset |
|---|---|
| `state_offload_staging_groups()` | returns `0` |
| `SubPoolSpec.staging_entries` (V4, GDN) | `0` → arena identical, `admission_entries == entries` |
| `BlockManager.state_offload` | `None`; no ring installed, no warning |
| `StateGroupPool.offload` | `None`; `_spill()` returns immediately |
| `StateGroupPool._resumable_from(h, tokens)` | reduces to the pre-branch `h in self.hash_to_group` |
| `ScheduledBatch.state_spill_pairs` | empty list → `CommonAttentionBuilder.build()` skips the whole block |
| `LMCacheOffloadConnector._state_tier` | `None`; `_submit_state_spills` returns on the first probe |
| `KVConnectorOutput.state_*` | empty sets; additive fields, the four other connectors never set them |

The two changes that are *not* gated on the flag, and are correct
independently:

1. **`per_request_state=True`** on the hybrid `KVCacheTensor`s (`gdn_attn.py`,
   `kimi_mla_gdn_attn.py`), which makes `ATOMKVByteCodec` skip them. Slot-
   addressed recurrent state has no per-block stride; including it either
   failed the divisibility check or — worse, when the slot count happened to
   divide `num_blocks` — inflated `bytes_per_block` past what the backend's own
   `block_regions` describe. This is a **paged-KV-offload bug fix** that
   predates the tier's usefulness.
2. **`prefix_cache_hit_tokens` now sourced from `seq.num_cached_tokens`** at
   all three scheduler sites, instead of `num_cached_blocks * block_size`. The
   old form under-reported by the DCP factor and did not see the disown path.

## 4. Configuration

| Env | Default | Meaning |
|---|:---:|---|
| `OFFLOAD_STATE` | `0` | Enable the tier. Truthy **by spelling**: `0/false/no/off`, empty, and whitespace-padded variants are off; `1/true/yes/on` are on, stripped and lowercased. |
| `OFFLOAD_STATE_STAGING_GROUPS` | `1` | Ring depth K, in **groups** not entries. Non-integer → warn, use 1. Negative → warn loudly, use 0 (tier off despite the flag). |

Both are read through the single function `state_offload_staging_groups()` in
`atom/model_engine/state_offload.py`. The arena sizing, the pool wiring, and
the worker-side tier construction all call that one function — a second
`os.environ` read anywhere would let the engine queue spills into a ring the
runner never drains.

Two env vars the original plan sketched are **not implemented and have no
effect**: `OFFLOAD_STATE_WORKERS` (only the `max_workers=1` default on
`StateOffloadTier`). `OFFLOAD_STATE_MIN_LOAD_TOKENS` was in that list too and
no longer is — it is read by `state_offload_min_load_tokens()` and consulted in
`_resumable_from`; see the load document.

### Refusals — the tier declines to build, loudly, in four cases

1. **Wrong connector.** `kv_connector_hosts_state_tier()` requires
   `kv_connector: "lmcache_offload"` (or a `multi` that lists it). Against
   anything else the ring would hand out slots and never get one back.
   `BlockManager.__init__` warns and installs nothing.
2. **Pipeline parallelism.** `pipeline_parallel_size > 1` → warn and return
   from `_maybe_build_state_tier`. See §8.
3. **No state backend.** `transfer_tensors.state_backend` is `None`, or the
   builder raises `NotImplementedError`/`AttributeError` from
   `state_entry_views`. `IndexError` is deliberately **not** caught — it means
   group 0 does not exist, i.e. a zero-entry state pool with the tier on,
   which is a sizing bug that must be loud.
4. **Entry larger than the staging buffer.** One state entry is MB-scale
   (**53.6 MiB measured** on the real model) while the shared GPU staging
   buffer is sized for KV chunks. If `entry_bytes > gpu_staging_buffer_bytes`,
   every spill would raise inside `StagedTransfer.ensure_buffer` and
   `_do_spill`'s broad `except` would turn it into a warning plus a slot
   release — a tier that looks healthy and stores nothing, forever. Refuse
   instead, naming both numbers. Raise `OFFLOAD_GPU_STAGING_CHUNKS` /
   `OFFLOAD_GPU_STAGING_MAX_BYTES` to fix.

## 5. Architecture

### 5.1 Process topology

```
ENGINE PROCESS                          SPAWNED RUNNER PROCESS (one per TP rank)
──────────────                          ────────────────────────────────────────
Scheduler                               ModelRunner
BlockManager                            AttentionMetadataBuilder  (state_entry_views)
  └ StateGroupPool  (_spill, ring copy) LMCacheOffloadConnector
  └ StateOffloadIndex (hashes, slots)     └ StateOffloadTier   (lmc-state thread)
                                              └ StateByteCodec → LMCache

  outbound:  ScheduledBatch.state_spill_pairs   ──────────────►
  inbound:   ◄────── KVConnectorOutput.state_*  (via KVOutputAggregator)
```

The engine owns all bookkeeping; the runner owns all bytes. Neither can touch
the other's half directly, which is why the worker **reports and the engine
applies**.

### 5.2 The spill, step by step

1. **Eviction.** `StateGroupPool.pop()` has no free group, so it takes the LRU
   checkpoint. Before `invalidate()` clears `group_hash[group]`, it calls
   `_spill(group)` — the hash is the key the tier stores under, and it is
   about to be erased.
2. **Reserve.** `_spill` asks `StateOffloadIndex.request_spill(h, group)` for a
   staging slot. Full ring → `-1`, `spills_dropped += 1`, the spill is dropped
   and `checkpoints_evicted` still counts it. **Dropping is identical to the
   no-tier behaviour, so there is no regression to reason about.** This is a
   deque pop and nothing else: `pop()` is on the scheduler's critical path.
3. **Queue the copy.** On success `_spill` appends `(group, slot)` to
   `_spill_copies`; the index separately holds `(hash, slot)` in `_pending`.
4. **Join and publish.** `BlockManager.state_spills_for_batch()` joins the two
   lists on the slot and emits `(src_group, dst_entry, staging_slot, hash)`,
   where `dst_entry = cache.num_groups + slot`. The join happens engine-side
   because `num_groups` is authoritative there. A half-present slot (in one
   list, not the other) is never observed in correct operation; it is warned
   about and released rather than guessed at.
5. **Travel.** The tuples ride `ScheduledBatch.state_spill_pairs` to the
   runner. `Scheduler` populates it at all four batch-construction sites
   (prefill, decode, `DecodeScheduler`, and the guarded decode path where
   `scheduled_seqs` may be empty — draining into a batch that is never
   forwarded would strand every slot it drained).
6. **Copy, then submit — in that order.** `CommonAttentionBuilder.build()` is
   the single point every batch shape passes through exactly once (prefill,
   decode, dummy, DP-sync, PP microbatch, TBO). It issues the spill copies
   **before** the checkpoint copies. This ordering is load-bearing: `pop()`
   spills a group and then hands that same group out as a checkpoint
   *destination*, so one group is routinely both this batch's spill source and
   this batch's copy destination. Copies first would store the new occupant's
   bytes under the evicted checkpoint's hash — present, valid-looking, and
   someone else's, undetectable on any load path.
7. **Fence and pack.** One `torch.cuda.Event` is recorded on the compute
   stream after all the copies, and handed to every `submit_spill` of the
   batch. The `lmc-state` worker synchronizes on it before packing — it packs
   on its own stream and would otherwise read the staging entry's previous
   occupant. Same shape as the KV path's `save_ready_event`.
8. **Store.** `StateByteCodec.put(h, entry_index)` allocates one flat uint8
   `MemoryObj` of `entry_bytes`, packs every view into it via
   `StagedTransfer.pack`, and does `batched_put([key], [obj])`.
9. **Report.** `_do_spill` records the hash in `_indexed` **or**
   `_index_failed`, and the slot in `_released` — always, stored or not,
   because a leaked slot shrinks the ring permanently.
10. **Aggregate.** `LMCacheOffloadConnector.get_finished` drains the three sets
    onto `KVConnectorOutput`. `KVOutputAggregator` waits for **all** TP ranks
    (see §5.5).
11. **Apply.** `Scheduler._update_from_kv_xfer_finished` calls
    `confirm_spill(h)` for each indexed hash, **then** `release_staging(slot)`.
    Index before release, always: the reverse order opens a window where the
    slot is reusable but its hash is not yet findable.

### 5.3 Spill by copy, not by pin

The staging ring is K groups appended to the state arena past the pool's own
range. `SubPoolSpec.staging_entries` allocates them; `PoolPlan.admission_entries`
withholds them from the BlockManager, so no request is ever handed one.
Backends keep sizing their tensors from `pool_plan.entries` (the allocation
count) while admission reads `admission_entries`.

Why copy rather than pin the evicted group until its D2H lands: `pop()` is
called **precisely when there is no free group to withhold**. Withholding one
is a deadlock shape. So the spill takes a copy, `pop()` hands the original out
immediately, and the ring bounds the cost at K groups of HBM. The original
2026-08-12 spec flagged this as "the least-settled part of the design"; it is
now settled, implemented, and tested (`tests/test_state_spill_copy_order.py`).

`staging_entries` is counted in **groups**, and `state_pool()` multiplies by
the class's own `entries_per_req` (the backends no longer pass it themselves;
`extra_entries` is a different cushion, `STATE_CKPT_EXTRA_ENTRIES`, and it *is*
admissible):

- V4: `entries_per_req=1`, so `staging_entries = K * 1`.
- GDN: `entries_per_req = span = 1 + num_spec`, so `staging_entries = K * span`.
  Sizing this in bare entries runs the last staging group off the end of the
  tensor — and `cache[layer, lo : lo + span]` **clamps** rather than raising,
  so the failure is a short segment, an `entry_bytes` mismatch, and a crash
  inside the packer. `tests/test_state_entry_views.py` pins the full span.

### 5.4 Two index spaces — do not conflate them

| Name | Space | Who uses it |
|---|---|---|
| `entry_index` | `state_entry_views` index; staging entries live at `num_groups + slot` | `StateByteCodec.put/get`, `state_entry_views` |
| `staging_slot` | ring slot, counts from 0 | `StateOffloadIndex.request_spill` / `release_staging` |

`StateByteCodec`'s parameter is called `entry_index` and not `group` for
exactly this reason: on the spill path it is a staging-ring entry, on the
load path it is a real pool group. Both resolve through the same
`state_entry_views` space.

### 5.5 Aggregation across TP ranks

`KVOutputAggregator` keys the state reports by **slot** and **hash**, not by
request id — the request that owned a spilled checkpoint is long gone by the
time its bytes land. All-ranks, not first-rank-wins, for two distinct reasons:
every rank packs its own shard out of the *same* staging slot (so the slot is
reusable only after the last rank's D2H), and a hash is loadable only if every
rank stored its shard (a load reads all of them back).

Quorum for a hash is taken on `state_indexed | state_index_failed`. That third
set exists **solely to resolve quorum** and is never forwarded to the engine:
without it a partial store (some ranks stored, some failed) would pin the hash
in the aggregator forever. A hash reaches `state_indexed` only if no rank
failed; partial shards are dropped silently and left to LMCache's LRU (the
engine never indexed them, so they are unreachable).

`MultiConnector` unions the three sets like the load/recv sets — there is no
pairing to withhold them for — and re-exposes the one sub-connector's
`_state_tier` on itself, because `_submit_state_spills` reads that attribute
off whatever connector the forward context holds. Two sub-connectors with a
tier is a `ValueError` at model load, not an arbitrary pick: the spill would go
to one tier and a future load could ask the other.

### 5.6 The key

`StateByteCodec.key(h)` builds
`CacheEngineKey(model_name, world_size, worker_id, h, torch.uint8)` with
`worker_id = tp.rank_in_group`. The ATOM hash goes in **unmodified** —
`_resumable_from(h, tokens)` looks up the same integer in HBM and in this tier, so
hashing, salting or stringifying it here would make the two branches ask
different questions.

`ChunkedTokenDatabase` is deliberately bypassed. State has a token range
(`[0, pos)`, which is why it can share KV's key) but its bytes cannot be sliced
by token — there is no such thing as "the first three chunks hit". Chunking
would produce N keys useful only all together, and would multiply the
partial-invalidation chance. It also sidesteps LMCache's chunk-alignment loss.

## 6. The defining bug class: "indexed" ≠ "reachable"

This is the one idea a new agent most needs. `StateGroupPool.resumable_hit`
scans block hashes **right to left** and stops at the **first** boundary
`_resumable_from` accepts. So accepting a boundary whose bytes nothing can
deliver does not merely waste a lookup — it *shadows* every shorter checkpoint
still resident in HBM that the walk-back would have reached. A resume thrown
away, growing with spill volume.

That is why `_resumable_from` gates the tier's `hashes` set behind
`STATE_OFFLOAD_LOADS_WIRED`. The flag is True today; with it False the
predicate is exactly the HBM-only test it was before the branch. `hashes` is
populated either way — the spill leg is live — so the flag changes *who may
vote on a hit*, not what is recorded.

Two tests are the control pair for this, and they must both survive any change
to the flag:

- `test_a_spilled_rung_does_not_shadow_a_resident_one` — the gate closed
  (`loads_unwired`): the shorter, resident rung wins.
- `test_the_gate_is_the_only_thing_holding_the_spilled_rung_back` — loads
  wired: the rightmost boundary wins again. This proves the first test passes
  because the spilled rung is *unreachable*, not because the scan stopped
  preferring the right.

**The second-line guard.** `BlockManager._attach_state_group` now returns a
bool, and `allocate` sets `seq.num_cached_tokens = 0` when it is False —
disowning the boundary so the forward recomputes over the already-claimed
blocks (which are kept; dropping them would leak a reference). This guard is
**not** made redundant by wiring loads: LMCache's own LRU can drop bytes under
a hash the index still advertises, so a load that finds nothing lands in
exactly this branch. Without it, a recycled group plus `num_cached_tokens > 0`
makes `has_initial_state` True over another request's leftovers — silent wrong
output, no exception.

## 7. Observability

The scheduler emits a `state checkpoints:` line every 100 ticks when any
counter is non-zero. `BlockManager.state_checkpoint_fates()` sums by key across
every state class, which is why the tier's counters are prefixed
`state_offload_` (an unprefixed `indexed` would collide with anything a future
pool names the same).

| Counter | Read it as |
|---|---|
| `checkpoints_evicted` | **The precondition.** Near zero → the pool is big enough and the tier is pure overhead. Do not enable. |
| `checkpoints_orphaned` | Partial signal — an orphaned checkpoint is spilled only when `kv_offload_enabled`. |
| `checkpoints_dropped` | No bytes to spill. |
| `state_offload_spills_requested` | Spills that got a slot. |
| `state_offload_spills_dropped` | Ring too shallow for the eviction rate → raise `OFFLOAD_STATE_STAGING_GROUPS`. **The starvation warning only fires after 256 consecutive drops, so a ring dropping one in three is silent there and visible only here.** |
| `state_offload_indexed` | Hashes every rank stored. |
| `state_offload_loads_attempted` / `_completed` / `_failed` | The load leg, added later; `failed / attempted` is the index's false-positive rate. See the companion document. |

Two latched warnings guard the contract that would otherwise fail silently:
`StateGroupPool` warns once if `take_spill_copies()` is not drained every step
(an exact leak detector — more than `staging_depth` outstanding copies is
proof), and `StateOffloadIndex` warns once after 256 consecutive drops with no
slot returned. Both are diagnoses, not repairs.

## 8. Pipeline parallelism is refused outright

`_maybe_build_state_tier` warns and returns when `pipeline_parallel_size > 1`.
Two **independent** reasons — fixing either alone leaves the other:

1. **The key has no PP component.** `worker_id = tp.rank_in_group`. Every PP
   stage holds a different slice of the layers, so stage 0 and stage 1 at the
   same TP rank write different bytes under an identical key and clobber each
   other. A later load restores one stage's state into another's layers: wrong
   output, no error.
2. **The reports are never drained.** `pp_engine_core.py` has every stage call
   `forward` on the head-pickled batch, so `_submit_state_spills` fires on all
   of them — but only the head runs `_poll_kv_transfer_progress`. Non-head
   stages' `state_staging_released` reports never reach the engine, the ring
   never gets those slots back, and the tier silently stops spilling after
   `staging_depth` evictions.

Lifting this needs a PP component in the key **and** a report path from the
non-head stages. It is a scheduler change, not a key change. Paged-KV offload
is unaffected — the guard returns before the tier exists and touches nothing
else.

## 9. HBM accounting

`StagedTransfer` keeps its staging buffers in `threading.local`, so handing the
tier the KV connector's `StagedTransfer` object does **not** share the buffer.
The tier packs on its own `lmc-state` worker and therefore adds one more
resident buffer per rank:

```
resident_HBM ≈ (1 load + OFFLOAD_COPY_WORKERS save
                + 1 if OFFLOAD_STATE else 0) * per_buffer_bytes
```

≈ **67 MiB** total at defaults, or ≈ **100 MiB** with the tier on. What reusing
the object buys is a single place the bound is configured — which is what makes
the `entry_bytes > staging_bytes` refusal in §4 meaningful.
`tests/test_staged_transfer.py::test_one_staged_transfer_still_means_one_buffer_per_thread`
pins this property, because both `connector.py` and the README's formula depend
on it.

On top of that, the ring itself costs **K groups of state-pool HBM** — out of
the very budget a small pool is the problem being solved with. K stays small
(default 1) and should be measured, not guessed.

## 10. Hybrid models still decline paged-KV loads

`_decide_load_after_alloc` refuses the offload KV load for any sequence with
`has_per_req_cache` (reason string `per_req_cache_state_boundary`), and **this
stays**. It is not blocked merely on a joint park existing.

The two boundaries come from independent matchers at different granularities:
the KV leg from LMCache's `lookup()` floored to `chunk_size` (256), the state
leg from `BlockManager._gated_hit` — a fixpoint over the state caches, snapped
to `hash_block_size`, then to a `state_checkpoint_interval_tokens` rung, then
gated by `min_fork_tokens`. They agree only by configuration coincidence, and
`L > P` (KV loaded further than the state covers) is **silent wrong output**:
the forward reads a compressed prefix whose raw KV was never loaded.
`_JointPark` makes `L == P` *representable*, not *guaranteed*.
`clamp_state_boundary` is the clamp for that world and has no production caller.

Refusing costs the hybrid nothing it could have had: ATOM runs one forward over
the whole batch, so the linear layers must walk `[hbm, lmc)` token by token
regardless of whether the full-attention layers' KV is present. The load buys
no work saving, only risk. **Saves are untouched** — a hybrid still populates
the tier for stateless readers. A startup warning says all of this once per
server, so a permanent 0% load rate reads as deliberate rather than broken.

## 11. What wiring the load path requires

For the agent picking this up. In dependency order:

1. **Give the worker an index, or a way to answer without one.** The tier is
   built `index=None` because `StateOffloadIndex` lives in the engine process.
   `submit_load`/`_do_load` write `loads_attempted`, `loads_failed`, and call
   `index.forget(h)` — all three need a home. The natural shape mirrors the
   spill: the worker reports, the engine applies.
2. **Carry `(state_hash, target_group)` outbound.** Add a field to
   `ScheduledBatch` alongside `state_spill_pairs`. The target is a **real pool
   group**, not a staging entry — the bytes land where the resuming request
   will read them. Only the spill direction needs the staging indirection.
3. **Carry a request-keyed completion inbound.** `StateOffloadTier.get_finished`
   already returns `(done, failed)` sets of request ids, and
   `KVConnectorOutput` already has `finished_loading` / `failed_loading`.
   Decide whether the state load shares those or gets its own pair; sharing
   means the aggregator's existing quorum applies for free.
4. **Park and unpark jointly.** Use `_JointPark`: both legs must land before
   unpark, and either failing fails the pair. Waking on the state transfer
   alone lets the model read KV blocks that are not yet filled — silent, not an
   error. `failed_loading` already means "wake and recompute over the blocks
   already allocated", which is exactly right for a failed state load.
5. **Clamp both legs to a common boundary and prove it.** This is the real
   work, and it gates lifting §10's guard. `clamp_state_boundary(state_blocks,
   kv_loaded_blocks)` is the helper; `should_load_state(hit, floor)` is the
   is-it-worth-it test (wire `OFFLOAD_STATE_MIN_LOAD_TOKENS` here).
6. **Flip `STATE_OFFLOAD_LOADS_WIRED = True`** — last, and only together with
   the above. This is the whole re-widening edit, and
   `test_the_gate_is_the_only_thing_holding_the_spilled_rung_back` is the test
   that says so.
7. **Delete the write-only startup warning** and its test.
8. **Keep `_attach_state_group`'s disown guard.** See §6 — the LRU can drop
   bytes under an advertised hash at any time.

## 12. Known limitations

- **No cross-restart reuse, and that is correct.** `LocalDiskBackend.__init__`
  starts from an empty dict and never scans its directory, so after a restart
  LMCache does not recognize its own files. An index recovered from disk would
  be a pure false-positive generator. The in-memory index and the bytes share
  one server lifetime (`LMCacheEngineBuilder.get_or_create` runs inside
  `register_kv_caches` at model load).
- **`_STATE_TIER_BACKENDS` is a name list**, not a capability probe. A future
  connector that grows a `_state_tier` must be added to the frozenset or the
  ring silently stays off for it. Deliberate: the check runs in the engine
  process at `BlockManager.__init__`, before any worker connector exists to
  ask. Fails safe.
- **`StateOffloadTier.shutdown()` has no caller.** Neither does the KV path's
  executor teardown; the connector has no `close`/`shutdown` hook.
  `ThreadPoolExecutor` registers its own interpreter-exit join, so a clean exit
  drains queued work — but nothing bounds how long, and nothing cancels
  in-flight transfers on a fast teardown.
- **`kv_offload_enabled`'s False arm is test-only.** `BlockManager` refuses to
  construct the index at all when the connector does not host the tier, so in
  production the flag is always True. It is kept because the two conditions are
  independent in principle, and because the failure it prevents (spending
  LMCache capacity on hashes no load can reach) is silent.

## 13. Verification state

- **427 tests passing** across the state, scheduler, and block-manager suites.
- Lint clean (`black`, `ruff`) on every changed file; CI's exact command
  reports `All checks passed`.
- 44 failures in `tests/plugin/` and `test_fused_compress_ragged.py` are
  **pre-existing** — verified by capturing the failure set against a stashed
  clean tree and diffing: identical, 44 both ways.
- Every finding at Important severity or above across five review packages is
  closed, and each fix was proven non-vacuous by deliberate sabotage (revert
  the fix, observe the exact expected failures, restore).
- Parked Minor findings are collected in
  `.superpowers/sdd/2026-08-12-state-cache-lmcache-offload/deferred-minors.md`.

### Test map

| File | Covers |
|---|---|
| `tests/test_state_offload_index.py` | The ring: reserve/drop/release, starvation latch, counters, env parsing. |
| `tests/test_state_tier.py` | Worker tier: submit/report, inflight pruning, the three report sets. |
| `tests/test_state_object.py` | `StateByteCodec`: one hash one key, per-worker keys, `busy_loop=False`, refcount discipline, allocation refusal. Skips without `lmcache`. |
| `tests/test_state_entry_views.py` | `state_entry_views` contiguity and full-span coverage for V4 and GDN, including staging groups past the admission count. |
| `tests/test_state_spill_copy_order.py` | Spills issued before checkpoint copies. |
| `tests/test_state_offload_resume.py` | Admission against a spilled hash; the shadowing control pair; the disown guard; reported `cached_tokens`. |
| `tests/test_state_offload_clamp.py` | `clamp_state_boundary` and `_JointPark`, both still uncalled. (`should_load_state` moved to `state_offload.py` and is wired; it is tested in `test_state_offload_index.py`.) |
| `tests/test_kv_aggregator.py` | Union quorum on `state_indexed \| state_index_failed`. |
| `tests/test_sub_pool_spec.py` | `extra_entries` vs `staging_entries` vs `admission_entries`. |
| `tests/test_staged_transfer.py` | Pack/unpack round trip, stream ordering, the per-thread buffer property. |

## 14. Open decisions — for the user, not the next agent

1. **Load-path scope.** Ship the branch write-only (the spill path is
   independently useful as measurement and as the foundation), or wire loads
   before merge? §11 is the work either way.
2. **Task 10, end-to-end GPU validation.** Not run — the machine is contended.
   Required before any claim about real-workload behaviour. Follow the
   project's rules when running it: `AITER_LOG_LEVEL=WARNING`,
   `rm -rf /root/.cache/atom/*` before restart, and verify with
   `rocm-smi --showmemuse` (VRAM% > 0), not just `curl /health`.

The precondition test comes first regardless: run the target workload with
`OFFLOAD_STATE` **off** and read `checkpoints_evicted` off the
`state checkpoints:` line. If it is near zero, this tier should not be turned
on at all.
