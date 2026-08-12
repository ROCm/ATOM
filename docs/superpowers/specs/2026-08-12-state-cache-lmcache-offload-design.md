# State-Cache LMCache Offload — Design

Date: 2026-08-12
Status: design approved, not yet planned or implemented
Scope: DeepSeek-V4 and GDN/Qwen3-Next, uniform from the start

## Problem

When `StateGroupPool` runs out of groups, `pop()` spends the least-recently-used
checkpoint and its bytes are **discarded** (`state_pool.py:263-267`). The
`checkpoints_evicted` counter measures exactly this loss. A later request that
hits the same prefix has to recompute it, because neither the SSM recurrent
state nor the V4 compressor ring can be rebuilt from cached KV blocks — the
cache holds the compressor's *output*, the state is its rolling *input window*
(`block_manager.py:320-324`).

This design adds a CPU/NVMe tier beneath the pool: an evicted checkpoint is
spilled to LMCache instead of dropped, and pulled back when a later request
hits its hash.

### Precondition — measure before building

The design only pays off when `checkpoints_evicted` is materially non-zero.
The three fates are not interchangeable (`state_pool.py:220-233`):

| counter | meaning | does this design help |
|---|---|---|
| `checkpoints_evicted` | pool too small for how long a checkpoint must last | **yes — this is the target** |
| `checkpoints_orphaned` | KV pool too small for the same span | partially, see §2 |
| `checkpoints_dropped` | pool full, checkpoint never established | no — there are no bytes to spill |

**If `checkpoints_evicted` is near zero on the target workload, do not build
this.** Measure first via `checkpoint_fates()`.

## Placement

`StateOffloadTier` is a backing store **owned by `StateGroupPool`**, not a
sibling in `BlockManager.state_caches`.

The reason is structural: every member of that tuple is a **veto** — it answers
"the rightmost boundary ≤ X that I accept" and `_gated_hit` runs them to a
fixpoint where each answer is ≤ its input (`block_manager.py:260-276`). An
offload tier does the opposite: it makes *more* boundaries reachable. As a
sibling it could only ever return identity. The widening must therefore happen
**inside** `StateGroupPool`, where the membership test lives.

## 1. When to spill

One hook point only: **`pop()` at `state_pool.py:263-267`**, in the branch that
spends a checkpoint, before `invalidate(group)` clears `group_hash[group]`.

`pop()` has two callers (`state_pool.py:513`, `:548`, both reached via
`block_manager.py:405,412`) and both are on the scheduler's critical path, so
the hook must be **non-blocking**: record the hash, enqueue the group, return.
No D2H on this thread.

### The staging hazard

The scheduler says "spill group 7" during step N, but group 7 is handed to a new
request in that same step; the worker reads it in step N+1 and gets the new
owner's bytes.

Pinning the group is not available: `pop()` is called *because* the pool has no
free group, so withholding one is a deadlock shape.

**Resolution: spill via copy, not via pin.** The hook copies the group into a
dedicated staging entry (allocated outside the pool, one entry wide) and `pop()`
hands the original out immediately. D2H reads from staging.

- Staging depth K is configurable, default small (1–2). Each entry is MB-scale.
- **Spills beyond depth K are dropped**, and `checkpoints_evicted` still counts
  them — identical to today's behaviour, so no regression.
- Cost: K entries of HBM, taken from the same budget the pool wants, plus one
  device-to-device copy on the compute stream before the forward (same path as
  `copy_state_entries`, `backends.py:519-527`).

**This is the least-settled part of the design.** The alternative — spill only
when `_pop_vacant()` succeeds, i.e. only when the pool is *not* full — needs no
staging at all, but misses exactly the high-pressure case the feature exists
for. Revisit if the K-entry cost measures worse than the spill benefit.

## 2. What is NOT spilled

**`checkpoints_dropped`** — `checkpoint()` found `has_free()` false and no
checkpoint was ever established (`state_pool.py:510-512`). No bytes exist.

**`unindex()` / `checkpoints_orphaned`** (`state_pool.py:621-644`) — spill these
**only when KV offload is also enabled**.

The reasoning, corrected twice during design: a state checkpoint is only useful
when the corresponding KV prefix is also reachable, because `resumable_hit`
scans `block_hashes`, which `can_allocate` builds from `self.kv.lookup(h)` —
HBM hits only (`block_manager.py:306-312`). If the KV prefix left HBM, the hash
never reappears and the bytes are wasted.

**But with KV offload on, the KV prefix can come back from LMCache**, and the
hash becomes reachable again. So the exclusion is conditional on KV offload
being off, not absolute. (Both earlier drafts of this document got this wrong:
first by asserting the exclusion unconditionally, then by flagging it as
unverified. It is verified now, and it is conditional.)

## 3. What is stored, and how

### The unit

One whole state entry per checkpoint. Both backends already build the byte views
in `copy_state_entries`; factor out the view-building half:

```python
def state_entry_views(self, group: int) -> list[torch.Tensor]
```

- **DeepSeek-V4** — contiguous: one `slot_span` slice per plane
  (`deepseek_v4_attn.py:783-830`).
- **GDN** — strided: `mamba_k_cache[:, g*span : (g+1)*span]` and the same for v,
  where `span = 1 + num_spec` (`gdn_attn.py:334-343`).

`copy_state_entries` is then expressed in terms of it. This is **refactoring out
the shared primitive, not bolting on a layer** — `backends.py:170-188` already
states that `copy_state_entries` is owed by every backend declaring a state
pool, fork backends included. This is why the design does not ride
`KVTransferRegion`/`get_kv_transfer_tensors`: GDN does not implement those.

### Sizing

`StateArena.entry_bytes` (`state_arena.py:254`) is authoritative. Order of
magnitude: V4 keeps 6 compressor fields across `n_csa`/`n_hca` layers plus an
optional window field; GDN keeps `2 × num_gdn_attn_state × (1+num_spec)` slots.
**Single-digit to double-digit MB per request, not KB.** Log `entry_bytes` at
startup; do not hard-code any size.

### Key

**ATOM's own `block_hashes[i]` — the same xxhash64 integer the HBM index uses**
(`BlockManager.compute_hash`, `block_manager.py:152-157`), written straight
through `storage_manager`, bypassing `ChunkedTokenDatabase`.

Three reasons, in order of importance:

1. **The two branches of the membership test must query the same integer.**
   `_resumable_from(h)` checks HBM and the tier; if the tier used a
   `ChunkedTokenDatabase`-derived key they would not be looking up the same
   thing. This is what makes the whole feature a one-line change.
2. **State is an indivisible whole-prefix snapshot, and `ChunkedTokenDatabase`
   is a splitter.** State *does* have a token range — `[0, pos)`, which is
   exactly why it can share KV's key — but unlike paged KV its bytes cannot be
   sliced by token: there is no such thing as "the first 3 chunks hit". Forcing
   it through the chunker would produce N keys that are only useful all
   together, multiplying the partial-invalidation probability for nothing.
3. It sidesteps LMCache's chunk-alignment loss (keys exist only at chunk
   boundaries, `token_database.py:387-391`).

### Transport — reuse the lower half of the KV path

`atom_lmcache_gpu_connector.py` is two layers stitched together:

| layer | contents | reusable for state |
|---|---|---|
| **upper: chunk orchestration** | `_iter_transfer_chunks`, `_iter_transfer_groups`, block_ids ↔ MemoryObj mapping | **no** |
| **lower: staging mechanism** | `_ensure_staging_buffer`, `_slice_to_memory_objs`, `_memory_objs_to_slice`, D2H, producer event, executors | **yes** |

The upper layer does not apply because the two caches differ in granularity by
two orders of magnitude. For an 8K prompt at `chunk_size=256` and
`interval=8192`: **32 KV chunk objects vs ~1 state object.** `_iter_transfer_chunks`
zips `memory_objs` against `block_id_groups` with `strict=True` and computes
`nbytes = block_count * bytes_per_block` from a startup constant
(`atom_kv_byte_codec.py:114`) — a 33rd object of a different size breaks both.
State is not a member of that loop.

Extract a `StagedTransfer` holding the staging buffer lifecycle, D2H/H2D, and
the producer event. KV and state each write their own orchestration on top.

**The existing Triton kernels need no modification.**
`_pack_chunk_major_kernel` / `_unpack_chunk_major_kernel`
(`triton_kv_staging.py:15,62`) are already fully parameterized gathers driven by
`segment_ptrs[]` + `segment_block_bytes[]` + `block_ids[]`, with nothing
KV-specific in them. State packs by passing `state_entry_views(group)` as the
segments with `block_ids=[group]`, `chunk_block_counts=[1]`.

**One constraint:** `_build_meta` requires `seg.is_contiguous()`
(`triton_kv_staging.py:135`). GDN's `cache[:, slot:slot+span]` is a **strided
view and is not contiguous**, so GDN must present its state as per-layer
contiguous slices rather than one strided block.

Also reused from the KV path: opaque `uint8` MemoryObjs (`MemoryFormat.BINARY`)
because the x-packed/strided/multi-plane layouts cannot be expressed in
LMCache's token-major model; and the producer `cuda.Event` recorded on the RPC
thread and `synchronize()`d on the save worker (`connector.py:240,407`) — from
commit `7427e05e`, which fixed KV corruption on reload.

## 4. Lookup and load

### Storage and lookup are separate; selection and admission are joint

| | KV | state |
|---|---|---|
| store | incremental, per chunk, `skip_leading_tokens` unchanged | **full entry**, once per checkpoint |
| key | `ChunkedTokenDatabase` (Python `hash()`) | `block_hashes[i]` (xxhash64) |
| LRU fate | independent | independent |

Neither store path changes for the other. This asymmetry is deliberate: KV is
naturally incremental, state is naturally whole, and packing them into one
object would force one to adopt the other's worst case. Physically bundling
`[0,pos)` KV with the state at `pos` was considered and **rejected**: with
`interval=8192` on a 32K prompt it rewrites 8K+16K+24K+32K = 80K tokens of KV
against 32K today, a **2.5× write amplification** of mutually-containing copies.

### Selection: the intersection's rightmost point, not the longest of two

`_gated_hit` (`block_manager.py:260-276`) already has the right shape — **KV is
the ceiling, state is the veto**:

```python
boundary = compressed_hit          # what KV can offer
while boundary > 0:
    for cache in self.state_caches:
        accepted = cache.resumable_hit(...)   # only ever ≤ boundary
```

The tier widens one line in `resumable_hit` (`state_pool.py:427-464`):

```python
if not assume_checkpointed and block_hashes[i] not in self.hash_to_group:
```
becomes
```python
if not assume_checkpointed and not self._resumable_from(block_hashes[i]):
```

with `_resumable_from(h) = h in self.hash_to_group or h in self._offload.hashes`.

Three properties follow without any new logic:

1. **HBM wins automatically.** The scan is right-to-left and both tiers are
   indexed by the same hash, so the rightmost boundary wins regardless of where
   it lives. No preference rule needed.
2. **The ladder is untouched.** Resume is a hash lookup, never arithmetic
   (`block_manager.py:744`), so a grid-placed and a demand-placed checkpoint are
   indistinguishable to the finder. The tier spills whatever entered the index.
3. **`min_fork_tokens` is not relaxed.** A spilled hash takes the identical
   successor-room test. Relaxing it would leave GDN's replacement group
   unfilled — a wrong state, not a slow one.

Note the scan skips a boundary for **two** distinct reasons — no checkpoint, or
failing `seq.num_tokens - (i+1)*hbs >= min_fork_tokens` — and `continue`s in
both cases. It returns on the first boundary passing both.

### The clamp — the one new correctness constraint

Let `P` be the state boundary and `L` the KV prefix actually loaded.

**`P ≤ L` must hold.** Today it is free, because both derive from
`block_hashes`. Once the tier admits spilled hashes and KV load length is
decided on a separate path, it is not.

| | consequence |
|---|---|
| `P ≤ L` | state is the compressed history of `[0,P)`, KV covers `[0,L)`, the forward continues correctly |
| **`P > L`** | state claims to have seen `[0,P)` but `[L,P)` KV does not exist — **silent wrong output, no error** |

The clamp must run **after** `update_state_after_alloc`, because before that
`seq.num_cached_tokens` is stale — often 0 — and loading below the true HBM
floor overwrites shared prefix blocks (`connector.py:834-840`, a documented past
corruption). If `P` clamps to 0, the sequence recomputes: the existing path, not
a new failure mode.

Conversely, **KV loaded past `P` is wasted bandwidth** on a hybrid model, since
nothing can resume from beyond the state boundary. `L` should converge to `P`.
This is an efficiency bound, not a correctness one.

### Why not unify `chunk_size` and `state_checkpoint_interval_tokens`

Considered and **rejected**. The premise that hybrid models can only resume at
interval granularity is false: the demand rung sits at `hash_block_size`
(`block_manager.py:637`), not at the interval, and `block_manager.py:614-620`
records why — gating the demand by the interval "left every prompt shorter than
an interval declining all the reuse it had."

Both directions lose:

- **`chunk_size` → 8192**: LMCache keys exist only at chunk boundaries, so the
  worst-case recompute tail grows from 255 to 8191 tokens, and any prompt under
  8192 gets no KV offload at all.
- **`interval` → 256**: `checkpointers_at` computes a 23% per-rung retention at
  V4's 256-token block versus effectively certain at the 8192 default
  (`block_manager.py:748-751`), and MB-scale entries every 256 tokens exhaust
  the pool.

The two constants optimize different things — `chunk_size` minimizes KV's
partial-hit loss (smaller is better), `interval` trades state storage cost
against checkpoint density (larger is cheaper). Binding them takes each one's
worst value. The correct mechanism is the **runtime** bound `L → P`, not a
startup constraint.

### Load: park jointly, wake jointly

Two objects, two transfers, **one park**. Both completions must land before
unpark. Waking on the state transfer alone lets the model read KV blocks that
are not yet filled — again silent, not an error.

Reload must **not** copy `_commit_pending`'s shape. That method indexes a
destination group *before its bytes exist* (`state_pool.py:525-553`), relying on
the copy reaching the very next batch; `scheduler.py:1490` guards the empty-batch
case specifically for it. A reload that indexes on lookup and lands H2D later
reproduces the #1417 failure. **Index only after H2D confirms** — go through the
offload connector's existing park/wake (`is_offload = True`), not `record_copy`.

### Load failure is a normal path

Three triggers: LMCache evicted the bytes on its own LRU, the spill never
succeeded, or the transfer errored. All three funnel into the existing
`failed_loading` path — the sequence falls back to whatever earlier boundary
`hash_to_group` offers (or 0) and recomputes. The hash is dropped from
`_offload.hashes` so the next request does not repeat the attempt.

**No spill acknowledgement.** LMCache's LRU can invalidate any receipt at any
time, so a receipt proves only that the spill once succeeded, never that the
bytes are still there. A reverse channel is not worth an expiring guarantee.

A load-length floor analogous to KV's `OFFLOAD_MIN_LOAD_TOKENS` (default 8192,
`connector.py:526`) should gate state loads too: a short hit prefix is not worth
a PCIe round trip. The same floor bounds the cost of false positives.

## 5. Lifetime and persistence

**The index is a plain in-memory set, and that is correct.**

`_offload.hashes` lives in the scheduler process; the bytes live in LMCache in
the worker process. They share one server lifetime: `LMCacheEngineBuilder.get_or_create`
runs inside `register_kv_caches` at model load (`forward_context.py:778`), so the
LMCache engine has no independent lifecycle — it cannot restart without the
worker restarting.

Persisting the index was considered and is **actively harmful**.
`LocalDiskBackend.__init__` starts from an empty dict and never scans its
directory — no `listdir`, `glob`, or `scandir` anywhere in the file. After a
restart LMCache does not recognize its own `.pt` files either. An ATOM-side
index recovered from disk would therefore be a pure false-positive generator:
every recovered hash triggers a load that is guaranteed to miss.

Cross-restart reuse is a property the entire dependency chain lacks, including
today's paged-KV offload. This design does not claim it.

(Related: `PYTHONHASHSEED=0`, documented in `offload/README.md:598`, makes
LMCache's `builtin` hash stable **across TP ranks within one run** — not across
restarts.)

The one real inconsistency remains: LMCache may evict bytes on its own LRU while
`_offload.hashes` still lists the hash. That is the false positive handled above
by `failed_loading`, and it is unrelated to restarts.

## 6. Impact

### Benefit

Recovers reuse currently counted as `checkpoints_evicted`, plus
`checkpoints_orphaned` when KV offload is on. Zero benefit against
`checkpoints_dropped`.

### Cost

1. **HBM: K staging entries**, MB-scale each, taken from the same budget the
   pool needs — and a small pool is the problem being solved. Whether one
   staging entry buys back more than it costs is an empirical question, not a
   derivable one; K large enough certainly inverts the sign.
2. **PCIe**: one D2H per spill, one H2D per load, MB-scale each, competing with
   KV offload for the same bus. **State should get its own executor**, for the
   same reason KV separates load from save (`connector.py:83-88`): a
   TTFT-critical load must not queue behind fire-and-forget spills.
3. **One device-to-device copy** per spill, on the compute stream ahead of that
   step's forward.
4. **TTFT**: sequences that were previously cut to 0 now hit and park waiting
   for H2D. Their TTFT rises in exchange for not recomputing. Worth it when the
   entry is small relative to the prefix; the load floor is what enforces that.
5. **False positives**: one wasted lookup plus a park/unpark. Bounded by the
   same floor.

### Unaffected

- Correctness bounds unchanged — `min_fork_tokens` / `successor_room` apply
  identically to spilled and resident checkpoints.
- Ladder policy untouched — `checkpointers_at` does not know the tier exists.
- Zero cost when disabled — `_resumable_from` degenerates to the original `in`,
  the spill hook is one `if`.
- Pure paged-KV models unchanged — `state_caches` is empty, `_gated_hit` is
  identity, no `P` bound exists.

## 7. Open questions

1. **Staging depth K, and whether staging is needed at all** (§1). The
   no-staging alternative — spill only when the pool is not full — is simpler
   but misses the target case. Decide with measurements.
2. **`checkpoints_evicted` on the real workload** (Precondition). Gates whether
   to build this at all.
3. **Whether `L → P` (KV load bounded by the state boundary) belongs in v1** or
   is a follow-up. It is an efficiency bound, not a correctness one; the
   correctness clamp `P ≤ L` is required in v1 regardless.

## Design history — corrections worth keeping

Recorded because a later reader will otherwise re-derive them:

- **"State has no token range" — wrong.** It has one, `[0, pos)`, which is
  precisely why it shares KV's hash. What it lacks is *sliceability by token*.
  The conclusion (bypass `ChunkedTokenDatabase`) survived; the reason changed.
- **The orphaned-checkpoint exclusion was wrong twice** — first stated
  unconditionally, then flagged unverified. It is conditional on KV offload
  being off (§2).
- **"The in-memory index is a design defect" — wrong**, and argued from a
  property (cross-restart durability) that no layer in the dependency chain
  has (§5).
- **Two independent A/B/C option sets were both lettered A/B/C**, making a
  consistent position read as a reversal. The architecture is "plan A"
  (tier beneath the pool); the persistence sub-question resolved to the
  in-memory set.
- **Physical bundling of KV+state into one object** was the user's proposal and
  was rejected on write amplification (§4), but its motivation — atomicity —
  was correct and is served by the joint park and the `P ≤ L` clamp.
