# ATOM scheduling & KV cache guide

ATOM (AiTer Optimized Model) uses a prefill-first scheduler with paged KV cache block management to drive LLM inference on AMD ROCm/HIP GPUs. This guide covers the scheduling algorithm, batch construction, block-level KV cache management, prefix caching, postprocessing, speculative decoding integration, and sequence lifecycle.

## Quick reference

| Class | File | Purpose |
|---|---|---|
| `Scheduler` | `atom/model_engine/scheduler.py` | Orchestrates prefill/decode scheduling, preemption, and postprocessing |
| `EngineStats` | `atom/model_engine/engine_stats.py` | MTP acceptance, prefix-cache hit, and throughput statistics |
| `ScheduledBatch` | `atom/model_engine/scheduler.py` | Immutable snapshot of a scheduled batch sent to the model runner |
| `ScheduledBatchOutput` | `atom/model_engine/scheduler.py` | Holds sampled token IDs and draft token IDs returned from forward pass |
| `BlockManager` | `atom/model_engine/block_manager.py` | Manages paged KV cache blocks with allocation, deallocation, and prefix caching |
| `Block` | `atom/model_engine/block_manager.py` | Single KV cache block with ID, reference count, hash, and token IDs |
| `Sequence` | `atom/model_engine/sequence.py` | Tracks a single request through its lifetime (tokens, blocks, status, timing) |
| `SequenceStatus` | `atom/model_engine/sequence.py` | Enum: `WAITING`, `RUNNING`, `FINISHED`, `EXIT_ENGINE` |
| `SequenceType` | `atom/model_engine/sequence.py` | Enum: `DUMMY`, `PREFILL`, `DECODE` |
| `RequestOutput` | `atom/model_engine/request.py` | Dataclass streamed to clients with new tokens and finish status |
| `Config` | `atom/config.py` | Scheduling-related fields: `max_num_seqs`, `max_num_batched_tokens`, `kv_cache_block_size`, etc. |

**Key config defaults:**

| Field | Default | Description |
|---|---|---|
| `max_num_seqs` | 512 | Maximum sequences in a single batch |
| `max_num_batched_tokens` | 16384 | Maximum tokens scheduled in a single step |
| `kv_cache_block_size` | 16 | Tokens per KV cache block (must be multiple of 16, or 1) |
| `enable_prefix_caching` | `False` | Enable hash-based prefix block sharing |
| `scheduler_delay_factor` | 0.0 | Delay factor for batching prompt requests (0 = no delay) |
| `gpu_memory_utilization` | 0.9 | Fraction of GPU memory for KV cache |

## Scheduling algorithm

The scheduler implements a **prefill-first** policy: all waiting (prefill) requests are scheduled before any running (decode) requests. The entry point is `Scheduler.schedule()`, which returns a `(ScheduledBatch, dict[int, Sequence])` tuple or `None` if both queues are empty.

### Scheduler initialization

```python
class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.stop_token_ids = config.stop_token_ids
        self.block_manager = BlockManager(config)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.prev_time = 0.0
        self.prev_prompt = False
        self.last_prompt_latency = 0.0
        self.delay_factor = config.scheduler_delay_factor
        self.use_spec = config.speculative_config is not None
        self.mtp_k: int = (
            config.speculative_config.num_speculative_tokens if self.use_spec else 0
        )
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0
```

The scheduler maintains two deques — `waiting` (pending prefill) and `running` (active decode) — plus a `BlockManager` for KV cache allocation.

### Schedule flow

`Scheduler.schedule()` proceeds in two phases:

**Phase 1 — Prefill scheduling:**

1. While the delay gate passes (`_passed_delay`), the waiting queue is non-empty, and `num_seqs_prefill < max_num_seqs`:
   - Peek the first waiting sequence.
   - Compute `num_new_tokens = seq.num_tokens - seq.num_cached_tokens` (prefix cache hits reduce new tokens).
   - If `num_batched_tokens + num_new_tokens > max_num_batched_tokens` or `block_manager.can_allocate(seq)` returns `False`, break.
   - Otherwise: allocate blocks, set `seq.status = RUNNING`, `seq.type = PREFILL`, move from `waiting` to `running`.
2. If any prefill sequences were scheduled, return the batch immediately (no decode mixing).

**Phase 2 — Decode scheduling (only when zero prefills were scheduled):**

1. Pop sequences from `running` up to `max_num_seqs`.
2. For each sequence, check `block_manager.can_append(seq)`.
3. If a block cannot be appended, **preempt** the last running sequence (move it back to `waiting` with status `WAITING` and deallocate its blocks).
4. If the sequence has speculative draft tokens (`seq.spec_token_ids`), record them in `scheduled_spec_decode_tokens`.
5. Call `block_manager.may_append(seq, num_new_tokens)` where `num_new_tokens = mtp_k + 1`.
6. Re-insert all scheduled sequences back into `running` (preserving order).

### Delay factor

When `scheduler_delay_factor > 0`, the scheduler delays prefill scheduling to allow the waiting queue to accumulate more requests for better batching:

```python
def _passed_delay(self, now: float) -> bool:
    if self.prev_prompt:
        self.last_prompt_latency = now - self.prev_time
    self.prev_time, self.prev_prompt = now, False
    if self.delay_factor > 0 and self.waiting:
        earliest_arrival_time = min([seq.arrive_time for seq in self.waiting])
        passed_delay = (now - earliest_arrival_time) > (
            self.delay_factor * self.last_prompt_latency
        ) or not self.running
    else:
        passed_delay = True
    return passed_delay
```

A new prefill is scheduled only when the earliest waiting request has waited longer than `delay_factor * last_prompt_latency`, or when there are no running decode requests.

### Preemption

When a decode step cannot extend a sequence's KV cache (no free blocks), the scheduler preempts the **last** running sequence:

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    # Strip placeholder + rejected draft tokens added by postprocess.
    if self.mtp_k > 0:
        strip = self.mtp_k + seq.num_rejected
        if strip > 0:
            del seq.token_ids[-strip:]
            del seq.output_tokens[-strip:]
            seq.num_tokens -= strip
    seq.num_rejected = 0
    seq.num_bonus_tokens = 0
    seq.spec_token_ids = np.array([], dtype=np.int32)
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

The scheduler pushes the preempted sequence to the front of the waiting queue, fully deallocates its blocks, and re-prefills it on the next scheduling cycle.

**MTP placeholder stripping:** When speculative decoding is active (`mtp_k > 0`), `postprocess()` appends placeholder tokens (EOS) to running sequences to reserve KV cache slots for the next step (see section 5.6). If a sequence is preempted before those placeholders are consumed, they must be removed so that re-prefill starts from the correct token history. The strip count is `mtp_k + seq.num_rejected` — this accounts for both the `mtp_k` placeholder slots and any tokens that were rejected during the last verification step. The method deletes that many trailing entries from both `seq.token_ids` and `seq.output_tokens` and decrements `seq.num_tokens` accordingly.

**Speculative state reset:** After stripping, the sequence's speculative decoding state is fully cleared: `num_rejected` and `num_bonus_tokens` are zeroed, and `spec_token_ids` is set to an empty array. This ensures the sequence re-enters the scheduling pipeline with a clean state — no stale draft predictions or acceptance metadata carry over across preemption.

## ScheduledBatch structure

`ScheduledBatch` is constructed by `Scheduler.schedule()` and passed to the model runner. It is a frozen snapshot of batch metadata.

### Constructor signature

```python
class ScheduledBatch:
    def __init__(
        self,
        seqs: dict[int, Sequence],
        num_scheduled_tokens: list[int],
        total_tokens_num: int,
        total_tokens_num_prefill: int = 0,
        total_tokens_num_decode: int = 0,
        total_seqs_num: int = 0,
        total_seqs_num_prefill: int = 0,
        total_seqs_num_decode: int = 0,
        is_dummy_run: bool = False,
        num_spec_step: int = 0,
        scheduled_spec_decode_tokens: dict[int, np.ndarray] | None = None,
    ):
```

### Fields

| Field | Type | Description |
|---|---|---|
| `req_ids` | `list[int]` | Sequence IDs in batch order (`list(seqs.keys())`) |
| `scheduled_tokens` | `np.ndarray[int32]` | The tokens to process, flattened in batch order and sliced by `num_scheduled_tokens` |
| `temperatures` | `list[float]` | Sampling temperature per sequence |
| `context_lens` | `list[int]` | Total token count per sequence (`seq.num_tokens`) |
| `block_tables` | `list[array("i")]` | Block ID tables for sequences that have block tables, held as `Sequence.block_table` gives them |
| `last_block_num_tokens` | `list[int]` | Number of valid tokens in each sequence's last block |
| `num_cached_tokens` | `list[int]` | Number of tokens served from prefix cache per sequence |
| `num_scheduled_tokens` | `list[int]` | Number of new tokens scheduled per sequence |
| `total_tokens_num` | `int` | Sum of all scheduled tokens across all sequences |
| `total_tokens_num_prefill` | `int` | Total scheduled tokens for prefill sequences |
| `total_tokens_num_decode` | `int` | Total scheduled tokens for decode sequences |
| `total_seqs_num` | `int` | Total number of sequences in the batch |
| `total_seqs_num_prefill` | `int` | Number of prefill sequences |
| `total_seqs_num_decode` | `int` | Number of decode sequences |
| `is_dummy_run` | `bool` | Whether this is a dummy/warmup run |
| `num_spec_step` | `int` | Number of speculative decode steps (`mtp_k`) |
| `scheduled_spec_decode_tokens` | `np.ndarray[int32]` | Draft tokens from the prior speculative step, densified to `[bs, num_spec_step]` in batch order (the constructor takes them keyed by request ID) |

### ScheduledBatchOutput

Returned by the model runner after a forward pass:

```python
class ScheduledBatchOutput:
    def __init__(
        self,
        token_ids: dict[int, tuple[int, ...]],
        draft_token_ids,
    ):
        self.req_ids = list(token_ids.keys())
        self.token_ids = token_ids        # {seq_id: (accepted_token_ids...)}
        self.draft_token_ids = draft_token_ids  # {seq_id: [draft_ids]} or None
```

- `token_ids` maps sequence ID to a tuple of accepted token IDs.
- `draft_token_ids` maps sequence ID to a list of speculative draft token IDs for the next step (when MTP is active).
- A special key `-1` in `token_ids` signals deferred output mode.

## Block manager

The `BlockManager` implements paged KV cache management with fixed-size blocks.

### Block class

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id   # Unique integer ID
        self.ref_count = 0         # Number of sequences referencing this block
        self.hash = -1             # xxhash64 digest for prefix caching (-1 = unhashed)
        self.token_ids = []        # Token IDs stored in this block
```

Methods:
- `update(hash, token_ids)` — Sets the block's hash and token content.
- `reset()` — Sets `ref_count = 1`, `hash = -1`, `token_ids = []` (used on fresh allocation).

### BlockManager initialization

```python
class BlockManager:
    def __init__(self, config: Config, *, state_runtime: StateRuntime = DEFAULT_STATE_RUNTIME):
        block_size = config.kv_cache_block_size      # Tokens per block (default 16)
        num_blocks = config.num_kvcache_blocks        # Total blocks in pool
        self.block_size = block_size
        self.enable_prefix_caching = config.enable_prefix_caching
        self.kv = BlockPool(num_blocks, on_evict=self._record_evicted)
        
        # Per-request cache slot pool. Used by attention types whose state
        # lives outside the paged KV pool (GDN recurrent state, the
        # DeepSeek-V4 compressor ring); they declare it as a STATE entry
        # class via AttentionMetadataBuilder.sub_pool_specs().
        # Slots are counted raw, not divided into per-request groups: what one
        # request occupies is a property of the request (`1 + num_spec` while
        # it speculates, 1 otherwise) and a checkpoint occupies exactly one
        # whatever the model does, so there is no single width to divide by.
        pool_entries: dict = getattr(config, "pool_entries", None) or {}
        pool_per_req: dict = getattr(config, "pool_entries_per_req", None) or {}
        self.num_state_slots = int(pool_entries.get(STATE_SLOT_CLASS, 0))
        self.state_slots_per_req = int(pool_per_req.get(STATE_SLOT_CLASS, 1)) or 1
        checkpoint_spec = state_runtime.checkpoint_spec
        self.paged_state_checkpoints = None
        if checkpoint_spec is not None:
            self.paged_state_checkpoints = PagedStateCheckpointCoordinator(
                self.kv,
                checkpoint_spec,
                enabled=self.enable_prefix_caching and self.num_state_slots > 0,
            )
        self.state = StateSlotPool(
            self.num_state_slots,
            transfer=(
                StateTransfer.none()
                if self.paged_state_checkpoints is not None
                else state_runtime.transfer
            ),
            hash_block_size=self.hash_block_size,
            enabled=self.enable_prefix_caching,
        )
```

The block pool is pre-allocated at startup. [`BlockPool`](../atom/model_engine/block_pool.py) holds the blocks, their ref counts, a free list (a deque for O(1) pop/push plus a set for membership, since a cache hit can claim an id the queue still lists) and the content-hash index.

`BlockManager` holds one, for the compressed KV. The sliding window used to hold a second — see **Sliding window: a ring, not a pool** below for why it no longer does.

#### Sliding window: a ring, not a pool

DeepSeek-V4's sliding window is a per-request ring inside that request's **slot** — a fixed run of rows at the high end of the shared row space, holding its compressor state and then every layer's window. `win_with_spec = window + max_spec_steps` positions are addressable per layer, and one formula serves every layer of a compress class ([`v4_pool_geometry.py`](../atom/model_ops/attentions/pool_layout/v4_pool_geometry.py)):

```
row = slot * slot_rows + ring_start + (q // ring_stride) * run_rows + q % ring_stride
```

where `q = pos % win_with_spec`. The layer term is not in it: a layer's view is anchored at its own base row, which is what lets one index buffer serve the whole class.

It was a content-addressed block pool until this change, which is worth writing down because the choice is not obvious and it is not permanent. **The question is where reuse comes from: a block pool reuses by sharing rows, a ring reuses by copying them.** Everything else follows.

Sharing rows was the only mechanism available before per-request state could be checkpointed by copying (`PagedStateCheckpointCoordinator` under `StateTransfer.copy(layout_id)`). A private ring at that time meant a request resuming someone else's cached prefix had never written that prefix into its own ring and read stale rows — issue #1417, which is exactly what replaced the ring with a pool. The ring is back only because `execute_paged_state_copies` now gathers the saved window and compressor state into the resumer's Active Slot. **Reverting the addressing without that copy reintroduces #1417 silently**, so the two are one change, not two.

What the ring buys:

- **Memory is sized in tokens, not blocks.** A block-addressed window straddling a boundary occupies `ceil(win/block)+1` blocks — at V4's 128-token window and 256-token block that is 512 tokens of full-resolution KV to hold 128. Measured on V4-Flash-DSpark tp2: the SWA sub-pool went 4.04 GB → 0.92 GB (fp8) and 6.47 GB → 1.58 GB (bf16), all of it returned to the paged pool as ~3,100 more blocks.
- **It shares a slot with the compressor state, so both are one entry class.** They are allocated and given up together and no request can have one without the other; pricing them apart would only invite a split that cannot happen. It also collapses a checkpoint copy to one range per plane.
- **The pool itself disappears** — free list, content index, window-freeing walk, per-request block table, `-1` out-of-window sentinels, and the admission term that had to account for all of it.
- **The bound is constructed, not measured.** The block pool carried a flat 64-block cushion because admission checked free blocks per request without reserving them, while materialization for the whole scheduling pass happened later. A ring is allocated with the request's state slot and cannot transiently exceed itself.
- **It stops gating prefix hits.** A block-addressed window vetoed any boundary whose trailing window was not resident. In the cache-stats line that veto is visible as `Lost-unrecoverable`; it moved to `Lost-to-checkpoint` (0.32% on V4-Flash-DSpark GSM8K) — the same reuse, now recoverable.

What it costs:

- **Positions alias.** `pos → row` is injective under block addressing and is not under a ring, so two invariants exist that did not before, and violating either corrupts silently rather than failing: `write_per_batch <= win_with_spec` (else one seq's own tokens race for a row) and, for the DSpark draft gather, `window <= win_with_spec` (the draft's `window_size` and the target's `win_with_spec` are separate configs). Both are asserted.
- **Ring size is coupled to speculation depth.** `win_with_spec` must grow with the draft count; a block pool did not care where drafts landed.
- **Nothing older than the window survives**, and every resume pays a copy where claiming a cached block paid nothing.

**When the trade reverses.** The memory win is entirely the `window / block_size` ratio: at V4's 128/256 a ring is 4× smaller, but at a 2048-token window a block pool needs `ceil(2048/256)+1 = 9` blocks = 2304 tokens for 2048, and the ring saves almost nothing while keeping all of its aliasing invariants. Note also that sharing rows was worth less than it looks: a resuming request shares only the trailing window and starts writing its own rows immediately, so a block pool never held one window for N requests either. **If V4's window ever grows past its block size, revisit this.**

**Per-Request Cache Pools (Stateful-Attention Models):** For models whose attention type maintains per-request state outside the paged KV pool (GDN: Qwen3-Next, Qwen3.5, Kimi-Linear; DeepSeek-V4's compressor ring):
- `state` — a [`StateSlotPool`](../atom/model_engine/state_pool.py), owning the Active Slot free list and the fork-checkpoint index. PAGE-copy checkpoints are owned separately by [`PagedStateCheckpointCoordinator`](../atom/model_engine/page_unit_checkpoint.py). Slots are handed out per need rather than in fixed-width groups: a live request takes `state_slots_per_req` of them (1 for a single committed state, `1 + num_speculative_tokens` where a rollback slot per speculated token is kept) while a checkpoint takes exactly one, because a resumed prefix has no speculation to roll back. See **State checkpoints** below.
- `num_state_slots` — total capacity, so callers can tell "all slots busy" (transient) from "the pool can never hold one request" (permanent). Compared against `state_slots_per_req`, not against zero.

The state class costs no paged blocks at admission time: sizing reserves every STATE class's floor before the paged class is sized (see [`sub_pool_spec.py`](../atom/model_ops/attentions/pool_layout/sub_pool_spec.py)), so a sequence only needs a free slot index. Because that floor is exactly `max_num_seqs` requests' worth, the slot pool never binds before `max_num_seqs` does.

### State checkpoints (stateful-attention prefix caching)

Neither the GDN recurrent state nor the V4 compressor ring can be rebuilt from cached KV blocks — the cache holds the compressor's *output*, the state is its rolling *input* window. So for a stateful model a prefix-cache hit is only resumable at a boundary where some earlier request saved its state, and `can_allocate` gates on that as a third shrink, chained after the SWA one:

```
for cache in state_caches:                        # to a fixpoint, not in series
    boundary = cache.resumable_hit(seq, boundary, hashes)
```

Run to a fixpoint rather than `min()`-ed or chained: the answer has to satisfy every class at once, and the largest boundary one class allows need not be one another class can resume from — nor is the nearest boundary below it necessarily acceptable to the first. Every answer is `<=` its input, so each round either terminates or strictly decreases. There is one member today (the compressor ring), which is why the loop currently converges in one pass; it is written for N because the next class is a matter of when, not whether.

**N classes, one protocol.** A `Pool.STATE` class (see [`sub_pool_spec.py`](../atom/model_ops/attentions/pool_layout/sub_pool_spec.py)) scales with in-flight requests rather than with history, and can therefore veto a prefix hit. [`StateCache`](../atom/model_engine/state_cache.py) is that shape — `resumable_hit` to answer how far back this class can resume from, `checkpoint` to keep a boundary that way, and one number saying what keeping one costs the forward that follows. The sliding window was the second member until it became a per-request ring carried by the checkpoint, which left it with nothing to veto; GDN's recurrent state becomes one the moment it stops forking. The tests exercise the multi-class behaviour through a stub rather than whichever class happens to exist, so they stay honest across that turnover.

That number, `successor_room`, is mutability quantified. A rolling state (GDN recurrence) is still being written by its owner and is not one range to duplicate, so keeping it means handing the group over and taking a fresh one — and the next forward has to refill the replacement, which is `min_fork_tokens` of it. An immutable entry, or one that can simply be copied, needs no hand-over and no successor, i.e. `0`. `inf` means the class cannot be checkpointed at all — it would gate hits and never keep one. No class reports it today; `StateTransfer.none()` decodes to it, so a backend with no transferable state lands there rather than being special-cased.

**Checkpoint capacity follows the transfer kind.** A fork checkpoint *is* a slot sitting on the free list with its content intact, indexed by the content hash of the last block it covers — the same lazy-eviction model the block pool uses, where hand-out (`StateSlotPool.pop`), not free, is the eviction event. The pool therefore never holds a slot back, and under full concurrency the fork checkpoint set drains on its own. One slot, whatever `state_slots_per_req` is: the rollback slots beside a live request's committed state are scratch a resumed prefix has no use for. A copy checkpoint does not occupy an Active Slot at all: it owns an ordered set of arbitrary PAGE units and is reclaimed only as a whole record. Active Slots remain reserved for resident requests.

**A store takes what its image needs and nothing more.** `begin_store` asks for `units_per_checkpoint`, and `ensure_free_units` spends checkpoints only for the shortfall — free units come first, and `pop` hands out never-used blocks before cached ones, so a store reaches for the cache only once the pool has nothing spare. A store whose units are not reachable is dropped and counted in `checkpoints_dropped`, and it is dropped *before* evicting anything: eviction gives up only after it has emptied the cache, so an unreachable request would destroy the cache on its way to refusing.

**There is no floor held back for live KV, and there cannot usefully be one.** `_fresh_block` raises when the pool is dry and nothing is evictable, so it is worth stating why the cache cannot take it there. A READY unpinned checkpoint is *already* available to live KV — `has_available_units` counts it and `ensure_free_units` will spend it — so the size of the cache is not the variable. What competes is the unevictable set, a checkpoint that is `COPYING` or held by a restore pin, and that set is confined to one pass: `schedule` publishes the previous batch's stores and releases its pins before it allocates anything, and this batch's stores are taken at batch construction, after every allocation. The one overlap is `allocate`, which pins a restore and then asks for fresh blocks in the same pass — and its own `can_allocate` counted that pin, protecting the checkpoint it is about to take and skipping the pins earlier admissions took. `may_append` never overlaps at all: the decode loop runs only in a pass that scheduled no prefill, so its pins were released at the top of it. Under contention the reachable outcome is therefore a refused admission, which the next pass retries, and never the raise. A floor would not improve on that in any case: live KV's demand is unbounded and legitimate — `allocate` takes a whole prompt's blocks, up to `max_model_len` of them — so no reserved quantity can promise it a block. *(The decode half of this rests on prefill and decode never sharing a pass; the mixed batch the scheduler has a TODO for would need the argument redone.)*

**Eligibility and eviction policy are separate.** `_is_evictable` says whether a checkpoint may be spent — READY, unpinned, not the one the caller is protecting — and is the single rule `has_available_units` and `ensure_free_units` share. `_next_victim` says which eligible one to spend first, and is the only place the policy lives: least recently used today, one method to replace for another. `has_available_units` asks whether the eligible set reaches a count, which the order it is walked in cannot change -- only how soon the loop gets there -- so a new policy can change which checkpoint is spent and never what the gate answers. The refusal for an unreachable count sits in `ensure_free_units` itself rather than in a caller, because the bare loop gives up only after it has emptied the cache.

The ladder asks a stricter question before it acts. A demand is an instruction to cut a prefill chunk onto a rung, and that cut costs the request a forward, so `_record_checkpoint_demand` records one only while `_checkpoint_has_room` holds — otherwise the forward is bought for a store `begin_store` is about to refuse. It is not "does an image fit": the admission asking takes its own block table first, so the question is `has_available_units(num_new_blocks + units_per_checkpoint)`, and it is asked with the same `protected_hash` `can_allocate` passes to `_has_page_units` on the next line, so the two gates of one pass agree on what eviction could reclaim. A pool with room for an image but not for this request *and* an image answers yes to the weaker question and refuses at `begin_store`, with the cut already bought. It is asked afresh on every attempt, because a demand recorded while the pool had room is not still affordable once it does not; what must not repeat is the counting, so the sequence carries a marker per counter rather than the gate reading a position it is about to overwrite. It remains a sample even so — the store happens many forwards later, at the rung this cut creates, against a pool that has moved — so what this gate removes is the loss knowable at admission and `checkpoints_dropped` counts the rest; the two are meant to be read together. What is *not* suppressed is the attribution: `num_wanted_hit_blocks` and hence `Lost-to-checkpoint` still say the reuse was declined for want of a checkpoint, because it was. `demands_declined_no_room` is where the difference shows, which keeps "the ladder is quiet because there is no demand" distinguishable from "the ladder is quiet because the pool is tight".

**The free list is two halves.** Groups carrying nothing sit in one container ordered by index; groups carrying a checkpoint sit in another ordered least-recently-used. `pop` always drains the first before touching the second, so a checkpoint can only be spent once there is nothing free left to take — a single release-ordered queue cannot express that, because a checkpoint handed back before a never-used group sits ahead of it and is spent first. Reuse counts as use: `claim` leaves the hash in place, so a resumed checkpoint returns through `release` to the LRU tail.

Index order in the vacant half is not a fairness choice. Allocating lowest-first keeps the top of the pool cold — a high index is only reached at a concurrency high-water mark — which is what lets `retire_top` hand the pool's top group back when the KV/state boundary moves. When something *is* sitting there, `retire_top` relocates it and spends the least recently used checkpoint instead, wherever that one lives; retiring by index alone would be anti-LRU, since an index records the high-water mark at hand-out and is never refreshed by use.

**Two ways to keep one.** How a group reaches the index is the backend's `StateTransfer`, declared by `AttentionMetadataBuilder.state_transfer()`, and it decides *where* that backend may checkpoint.

After pool sizing, the runner combines that capability with the optional `PagedStateCheckpointSpec` in one validated `StateRuntime`. COPY requires a spec with the same versioned layout id; FORK and NONE forbid one. The nested runtime wire payload is reconstructed once in `EngineCore` and passed explicitly through the scheduler to `BlockManager`, so neither `Config` nor downstream constructors can observe an invalid transfer/spec combination.

*Fork* (`StateTransfer.fork(n)`, GDN). At a rung the request hands its group to the index and takes a fresh one; for exactly one forward it then reads the handed-over group and writes the new one (`non_spec_state_indices_in_tensor` / `non_spec_state_indices_tensor`). A checkpointed group is never written again, which is what makes it safe to share. Resuming is the same move in reverse. The cost is that the *next* forward is bound: it has to leave the replacement self-contained, which takes `n` committed tokens.

*Copy* (`StateTransfer.copy(layout_id)`, DeepSeek-V4). The resident request keeps one contiguous Active Slot ([`StateArena`](../atom/model_ops/attentions/pool_layout/state_arena.py)); an immutable checkpoint scatters that slot's canonical byte stream across an ordered set of arbitrary PAGE units, so the owner is not disturbed and checkpoint allocation does not require a large contiguous extent.

An image holds part of a slot, not all of it (`PagedStateCheckpointSpec.image_bytes`, sized from `AttentionMetadataBuilder.checkpoint_image_bytes`). A resumer starts at the boundary the checkpoint was taken on, and a compressor that pools `ratio` tokens with no overlap begins its first pool exactly there — so every row it reads is one it writes, and the checkpoint owes it nothing. DeepSeek-V4's HCA state is 52% of a slot and entirely dead on that argument, which is what `StateField.in_checkpoint=False` declares; the sliding windows next to it are a sliding window, so every row a window position reaches is carried. The rows *between* them are not: a class interleaves its layers' windows so one index formula can serve every layer of it, and the rows that construction skips are reachable by no `(layer, position)` pair at all, so nothing writes or reads them (`UnifiedPoolGeometry.entry_row_runs`). Leaving them out costs 17.3% of the entry on the DSpark configuration and makes the image no longer a subsequence of the slot's rows — which is why a reader at the wrong version would gather every window row shifted. Both rules are named in `layout_id` (`nocopy=`, `entry=packed`) and fenced by its version, because two workers disagreeing about either would read one image at two layouts. It holds only while every ratio the model declares divides the quantity a checkpoint is aligned to -- `kv_cache_block_size * decode_context_parallel_size`, not `block_size` -- which `_assert_ratios_divide_the_alignment` enforces at startup against `hf_config.compress_ratios`. Nothing downstream has to cooperate, which is what makes a decode boundary checkpointable at all — see below. The bytes still need a forward to move them, so `checkpoint` records the intent and keeps the record invisible in `COPYING` state. When the next real batch is built, `BlockManager.take_state_maintenance_ops` is the only drain: it returns one typed `StateMaintenanceOps` containing Active Slot relocations, checkpoint stores and checkpoint restores. `ScheduledBatch.state_maintenance_ops` carries that bundle, and `AttentionMetadataBuilder.build` issues every operation on the compute stream before the forward — one place every execution path passes through exactly once per batch. An empty batch does not drain the bundle. Publishing the checkpoint only after its store has ridden a real batch stops a resumer claiming bytes that do not exist yet.

Under fork, when no second group is free the request can adopt the checkpoint group, spending it rather than sharing it. Copy never adopts PAGE fragments as an Active Slot: a hit first obtains a complete contiguous Active Slot and then gathers the checkpoint into it; without a free Active Slot, admission waits.

**Where checkpoints land.** One ladder for every state class: a rung every `--state-checkpoint-interval-tokens` (default 8192) of context. Whether a class takes a given rung comes down to one comparison — how many tokens the *next* forward carries, against that class's `successor_room`. `BlockManager.checkpointers_at` takes the first as an argument (prefill passes what is left of the prompt, decode passes one token) and returns the classes that qualify; `checkpoint_limit` is the same rule solved for prefill's last qualifying rung and `checkpoint_cut` turns it into a chunk boundary, which the scheduler needs up front (`_finalize_prefill_chunk`). Everything else follows from the one comparison: GDN's `fork(1)` always qualifies — its `causal_conv1d` write paths all store the full window to the output slot — V4's `copy(layout_id)` reports 0 and so does too, and a rolling class needing a long hand-over simply never qualifies mid-generation. A backend with no transferable state at all declares `StateTransfer.none()`, which is `inf` on this scale; it is a separate kind rather than a token count precisely because `copy(layout_id)` has to report a real 0 and the two would otherwise be the same number. `hash_blocks` calls `checkpoint` only on an exact position match: a forward that overshoots a rung holds state ahead of the hash it would be filed under. An interval off the hash-block grid is snapped down in `BlockManager.__init__` so every rung has a block hash.


**Checkpoints past the prompt.** A long answer crosses rungs the prompt never reached, and a follow-up turn replaying the conversation wants to resume from them — which is also why generated blocks enter the prefix cache at all (`hash_decode_blocks`, bounded by the committed KV length). The room test gates this with no special case: one decode token satisfies GDN's `fork(1)` and V4's `copy()` alike. Two things are gated explicitly in `Scheduler._checkpoint_room` — a request stopping on this step (nothing follows it: no forward to fork into, no batch to copy on) and speculative decode *for a forking class*. The spec exclusion has two independent reasons and either alone is decisive: the spec path's state index tensor has no read-side counterpart, so a fork must never reach it; and a spec step commits `1 + accepted_drafts` tokens, which is what a fork's successor actually gets — the rest is rolled back and re-forwarded — so no promise made when the checkpoint is decided can be kept, and by the time acceptance is known the state is already split across two groups that no single read index spans. That second reason is why DeepSeek-V4 copies rather than forks: it is the only way to checkpoint a decode boundary, which is exactly the boundary multi-turn reuse resumes from. Prefill checkpointing stays live on forking models because `min_fork_tokens` keeps prompt behind every rung and prompt always forwards down the non-spec path. Arithmetic for both compressor rings, replayed from `compress_plan.py`: `logs_claude/verify_v4_min_fork.py`.

**Checkpoints where someone asked for one.** The grid is a guess about where reuse will want to resume; the requests themselves know. Whenever the state gates cut a hit short, `can_allocate` asks the same question a second time with every ladder assumed dense (`resumable_hit(..., assume_checkpointed=True)`), and the gap between the two answers is reuse that exists and is being declined only for want of a checkpoint. `BlockManager._record_checkpoint_demand` turns that into one extra rung for that seq (`Sequence.checkpoint_demand_pos`), which `checkpoint_cut` cuts a chunk at and `checkpointers_at` accepts — off the same field, so the cut and the keep cannot drift. It is decided at admission, where the counterfactual and the admitted hit are both in hand: the hit survives only as `num_cached_tokens`, which the scheduler advances as chunks land, and under pipeline parallelism is already past the chunk by the time `hash_blocks` runs. The request that discovers the gap is the one that pays for it, which is the right way round: it collects none of that reuse and has to compute the prefix anyway. The counterfactual must keep every *other* class's gate applied — a boundary some other class cannot resume from either is not worth checkpointing this one at — and demand below one interval is dropped, so a workload that keeps no checkpoints today gains no chunk cuts from this. The property is self-limiting: the first request finds nothing cached, the second finds the gap and pays one cut, and the third hits outright and finds no gap. `Lost-to-checkpoint` in the cache-stats line is the gap, measured; it falling to zero is the feature working.

**What the interval is pacing.** A checkpoint costs the request that takes it a forward: its prompt gets cut at the rung, and the extra forward is paid whether or not anyone ever resumes from it. That cost is the same under both mechanisms — a checkpoint holds the state as of the end of a forward, so the forward has to end on the boundary either way. Copy additionally consumes PAGE capacity and runs one descriptor-driven scatter or gather per checkpoint or resume; the physical PAGE ids need not be contiguous. That is why the interval counts tokens rather than blocks, and why a prompt shorter than one interval checkpoints nothing at all — on a workload of short, mutually-distinct prompts the hit rate is 0 by construction, so the feature has to be free there. Measured on Qwen3.5-27B tp2 at ISL/OSL 1024/1024, checkpointing unconditionally at the last eligible boundary cost 17.5% of total throughput for zero resumes.

### Allocation (`allocate`)

Called during prefill scheduling for new sequences:

```python
def allocate(self, seq: Sequence):
```

**KV Cache allocation:**

1. Iterates over `seq.num_blocks` blocks.
2. For each block, computes hash if the block is full (`len(token_ids) == block_size`). Partial (last) blocks get `hash = -1`.
3. If prefix caching is enabled, looks up `kv.lookup(h)`:
   - **Cache hit:** Verifies `token_ids` match, then `kv.claim(block_id)` — `ref_count += 1` if live, otherwise take it off the free list with its contents intact. Deliberately not `kv.allocate`, whose reset would drop the hash and destroy the entry for everyone else. Increments `seq.num_cached_tokens` by `block_size`.
   - **Cache miss:** `kv.pop()` then `kv.allocate()`.
4. Full blocks are registered by `kv.publish(block_id, h, token_ids)`.

**Per-request cache allocation (if `seq.has_per_req_cache`):**

Pops `state_slots_per_req` indices from the state pool's free list into `seq.state_slots` — one committed state plus a rollback slot per speculated token. They need not be adjacent, and nothing downstream may reconstruct the set by arithmetic on a base. `seq.state_slot` is a property over element 0: the one every non-speculative path reads and writes, and the one a checkpoint is. The resident Active Slot's bytes were already taken out of the budget at sizing time.

On a fork checkpoint hit, the slot holding the checkpoint is claimed as `seq.state_fork_src` and the request writes a fresh slot for one forward. On a copy checkpoint hit, the request keeps its newly allocated Active Slot and queues a PAGE gather into it in the batch's `StateMaintenanceOps`. Either way the checkpoint is **one** slot wide — a resumed prefix has no speculation to roll back — so a resume costs the pool exactly what a cold start does.

### Deallocation (`deallocate`)

Called when a sequence finishes or is preempted:

```python
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        self.kv.free(block_id)
    seq.num_cached_tokens = 0
    # ... checkpoint intents that describe this slot's bytes die with it:
    # `forget_pending` for a queued PAGE store, `cancel_midstep` for a
    # reservation whose forward is not going to run.
    del seq.block_table[:]
    if seq.has_per_req_cache and seq.state_slots:
        self.state.release_many(seq.state_slots)
        self.state.drop_reader(seq.state_fork_src)
        seq.state_slots = []
        seq.state_fork_src = -1
```

**KV Cache deallocation:** Blocks are released in reverse order. Shared blocks (with `ref_count > 1` from prefix caching) are not freed until all referencing sequences release them.

**Per-request cache deallocation (if `seq.has_per_req_cache`):**

1. Returns **every** slot the seq held — `release_many`, not one `release`. The committed slot and its speculation scratch go back together; the scratch is this request's alone and means nothing to the next one. Releasing only `state_slots[0]` leaks `num_spec` slots per request, which admission then cannot see.
2. Drops the pending fork source: no next forward is going to read it, so it should not sit out a pass for a reader that no longer exists.
3. Clears `state_slots` and `state_fork_src`.

A checkpoint the seq resumed from is not released here — it went back to the free list under the state index when it was taken, and it is read-only.

### Can-allocate and can-append checks

```python
def can_allocate(self, seq: Sequence) -> int:
    """Return the number of cache-hit blocks (>=0) if seq fits, else -1."""
    # State cache has its own reservation; admission only needs a free slot
    # index, not extra paged blocks.
    if seq.has_per_req_cache and not self.state.has_free():
        return -1
    if not self.enable_prefix_caching:
        if not self.kv.has_free(self.num_pool_blocks(len(seq))):
            return -1
    # ... (prefix caching dry-run returns the contiguous hit-block count)

def can_append(self, seq: Sequence, num_new_tokens: int = 1) -> bool:
    seq_len = len(seq)
    current_blocks = len(seq.block_table)
    needed_blocks = (seq_len + num_new_tokens + self.block_size - 1) // self.block_size
    new_blocks_needed = max(0, needed_blocks - current_blocks)
    return self.kv.has_free(new_blocks_needed)
```

- `can_allocate` checks that:
  - Enough free KV blocks exist for the full sequence. A windowed architecture adds nothing here: its window is a ring inside the per-request state slot, so the slot check below covers it.
  - At least one per-request cache slot group is available if the sequence has `has_per_req_cache=True`. Per-request state costs no paged blocks — its bytes were reserved ahead of the paged pool at sizing time.
  
- `can_append` checks whether a decode step needs a new block. Calculates the required block count given `num_new_tokens` (typically `mtp_k + 1` for speculative decode) and returns whether enough free blocks remain.

### May-append (decode extension)

```python
def may_append(self, seq: Sequence, num_new_tokens: int = 1):
```

Called during decode scheduling to extend a sequence's block table:

1. If the sequence length modulo `block_size` falls within `(0, num_new_tokens]`, or `block_size == 1`, a new block is needed:
   - Takes a block via `kv.pop()` + `kv.allocate()` and appends to `block_table`.
   - For `block_size == 1`, immediately computes and stores the hash.
2. If `seq_len % block_size == 0`, the last block is now full — computes and stores its hash using the chained prefix.
3. Otherwise the last block is partially filled with `hash = -1` (hash deferred until full).

## Prefix caching

Prefix caching enables sharing KV cache blocks across sequences that share a common prompt prefix, avoiding redundant computation.

### Hash function

ATOM uses `xxhash64` (via the `xxhash` Python library) for fast, collision-resistant block hashing:

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

### Hash chaining

Blocks form a hash chain: each block's hash incorporates the previous block's hash as a prefix. This ensures that two blocks with identical token content but different preceding context produce different hashes.

- First block: `compute_hash(token_ids, prefix=-1)` (no prefix).
- Subsequent blocks: `compute_hash(token_ids, prefix=prev_block.hash)`.
- Only **full** blocks (where `len(token_ids) == block_size`) receive a hash. Partial blocks have `hash = -1` and are not cached.

### Cache lookup during allocation

During `allocate()`, for each full block:

1. Compute the block hash via the chain.
2. Look up `kv.lookup(h)` (-1 on a miss).
3. If found, verify `kv.block(block_id).token_ids == token_ids` (guard against hash collisions).
4. **Hit:** `kv.claim(block_id)`. Add `block_size` to `seq.num_cached_tokens`.
5. **Miss (or first miss in chain):** Once a cache miss occurs, all subsequent blocks in the sequence are also misses (`cache_miss = True` is sticky). Allocate fresh blocks from the free list.

### Reference counting

- On allocation: `block.reset()` sets `ref_count = 1`.
- On cache hit for an in-use block: `ref_count += 1`.
- On deallocation: `ref_count -= 1`. Block returns to free list only when `ref_count == 0`.
- Shared blocks (prefix cache hits) have `ref_count > 1`.

### Enabling prefix caching

Set `enable_prefix_caching=True` in `Config`. When disabled, the hash lookup in `allocate()` is skipped entirely (`block_id` is always `-1`).

## Postprocessing

`Scheduler.postprocess()` is called after the model forward pass to update sequences with sampled tokens, check stop conditions, generate streaming output, and clean up finished sequences.

### Signature

```python
def postprocess(
    self,
    seqs: list[Sequence],
    fwd_output: ScheduledBatchOutput,
    stream_output_queue=None,
) -> list[Sequence]:
```

### Token appending

For each running sequence whose ID appears in `fwd_output.req_ids`:

- **Deferred output or speculative decode with EOS:** Replaces placeholder tokens in-place:
  ```python
  seq.token_ids[-num_placeholder:] = token_ids
  seq.output_tokens[-num_placeholder:] = token_ids
  ```
- **Normal path:** Calls `seq.append_token(token_id)` for each accepted token, which appends to `token_ids`, updates `output_tokens`, `last_token`, and `num_tokens`.

### Stop condition checking

The postprocessor checks stop conditions in priority order:

1. **Stop token sequences:** Compares the tail of `seq.token_ids` against each entry in `seq.stop_token_sequences`. Also checks the MTP-adjusted position for speculative decode. Sets `leave_reason = "stop_sequence"`.
2. **EOS token:** If `self.eos_token_id` appears in the accepted tokens and `seq.ignore_eos` is `False`. Sets `leave_reason = "eos"`.
3. **Stop token IDs:** If any accepted token is in `self.stop_token_ids` (from `Config.stop_token_ids`, derived from the model's generation config). Sets `leave_reason = "stop_{token_id}"`.
4. **Max tokens:** If `seq.num_completion_tokens >= seq.max_tokens`. Sets `leave_reason = "max_tokens"`.

### Stream output

When `stream_output_queue` is provided, the scheduler creates a `RequestOutput` for each processed sequence:

```python
request_output = RequestOutput(
    request_id=seq.id,
    output_tokens=output_tokens_list,
    finished=(leave_reason is not None),
    finish_reason=leave_reason,
)
```

`RequestOutput` fields:

| Field | Type | Description |
|---|---|---|
| `request_id` | `int` | Sequence ID |
| `output_tokens` | `list[int]` | Newly generated tokens since last callback |
| `finished` | `bool` | Whether the sequence is done |
| `finish_reason` | `Optional[str]` | One of: `"eos"`, `"max_tokens"`, `"stop_sequence"`, `"stop_{token_id}"`, or `None` |

Stream outputs are batched and put onto `stream_output_queue` via `put_nowait`.

### Sequence cleanup

For finished sequences:
1. Set `seq.status = SequenceStatus.FINISHED`.
2. Call `block_manager.deallocate(seq)` to free KV cache blocks.
3. Remove from the `running` deque.
4. Return in the `finished_seqs` list.

### Placeholder insertion

When speculative decoding or deferred output is active, placeholder EOS tokens are appended to still-running sequences to reserve KV cache slots for the next step:

```python
if need_placeholder:
    for seq in seqs:
        if seq.status == SequenceStatus.RUNNING:
            for _ in range(seq.num_placeholder):
                seq.append_token(self.eos_token_id)
```

The placeholder count is determined as follows:

- **For sequences processed in this step** (had output in `fwd_output`): always `1 + mtp_k`, regardless of mode.
- **For sequences not processed** (skipped in this step): the count depends on the batch-level mode:
  - Deferred output + speculative: `mtp_k + 1`
  - Deferred output only: `1`
  - Speculative only: `mtp_k`

## Speculative decoding integration

ATOM supports Multi-Token Prediction (MTP) speculative decoding, where a draft model proposes `mtp_k` additional tokens per step.

### Scheduler tracking

```python
self.use_spec = config.speculative_config is not None
self.mtp_k: int = config.speculative_config.num_speculative_tokens if self.use_spec else 0
self.total_draft_tokens = 0
self.total_accepted_tokens = 0
```

Note: `SpeculativeConfig` currently enforces `num_speculative_tokens == 1`.

### Draft tokens in scheduling

During decode scheduling:
- If `seq.spec_token_ids` is non-empty, the draft tokens are recorded in `scheduled_spec_decode_tokens[seq.id]`.
- `num_new_tokens = mtp_k + 1` (1 target + `mtp_k` draft tokens), so `may_append` reserves enough block space.
- The `ScheduledBatch` carries `num_spec_step = mtp_k` and the `scheduled_spec_decode_tokens` dict.

### Acceptance statistics

```python
def update_spec_stats(self, num_accepted_tokens):
    self.total_draft_tokens += self.mtp_k
    self.total_accepted_tokens += num_accepted_tokens - self.mtp_k
```

Every 1000 draft tokens, the acceptance rate is logged:

```
[MTP Stats] Total draft tokens: 5000, Accepted: 3750, Acceptance rate: 75.00%
```

### Draft token storage on sequences

After postprocessing, accepted draft token IDs for the next step are stored on the sequence:

```python
if draft_token_ids and seq.id in draft_token_ids:
    seq.spec_token_ids = draft_token_ids[seq.id]
```

These are picked up by the scheduler on the next `schedule()` call.

## Sequence management

The `Sequence` class represents a single request throughout its lifecycle.

### Constructor

```python
class Sequence:
    def __init__(
        self,
        token_ids: list[int],
        block_size: int,
        sampling_params=SamplingParams(),
        stop_token_sequences: list[list[int]] = None,
        stream_callback: Optional[Callable[[Any], None]] = None,
        id=None,
    ):
```

### Core fields

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Auto-incrementing unique ID (from `itertools.count`) |
| `token_ids` | `list[int]` | Full token sequence (prompt + completion) |
| `block_size` | `int` | KV cache block size (from config) |
| `status` | `SequenceStatus` | Current lifecycle state |
| `type` | `SequenceType` | Current step type (`DUMMY`, `PREFILL`, `DECODE`) |
| `num_tokens` | `int` | Total tokens (prompt + completion); property with setter that also updates `num_blocks` and `last_block_num_tokens` |
| `num_prompt_tokens` | `int` | Number of prompt tokens (fixed at init) |
| `num_cached_tokens` | `int` | Tokens served from prefix cache |
| `block_table` | `list[int]` | Ordered list of block IDs assigned to this sequence |
| `has_per_req_cache` | `bool` | Whether the model's attention type maintains per-request state outside the paged KV pool (set at sequence init; True for GDN-based models, future stateful attentions) |
| `state_slots` | `list[int]` | Every stateful-attention slot the sequence holds, in allocation order: `[0]` is the committed state, `[1:]` is speculation rollback. Not adjacent, and no backend may rebuild the set by arithmetic on a base. Assigned by BlockManager during allocation, `[]` if unallocated |
| `state_slot` | `int` | Property over `state_slots[0]` — the slot the forward reads and writes, the one a fork gives away, the one a checkpoint is. `-1` when the sequence holds none |
| `state_fork_src` | `int` | Slot the next forward READS its incoming state from when a state fork is pending; `-1` (read == write) otherwise. Always a single slot: a checkpoint is one slot wide. Set by BlockManager on publish/resume, cleared by the scheduler once a batch has carried it |
| `last_token` | `int` | Most recently appended token ID |
| `temperature` | `float` | Sampling temperature (from `SamplingParams`) |
| `max_tokens` | `int` | Max completion tokens (from `SamplingParams`, default 64) |
| `ignore_eos` | `bool` | Whether to ignore EOS tokens (from `SamplingParams`) |
| `stop_strings` | `Optional[list[str]]` | Stop strings (from `SamplingParams`) |
| `stop_token_sequences` | `list[list[int]]` | Token-level stop sequences |
| `stream_callback` | `Optional[Callable]` | Per-sequence stream callback |
| `output_tokens` | `list[int]` | Cache of newly generated tokens |
| `spec_token_ids` | `list[int]` | Speculative draft token IDs for next step |
| `num_placeholder` | `int` | Number of placeholder tokens inserted for speculative/deferred output |

### Timing fields

| Field | Type | Description |
|---|---|---|
| `arrive_time` | `float` | Timestamp when the sequence entered the scheduler |
| `first_token_time` | `float` | Timestamp of the first completion token (TTFT measurement) |
| `leave_time` | `float` | Timestamp when the sequence finished |
| `leave_reason` | `str` | Reason for finishing (e.g., `"eos"`, `"max_tokens"`, `"stop_sequence"`) |

### Computed properties

| Property | Returns |
|---|---|
| `num_completion_tokens` | `num_tokens - num_prompt_tokens` |
| `prompt_token_ids` | `token_ids[:num_prompt_tokens]` |
| `completion_token_ids` | `token_ids[num_prompt_tokens:]` |
| `num_cached_blocks` | `num_cached_tokens // block_size` |
| `is_finished` | `status == SequenceStatus.FINISHED` |

### num_tokens setter

Setting `num_tokens` triggers derived field updates:

```python
@num_tokens.setter
def num_tokens(self, value):
    self._num_tokens = value
    self.num_blocks = (value + self.block_size - 1) // self.block_size
    self.last_block_num_tokens = self._num_tokens - (self.num_blocks - 1) * self.block_size
```

### Lifecycle

```
                          allocate blocks
   add(seq) ---------> WAITING ---------> RUNNING (PREFILL)
                          ^                    |
                          |                    | next schedule() step
                     preempt()                 v
                          |              RUNNING (DECODE) <--+
                          +--- can't append    |             |
                                               | stop condition met
                                               v
                                           FINISHED
                                               |
                                               | deallocate blocks
                                               v
                                         (removed from running)
```

### SequenceStatus enum

| Value | Meaning |
|---|---|
| `WAITING` | In the waiting queue, pending prefill |
| `RUNNING` | Actively being processed (prefill or decode) |
| `FINISHED` | Stop condition met, blocks deallocated |
| `EXIT_ENGINE` | Sentinel for engine shutdown |

### SequenceType enum

| Value | Meaning |
|---|---|
| `DUMMY` | Initial state before scheduling |
| `PREFILL` | Currently in prefill phase |
| `DECODE` | Currently in decode phase |

## Source files

| File | Description |
|---|---|
| `atom/model_engine/scheduler.py` | `Scheduler`, `ScheduledBatch`, `ScheduledBatchOutput` — scheduling algorithm, postprocessing |
| `atom/model_engine/engine_stats.py` | `EngineStats` — speculative-decode acceptance, prefix-cache hit, and throughput statistics |
| `atom/model_engine/block_manager.py` | `Block`, `BlockManager` — paged KV cache block pool, allocation/deallocation, prefix caching with xxhash64 |
| `atom/model_engine/sequence.py` | `Sequence`, `SequenceStatus`, `SequenceType` — request lifecycle, token management, timing |
| `atom/model_engine/request.py` | `RequestOutput` — streaming output dataclass with `request_id`, `output_tokens`, `finished`, `finish_reason` |
| `atom/config.py` | `Config` — scheduling-related fields (`max_num_seqs`, `max_num_batched_tokens`, `kv_cache_block_size`, `enable_prefix_caching`, `scheduler_delay_factor`), `SpeculativeConfig` |
| `atom/sampling_params.py` | `SamplingParams` — `temperature`, `max_tokens`, `ignore_eos`, `stop_strings` |
