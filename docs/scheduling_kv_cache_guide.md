# ATOM Scheduling & KV Cache Guide

ATOM (AiTer Optimized Model) uses a prefill-first scheduler with paged KV cache block management to drive LLM inference on AMD ROCm/HIP GPUs. This guide covers the scheduling algorithm, batch construction, block-level KV cache management, prefix caching, postprocessing, speculative decoding integration, and sequence lifecycle.

## Quick Reference

| Class | File | Purpose |
|---|---|---|
| `Scheduler` | `atom/model_engine/scheduler.py` | Orchestrates prefill/decode scheduling, preemption, and postprocessing |
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

---

## 1. Scheduling Algorithm

The scheduler implements a **prefill-first** policy: all waiting (prefill) requests are scheduled before any running (decode) requests. The entry point is `Scheduler.schedule()`, which returns a `(ScheduledBatch, dict[int, Sequence])` tuple or `None` if both queues are empty.

### 1.1 Scheduler Initialization

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

The scheduler maintains two deques -- `waiting` (pending prefill) and `running` (active decode) -- plus a `BlockManager` for KV cache allocation.

### 1.2 Schedule Flow

`Scheduler.schedule()` proceeds in two phases:

**Phase 1 -- Prefill scheduling:**

1. While the delay gate passes (`_passed_delay`), the waiting queue is non-empty, and `num_seqs_prefill < max_num_seqs`:
   - Peek the first waiting sequence.
   - Compute `num_new_tokens = seq.num_tokens - seq.num_cached_tokens` (prefix cache hits reduce new tokens).
   - If `num_batched_tokens + num_new_tokens > max_num_batched_tokens` or `block_manager.can_allocate(seq)` returns `False`, break.
   - Otherwise: allocate blocks, set `seq.status = RUNNING`, `seq.type = PREFILL`, move from `waiting` to `running`.
2. If any prefill sequences were scheduled, return the batch immediately (no decode mixing).

**Phase 2 -- Decode scheduling (only when zero prefills were scheduled):**

1. Pop sequences from `running` up to `max_num_seqs`.
2. For each sequence, check `block_manager.can_append(seq)`.
3. If a block cannot be appended, **preempt** the last running sequence (move it back to `waiting` with status `WAITING` and deallocate its blocks).
4. If the sequence has speculative draft tokens (`seq.spec_token_ids`), record them in `scheduled_spec_decode_tokens`.
5. Call `block_manager.may_append(seq, num_new_tokens)` where `num_new_tokens = mtp_k + 1`.
6. Re-insert all scheduled sequences back into `running` (preserving order).

### 1.3 Delay Factor

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

### 1.4 Preemption

When a decode step cannot extend a sequence's KV cache (no free blocks), the scheduler preempts the **last** running sequence:

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
```

The preempted sequence is pushed to the front of the waiting queue and its blocks are fully deallocated, so it will be re-prefilled on the next scheduling cycle.

---

## 2. ScheduledBatch Structure

`ScheduledBatch` is constructed by `Scheduler.schedule()` and passed to the model runner. It is a frozen snapshot of batch metadata.

### 2.1 Constructor Signature

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
        scheduled_spec_decode_tokens: dict[int, list[int]] = {},
    ):
```

### 2.2 Fields

| Field | Type | Description |
|---|---|---|
| `req_ids` | `list[int]` | Sequence IDs in batch order (`list(seqs.keys())`) |
| `scheduled_tokens` | `list[list[int]]` | Last `num_tokens` token IDs per sequence (the tokens to process) |
| `temperatures` | `list[float]` | Sampling temperature per sequence |
| `context_lens` | `list[int]` | Total token count per sequence (`seq.num_tokens`) |
| `block_tables` | `list[list[int]]` | Block ID tables for sequences that have block tables |
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
| `scheduled_spec_decode_tokens` | `dict[int, list[int]]` | Draft token IDs per sequence ID from prior speculative step |

### 2.3 ScheduledBatchOutput

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

---

## 3. Block Manager

The `BlockManager` implements paged KV cache management with fixed-size blocks.

### 3.1 Block Class

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id   # Unique integer ID
        self.ref_count = 0         # Number of sequences referencing this block
        self.hash = -1             # xxhash64 digest for prefix caching (-1 = unhashed)
        self.token_ids = []        # Token IDs stored in this block
```

Methods:
- `update(hash, token_ids)` -- Sets the block's hash and token content.
- `reset()` -- Sets `ref_count = 1`, `hash = -1`, `token_ids = []` (used on fresh allocation).

### 3.2 BlockManager Initialization

```python
class BlockManager:
    def __init__(self, config: Config):
        block_size = config.kv_cache_block_size      # Tokens per block (default 16)
        num_blocks = config.num_kvcache_blocks        # Total blocks in pool
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.enable_prefix_caching = config.enable_prefix_caching
```

The block pool is pre-allocated at startup. `free_block_ids` is a deque for O(1) pop/push, `used_block_ids` tracks active blocks, and `hash_to_block_id` maps content hashes to block IDs for prefix caching.

### 3.3 Allocation (`allocate`)

Called during prefill scheduling for new sequences:

```python
def allocate(self, seq: Sequence):
```

1. Iterates over `seq.num_blocks` blocks.
2. For each block, computes hash if the block is full (`len(token_ids) == block_size`). Partial (last) blocks get `hash = -1`.
3. If prefix caching is enabled, looks up `hash_to_block_id`:
   - **Cache hit:** Verifies `token_ids` match. If the block is already in `used_block_ids`, increments `ref_count`. If it was evicted but still in the free list, re-allocates it. Increments `seq.num_cached_tokens` by `block_size`.
   - **Cache miss:** Allocates from `free_block_ids[0]`.
4. Full blocks are registered in `hash_to_block_id`.

### 3.4 Deallocation (`deallocate`)

Called when a sequence finishes or is preempted:

```python
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

Blocks are released in reverse order. Shared blocks (with `ref_count > 1` from prefix caching) are not freed until all referencing sequences release them.

### 3.5 Can-Allocate and Can-Append Checks

```python
def can_allocate(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= seq.num_blocks

def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

- `can_allocate` checks that enough free blocks exist for the full sequence.
- `can_append` checks whether a decode step needs a new block. A new block is needed only when `len(seq) % block_size == 1` (the previous block just filled up), requiring exactly 1 free block.

### 3.6 May-Append (Decode Extension)

```python
def may_append(self, seq: Sequence, num_new_tokens: int = 1):
```

Called during decode scheduling to extend a sequence's block table:

1. If the sequence length modulo `block_size` falls within `(0, num_new_tokens]`, or `block_size == 1`, a new block is needed:
   - Allocates from `free_block_ids` and appends to `block_table`.
   - For `block_size == 1`, immediately computes and stores the hash.
2. If `seq_len % block_size == 0`, the last block is now full -- computes and stores its hash using the chained prefix.
3. Otherwise the last block is partially filled with `hash = -1` (hash deferred until full).

---

## 4. Prefix Caching

Prefix caching enables sharing KV cache blocks across sequences that share a common prompt prefix, avoiding redundant computation.

### 4.1 Hash Function

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

### 4.2 Hash Chaining

Blocks form a hash chain: each block's hash incorporates the previous block's hash as a prefix. This ensures that two blocks with identical token content but different preceding context produce different hashes.

- First block: `compute_hash(token_ids, prefix=-1)` (no prefix).
- Subsequent blocks: `compute_hash(token_ids, prefix=prev_block.hash)`.
- Only **full** blocks (where `len(token_ids) == block_size`) receive a hash. Partial blocks have `hash = -1` and are not cached.

### 4.3 Cache Lookup During Allocation

During `allocate()`, for each full block:

1. Compute the block hash via the chain.
2. Look up `hash_to_block_id.get(h, -1)`.
3. If found, verify `self.blocks[block_id].token_ids == token_ids` (guard against hash collisions).
4. **Hit:** Reuse the block. If already in `used_block_ids`, increment `ref_count`. Add `block_size` to `seq.num_cached_tokens`.
5. **Miss (or first miss in chain):** Once a cache miss occurs, all subsequent blocks in the sequence are also misses (`cache_miss = True` is sticky). Allocate fresh blocks from the free list.

### 4.4 Reference Counting

- On allocation: `block.reset()` sets `ref_count = 1`.
- On cache hit for an in-use block: `ref_count += 1`.
- On deallocation: `ref_count -= 1`. Block returns to free list only when `ref_count == 0`.
- Shared blocks (prefix cache hits) have `ref_count > 1`.

### 4.5 Enabling Prefix Caching

Set `enable_prefix_caching=True` in `Config`. When disabled, the hash lookup in `allocate()` is skipped entirely (`block_id` is always `-1`).

---

## 5. Postprocessing

`Scheduler.postprocess()` is called after the model forward pass to update sequences with sampled tokens, check stop conditions, generate streaming output, and clean up finished sequences.

### 5.1 Signature

```python
def postprocess(
    self,
    seqs: list[Sequence],
    fwd_output: ScheduledBatchOutput,
    stream_output_queue=None,
) -> list[Sequence]:
```

### 5.2 Token Appending

For each running sequence whose ID appears in `fwd_output.req_ids`:

- **Deferred output or speculative decode with EOS:** Replaces placeholder tokens in-place:
  ```python
  seq.token_ids[-num_placeholder:] = token_ids
  seq.output_tokens[-num_placeholder:] = token_ids
  ```
- **Normal path:** Calls `seq.append_token(token_id)` for each accepted token, which appends to `token_ids`, updates `output_tokens`, `last_token`, and `num_tokens`.

### 5.3 Stop Condition Checking

The postprocessor checks stop conditions in priority order:

1. **Stop token sequences:** Compares the tail of `seq.token_ids` against each entry in `seq.stop_token_sequences`. Also checks the MTP-adjusted position for speculative decode. Sets `leave_reason = "stop_sequence"`.
2. **EOS token:** If `self.eos_token_id` appears in the accepted tokens and `seq.ignore_eos` is `False`. Sets `leave_reason = "eos"`.
3. **Stop token IDs:** If any accepted token is in `self.stop_token_ids` (from `Config.stop_token_ids`, derived from the model's generation config). Sets `leave_reason = "stop_{token_id}"`.
4. **Max tokens:** If `seq.num_completion_tokens >= seq.max_tokens`. Sets `leave_reason = "max_tokens"`.

### 5.4 Stream Output

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

### 5.5 Sequence Cleanup

For finished sequences:
1. Set `seq.status = SequenceStatus.FINISHED`.
2. Call `block_manager.deallocate(seq)` to free KV cache blocks.
3. Remove from the `running` deque.
4. Return in the `finished_seqs` list.

### 5.6 Placeholder Insertion

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

---

## 6. Speculative Decoding Integration

ATOM supports Multi-Token Prediction (MTP) speculative decoding, where a draft model proposes `mtp_k` additional tokens per step.

### 6.1 Scheduler Tracking

```python
self.use_spec = config.speculative_config is not None
self.mtp_k: int = config.speculative_config.num_speculative_tokens if self.use_spec else 0
self.total_draft_tokens = 0
self.total_accepted_tokens = 0
```

Note: `SpeculativeConfig` currently enforces `num_speculative_tokens == 1`.

### 6.2 Draft Tokens in Scheduling

During decode scheduling:
- If `seq.spec_token_ids` is non-empty, the draft tokens are recorded in `scheduled_spec_decode_tokens[seq.id]`.
- `num_new_tokens = mtp_k + 1` (1 target + `mtp_k` draft tokens), so `may_append` reserves enough block space.
- The `ScheduledBatch` carries `num_spec_step = mtp_k` and the `scheduled_spec_decode_tokens` dict.

### 6.3 Acceptance Statistics

```python
def update_spec_stats(self, num_accepted_tokens):
    self.total_draft_tokens += self.mtp_k
    self.total_accepted_tokens += num_accepted_tokens - self.mtp_k
```

Every 1000 draft tokens, the acceptance rate is logged:

```
[MTP Stats] Total draft tokens: 5000, Accepted: 3750, Acceptance rate: 75.00%
```

### 6.4 Draft Token Storage on Sequences

After postprocessing, accepted draft token IDs for the next step are stored on the sequence:

```python
if draft_token_ids and seq.id in draft_token_ids:
    seq.spec_token_ids = draft_token_ids[seq.id]
```

These are picked up by the scheduler on the next `schedule()` call.

---

## 7. Sequence Management

The `Sequence` class represents a single request throughout its lifecycle.

### 7.1 Constructor

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

### 7.2 Core Fields

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

### 7.3 Timing Fields

| Field | Type | Description |
|---|---|---|
| `arrive_time` | `float` | Timestamp when the sequence entered the scheduler |
| `first_token_time` | `float` | Timestamp of the first completion token (TTFT measurement) |
| `leave_time` | `float` | Timestamp when the sequence finished |
| `leave_reason` | `str` | Reason for finishing (e.g., `"eos"`, `"max_tokens"`, `"stop_sequence"`) |

### 7.4 Computed Properties

| Property | Returns |
|---|---|
| `num_completion_tokens` | `num_tokens - num_prompt_tokens` |
| `prompt_token_ids` | `token_ids[:num_prompt_tokens]` |
| `completion_token_ids` | `token_ids[num_prompt_tokens:]` |
| `num_cached_blocks` | `num_cached_tokens // block_size` |
| `is_finished` | `status == SequenceStatus.FINISHED` |

### 7.5 num_tokens Setter

Setting `num_tokens` triggers derived field updates:

```python
@num_tokens.setter
def num_tokens(self, value):
    self._num_tokens = value
    self.num_blocks = (value + self.block_size - 1) // self.block_size
    self.last_block_num_tokens = self._num_tokens - (self.num_blocks - 1) * self.block_size
```

### 7.6 Lifecycle

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

### 7.7 SequenceStatus Enum

| Value | Meaning |
|---|---|
| `WAITING` | In the waiting queue, pending prefill |
| `RUNNING` | Actively being processed (prefill or decode) |
| `FINISHED` | Stop condition met, blocks deallocated |
| `EXIT_ENGINE` | Sentinel for engine shutdown |

### 7.8 SequenceType Enum

| Value | Meaning |
|---|---|
| `DUMMY` | Initial state before scheduling |
| `PREFILL` | Currently in prefill phase |
| `DECODE` | Currently in decode phase |

---

## Source Files

| File | Description |
|---|---|
| `atom/model_engine/scheduler.py` | `Scheduler`, `ScheduledBatch`, `ScheduledBatchOutput` -- scheduling algorithm, postprocessing, speculative decode stats |
| `atom/model_engine/block_manager.py` | `Block`, `BlockManager` -- paged KV cache block pool, allocation/deallocation, prefix caching with xxhash64 |
| `atom/model_engine/sequence.py` | `Sequence`, `SequenceStatus`, `SequenceType` -- request lifecycle, token management, timing |
| `atom/model_engine/request.py` | `RequestOutput` -- streaming output dataclass with `request_id`, `output_tokens`, `finished`, `finish_reason` |
| `atom/config.py` | `Config` -- scheduling-related fields (`max_num_seqs`, `max_num_batched_tokens`, `kv_cache_block_size`, `enable_prefix_caching`, `scheduler_delay_factor`), `SpeculativeConfig` |
| `atom/sampling_params.py` | `SamplingParams` -- `temperature`, `max_tokens`, `ignore_eos`, `stop_strings` |
