# State-Cache LMCache Offload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Spill an evicted per-request state checkpoint (DeepSeek-V4 compressor state, GDN recurrent state) to LMCache instead of discarding it, and pull it back when a later request hits the same prefix hash.

**Architecture:** A `StateOffloadTier` owned by `StateGroupPool` (not a sibling in `BlockManager.state_caches` — every sibling is a *veto*, this tier *widens*). Spill hook at `pop()`'s eviction branch copies the group into a staging entry and enqueues it; a dedicated executor does D2H and writes one opaque `uint8` object to LMCache keyed by **ATOM's own `block_hashes[i]` xxhash64 integer**, bypassing `ChunkedTokenDatabase`. Lookup widens one line of `resumable_hit`. Load parks jointly with the KV load and indexes the destination group **only after H2D confirms**.

**Tech Stack:** Python 3.10+, PyTorch (ROCm), Triton (existing `triton_kv_staging.py` kernels, unmodified), LMCache v1 (`storage_manager`, `CacheEngineKey`, `MemoryFormat.BINARY`), pytest.

## Global Constraints

- **Scope: DeepSeek-V4 and GDN/Qwen3-Next, uniform from the start.** No backend-specific branches outside `state_entry_views`.
- **Key is `block_hashes[i]`** — the same xxhash64 integer `BlockManager.compute_hash` produces (`block_manager.py:152-157`). Never `ChunkedTokenDatabase`, never Python `hash()`.
- **Never hard-code an entry size.** `StateArena.entry_bytes` (`state_arena.py:254`) is authoritative; log it at startup.
- **The staging ring is `K` extra *groups* appended to the state arena**, addressed by `num_groups + slot`. They are excluded from the pool's group range, so `state_entry_views(num_groups + slot)` names a staging group with no new addressing scheme — which is why `put` takes an entry index rather than a group.
- **K groups cost `K * entries_per_req` entries.** `state_entry_views` takes a *group* index and GDN slices `span = 1 + num_spec` rows per group (`gdn_attn.py:318-343`), so `extra_entries=K` alone would under-allocate the ring by `K * num_spec` rows and the last staging group would run off the tensor. Every sizing site passes `extra_entries=K * entries_per_req`. V4 declares `entries_per_req=1`, so there K entries and K groups coincide; GDN is where the distinction bites.
- **Two counts, not one — allocation vs. admission.** `state_pool(..., extra_entries=K)` already exists (`sub_pool_spec.py:89-97`) and `plan_pools` folds it straight into `plan.entries[name]` (`:221`), so `extra_entries` alone would inflate the *same* number `BlockManager` divides into groups (`block_manager.py:91-95`) — admission would hand out staging entries, which is the bug this constraint exists to prevent. The split is therefore made explicit on the plan:
  - `PoolPlan.entries[STATE_SLOT_CLASS]` — **allocation count**, `(max_num_seqs + K) * per_req`. Backends size tensors from it; `total_reserved_bytes` (checked to 3% at `model_runner.py:1915`) stays truthful because the extra entries are real HBM.
  - `PoolPlan.admission_entries[STATE_SLOT_CLASS]` — **admission count**, `max_num_seqs * per_req`, i.e. `entries - extra_entries`. Shipped to the engine process as `config.pool_entries` so `BlockManager.num_per_req_cache_groups` (`entries // per_req`) reads back exactly `max_num_seqs`, unchanged from today.
  Sizing declares K via `extra_entries`; nothing computes `+ K` in two places. Task 9 Step 4 implements this.
- **`num_state_slots` names the allocation count.** `deepseek_v4_attn.py:846-848` reads `pool_plan.entries[STATE_SLOT_CLASS]`, and with the split that is `num_groups + K` — exactly what `_slot_views()` must span for `state_entry_views(num_groups + slot)` to resolve. Its three other readers (`:1356`, `:1819`, `:3279`) are warmup guards and region builders that want the allocated extent, so they are correct unchanged. Any site that needs the *admission* count must read `admission_entries` and must say so.
- **The spill hook must be non-blocking.** `pop()` runs on the scheduler critical path (`state_pool.py:513`, `:548` via `block_manager.py:405,412`). Record hash, enqueue group, return. No D2H on that thread.
- **`P ≤ L` must hold** (P = state boundary, L = KV prefix loaded). Violation is silent wrong output. The clamp runs **after** `update_state_after_alloc` (before it `seq.num_cached_tokens` is stale — `connector.py:834-840`).
- **Index only after H2D confirms.** Never reproduce `_commit_pending`'s index-then-copy shape (`state_pool.py:525-553`) — that is the #1417 failure.
- **Triton kernels are not modified.** `_pack_chunk_major_kernel` / `_unpack_chunk_major_kernel` (`triton_kv_staging.py:15,62`) are already generic gathers.
- **`_build_meta` requires `seg.is_contiguous()`** (`triton_kv_staging.py:135`). GDN's `cache[:, slot:slot+span]` is strided → GDN must present per-layer contiguous slices.
- **Zero cost when disabled.** `_resumable_from` degenerates to the original `in`; the spill hook is one `if`.
- **`checkpoints_orphaned` (`unindex()`, `state_pool.py:621-644`) is spilled only when KV offload is also enabled.** `checkpoints_dropped` is never spilled — no bytes exist.
- **`min_fork_tokens` is not relaxed** for spilled hashes.
- Env vars follow the offload module's local `os.environ.get` / `_env_int` pattern (`connector.py:82`, `atom_lmcache_gpu_connector.py:93`), not `atom/utils/envs.py`.
- Tests must run without a GPU where possible (`tests/conftest.py` mocks nothing — import real classes; gate GPU-only tests with `pytest.importorskip("aiter")`).
- `black . && ruff check .` clean before every commit.

---

## File Structure

**New:**

| File | Responsibility |
|---|---|
| `atom/model_engine/state_offload.py` | `StateOffloadIndex` — the in-memory `hashes` set, the spill queue, the staging-entry ring. Scheduler-process only, no torch device work. |
| `atom/kv_transfer/offload/staged_transfer.py` | `StagedTransfer` — staging-buffer lifecycle, D2H/H2D, producer event, pipeline. Extracted verbatim from `atom_lmcache_gpu_connector.py`'s lower half. |
| `atom/kv_transfer/offload/state_object.py` | `StateByteCodec` — `CacheEngineKey` construction from an ATOM hash, pack/unpack of one state entry through `StagedTransfer`, `storage_manager.batched_put/get/contains`. |
| `atom/kv_transfer/offload/state_tier.py` | `StateOffloadTier` — worker-side spill/load driver, the dedicated executor, completion sets. |
| `tests/test_state_entry_views.py` | View-building for both backends. |
| `tests/test_state_offload_index.py` | Index, spill queue, staging depth K, `_resumable_from`. |
| `tests/test_staged_transfer.py` | Extraction is behaviour-preserving. |
| `tests/test_state_offload_clamp.py` | `P ≤ L`, joint park/wake, failure fallback. |

**Modified:**

| File | Change |
|---|---|
| `atom/model_ops/attentions/backends.py:170-188` | Add `state_entry_views` abstract method next to `copy_state_entries`. |
| `atom/model_ops/attentions/deepseek_v4_attn.py:783-830` | Implement `state_entry_views`; express `copy_state_entries` in terms of it. |
| `atom/model_ops/attentions/gdn_attn.py:318-343` | Same, with per-layer contiguous slices. |
| `atom/model_engine/state_pool.py:249-270, 427-464, 621-644` | Spill hook in `pop()`, `_resumable_from` in `resumable_hit`, conditional spill in `unindex()`. |
| `atom/model_engine/block_manager.py` | Thread the tier into `StateGroupPool` construction; expose it to the connector. |
| `atom/kv_transfer/offload/atom_lmcache_gpu_connector.py` | Delegate the lower half to `StagedTransfer`; KV orchestration unchanged. |
| `atom/kv_transfer/offload/connector.py` | State executor, state lookup in the scheduler path, `P ≤ L` clamp, joint park/wake, `failed_loading` fallback. |
| `atom/kv_transfer/offload/README.md` | Module Map table gains the three new modules + the new env vars. |

---

## Task 0: Measurement gate

The spec's Precondition: **do not build this if `checkpoints_evicted` is near zero on the target workload.** This task makes that measurable and is the gate on Tasks 1–9.

**Files:**
- Modify: `atom/model_engine/state_pool.py` (`checkpoint_fates`, `state_pool.py:555`)
- Modify: `atom/model_engine/scheduler.py` (periodic stats log)
- Test: `tests/test_state_checkpoint.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `StateGroupPool.checkpoint_fates() -> dict[str, int]` gains key `"evicted"` already present; adds nothing new to the API. Emits log line `state checkpoints: kept=%d evicted=%d orphaned=%d dropped=%d`.

- [ ] **Step 1: Write the failing test**

```python
def test_fates_report_every_counter():
    pool = StateGroupPool(num_groups=2, transfer=StateTransfer.copy(), hash_block_size=4)
    fates = pool.checkpoint_fates()
    assert set(fates) == {"kept", "evicted", "orphaned", "dropped"}
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_state_checkpoint.py::test_fates_report_every_counter -v`
Expected: PASS if the four keys already exist (likely), FAIL naming the missing key otherwise. If it passes, this step confirms the surface and you move to Step 3 without editing `state_pool.py`.

- [ ] **Step 3: Add the periodic log**

In the scheduler's existing stats-logging site, alongside the KV-cache-usage line:

```python
fates = self.block_manager.state_checkpoint_fates()
if any(fates.values()):
    logger.info(
        "state checkpoints: kept=%d evicted=%d orphaned=%d dropped=%d",
        fates["kept"], fates["evicted"], fates["orphaned"], fates["dropped"],
    )
```

and in `BlockManager`:

```python
def state_checkpoint_fates(self) -> dict[str, int]:
    """Summed fates across every state class, for the periodic stats line."""
    totals = {"kept": 0, "evicted": 0, "orphaned": 0, "dropped": 0}
    for cache in self.state_caches:
        for k, v in cache.checkpoint_fates().items():
            totals[k] = totals.get(k, 0) + v
    return totals
```

- [ ] **Step 4: Run the suite**

Run: `python -m pytest tests/test_state_checkpoint.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add atom/model_engine/state_pool.py atom/model_engine/block_manager.py atom/model_engine/scheduler.py tests/test_state_checkpoint.py
git commit -m "feat(state-pool): log checkpoint fates periodically"
```

- [ ] **Step 6: Gate**

Run the target workload, read the line. **If `evicted` is near zero, stop here and report that to the user.** Otherwise continue to Task 1.

---

## Task 1: `state_entry_views` on both backends

Factor out the view-building half of `copy_state_entries` so the offload path can address a group's bytes without duplicating layout knowledge. This is refactoring out the shared primitive — `backends.py:170-188` already states every backend declaring a state pool owes `copy_state_entries`.

**Files:**
- Modify: `atom/model_ops/attentions/backends.py:170-188`
- Modify: `atom/model_ops/attentions/deepseek_v4_attn.py:783-830`
- Modify: `atom/model_ops/attentions/gdn_attn.py:318-343`
- Test: `tests/test_state_entry_views.py` (create), `tests/test_gdn_state_copy.py` (must keep passing)

**Interfaces:**
- Consumes: nothing.
- Produces: `AttentionBackend.state_entry_views(self, group: int) -> list[torch.Tensor]` — every tensor **contiguous**, covering the whole of that group's per-request state. Order is stable across calls and identical for src/dst of the same class. Used by Task 4 as the segment list.

- [ ] **Step 1: Write the failing test**

Create `tests/test_state_entry_views.py`:

```python
# SPDX-License-Identifier: MIT
# state_entry_views must cover a group's whole state, contiguously.
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_ops.attentions.gdn_attn import GDNStateMixin

LAYERS, GROUPS = 3, 4
SHAPE_K, SHAPE_V = (2, 5), (2, 3, 4)


def build(num_spec: int):
    span = 1 + num_spec
    slots = GROUPS * span
    k = torch.arange(LAYERS * slots * 10, dtype=torch.float32)[
        : LAYERS * slots * SHAPE_K[0] * SHAPE_K[1]
    ].reshape((LAYERS, slots) + SHAPE_K)
    v = torch.zeros((LAYERS, slots) + SHAPE_V)
    stub = SimpleNamespace(
        num_spec=num_spec,
        model_runner=SimpleNamespace(mamba_k_cache=k, mamba_v_cache=v),
    )
    return stub, k, v, span


@pytest.mark.parametrize("num_spec", [0, 2])
def test_every_view_is_contiguous(num_spec):
    """_build_meta rejects a strided segment (triton_kv_staging.py:135)."""
    stub, _, _, _ = build(num_spec)
    views = GDNStateMixin.state_entry_views(stub, 1)
    assert views
    assert all(v.is_contiguous() for v in views)


def test_views_cover_the_whole_group_and_nothing_else():
    stub, k, v, span = build(num_spec=2)
    views = GDNStateMixin.state_entry_views(stub, 1)
    total = sum(int(x.numel()) for x in views)
    expected = LAYERS * span * (k[0, 0].numel() + v[0, 0].numel())
    assert total == expected


def test_writing_through_the_views_writes_the_cache():
    """Views must alias, not copy — the packer reads them in place."""
    stub, k, _, span = build(num_spec=1)
    for view in GDNStateMixin.state_entry_views(stub, 2):
        view.fill_(7.0)
    assert torch.all(k[:, 2 * span : 3 * span] == 7.0)
    assert not torch.all(k[:, 0:span] == 7.0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_state_entry_views.py -v`
Expected: FAIL — `AttributeError: state_entry_views` (or `type object 'GDNStateMixin' has no attribute`).

- [ ] **Step 3: Add the abstract method**

In `atom/model_ops/attentions/backends.py`, directly above `copy_state_entries`:

```python
    def state_entry_views(self, group: int) -> list["torch.Tensor"]:
        """Contiguous views covering the whole of `group`'s per-request state.

        The byte-level counterpart of `copy_state_entries`: that method moves a
        group between two indices, this one names the same bytes so something
        outside the pool — the LMCache offload tier — can read or write them.
        `copy_state_entries` is expressed in terms of this where the layout
        allows, so the two cannot drift apart.

        Every returned tensor must be contiguous. The Triton staging packer
        builds its segment table from `seg.is_contiguous()` and refuses a
        strided view, so a class whose group is strided (GDN, slot on axis 1)
        returns one view per layer rather than one strided block.
        """
        raise NotImplementedError(
            f"{type(self).__name__} owns per-request state but does not "
            "implement state_entry_views"
        )
```

- [ ] **Step 4: Implement for GDN**

In `gdn_attn.py`, above `copy_state_entries`:

```python
    def state_entry_views(self, group: int) -> list[torch.Tensor]:
        """One contiguous slice per (cache, layer) — the group's whole state.

        Both caches are layer-major with the slot on axis 1, so a group's rows
        are strided and there is no single range covering them. Slicing per
        layer makes each piece contiguous, which is what the staging packer
        requires; `copy_state_entries` keeps its own strided views because
        `_foreach_copy_` has no such constraint and one launch beats `LAYERS`.
        """
        span = 1 + self.num_spec
        lo = group * span
        views = []
        for cache in (self.model_runner.mamba_k_cache, self.model_runner.mamba_v_cache):
            for layer in range(cache.shape[0]):
                views.append(cache[layer, lo : lo + span])
        return views
```

- [ ] **Step 5: Implement for DeepSeek-V4**

In `deepseek_v4_attn.py`, and rewrite `copy_state_entries` on top of it:

```python
    def state_entry_views(self, group: int) -> list[torch.Tensor]:
        """One contiguous slice per plane — a V4 group is one slot per plane.

        A slot holds the compressor state and then every layer's windows
        contiguously (see `copy_state_entries`), so a plane's whole
        contribution is one range and no per-layer split is needed.
        """
        return self._slot_views()[group]

    def copy_state_entries(self, pairs: list[tuple[int, int]]) -> None:
        views = self._slot_views()
        dsts, srcs = [], []
        for src, dst in pairs:
            dsts += views[dst]
            srcs += views[src]
        if dsts:
            torch._foreach_copy_(dsts, srcs)
```

(The `copy_state_entries` docstring at `deepseek_v4_attn.py:784-806` stays — it explains why the whole slot moves, which is still true.)

- [ ] **Step 6: Run both test files**

Run: `python -m pytest tests/test_state_entry_views.py tests/test_gdn_state_copy.py -v`
Expected: PASS. On a CPU-only runner both skip; run them on a GPU box before committing.

- [ ] **Step 7: Lint and commit**

```bash
black . && ruff check .
git add atom/model_ops/attentions/ tests/test_state_entry_views.py
git commit -m "refactor(state): factor state_entry_views out of copy_state_entries"
```

---

## Task 2: `StateOffloadIndex` and the widened membership test

The scheduler-side half: the in-memory hash set, the bounded spill queue, and the one-line widening of `resumable_hit`. No torch, no LMCache, no I/O — this task is fully unit-testable on CPU.

**Files:**
- Create: `atom/model_engine/state_offload.py`
- Modify: `atom/model_engine/state_pool.py:249-270` (`pop`), `:427-464` (`resumable_hit`), `:621-644` (`unindex`), `:151-235` (`__init__`)
- Test: `tests/test_state_offload_index.py` (create)

**Interfaces:**
- Consumes: `state_entry_views` exists (Task 1) but is **not** called here — this task only moves integers.
- Produces:
  - `StateOffloadIndex(staging_depth: int, kv_offload_enabled: bool)`
  - `.hashes: set[int]` — hashes believed present in LMCache.
  - `.request_spill(h: int, group: int) -> int` — returns the staging slot index the caller must copy into, or `-1` when the queue is at depth K (spill dropped).
  - `.take_pending() -> list[tuple[int, int]]` — `(hash, staging_slot)` pairs the worker should D2H, draining the queue.
  - `.confirm_spill(h: int) -> None` / `.release_staging(slot: int) -> None`
  - `.forget(h: int) -> None` — remove after a failed load.
  - `.spills_requested`, `.spills_dropped`, `.loads_attempted`, `.loads_failed` counters.
  - `StateGroupPool._resumable_from(h: int) -> bool`
  - `StateGroupPool._spill(group: int) -> None`
  - `StateGroupPool.take_spill_copies() -> list[tuple[int, int]]` — `(group, staging_slot)`
  - `StateGroupPool.offload: StateOffloadIndex | None`
  - `state_offload_staging_groups() -> int` (added in Task 9 Step 3, listed here because it lives in this module)

- [ ] **Step 1: Write the failing test**

Create `tests/test_state_offload_index.py`:

```python
# SPDX-License-Identifier: MIT
# The scheduler-side half of the state offload tier: a hash set and a bounded
# spill queue. No device work happens here, so this runs anywhere.

import pytest

from atom.model_engine.state_offload import StateOffloadIndex
from atom.model_engine.state_pool import StateGroupPool, StateTransfer


def index(depth=2):
    return StateOffloadIndex(staging_depth=depth, kv_offload_enabled=False)


def test_a_spill_reserves_a_distinct_staging_slot():
    idx = index(depth=2)
    a = idx.request_spill(11, group=3)
    b = idx.request_spill(22, group=4)
    assert a >= 0 and b >= 0 and a != b


def test_spills_beyond_depth_k_are_dropped():
    """Dropping is not a regression: checkpoints_evicted counts them either
    way, which is exactly today's behaviour."""
    idx = index(depth=1)
    assert idx.request_spill(11, group=3) >= 0
    assert idx.request_spill(22, group=4) == -1
    assert idx.spills_dropped == 1


def test_a_released_slot_is_reusable():
    idx = index(depth=1)
    slot = idx.request_spill(11, group=3)
    idx.take_pending()
    idx.release_staging(slot)
    assert idx.request_spill(22, group=4) == slot


def test_take_pending_drains():
    idx = index(depth=2)
    idx.request_spill(11, group=3)
    assert [h for h, _ in idx.take_pending()] == [11]
    assert idx.take_pending() == []


def test_only_a_confirmed_spill_enters_the_index():
    idx = index()
    idx.request_spill(11, group=3)
    assert 11 not in idx.hashes
    idx.confirm_spill(11)
    assert 11 in idx.hashes


def test_forget_drops_a_hash_that_failed_to_load():
    idx = index()
    idx.confirm_spill(11)
    idx.forget(11)
    assert 11 not in idx.hashes


def test_resumable_from_is_hbm_or_tier():
    pool = StateGroupPool(num_groups=2, transfer=StateTransfer.copy(), hash_block_size=4)
    pool.offload = index()
    assert not pool._resumable_from(99)
    pool.offload.confirm_spill(99)
    assert pool._resumable_from(99)


def test_resumable_from_without_a_tier_is_the_plain_lookup():
    """Zero cost when disabled is a stated constraint, so the None path must
    behave exactly like the original `h in self.hash_to_group`."""
    pool = StateGroupPool(num_groups=2, transfer=StateTransfer.copy(), hash_block_size=4)
    assert pool.offload is None
    assert not pool._resumable_from(99)
    pool.hash_to_group[99] = 0
    assert pool._resumable_from(99)


def test_a_spilled_hash_still_takes_the_fork_test():
    """min_fork_tokens is not relaxed for spilled hashes: a boundary too close
    to the end of the prompt leaves GDN's replacement group unfilled, which is
    a wrong state, not a slow one."""

    class Seq:
        has_per_req_cache = True
        num_tokens = 8

    pool = StateGroupPool(
        num_groups=2, transfer=StateTransfer.fork(tokens=64), hash_block_size=4
    )
    pool.offload = index()
    pool.offload.confirm_spill(7)
    assert pool.resumable_hit(Seq(), 2, [3, 7]) == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_state_offload_index.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'atom.model_engine.state_offload'`.

- [ ] **Step 3: Write `state_offload.py`**

```python
# SPDX-License-Identifier: MIT
"""Scheduler-side bookkeeping for the state-cache offload tier.

Two things live here and neither touches a device: the set of hashes believed
to be in LMCache, and the bounded queue of groups waiting to be read out of
HBM. The bytes and the transfers are the worker's business
(`kv_transfer/offload/state_tier.py`).

Why a plain in-memory set is right, and persisting it would be harmful:
`LocalDiskBackend.__init__` starts from an empty dict and never scans its
directory, so after a restart LMCache does not recognize its own files. An
index recovered from disk would be a pure false-positive generator. The index
and the bytes share one server lifetime — `LMCacheEngineBuilder.get_or_create`
runs inside `register_kv_caches` at model load, so the LMCache engine cannot
restart without the worker restarting.
"""

from collections import deque


class StateOffloadIndex:
    """What has been spilled, and what is queued to be.

    `hashes` answers the membership half of `StateGroupPool._resumable_from`.
    It is deliberately optimistic: LMCache's own LRU can drop bytes at any
    time, so a hash here means "was spilled once", never "is still there".
    The false positive costs one lookup and a park/unpark and is handled by
    the `failed_loading` path, which calls `forget`.
    """

    def __init__(self, staging_depth: int, kv_offload_enabled: bool) -> None:
        self.staging_depth = max(0, int(staging_depth))
        # Orphaned checkpoints (`unindex`) are worth spilling only when the KV
        # prefix can also come back: `resumable_hit` scans `block_hashes`, which
        # `can_allocate` builds from HBM `kv.lookup` hits only. With KV offload
        # off, a hash whose KV left HBM never reappears and the bytes are wasted.
        self.kv_offload_enabled = bool(kv_offload_enabled)
        self.hashes: set[int] = set()
        self._free_slots: deque[int] = deque(range(self.staging_depth))
        self._pending: deque[tuple[int, int]] = deque()
        self.spills_requested = 0
        self.spills_dropped = 0
        self.loads_attempted = 0
        self.loads_failed = 0

    @property
    def enabled(self) -> bool:
        return self.staging_depth > 0

    def request_spill(self, h: int, group: int) -> int:
        """Reserve a staging slot for `group`, or -1 if the ring is full.

        Called from `pop()` on the scheduler's critical path, so this does no
        work beyond a deque pop. The caller copies `group` into the returned
        slot on the compute stream and `pop()` hands the original out
        immediately: spilling by copy rather than by pin, because `pop()` is
        called precisely when there is no free group to withhold.
        """
        if h < 0 or not self._free_slots:
            self.spills_dropped += 1
            return -1
        slot = self._free_slots.popleft()
        self._pending.append((h, slot))
        self.spills_requested += 1
        return slot

    def take_pending(self) -> list[tuple[int, int]]:
        """Drain the queue as `(hash, staging_slot)` pairs."""
        out = list(self._pending)
        self._pending.clear()
        return out

    def confirm_spill(self, h: int) -> None:
        """Index `h` once its bytes reached LMCache."""
        self.hashes.add(h)

    def release_staging(self, slot: int) -> None:
        if 0 <= slot < self.staging_depth and slot not in self._free_slots:
            self._free_slots.append(slot)

    def forget(self, h: int) -> None:
        """Drop a hash whose load failed, so the next request does not retry."""
        self.hashes.discard(h)

    def stats(self) -> dict[str, int]:
        return {
            "spills_requested": self.spills_requested,
            "spills_dropped": self.spills_dropped,
            "loads_attempted": self.loads_attempted,
            "loads_failed": self.loads_failed,
            "indexed": len(self.hashes),
        }
```

- [ ] **Step 4: Wire it into `StateGroupPool`**

In `state_pool.py` `__init__`, after `self.group_hash = [-1] * num_groups`:

```python
        # Backing store beneath the pool, or None. Owned here rather than
        # placed in `BlockManager.state_caches` because every member of that
        # tuple is a veto — it answers "the rightmost boundary <= X that I
        # accept" — and this tier does the opposite: it makes more boundaries
        # reachable. As a sibling it could only ever return identity.
        self.offload: "StateOffloadIndex | None" = None
```

Add the two methods:

```python
    def _resumable_from(self, h: int) -> bool:
        """Whether a checkpoint for `h` can be reached, in HBM or beneath it.

        HBM wins without a preference rule: `resumable_hit` scans right to left
        and both tiers are indexed by the same hash, so the rightmost boundary
        wins wherever it lives.
        """
        if h in self.hash_to_group:
            return True
        return self.offload is not None and h in self.offload.hashes

    def _spill(self, group: int) -> None:
        """Stage `group`'s bytes for the tier, if there is room. Never blocks.

        Beyond the staging depth the spill is simply dropped and
        `checkpoints_evicted` still counts it — identical to the behaviour
        without a tier, so there is no regression to reason about.
        """
        if self.offload is None or not self.offload.enabled:
            return
        h = self.group_hash[group]
        if h == -1:
            return
        slot = self.offload.request_spill(h, group)
        if slot >= 0:
            self._spill_copies.append((group, slot))
```

Initialise `self._spill_copies: list[tuple[int, int]] = []` in `__init__` and add a drain:

```python
    def take_spill_copies(self) -> list[tuple[int, int]]:
        """`(group, staging_slot)` pairs the next forward must copy.

        The consumer turns each into `copy_state_entries([(group,
        num_groups + slot)])` — the staging ring is K groups appended to the
        arena past the pool's own range, so a staging slot is addressable by
        the existing group-indexed copy with no second addressing scheme. One
        device-to-device copy per spill, on the compute stream ahead of that
        step's forward, the same path relocation already uses.
        """
        out, self._spill_copies = self._spill_copies, []
        return out
```

- [ ] **Step 5: Hook `pop()` and widen `resumable_hit`**

In `pop()` (`state_pool.py:263-267`), in the eviction branch **before** `invalidate(group)`:

```python
        group = self._pop_vacant()
        if group < 0:
            group = self._checkpointed.popleft()
            self._free.discard(group)
            self.checkpoints_evicted += 1
            # Before invalidate() clears group_hash[group]: the hash is the key
            # the tier stores under.
            self._spill(group)
        self.invalidate(group)
        return group
```

In `resumable_hit`, one line:

```python
            if not assume_checkpointed and not self._resumable_from(block_hashes[i]):
```

In `unindex()` (`state_pool.py:621-644`), before the hash is dropped:

```python
        if self.offload is not None and self.offload.kv_offload_enabled:
            self._spill(group)
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_state_offload_index.py tests/test_state_checkpoint.py -v`
Expected: PASS, including the pre-existing checkpoint tests (the `None` tier path must be exactly the old behaviour).

- [ ] **Step 7: Lint and commit**

```bash
black . && ruff check .
git add atom/model_engine/state_offload.py atom/model_engine/state_pool.py tests/test_state_offload_index.py
git commit -m "feat(state-pool): add offload index and widen the membership test"
```

---

## Task 3: Extract `StagedTransfer`

The KV connector is two layers stitched together and only the lower one is reusable. Extract it so state can write its own orchestration on top without touching `_iter_transfer_chunks` — whose `strict=True` zip and `nbytes = block_count * bytes_per_block` from a startup constant would both break on a differently-sized object.

**Files:**
- Create: `atom/kv_transfer/offload/staged_transfer.py`
- Modify: `atom/kv_transfer/offload/atom_lmcache_gpu_connector.py:126-392`
- Test: `tests/test_staged_transfer.py` (create), `tests/test_lmcache_offload_connector.py` (must keep passing)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `StagedTransfer(device: torch.device, staging_buffer_bytes: int, *, release_after_transfer: bool = False)`
  - `.thread_state() -> _ThreadTransferState`
  - `.ensure_buffer(staging_buffer, nbytes: int) -> torch.Tensor`
  - `.release_buffer_if_requested(staging_buffer) -> None`
  - `.memory_tensor(memory_obj, nbytes: int) -> torch.Tensor`
  - `.run_pipeline(stages: list[_PipelineStage], *, to_gpu: bool) -> dict`
  - `.staging_buffer_bytes: int`
  - `.pack(segments: list[torch.Tensor], dst) -> None` and `.unpack(src, segments: list[torch.Tensor]) -> None` — added in Task 4 Step 4, listed here because Task 4 consumes them by these exact names.
  - Re-exports `_PipelineStage`, `_ThreadTransferState`, `_StagingBuffer`.

- [ ] **Step 1: Write the characterization test first**

Create `tests/test_staged_transfer.py`:

```python
# SPDX-License-Identifier: MIT
# The staging half of the KV path, now shared with the state tier. These are
# characterization tests: the extraction must not change behaviour.

import pytest
import torch

from atom.kv_transfer.offload.staged_transfer import StagedTransfer, _StagingBuffer

CPU = torch.device("cpu")


def test_buffer_is_allocated_once_and_reused():
    st = StagedTransfer(CPU, staging_buffer_bytes=1024)
    buf = _StagingBuffer()
    first = st.ensure_buffer(buf, 512)
    second = st.ensure_buffer(buf, 256)
    assert first.data_ptr() == second.data_ptr()
    assert int(second.numel()) == 256


def test_a_request_larger_than_the_buffer_is_an_error_not_a_realloc():
    """The buffer is bounded on purpose — silently growing it would put the
    HBM ceiling back in the hands of whatever the largest group happened to be."""
    st = StagedTransfer(CPU, staging_buffer_bytes=1024)
    with pytest.raises(RuntimeError, match="exceeds bounded GPU staging buffer"):
        st.ensure_buffer(_StagingBuffer(), 2048)


def test_release_drops_the_tensor_when_asked():
    st = StagedTransfer(CPU, staging_buffer_bytes=1024, release_after_transfer=True)
    buf = _StagingBuffer()
    st.ensure_buffer(buf, 512)
    st.release_buffer_if_requested(buf)
    assert buf.tensor is None


def test_release_is_a_no_op_by_default():
    st = StagedTransfer(CPU, staging_buffer_bytes=1024)
    buf = _StagingBuffer()
    st.ensure_buffer(buf, 512)
    st.release_buffer_if_requested(buf)
    assert buf.tensor is not None


def test_memory_tensor_rejects_a_non_uint8_object():
    st = StagedTransfer(CPU, staging_buffer_bytes=1024)

    class Obj:
        tensor = torch.zeros(64, dtype=torch.float16)

    with pytest.raises(TypeError, match="must be uint8"):
        st.memory_tensor(Obj(), 64)


def test_memory_tensor_rejects_an_object_that_is_too_small():
    st = StagedTransfer(CPU, staging_buffer_bytes=1024)

    class Obj:
        tensor = torch.zeros(16, dtype=torch.uint8)

    with pytest.raises(ValueError, match="too small"):
        st.memory_tensor(Obj(), 64)


def test_thread_state_is_per_device_and_cached():
    st = StagedTransfer(CPU, staging_buffer_bytes=1024)
    assert st.thread_state() is st.thread_state()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_staged_transfer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'atom.kv_transfer.offload.staged_transfer'`.

- [ ] **Step 3: Move the lower half**

Create `staged_transfer.py` and move, **unchanged in body**, from `atom_lmcache_gpu_connector.py`:
`_StagingBuffer`, `_ThreadTransferState`, `_PipelineStage` (`:50`), `_thread_state` (`:134`), `_ensure_staging_buffer` (`:149`), `_release_staging_buffer_if_requested` (`:173`), `_memory_tensor` (`:191`), `_run_staged_pipeline` (`:354`), and the `_env_int` / `_env_optional_int` / `_env_flag` helpers.

Rename to the public names in the Interfaces block (`_thread_state` → `thread_state`, etc.), and give the class this docstring:

```python
class StagedTransfer:
    """Bounded GPU staging buffer, D2H/H2D, and the producer event.

    The half of the LMCache GPU connector that is not about chunks. KV and
    state both need a bounded device buffer, a copy stream, and an event the
    save worker synchronizes on; neither needs the other's orchestration. The
    chunk layer stays in `ATOMLMCacheGPUConnector` because it is genuinely
    KV-specific: `_iter_transfer_chunks` zips MemoryObjs against block-id
    groups with `strict=True` and sizes each from a startup per-block
    constant, so a single object of a different size breaks both invariants.
    State is not a member of that loop.

    The producer `cuda.Event` recorded on the RPC thread and `synchronize()`d
    on the save worker is load-bearing — it is what commit 7427e05e added to
    fix KV corruption on reload. Do not drop it from either caller.
    """
```

- [ ] **Step 4: Delegate from the KV connector**

In `ATOMLMCacheGPUConnector.__init__`, after `self._gpu_staging_buffer_bytes` is computed:

```python
        self._staged = StagedTransfer(
            self.device,
            staging_buffer_bytes=self._gpu_staging_buffer_bytes,
            release_after_transfer=_env_flag(
                "OFFLOAD_RELEASE_GPU_STAGING_AFTER_TRANSFER"
            ),
        )
```

Replace each moved method with a one-line delegation, e.g.:

```python
    def _thread_state(self) -> "_ThreadTransferState":
        return self._staged.thread_state()
```

Keep `_assert_fused_chunk_major_available`, `_range_block_ids`, `_ranges_to_block_ids`, `_iter_transfer_chunks`, `_iter_transfer_groups`, `_group_block_ids`, `_slice_to_memory_objs`, `_memory_objs_to_slice`, `_prepare_transfer`, `from_gpu`, `to_gpu`, `batched_from_gpu`, `batched_to_gpu` **where they are**.

- [ ] **Step 5: Run both suites**

Run: `python -m pytest tests/test_staged_transfer.py tests/test_lmcache_offload_connector.py -v`
Expected: PASS. The second file is the regression gate — extraction must not change KV behaviour.

- [ ] **Step 6: Lint and commit**

```bash
black . && ruff check .
git add atom/kv_transfer/offload/ tests/test_staged_transfer.py
git commit -m "refactor(offload): extract StagedTransfer from the KV GPU connector"
```

---

## Task 4: `StateByteCodec` — one entry, one object, one key

Pack a group's state into one opaque `uint8` MemoryObj and write it through `storage_manager` under ATOM's own hash. This bypasses `ChunkedTokenDatabase` entirely: state has a token range (`[0, pos)`) but is not sliceable by token, so a chunker would produce N keys useful only all together.

**Files:**
- Create: `atom/kv_transfer/offload/state_object.py`
- Test: `tests/test_state_object.py` (create)

**Interfaces:**
- Consumes: `StagedTransfer` (Task 3), `state_entry_views` (Task 1).
- Produces:
  - `StateByteCodec(backend, staged: StagedTransfer, entry_bytes: int, *, model_name: str, world_size: int, worker_id: int)`
  - `.key(h: int) -> CacheEngineKey`
  - `.entry_bytes: int`
  - `.put(h: int, entry_index: int) -> bool` — packs `state_entry_views(entry_index)` into one object, `storage_manager.batched_put`. Returns success. On the spill path `entry_index` is the **staging slot**, not a pool group: the hook copied the group's bytes there so `pop()` could hand the original out immediately.
  - `.get(h: int, entry_index: int) -> bool` — `storage_manager.get`, unpack into `entry_index`. On the load path this is a real pool group. Returns False on miss.
  - `.contains(h: int) -> bool`
  - `.bind_storage_manager(storage_manager) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_state_object.py`:

```python
# SPDX-License-Identifier: MIT
# The state object: one key per checkpoint, ATOM's own hash, no chunking.

from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload.state_object import StateByteCodec


class FakeStaged:
    """Records what the packer was asked to move; no device involved."""

    def __init__(self):
        self.packed = []
        self.unpacked = []

    def pack(self, segments, dst):
        self.packed.append((list(segments), dst))

    def unpack(self, src, segments):
        self.unpacked.append((src, list(segments)))


class FakeBackend:
    def __init__(self):
        self.views = {g: [f"view-{g}-{i}" for i in range(3)] for g in range(4)}

    def state_entry_views(self, group):
        return self.views[group]


class FakeStorageManager:
    def __init__(self):
        self.store = {}
        self.puts = []

    def contains(self, key, pin=False):
        return key in self.store

    def batched_put(self, keys, objs):
        for key, obj in zip(keys, objs, strict=True):
            self.store[key] = obj
            self.puts.append(key)

    def get(self, key):
        return self.store.get(key)

    def allocate(self, shape, dtype, fmt=None):
        return SimpleNamespace(shape=shape, dtype=dtype, fmt=fmt)


def codec():
    c = StateByteCodec(
        FakeBackend(),
        FakeStaged(),
        entry_bytes=4096,
        model_name="m",
        world_size=8,
        worker_id=0,
    )
    c.bind_storage_manager(FakeStorageManager())
    return c


def test_one_hash_is_one_key():
    """State is an indivisible whole-prefix snapshot: N keys would be useful
    only all together and would multiply the partial-invalidation chance."""
    c = codec()
    assert c.key(1234) == c.key(1234)
    assert c.key(1234) != c.key(5678)


def test_the_key_carries_the_atom_hash_unmodified():
    c = codec()
    assert c.key(1234).chunk_hash == 1234


def test_the_key_is_per_worker():
    """Each TP rank stores its own shard of the state."""
    a = StateByteCodec(
        FakeBackend(), FakeStaged(), 4096, model_name="m", world_size=8, worker_id=0
    )
    b = StateByteCodec(
        FakeBackend(), FakeStaged(), 4096, model_name="m", world_size=8, worker_id=1
    )
    assert a.key(1234) != b.key(1234)


def test_put_packs_every_view_of_the_group():
    c = codec()
    c.put(1234, entry_index=2)
    segments, _ = c._staged.packed[-1]
    assert segments == c._backend.state_entry_views(2)


def test_put_writes_exactly_one_object():
    c = codec()
    c.put(1234, entry_index=2)
    assert len(c._storage.puts) == 1


def test_get_on_a_miss_is_false_and_moves_nothing():
    c = codec()
    assert c.get(9999, entry_index=1) is False
    assert c._staged.unpacked == []


def test_get_after_put_unpacks_into_the_destination_group():
    c = codec()
    c.put(1234, entry_index=2)
    assert c.get(1234, entry_index=3) is True
    _, segments = c._staged.unpacked[-1]
    assert segments == c._backend.state_entry_views(3)


def test_contains_answers_from_storage_not_from_a_local_set():
    """A local set would go stale the moment LMCache's LRU evicted."""
    c = codec()
    assert c.contains(1234) is False
    c.put(1234, entry_index=2)
    assert c.contains(1234) is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_state_object.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'atom.kv_transfer.offload.state_object'`.

- [ ] **Step 3: Write `state_object.py`**

```python
# SPDX-License-Identifier: MIT
"""One state checkpoint, one opaque object, keyed by ATOM's own hash.

`ChunkedTokenDatabase` is deliberately bypassed. Two reasons, in order:

1. The two branches of the membership test must query the same integer.
   `StateGroupPool._resumable_from(h)` checks HBM and this tier; a
   chunker-derived key would not be the same thing being looked up, and the
   whole feature stops being a one-line widening.
2. State has a token range -- `[0, pos)`, which is exactly why it can share
   KV's key -- but its bytes cannot be sliced by token. There is no such thing
   as "the first three chunks hit". Chunking would produce N keys useful only
   all together.

It also sidesteps LMCache's chunk-alignment loss: keys exist only at chunk
boundaries (`token_database.py:387-391`).
"""

import logging

import torch

logger = logging.getLogger(__name__)


class StateByteCodec:
    """Pack/unpack one group's state and move it through `storage_manager`.

    The object is `MemoryFormat.BINARY` uint8, like the KV path's, because the
    x-packed / strided / multi-plane state layouts cannot be expressed in
    LMCache's token-major model.
    """

    def __init__(
        self,
        backend,
        staged,
        entry_bytes: int,
        *,
        model_name: str,
        world_size: int,
        worker_id: int,
    ) -> None:
        self._backend = backend
        self._staged = staged
        self.entry_bytes = int(entry_bytes)
        if self.entry_bytes <= 0:
            raise ValueError("state entry bytes must be > 0")
        self._model_name = model_name
        self._world_size = int(world_size)
        self._worker_id = int(worker_id)
        self._storage = None
        # Never hard-code a size: V4 keeps six compressor fields across
        # n_csa/n_hca layers plus an optional window; GDN keeps
        # 2 * num_gdn_attn_state * (1 + num_spec) slots. MB-scale, not KB.
        logger.info(
            "state offload: entry_bytes=%d (%.2f MiB) per request",
            self.entry_bytes,
            self.entry_bytes / (1 << 20),
        )

    def bind_storage_manager(self, storage_manager) -> None:
        self._storage = storage_manager

    def key(self, h: int):
        from lmcache.utils import CacheEngineKey

        return CacheEngineKey(
            fmt="binary",
            model_name=self._model_name,
            world_size=self._world_size,
            worker_id=self._worker_id,
            chunk_hash=int(h),
        )

    def put(self, h: int, group: int) -> bool:
        if self._storage is None:
            return False
        obj = self._allocate(self.entry_bytes)
        if obj is None:
            return False
        self._staged.pack(self._backend.state_entry_views(group), obj)
        self._storage.batched_put([self.key(h)], [obj])
        return True

    def get(self, h: int, group: int) -> bool:
        if self._storage is None:
            return False
        obj = self._storage.get(self.key(h))
        if obj is None:
            return False
        self._staged.unpack(obj, self._backend.state_entry_views(group))
        return True

    def contains(self, h: int) -> bool:
        if self._storage is None:
            return False
        return bool(self._storage.contains(self.key(h)))

    def _allocate(self, nbytes: int):
        from lmcache.v1.memory_management import MemoryFormat

        return self._storage.allocate(
            torch.Size([nbytes]), torch.uint8, fmt=MemoryFormat.BINARY
        )
```

**Before implementing, check two signatures against the pinned LMCache and match them exactly** — both are third-party surfaces this plan cannot pin down for you:

```bash
python -c "import inspect, lmcache.utils as u; print(inspect.signature(u.CacheEngineKey.__init__))"
python -c "import inspect
from lmcache.v1.storage_backend.storage_manager import StorageManager as S
for m in ('allocate', 'batched_put', 'get', 'contains'):
    print(m, inspect.signature(getattr(S, m)))"
```

`CacheEngineKey`'s field order and whether it takes `request_configs` vary by version; the only thing this design requires is that **`chunk_hash` carries ATOM's integer unmodified** and that `worker_id` distinguishes TP ranks. Adjust the constructor call and `FakeStorageManager` to whatever the real signatures are — the tests assert behaviour, not the vendor's argument names.

- [ ] **Step 4: Add `pack` / `unpack` to `StagedTransfer`**

```python
    def pack(self, segments: list[torch.Tensor], dst) -> None:
        """Gather `segments` into one contiguous object via the Triton packer.

        The existing kernel needs no modification: it is already a fully
        parameterized gather driven by segment_ptrs[] + segment_block_bytes[] +
        block_ids[]. State passes its own views as the segments with
        block_ids=[0] and chunk_block_counts=[1] -- a single "chunk" of one
        "block", which is what a whole-entry snapshot is.
        """
```

Implement by calling the same `_build_meta` + `_pack_chunk_major_kernel` path the KV codec uses, with `block_ids=[0]`, `chunk_block_counts=[1]`. `unpack` is the mirror through `_unpack_chunk_major_kernel`.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_state_object.py -v`
Expected: PASS.

- [ ] **Step 6: Lint and commit**

```bash
black . && ruff check .
git add atom/kv_transfer/offload/ tests/test_state_object.py
git commit -m "feat(offload): store one state entry per ATOM hash, bypassing the chunker"
```

---

## Task 5: `StateOffloadTier` — the worker-side driver

Drain the spill queue on a dedicated executor, run the D2H, confirm the hash. State gets its own executor for the same reason KV separates load from save (`connector.py:83-88`): a TTFT-critical load must not queue behind fire-and-forget spills.

**Files:**
- Create: `atom/kv_transfer/offload/state_tier.py`
- Modify: `atom/kv_transfer/offload/connector.py:82-95` (executor construction)
- Test: `tests/test_state_tier.py` (create)

**Interfaces:**
- Consumes: `StateByteCodec` (Task 4), `StateOffloadIndex` (Task 2).
- Produces:
  - `StateOffloadTier(codec: StateByteCodec, index: StateOffloadIndex, *, max_workers: int = 1)`
  - `.submit_spill(h: int, entry_index: int, staging_slot: int) -> None` — `entry_index` is what the codec packs (`num_groups + slot`); `staging_slot` is what the ring releases. Two arguments because they are two different index spaces and collapsing them leaks the ring.
  - `.submit_load(req_id: str, h: int, group: int) -> None`
  - `.get_finished() -> tuple[set[str], set[str]]` — `(done_loads, failed_loads)` by req_id.
  - `.shutdown() -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_state_tier.py`:

```python
# SPDX-License-Identifier: MIT
# The worker-side driver: its own executor, and completions reported by req_id.

from atom.kv_transfer.offload.state_tier import StateOffloadTier
from atom.model_engine.state_offload import StateOffloadIndex


class FakeCodec:
    entry_bytes = 4096

    def __init__(self, put_ok=True, get_ok=True):
        self.put_ok, self.get_ok = put_ok, get_ok
        self.puts, self.gets = [], []

    def put(self, h, entry_index):
        self.puts.append((h, entry_index))
        return self.put_ok

    def get(self, h, entry_index):
        self.gets.append((h, entry_index))
        return self.get_ok


def tier(codec):
    return StateOffloadTier(codec, StateOffloadIndex(2, kv_offload_enabled=False))


def test_a_successful_spill_indexes_the_hash():
    codec = FakeCodec()
    t = tier(codec)
    t.submit_spill(11, entry_index=64, staging_slot=0)
    t.drain()
    assert 11 in t.index.hashes


def test_the_codec_packs_the_staging_entry_not_the_pool_group():
    """The hook copied the group's bytes into staging precisely so `pop()`
    could hand the original out immediately; packing the group would race the
    new owner."""
    codec = FakeCodec()
    t = tier(codec)
    t.submit_spill(11, entry_index=64, staging_slot=0)
    t.drain()
    assert codec.puts == [(11, 64)]


def test_a_failed_spill_does_not_index():
    """No spill acknowledgement is sent back to the scheduler, so the index is
    the only record — indexing a spill that did not land is a guaranteed
    false positive on every later request."""
    codec = FakeCodec(put_ok=False)
    t = tier(codec)
    t.submit_spill(11, entry_index=64, staging_slot=0)
    t.drain()
    assert 11 not in t.index.hashes


def test_a_spill_always_releases_its_staging_slot():
    codec = FakeCodec(put_ok=False)
    t = tier(codec)
    t.submit_spill(11, entry_index=64, staging_slot=0)
    t.drain()
    t.submit_spill(22, entry_index=65, staging_slot=1)
    t.drain()
    assert t.index._free_slots  # the ring did not leak


def test_a_successful_load_reports_done():
    t = tier(FakeCodec())
    t.index.confirm_spill(11)
    t.submit_load("req-a", 11, group=3)
    t.drain()
    assert t.get_finished() == ({"req-a"}, set())


def test_a_failed_load_reports_failed_and_forgets_the_hash():
    """Three triggers funnel here — LMCache's own LRU, a spill that never
    landed, a transfer error — and all three are the same normal path."""
    t = tier(FakeCodec(get_ok=False))
    t.index.confirm_spill(11)
    t.submit_load("req-a", 11, group=3)
    t.drain()
    assert t.get_finished() == (set(), {"req-a"})
    assert 11 not in t.index.hashes


def test_get_finished_drains():
    t = tier(FakeCodec())
    t.submit_load("req-a", 11, group=3)
    t.drain()
    t.get_finished()
    assert t.get_finished() == (set(), set())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_state_tier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'atom.kv_transfer.offload.state_tier'`.

- [ ] **Step 3: Write `state_tier.py`**

```python
# SPDX-License-Identifier: MIT
"""Worker-side spill and load driver for the state offload tier.

Its own executor, separate from the KV connector's `_load_executor` and
`_save_executor`. The reason is the one recorded at `connector.py:83-88`: a
load is on the TTFT critical path -- a parked sequence is waiting for it --
and must never queue behind a backlog of fire-and-forget spills.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class StateOffloadTier:
    def __init__(self, codec, index, *, max_workers: int = 1) -> None:
        self.codec = codec
        self.index = index
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="lmc-state"
        )
        self._lock = threading.Lock()
        self._done: set[str] = set()
        self._failed: set[str] = set()
        self._inflight: list = []

    def submit_spill(self, h: int, entry_index: int, staging_slot: int) -> None:
        """`entry_index` is what the codec packs, `staging_slot` what the ring
        releases. Two index spaces: the staging entries sit past the pool's
        range in the arena (`num_groups + slot`), while the ring counts from 0.
        """
        self._inflight.append(
            self._executor.submit(self._do_spill, h, entry_index, staging_slot)
        )

    def submit_load(self, req_id: str, h: int, group: int) -> None:
        self.index.loads_attempted += 1
        self._inflight.append(
            self._executor.submit(self._do_load, req_id, h, group)
        )

    def drain(self) -> None:
        """Block until every submitted transfer has settled. Tests and shutdown
        only -- the serving path polls `get_finished` instead."""
        inflight, self._inflight = self._inflight, []
        for fut in inflight:
            fut.result()

    def get_finished(self) -> tuple[set[str], set[str]]:
        with self._lock:
            done, failed = set(self._done), set(self._failed)
            self._done.clear()
            self._failed.clear()
        return done, failed

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def _do_spill(self, h: int, entry_index: int, staging_slot: int) -> None:
        try:
            if self.codec.put(h, entry_index):
                self.index.confirm_spill(h)
        except Exception:  # noqa: BLE001  # a spill is best effort by design
            logger.warning("state offload: spill of hash %d failed", h, exc_info=True)
        finally:
            # Always: a leaked slot shrinks the ring permanently and the
            # feature quietly stops spilling.
            self.index.release_staging(staging_slot)

    def _do_load(self, req_id: str, h: int, group: int) -> None:
        # A load target is a real pool group, not a staging entry: the bytes
        # land where the resuming request will read them. Only the spill
        # direction needs the staging indirection.
        ok = False
        try:
            ok = bool(self.codec.get(h, group))
        except Exception:  # noqa: BLE001  # a failed load is a normal path
            logger.warning("state offload: load of hash %d failed", h, exc_info=True)
        with self._lock:
            if ok:
                self._done.add(req_id)
            else:
                self.index.loads_failed += 1
                # So the next request does not repeat the attempt.
                self.index.forget(h)
                self._failed.add(req_id)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_state_tier.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
black . && ruff check .
git add atom/kv_transfer/offload/state_tier.py tests/test_state_tier.py
git commit -m "feat(offload): add the state tier's dedicated spill/load executor"
```

---

## Task 6: The `P ≤ L` clamp

The one new correctness constraint. `P` (state boundary) claiming history that `L` (KV loaded) does not cover is **silent wrong output, no error**.

**Files:**
- Modify: `atom/kv_transfer/offload/connector.py` (after `update_state_after_alloc`, near `connector.py:834-840`)
- Test: `tests/test_state_offload_clamp.py` (create)

**Interfaces:**
- Consumes: `StateOffloadIndex` (Task 2).
- Produces: `clamp_state_boundary(state_blocks: int, kv_loaded_blocks: int) -> int` in `atom/kv_transfer/offload/state_tier.py` — pure, `min(P, L)`, floors at 0.

- [ ] **Step 1: Write the failing test**

Create `tests/test_state_offload_clamp.py`:

```python
# SPDX-License-Identifier: MIT
# P <= L. Violating it is silent wrong output, so it gets its own test file.

from atom.kv_transfer.offload.state_tier import clamp_state_boundary


def test_a_state_boundary_within_the_loaded_kv_is_kept():
    assert clamp_state_boundary(4, 8) == 4


def test_a_state_boundary_past_the_loaded_kv_is_cut_to_it():
    """State claims to have seen [0,P) but [L,P) KV does not exist. The forward
    would produce wrong output and raise nothing."""
    assert clamp_state_boundary(8, 4) == 4


def test_equal_is_the_ideal_and_is_kept():
    assert clamp_state_boundary(4, 4) == 4


def test_no_kv_loaded_clamps_to_zero_which_means_recompute():
    """0 is always a valid boundary — a request starting from scratch needs no
    prior state. This is the existing path, not a new failure mode."""
    assert clamp_state_boundary(8, 0) == 0


def test_negatives_floor_at_zero():
    assert clamp_state_boundary(-1, 8) == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_state_offload_clamp.py -v`
Expected: FAIL — `ImportError: cannot import name 'clamp_state_boundary'`.

- [ ] **Step 3: Implement**

Append to `state_tier.py`:

```python
def clamp_state_boundary(state_blocks: int, kv_loaded_blocks: int) -> int:
    """`P <= L`: a state boundary may not claim history the KV does not cover.

    Today this holds for free because both derive from `block_hashes`. Once the
    tier admits spilled hashes and the KV load length is decided on a separate
    path, it does not, and the failure is silent: state is the compressed
    history of [0,P), so with P > L the forward reads a compressed prefix whose
    raw KV was never loaded and produces wrong output without raising.

    Clamping to 0 means the sequence recomputes -- the existing path.
    """
    return max(0, min(int(state_blocks), int(kv_loaded_blocks)))
```

- [ ] **Step 4: Apply it at the call site**

In `connector.py`, **after** `update_state_after_alloc` returns:

```python
            # After update_state_after_alloc, never before: until it runs
            # `seq.num_cached_tokens` is stale -- often 0 -- and a load decided
            # below the true HBM floor overwrites shared prefix blocks, a
            # corruption this file has already paid for once.
            state_blocks = clamp_state_boundary(state_blocks, kv_loaded_blocks)
```

- [ ] **Step 5: Run**

Run: `python -m pytest tests/test_state_offload_clamp.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
black . && ruff check .
git add atom/kv_transfer/offload/ tests/test_state_offload_clamp.py
git commit -m "feat(offload): clamp the state boundary to the loaded KV prefix"
```

---

## Task 7: Joint park, joint wake

Two objects, two transfers, **one park**. Waking on the state transfer alone lets the model read KV blocks that are not yet filled — silent again, not an error. And the destination group is indexed **only after H2D confirms**: indexing on lookup and landing H2D later reproduces #1417.

**Files:**
- Modify: `atom/kv_transfer/offload/connector.py` (`get_finished`, `:440-475`; the recv-queue path around `_reqs_need_recv`)
- Test: `tests/test_state_offload_clamp.py` (extend)

**Interfaces:**
- Consumes: `StateOffloadTier.get_finished()` (Task 5).
- Produces:
  - `_JointPark` in `connector.py` — `.arm(req_id, *, needs_kv: bool, needs_state: bool)`, `.settle_kv(req_id, ok: bool)`, `.settle_state(req_id, ok: bool)`, `.take_ready() -> tuple[set[str], set[str]]` returning `(finished_loading, failed_loading)`.
  - `ATOMLMCacheConnector._index_state_group(req_id: str) -> None` — looks up the `(hash, group)` pair recorded when `submit_load` was issued (keep it in `self._state_load_targets: dict[str, tuple[int, int]]`, populated at submit, popped here) and calls the pool's existing `_index(h, group)`.

**`KVOutputAggregator` requires every TP rank to report** before a request is considered finished, so a rank that skips the state leg entirely must still `arm` with `needs_state=False` rather than not arming at all.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_state_offload_clamp.py`:

```python
from atom.kv_transfer.offload.connector import _JointPark


def test_a_request_needing_both_wakes_on_neither_alone():
    park = _JointPark()
    park.arm("r", needs_kv=True, needs_state=True)
    park.settle_state("r", ok=True)
    assert park.take_ready() == (set(), set())
    park.settle_kv("r", ok=True)
    assert park.take_ready() == ({"r"}, set())


def test_either_failing_fails_the_pair():
    """Half a load is not a partial success: the state would claim a prefix
    whose KV never arrived."""
    park = _JointPark()
    park.arm("r", needs_kv=True, needs_state=True)
    park.settle_state("r", ok=False)
    park.settle_kv("r", ok=True)
    assert park.take_ready() == (set(), {"r"})


def test_a_kv_only_request_is_unchanged():
    park = _JointPark()
    park.arm("r", needs_kv=True, needs_state=False)
    park.settle_kv("r", ok=True)
    assert park.take_ready() == ({"r"}, set())


def test_take_ready_drains():
    park = _JointPark()
    park.arm("r", needs_kv=True, needs_state=False)
    park.settle_kv("r", ok=True)
    park.take_ready()
    assert park.take_ready() == (set(), set())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_state_offload_clamp.py -v`
Expected: FAIL — `ImportError: cannot import name '_JointPark'`.

- [ ] **Step 3: Implement `_JointPark`**

```python
class _JointPark:
    """One park for the KV load and the state load of the same request.

    Both completions must land before unpark. Waking on the state transfer
    alone lets the model read KV blocks that are not yet filled, which is
    silent rather than an error.

    Either side failing fails the pair: half a load leaves state claiming a
    prefix whose KV never arrived, and `failed_loading` already means "wake for
    recompute using the blocks already allocated", which is exactly right here.
    """

    def __init__(self) -> None:
        self._need: dict[str, set[str]] = {}
        self._failed: set[str] = set()
        self._ready: set[str] = set()
        self._ready_failed: set[str] = set()

    def arm(self, req_id: str, *, needs_kv: bool, needs_state: bool) -> None:
        need = set()
        if needs_kv:
            need.add("kv")
        if needs_state:
            need.add("state")
        self._need[req_id] = need
        if not need:
            self._ready.add(req_id)

    def settle_kv(self, req_id: str, ok: bool) -> None:
        self._settle(req_id, "kv", ok)

    def settle_state(self, req_id: str, ok: bool) -> None:
        self._settle(req_id, "state", ok)

    def _settle(self, req_id: str, leg: str, ok: bool) -> None:
        need = self._need.get(req_id)
        if need is None:
            return
        need.discard(leg)
        if not ok:
            self._failed.add(req_id)
        if need:
            return
        del self._need[req_id]
        if req_id in self._failed:
            self._failed.discard(req_id)
            self._ready_failed.add(req_id)
        else:
            self._ready.add(req_id)

    def take_ready(self) -> tuple[set[str], set[str]]:
        ready, failed = set(self._ready), set(self._ready_failed)
        self._ready.clear()
        self._ready_failed.clear()
        return ready, failed
```

- [ ] **Step 4: Route it through `get_finished`**

In `connector.py`'s `get_finished`, feed both legs in before building the output, and index the state group only here:

```python
        if self._state_tier is not None:
            done_state, failed_state = self._state_tier.get_finished()
            for rid in done_state:
                # Only now: the group's bytes exist. Indexing on lookup and
                # landing H2D later is the #1417 shape.
                self._index_state_group(rid)
                self._park.settle_state(rid, ok=True)
            for rid in failed_state:
                self._park.settle_state(rid, ok=False)
        for rid in dl:
            self._park.settle_kv(rid, ok=True)
        for rid in fl:
            self._park.settle_kv(rid, ok=False)
        ready, ready_failed = self._park.take_ready()
        return KVConnectorOutput(
            finished_sending=set(),
            finished_loading=ready,
            failed_loading=ready_failed,
            finished_saving=ds,
        )
```

`_index_state_group(req_id)` calls the pool's existing `_index(h, group)` for the pair recorded when the load was submitted. **Do not** use `record_copy` — that is `_commit_pending`'s path and carries its index-before-bytes assumption.

- [ ] **Step 5: Run**

Run: `python -m pytest tests/test_state_offload_clamp.py tests/test_lmcache_offload_connector.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
black . && ruff check .
git add atom/kv_transfer/offload/connector.py tests/test_state_offload_clamp.py
git commit -m "feat(offload): park KV and state loads jointly, index after H2D"
```

---

## Task 8: The load floor and the failure path

A short hit prefix is not worth a PCIe round trip, and the same floor bounds the cost of a false positive.

**Files:**
- Modify: `atom/kv_transfer/offload/connector.py` (state load decision)
- Test: `tests/test_state_tier.py` (extend)

**Interfaces:**
- Consumes: `StateOffloadIndex` (Task 2), `_JointPark` (Task 7).
- Produces: `should_load_state(hit_tokens: int, floor_tokens: int) -> bool` in `state_tier.py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_state_tier.py`:

```python
from atom.kv_transfer.offload.state_tier import should_load_state


def test_a_hit_at_or_above_the_floor_loads():
    assert should_load_state(8192, 8192) is True


def test_a_short_hit_is_not_worth_a_pcie_round_trip():
    assert should_load_state(4096, 8192) is False


def test_a_zero_floor_loads_anything_positive():
    assert should_load_state(1, 0) is True


def test_a_zero_hit_never_loads():
    assert should_load_state(0, 0) is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_state_tier.py -v`
Expected: FAIL — `ImportError: cannot import name 'should_load_state'`.

- [ ] **Step 3: Implement**

```python
def should_load_state(hit_tokens: int, floor_tokens: int) -> bool:
    """Whether a state hit of `hit_tokens` is worth an H2D.

    Mirrors KV's OFFLOAD_MIN_LOAD_TOKENS (`connector.py:526`). Two jobs: a
    short prefix does not repay the round trip, and the same floor bounds what
    a false positive costs -- the index cannot know LMCache's LRU dropped the
    bytes until the load misses.
    """
    hit_tokens = int(hit_tokens)
    return hit_tokens > 0 and hit_tokens >= int(floor_tokens)
```

- [ ] **Step 4: Apply at the call site**

In `connector.py`, alongside `self._min_load_tokens`:

```python
        try:
            self._state_min_load_tokens = max(
                0, int(os.environ.get("OFFLOAD_STATE_MIN_LOAD_TOKENS", "8192"))
            )
        except ValueError:
            logger.warning(
                "LMCache offload scheduler: invalid "
                "OFFLOAD_STATE_MIN_LOAD_TOKENS=%r; using 8192",
                os.environ.get("OFFLOAD_STATE_MIN_LOAD_TOKENS"),
            )
            self._state_min_load_tokens = 8192
```

- [ ] **Step 5: Issue the state load at the scheduler's match site**

`submit_load` has had no caller until now. Where the connector already decides a KV load for a matched request (alongside the `LoadSpec` it puts in `self._load_specs`), add the state leg. This is the site the `P ≤ L` clamp from Task 6 guards:

```python
        state_group = -1
        if self._state_tier is not None and state_hash != -1:
            # Storage and lookup are separate paths, but selection and
            # admission are joint: the boundary was already chosen by
            # `resumable_hit`, which the tier widened. All that is decided here
            # is whether the H2D is worth issuing.
            state_tokens = state_blocks * self.block_size
            if should_load_state(state_tokens, self._state_min_load_tokens):
                state_group = seq.per_req_cache_group
                self._state_load_targets[req_id] = (state_hash, state_group)
                self._state_tier.submit_load(req_id, state_hash, state_group)
        self._park.arm(req_id, needs_kv=True, needs_state=state_group >= 0)
```

Every rank arms — including one that skipped the state leg — because `KVOutputAggregator` will not consider the request finished until all TP ranks report.

- [ ] **Step 6: Run**

Run: `python -m pytest tests/test_state_tier.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
black . && ruff check .
git add atom/kv_transfer/offload/ tests/test_state_tier.py
git commit -m "feat(offload): gate state loads on a minimum hit length"
```

---

## Task 9: Wiring, env vars, and docs

Construct the tier where the pool is built, and document it.

**Files:**
- Modify: `atom/model_engine/block_manager.py` (construct `StateOffloadIndex`, assign `pool.offload`)
- Modify: `atom/kv_transfer/offload/connector.py` (construct `StateByteCodec` + `StateOffloadTier` in `register_kv_caches`)
- Modify: `atom/kv_transfer/offload/README.md` (Module Map + env var table)
- Test: `tests/test_state_offload_index.py` (extend)

**Interfaces:**
- Consumes: everything from Tasks 1–8.
- Produces: env vars
  - `OFFLOAD_STATE=0|1` (default `0` — off)
  - `OFFLOAD_STATE_STAGING_GROUPS` (default `1`, the staging depth K)
  - `OFFLOAD_STATE_MIN_LOAD_TOKENS` (default `8192`)
  - `OFFLOAD_STATE_WORKERS` (default `1`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_state_offload_index.py`:

```python
def test_disabled_by_default_costs_nothing():
    """Zero cost when disabled is a stated constraint: depth 0 means every
    request_spill is refused and `hashes` stays empty, so `_resumable_from`
    degenerates to the original `in`."""
    idx = StateOffloadIndex(staging_depth=0, kv_offload_enabled=False)
    assert idx.enabled is False
    assert idx.request_spill(11, group=1) == -1
    assert idx.hashes == set()


def test_kv_offload_flag_is_carried_for_the_orphan_decision():
    assert StateOffloadIndex(1, kv_offload_enabled=True).kv_offload_enabled is True
    assert StateOffloadIndex(1, kv_offload_enabled=False).kv_offload_enabled is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_state_offload_index.py -v`
Expected: PASS if Task 2's implementation already satisfies both (likely). If it fails, fix `state_offload.py` — do not weaken the test.

- [ ] **Step 3: Construct the index in `BlockManager`**

Where `StateGroupPool` instances are built:

```python
        from atom.model_engine.state_offload import (
            StateOffloadIndex,
            state_offload_staging_groups,
        )

        staging = state_offload_staging_groups()
        if staging > 0:
            index = StateOffloadIndex(
                staging_depth=staging,
                kv_offload_enabled=bool(getattr(config, "kv_transfer_config", None)),
            )
            for cache in self.state_caches:
                cache.offload = index
            self.state_offload = index
```

`state_offload_staging_groups` is the single place the two env vars are read, added to `state_offload.py` in this step so the arena sizing (Step 4) and the pool wiring cannot disagree about K:

```python
import os


def state_offload_staging_groups() -> int:
    """K: staging *groups* to reserve, or 0 when the tier is off.

    Groups, not entries: a sizing site multiplies by its own
    `entries_per_req` to get rows, because `state_entry_views` is indexed by
    group. Returning entries here would make every caller divide back out.

    One function rather than two `os.environ` reads because the arena sizes
    itself from this and the pool wires itself from this; if they disagreed,
    the arena would be short exactly the rows the spill path addresses.
    """
    if os.environ.get("OFFLOAD_STATE", "0") != "1":
        return 0
    try:
        return max(0, int(os.environ.get("OFFLOAD_STATE_STAGING_GROUPS", "1")))
    except ValueError:
        return 1
```

- [ ] **Step 4: Reserve the K staging entries — one count for allocation, one for admission**

Both backends size their state tensors from `plan.entries[STATE_SLOT_CLASS]` (`deepseek_v4_attn.py:848` via `num_state_slots`, `gdn_attn.py:305`), and `BlockManager` divides *that same published number* into request groups (`block_manager.py:91-95`). One number cannot mean both "rows to allocate" and "rows admission may hand out". Split it on `PoolPlan`.

**Step 4a — write the failing test.**

```python
# tests/test_sub_pool_spec.py
def test_extra_entries_are_allocated_but_not_admissible():
    """`extra_entries` buys rows the pool must not lease out. Allocation
    sizes tensors from `entries`; admission divides `admission_entries`."""
    plan = plan_pools(
        [page_pool(1000), state_pool(ENTRY_STATE, 500, entries_per_req=2, extra_entries=3)],
        available_bytes=1_000_000,
        max_num_seqs=8,
    )
    assert plan.entries[ENTRY_STATE] == 8 * 2 + 3
    assert plan.admission_entries[ENTRY_STATE] == 8 * 2
    # The reservation covers every allocated row, staging included.
    assert plan.reserved_bytes[ENTRY_STATE] == (8 * 2 + 3) * 500


def test_admission_entries_defaults_to_entries_without_extras():
    plan = plan_pools(
        [page_pool(1000), state_pool(ENTRY_STATE, 500, entries_per_req=2)],
        available_bytes=1_000_000,
        max_num_seqs=8,
    )
    assert plan.admission_entries[ENTRY_STATE] == plan.entries[ENTRY_STATE] == 16


def test_staging_groups_are_sized_by_multiplicity():
    """A staging *group* is `entries_per_req` rows. Sizing the ring in bare
    entries would run the last staging group off the end of the tensor."""
    span, k = 3, 2  # GDN: span = 1 + num_spec
    plan = plan_pools(
        [page_pool(1000), state_pool(ENTRY_STATE, 500, entries_per_req=span, extra_entries=k * span)],
        available_bytes=1_000_000,
        max_num_seqs=8,
    )
    num_groups = plan.admission_entries[ENTRY_STATE] // span
    assert num_groups == 8
    # The highest staging group's last row must still be inside the tensor.
    assert (num_groups + k - 1) * span + span <= plan.entries[ENTRY_STATE]


def test_paged_class_admission_equals_entries():
    """PAGE has no per-request reservation, so the two counts coincide and
    `with_paged_entries` must keep them in step."""
    plan = plan_pools([page_pool(1000)], available_bytes=10_500, max_num_seqs=8)
    assert plan.admission_entries[ENTRY_KV] == plan.entries[ENTRY_KV] == 10
    assert plan.with_paged_entries(4).admission_entries[ENTRY_KV] == 4
```

Run: `python -m pytest tests/test_sub_pool_spec.py -k admission -v` → FAIL, `PoolPlan` has no attribute `admission_entries`.

**Step 4b — add the second count to `PoolPlan`** (`atom/model_ops/attentions/sub_pool_spec.py`). Add the field after `entries_per_req`:

```python
    # What allocation buys vs. what admission may lease. They differ only by
    # `extra_entries`: a flat cushion the declaring backend allocates for its
    # own use (the offload staging ring) and that no request may be given.
    # Kept as a computed table rather than a subtraction at each call site so
    # a consumer picks a meaning by the name it reads.
    admission_entries: dict[str, int] = field(default_factory=dict)
```

with `from dataclasses import dataclass, field, replace` at the top. `empty()` gains `admission_entries={}`; `with_paged_entries` adds it to the `replace(...)` so the reconciled paged count lands in both tables:

```python
        return replace(
            self,
            entries={**self.entries, self.paged_class: count},
            admission_entries={**self.admission_entries, self.paged_class: count},
            reserved_bytes={**self.reserved_bytes, self.paged_class: count * bytes_per},
        )
```

In `plan_pools`, populate it in both loops. STATE (replacing the body at `:220-223`):

```python
    for name, spec in state.items():
        admissible = max_num_seqs * spec.entries_per_req
        count = admissible + spec.extra_entries
        cost = count * spec.entry_bytes
        entries[name], reserved[name] = count, cost
        # The cushion is allocated, never leased.
        admissible_entries[name] = admissible
        remaining -= cost
```

PAGE (`:235-238`) sets `admissible_entries[name] = count` — no cushion exists there. Declare `admissible_entries: dict[str, int] = {}` beside `entries`, and pass `admission_entries=admissible_entries` to the returned `PoolPlan`.

Run: `python -m pytest tests/test_sub_pool_spec.py -v` → PASS, including the pre-existing `extra_entries` cases (`:80,106,138,153,240`), which assert on `entries` and are unaffected.

**Step 4c — publish the admission count, not the allocation count.** `model_runner.py:1642` and `:1719` both ship `dict(plan.entries)` as `pool_entries`, and `BlockManager` is its only consumer for the state class. Change both to the admission table:

```python
        # BlockManager divides this into request groups; the offload staging
        # cushion is allocated but not leasable, so admission must not see it.
        # Backends keep sizing their tensors from `pool_plan.entries`.
        config.pool_entries = dict(plan.admission_entries)
```

and identically in the returned dict at `:1719`. `pool_plan` itself (`:1641`) still carries both tables, so `num_state_slots` and `allocate_per_req_cache(self.pool_plan.entries)` (`:1806`) keep seeing the allocation count with no change. `total_reserved_bytes` already counts the cushion, so the 3% allocation cross-check at `:1915` stays truthful.

**Step 4d — declare K at the two sizing sites.** `deepseek_v4_attn.py:903`:

```python
            state_pool(
                STATE_SLOT_CLASS,
                geo.slot_bytes(row_bytes),
                entries_per_req=1,
                # K staging groups past the pool's group range: inside the
                # arena so `state_entry_views(num_groups + slot)` addresses
                # them with no second scheme, outside admission so no request
                # is ever handed one. Cost is K groups of HBM from the very
                # budget the pool wants -- and a small pool is the problem
                # being solved -- so K stays small (default 1) and measured.
                # Groups, not entries: `state_entry_views` indexes by group,
                # so a group costs `entries_per_req` rows (1 here).
                extra_entries=state_offload_staging_groups() * 1,
            ),
```

GDN's site multiplies by its own multiplicity, which is not 1:

```python
        span = 1 + self.num_spec
        return state_pool(
            STATE_SLOT_CLASS,
            self.model_runner.num_gdn_attn_state * per_layer,
            entries_per_req=span,
            # `span` rows per staging group -- `copy_state_entries` slices
            # `cache[:, dst_slot : dst_slot + span]`, so a ring sized in bare
            # entries would run the last staging group off the tensor.
            extra_entries=state_offload_staging_groups() * span,
        )
```

and the same `extra_entries=` argument on GDN's `state_pool(...)` (`gdn_attn.py:291-295`).

Import it **function-locally in each `sub_pool_specs()`**, not at module top:

```python
        from atom.kv_transfer.offload.state_offload import (
            state_offload_staging_groups,
        )
```

This matches how `model_ops/attentions` already reaches into `kv_transfer` (`deepseek_v4_attn.py:1321`, `aiter_mla.py:917`, `aiter_attention.py:733` are all function-local) and keeps the attention modules importable when the offload extra is absent.

Note `merge_specs` (`:180-192`) requires two specs naming one class to agree on `extra_entries`. Both sites call the same function reading the same env var, so they agree by construction — and a hybrid like Kimi-K3, which mixes MLA and KDA layers, is exactly the case that would otherwise raise.

**Step 4e — log both counts** so the HBM cost is visible. `model_runner.py:1649-1654` already logs per class; extend that line:

```python
        for name in sorted(plan.entries):
            extra = plan.entries[name] - plan.admission_entries[name]
            logger.info(
                f"sub-pool {name}: entries={plan.entries[name]}"
                + (f" ({plan.admission_entries[name]} admissible + {extra} staging)" if extra else "")
                + f", entry_bytes={plan.entry_bytes[name]}, "
                f"reserved={plan.reserved_bytes[name] / (1 << 30):.2f}GB"
            )
```

**Step 4f — verify the disabled path is byte-identical.** With `OFFLOAD_STATE=0`, `state_offload_staging_groups()` returns 0, `extra_entries=0`, and `admission_entries == entries` for every class.

Run: `python -m pytest tests/test_sub_pool_spec.py tests/test_block_manager.py tests/test_state_checkpoint.py -v` → PASS.

**Step 4g — commit.**

```bash
git add atom/model_ops/attentions/sub_pool_spec.py atom/model_engine/model_runner.py \
        atom/model_ops/attentions/deepseek_v4_attn.py atom/model_ops/attentions/gdn_attn.py \
        tests/test_sub_pool_spec.py
git commit -m "feat(state-offload): split allocation and admission entry counts"
```

- [ ] **Step 5: Construct the tier in `register_kv_caches`**

After `self._codec` is built and `storage_manager` is reachable:

```python
        if os.environ.get("OFFLOAD_STATE", "0") == "1":
            from atom.kv_transfer.offload.state_object import StateByteCodec
            from atom.kv_transfer.offload.state_tier import StateOffloadTier

            state_codec = StateByteCodec(
                backend,
                staged,
                entry_bytes=arena.entry_bytes,
                model_name=meta.model_name,
                world_size=world,
                worker_id=rank,
            )
            state_codec.bind_storage_manager(self._engine.storage_manager)
            self._state_tier = StateOffloadTier(
                state_codec,
                index,
                max_workers=int(os.environ.get("OFFLOAD_STATE_WORKERS", "1")),
            )
```

- [ ] **Step 6: Issue the staging copy and submit the spill**

`take_spill_copies` has no consumer yet. Drain it where `take_copies` is already drained into the forward's copy list — the same batch-construction site, so the staging copy lands on the compute stream before that step's forward:

```python
        spills = pool.take_spill_copies()
        if spills:
            # (group -> staging entry) first, on the compute stream, then hand
            # the D2H to the tier's own executor. The scheduler thread does no
            # device work: `pop()` is on its critical path.
            backend.copy_state_entries(
                [(g, pool.num_groups + slot) for g, slot in spills]
            )
            for h, slot in pool.offload.take_pending():
                self._state_tier.submit_spill(
                    h, entry_index=pool.num_groups + slot, staging_slot=slot
                )
```

`take_spill_copies` and `take_pending` are drained in that order and both are appended to by the same `_spill` call, so the copy for a slot is always issued before the D2H that reads it. Draining `take_pending` first would submit a read of bytes not yet copied.

- [ ] **Step 7: Document**

Extend the Module Map table in `atom/kv_transfer/offload/README.md`:

```markdown
| `staged_transfer.py` | Bounded GPU staging buffer, D2H/H2D, producer event. Shared by the KV chunk path and the state tier. |
| `state_object.py` | One state checkpoint -> one opaque uint8 object, keyed by ATOM's own xxhash64. Bypasses `ChunkedTokenDatabase`. |
| `state_tier.py` | Worker-side spill/load driver for the state tier, on its own executor. |
```

and add the four env vars to the env var table, each with default and one-line meaning. Add a short section noting: the tier is off by default; it only pays off when `checkpoints_evicted` is materially non-zero (check the `state checkpoints:` log line from Task 0); and cross-restart reuse does not exist — `LocalDiskBackend` never scans its own directory, so the in-memory index is correct and persisting it would be a false-positive generator.

- [ ] **Step 8: Full suite**

Run: `python -m pytest tests/ -q`
Expected: PASS (GPU-gated files skip on a CPU runner).

- [ ] **Step 9: Commit**

```bash
black . && ruff check .
git add atom/ tests/
git commit -m "feat(offload): wire the state offload tier and document it"
```

---

## Task 10: End-to-end validation on GPU

**Files:** none — this is a measurement task.

- [ ] **Step 1: Baseline**

```bash
rm -rf /root/.cache/atom/*
AITER_LOG_LEVEL=WARNING python -m atom.entrypoints.openai_server \
  --model <hybrid-model> --kv-cache-dtype fp8 -tp 8
```

Run the target workload, record the `state checkpoints:` line and TTFT.

- [ ] **Step 2: With the tier**

```bash
rm -rf /root/.cache/atom/*
AITER_LOG_LEVEL=WARNING OFFLOAD_STATE=1 OFFLOAD_STATE_STAGING_GROUPS=1 \
python -m atom.entrypoints.openai_server \
  --model <hybrid-model> --kv-cache-dtype fp8 -tp 8
```

Confirm VRAM with `rocm-smi --showmemuse` (VRAM% > 0) before trusting `/health`. On any server or GPU error, run `/debug-guide` — do not blindly retry.

- [ ] **Step 3: Accuracy gate**

Run `lm_eval` per `/ci-pr-guide` with `OFFLOAD_STATE=1`. A `P > L` bug shows up here and nowhere else — it produces wrong output with no error.

- [ ] **Step 4: Record the three open questions**

Append measurements to the spec's §7:
1. Staging depth K — does K=1 buy back more than its MB of HBM costs? Compare `spills_requested` vs `spills_dropped` and the eviction delta.
2. `checkpoints_evicted` on the real workload.
3. Whether `L → P` (bounding the KV load by the state boundary) is worth a follow-up. It is efficiency, not correctness.

- [ ] **Step 5: Commit the findings**

```bash
git add docs/superpowers/specs/2026-08-12-state-cache-lmcache-offload-design.md
git commit -m "docs(state-offload): record measured answers to the open questions"
```
