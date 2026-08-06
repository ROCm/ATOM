# SparseKV decode 0x1016 — root cause & fix

GLM-5.2 PP4×PD + SparseKV decode crashes probabilistically under the aiperf
agentic-trace replay with `HSA_STATUS_ERROR_EXCEPTION code: 0x1016`
(`ModelRunner proc died, exitcode=-6`). This document is the authoritative
root-cause summary and the fix plan. The blow-by-blow investigation (all the
ruled-out hypotheses, every instrumentation attempt) lives in
`docs/sparsekv_hsa_fault_0x1016_report.md` (Sessions 1, 2, 2.6).

## 1. Confirmed fault signature

The faulting GPU kernel (captured 3× under `rocm-debug-agent`, consistent across
all 4 TP ranks):

```
at::native::_scatter_gather_elementwise_kernel<256, 4,
  _cuda_scatter_gather_internal_kernel<false, OpaqueType<4>, long>
    ::operator()<TensorAssign>>          (stopped, reason: ASSERT_TRAP)
```

= a PyTorch `at::gather` on a **4-byte** tensor (int32/float32) with an **int64**
index, whose device-side bounds assert `0 <= idx < size` fired → **out-of-bounds
index**. Register capture at the trap: index value `v2 = 0x2040 = 8256` (note
`8256 = 8192 + 64`, and `hot_buffer_size H = 8192`, `padded_hot_size H1 = 8193`);
nearby dims/strides `0x200=512` (`kv_lora_rank`), `0x240=576` (`kv_dim`).

## 2. Root cause — a deferred-out pipeline write-after-read race on
## single-instance per-step buffers, unprotected on the pp=1 decode node

Established by elimination + two decisive experiments (see §3):

- ATOM protects the decode pipeline from "step N+1's host-side buffer writes
  clobbering step N's still-in-flight GPU reads" with a **`forward_vars` ring +
  per-slot CUDA event** (`model_runner.py: _init_forward_vars_ring` /
  `_advance_forward_vars` / `_record_forward_vars_event`).
- **The ring is sized to `pp_size`.** The decode node runs **TP4, pp=1**, so the
  ring has a **single slot**, and `_advance_forward_vars` / the event gate are an
  explicit **no-op** (`if len(self._fv_ring) == 1: return`). There is no
  cross-step double-buffering or event gating on the decode node.
- The decode node uses **deferred output** (`is_deferred_out`, TP>1): it does not
  synchronize the sampled token each step, deliberately letting the CPU **run
  ahead** of the GPU to overlap step N+1's host-side preparation with step N's
  in-flight forward.
- The forward reads a set of **single-instance, stateful GPU buffers** that the
  next step's host-side preparation **overwrites**:
  - the SparseKV coordinator tables (`slot_token`, `last_used`, `token_to_slot`,
    `recency`, `req_to_host_pool`) — rewritten each step by
    `_sparsekv_stage_and_sync` → `sync_active`/`release`/`unregister_request`
    (fills `-1`) and `acquire`/`register_request` (resets a slot) and
    `alloc_host_pages`;
  - the SparseKV per-token metadata (`attn_metadata.sparsekv_req_slots` /
    `sparsekv_token_pos` / `sparsekv_src_slots`) and `coord.topk_buffer` /
    `sparse_kv_indices_buffer`;
  - `input_ids_loc` (the deferred-token gather index buffer).
  None of these are in the `forward_vars` ring, so even on a PP node they would be
  unprotected; on the pp=1 decode node nothing is protected.

Under concurrent long-context decode the CPU runs far enough ahead that step
N+1's preparation **frees/reassigns a request slot and rewrites that slot's
coordinator rows / metadata while step N's forward is still reading them**. The
in-flight swap/translate then resolves an index that was valid for the slot's
previous occupant to an out-of-range **absolute** hot-buffer row
(`req_slot*H1 + slot ≈ 8256`, overrunning a per-request-scoped dimension), which
the consuming `at::gather` reads out of bounds → device assert → 0x1016.

This is why the faulting `at::gather` is **absent from steady-state decode**
(10 s of burst decode captured 0 `_scatter_gather` kernels via kineto) — it only
materializes on the specific step where a slot is reassigned mid-flight.

## 3. Decisive evidence

- **`AMD_SERIALIZE_KERNEL=3` fully masks it.** Under serialize (+debug-agent),
  686 decode steps / req 590 / 518K-token contexts ran with **zero** crashes
  (bare crash rate ≈ 2/3, fires by req 48–150). Forcing a host sync after every
  kernel removes the CPU run-ahead → no overwrite-in-flight → no crash. This
  proves it is a timing race of exactly this "CPU ahead of GPU" shape, not a
  data-/composition-deterministic OOB.
- **Every dispatch/stream-touching instrumentation also masks it** (a global
  `torch.gather` sync hook; a per-gather `index.max()` check; a bare
  `TorchFunctionMode`; `rocgdb` attach stalls the TP collective outright) — all
  consistent with a fine timing race.
- **`ATOM_SPARSEKV_PREFETCH=0` still crashes (earlier, req 48):** the IndexShare
  prefetch side stream is exonerated; the race is in the **main** swap path.
- The report's earlier suspects are disproven: `coordinator.plan_swap_for_request`
  (advanced-index → `at::index`, int64 data, reference/MTP-only path — not run at
  level 3) and `model_runner.py:522` (zero-sync host bounds probe fired 0× across
  a natural crash).

## 4a. Shipped fix (implemented + validated)

`atom/model_engine/model_runner.py`:
- `ModelRunner.__init__`: added `self._sparsekv_prev_forward_event` (a
  `torch.cuda.Event`, lazily created) and `self._sparsekv_last_active_reqids`
  (set of the previous SparseKV decode step's req_ids).
- `ModelRunner.forward`: after `run_model`, for a real SparseKV decode batch
  (`coord is not None and not is_dummy_run and total_tokens_num_prefill == 0`),
  `record()` the event on the (default) forward stream.
- `_sparsekv_stage_and_sync`: before `coord.sync_active(...)`, if
  `set(req_ids) != self._sparsekv_last_active_reqids` (batch composition changed →
  a slot is freed/reassigned → the write that can clobber in-flight reads),
  `self._sparsekv_prev_forward_event.synchronize()`; then update the tracked set.
  Steady-state steps (unchanged composition) never mutate the coordinator tables,
  so they skip the drain and keep the deferred-out CPU/GPU overlap.
- `atom/sparsekv/coordinator.py: release()`: a note that this is the sole path a
  slot re-enters the free pool, on which the above gate's safety depends.

Why the drain (not a GPU stream-wait): the base decode forward runs on the default
stream, so a same-stream `event.wait(current_stream)` is a program-order no-op and
does NOT bound the CPU run-ahead — validated: the `.wait()` variant re-crashed at
req 176. The CPU `synchronize()` bounds the run-ahead, which is what removes the
race (same mechanism as `AMD_SERIALIZE_KERNEL=3`, but scoped to the one dangerous
step). Gating it on batch-composition change keeps the overlap on steady steps.

Validation (bare, no serialize / agent / debug):
- Crash **eliminated**: reproduces bare by req 48–150 (≈2/3); with the fix,
  multiple runs reached **req 460–515 across repeated 518K-token-context bursts
  with 0 crashes**.
- Independent code review: fix is correct and sufficient for this config; the
  composition-change gate covers every reachable dangerous coordinator-table
  write; the end-of-forward event correctly captures the forward's reads on the
  default stream (and, with prefetch on, transitively via `wait_prefetch`).
- Accuracy: GSM8K 5-shot (limit 100, via the mesh) = flexible-extract 0.96 /
  strict-match 0.95, 0 crashes — at/above the SparseKV baseline (~0.93).

Rejected alternatives: `event.wait(current_stream)` (no-op on single stream →
re-crashed); a blanket `AMD_SERIALIZE_KERNEL` / unconditional per-step
`synchronize()` (correct but kills overlap every step).

## 4b. Original fix plan (superseded by 4a)

Goal: guarantee that step N+1's host-side rewrites of the single-instance,
in-flight-read decode buffers happen **after** step N's forward has finished
reading them, on the pp=1 deferred-out decode node — without a blanket
`AMD_SERIALIZE_KERNEL`-style global serialize.

**Stage A (minimal, provably correct — land first).** Gate the next step's
SparseKV coordinator mutations on the previous forward's completion: record a
CUDA event at the end of each decode forward, and before `_sparsekv_stage_and_sync`
performs any coordinator/slot mutation for the next step, wait on that event.
Scope strictly to `sparsekv_coordinator is not None and is_deferred_out and
pp_size == 1` so non-SparseKV / PP paths are byte-for-byte unchanged. Validate:
loop the aiperf trace bare (no serialize, no agent) to confirm the crash is gone;
GSM8K to confirm accuracy is unaffected.

**Stage B (restore overlap — after A is proven).** Replace the coarse
event-wait with true double-buffering / per-slot event gating for the *small*
per-step index buffers (`sparsekv_req_slots`/`token_pos`/`src_slots`,
`topk_buffer`, `sparse_kv_indices_buffer`, `input_ids_loc`) so the CPU can run
ahead again, while the *stateful* coordinator tables (LRU state, which cannot be
double-buffered) stay event-gated. Optionally: enable the existing `forward_vars`
ring with ≥2 slots on the decode node when deferred-out is active, and extend it
to cover the SparseKV per-step buffers.

Do NOT "fix" this with a bare `torch.cuda.synchronize()` sprinkled in the hot
path, and do NOT rely on `AMD_SERIALIZE_KERNEL` in production — both are global
masks, not scoped ordering.
