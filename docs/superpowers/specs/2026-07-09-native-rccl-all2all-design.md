# Native RCCL All2All Backend for ATOM MoE — Design

**Date:** 2026-07-09
**Status:** Approved for planning
**Author:** brainstorming session

## 1. Problem & Goal

ATOM's expert-parallel (EP) MoE path currently exchanges tokens between EP ranks
exclusively through the **MoRI** kernels (`MoriPrepareAndFinalize`, driving
`mori.ops.EpDispatchCombineOp.dispatch/combine`). MoRI is an optional dependency
that requires a separate build (ROCm/mori). On deployments where MoRI is not
installed, or where we want a dependency-free, portable baseline for
benchmarking and debugging, there is no native all2all path.

**Goal:** implement a native all2all backend built only on RCCL collectives
exposed through `torch.distributed` (`all_gather_into_tensor`,
`all_to_all_single`). It must be a drop-in alternative to `MoriPrepareAndFinalize`
producing byte-compatible dispatch output for the downstream aiter `fused_moe`
call, and it must support both:

- **Prefill** — variable token counts; host synchronization is allowed because
  prefill does not run under a CUDA graph.
- **Decode** — fixed shapes with padding, **no host synchronization** on routing
  tensors, so the path captures and replays inside a CUDA graph.

Non-goals: replacing MoRI as the default; TBO/async overlap (v1 is synchronous);
inter-node RDMA-specific tuning beyond what RCCL provides transparently.

## 2. Background: the interface we must match

The MoE caller is `FusedMoEModularKernel.forward()`
(`atom/model_ops/fused_moe/modular_kernel.py`). It calls, in order:

1. `prepare(a1, topk_weights, topk_ids, num_experts, expert_map,
   apply_router_weight_on_input, quant_config, quant_type)`
   → returns the 5-tuple
   `(dispatch_a1, dispatch_scale, ExpertTokensMetadata, dispatch_ids,
   dispatch_weights)`.
2. `fused_moe(dispatch_a1, w1, w2, dispatch_weights, dispatch_ids, ...,
   num_local_tokens=expert_tokens_meta.expert_num_tokens, ...)`.
3. `finalize(output, fused_expert_output, topk_weights, topk_ids,
   apply_router_weight_on_input)` → returns reduced `[num_tokens, hidden]`.

Key semantic requirements copied from `MoriPrepareAndFinalize`:

- `topk_indices_dtype() == torch.int32`.
- `output_is_reduced() == True` (finalize returns the reduced result; the caller
  does not reduce again).
- `max_num_tokens_per_rank()` returns the static per-rank token bound
  (`moe.max_num_tokens`).
- `num_dispatchers()` returns EP `world_size`.
- The dispatched activation buffer is grouped/ordered so that
  `fused_moe(..., num_local_tokens=expert_num_tokens)` reads only valid rows.
  `expert_num_tokens` is a **device** tensor (no `.item()` on the hot path).

The expert-ownership map is static: global expert `e` is owned by EP rank
`e // num_local_experts`, where `num_local_experts = num_experts // world_size`.

## 3. Architecture & integration

### 3.1 New class

Mirroring vLLM's DeepEP HT/LL split, the native backend is **two algorithm
classes over a shared base, behind a thin per-step dispatcher**. The two
algorithms are genuinely different (variable-length `all_to_all_single` vs
fixed-capacity graph-safe exchange), but they share FP8 quant, expert-ownership
math, and the finalize scatter-add — which live on the base.

`atom/model_ops/fused_moe/rccl_prepare_finalize.py`:

```
class RcclAll2AllBase:
    """Shared helpers: FP8 quant, dest_rank(expert), scatter-add combine,
    EP-group collective wrappers, routing-state stack. Holds no path policy."""
    def __init__(self, rank, world_size, hidden_dim, scale_dim,
                 max_tokens_per_rank, num_local_experts, num_experts_per_token,
                 in_dtype, use_fp8_dispatch, quant_type, ep_group): ...

class RcclHTPrepareAndFinalize(RcclAll2AllBase, mk.FusedMoEPrepareAndFinalize):
    """High-throughput / prefill: allgather topk -> host counts ->
    variable-length all_to_all_single (§4)."""

class RcclLLPrepareAndFinalize(RcclAll2AllBase, mk.FusedMoEPrepareAndFinalize):
    """Low-latency / decode: fixed cross-DP-unified capacity C, equal-split
    all_to_all_single, no host sync, CUDA-graph safe (§5)."""

class RcclPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """Dispatcher held by the MoE layer. Constructs one HT and one LL instance
    (sharing config) and forwards each prepare()/finalize() to the right one
    based on forward_context, because a served model alternates prefill and
    decode steps through the same MoE module."""
    def prepare(self, ...):
        ctx = get_forward_context().context
        impl = self._ll if (not ctx.is_prefill and ctx.dp_uniform_decode) \
               else self._ht
        return impl.prepare(...)
    # finalize() mirrors prepare(); the interface passthroughs
    # (topk_indices_dtype, output_is_reduced, num_dispatchers,
    #  max_num_tokens_per_rank) delegate to either impl (identical values).
```

Why a dispatcher rather than two independently-selected objects: the MoE layer
holds exactly one `prepare_finalize` object for the model's lifetime, yet each
forward step may be prefill **or** decode. The dispatcher is the object that
holds that single reference and routes per step. The HT/LL classes stay
single-responsibility and independently testable.

Non-uniform decode (`not dp_uniform_decode`) is forced eager and also routes to
the HT (variable-length) implementation — see §5 intro — so the LL class is only
ever entered under the graph-safe uniform-decode contract.

All three hold a reference to the EP `GroupCoordinator` (`get_ep_group()`), using
`all_gather_into_tensor` and `all_to_all_single` via `torch.distributed`. No MoRI
handle and no `all2all_manager` handle are required, so the classes are usable
when `_has_module("mori")` is False.

### 3.2 Selection

- New env var in `atom/utils/envs.py`:
  `ATOM_ALL2ALL_BACKEND` ∈ {`mori` (default), `rccl`}.
- In `atom/model_ops/moe.py:_maybe_make_prepare_finalize`, after computing the
  common MoE dimensions, branch:
  - `rccl` → build the `RcclPrepareAndFinalize` dispatcher (which internally
    constructs the HT and LL implementations sharing one config).
  - `mori` (default) → existing `MoriPrepareAndFinalize` path, unchanged.
- `FusedMoEParallelConfig.use_all2all_kernels` currently requires
  `_has_module("mori")`. Relax it so that when
  `ATOM_ALL2ALL_BACKEND == "rccl"` the all2all path is enabled without MoRI
  present. (`dp_size > 1 and use_ep and (mori_present or backend == "rccl")`.)

### 3.3 Path selection & routing-state hand-off

The dispatcher (§3.1) selects HT vs LL per step from
`get_forward_context().context` (`is_prefill`, `dp_uniform_decode`). The chosen
implementation runs the algorithm in §4 (HT) or §5 (LL). FP8 quant (§6) and the
expert-ownership arithmetic live on `RcclAll2AllBase` and are shared by both.

**Routing state hand-off.** `finalize()` must invert the exact routing computed
in `prepare()` (send/recv split sizes and the per-(token,expert) pack index).
Because `FusedMoEModularKernel` calls `prepare` then `fused_moe` then `finalize`
synchronously within one layer's forward, each HT/LL instance stores its routing
state in a small stack on the instance (one entry pushed by `prepare`, popped by
`finalize`) — a helper on `RcclAll2AllBase`. This mirrors how MoRI caches its
routing map internally on the op handle. The stack (not a single slot) keeps
shared-expert-overlap and repeated calls correct; entries hold only device
tensors + the cpu split lists (HT). Because HT and LL are separate instances,
their stacks never interleave even if a (pathological) mixed step occurred.

## 4. Prefill / HT dispatch (variable-length, host sync allowed)

Implemented by `RcclHTPrepareAndFinalize`. Entered for prefill steps and for
non-uniform decode (§5 intro). `prepare()`:

1. **Quantize (optional).** If `use_fp8_dispatch`, quantize `a1` to fp8 with the
   configured `quant_type` (`per_1x128` or `per_Token`) via `get_hip_quant`,
   producing `a1_fp8` and `scale`. Otherwise dispatch in `in_dtype` (bf16).
2. **All-gather `topk_ids`.** `all_gather_into_tensor` over the EP group so every
   rank has the global `[world_size * local_tokens, topk]` routing (each rank
   contributes its own local `topk_ids`; local token counts are gathered too).
3. **Host-side counts.** For this rank's experts, count how many (token, expert)
   pairs each source rank sends here (recv counts) and how many pairs this rank
   sends to each destination (send counts). This uses `.cpu()`/`.tolist()` —
   allowed in prefill.
4. **Pack send buffer.** Build a device gather-index that, for each local token
   replicated once per selected expert, writes it into a contiguous send buffer
   ordered by destination rank. Gather `hidden_states` (and `scale`) with it.
5. **all_to_all_single (variable split).** Exchange the packed hidden_states
   with the computed send/recv split sizes; likewise for scales (FP8) and for
   the per-pair `topk_weights` and `expert_ids`.
6. **Return** `(recv_a1, recv_scale, ExpertTokensMetadata(expert_num_tokens=...,
   expert_num_tokens_cpu=...), recv_expert_ids, recv_weights)`. The recv buffer
   is ordered grouped-by-local-expert so `num_local_tokens` selects valid rows.

`finalize()` (prefill): run the inverse `all_to_all_single` (swap send/recv
counts) to return each expert output to its originating rank, then scatter-add
the per-(token,expert) contributions back to `[num_tokens, hidden]`, weighting by
`topk_weights` unless `apply_router_weight_on_input`. Save the routing index from
`prepare()` (per-layer, per-forward) to drive the inverse scatter.

## 5. Decode / LL dispatch (fixed-shape, CUDA-graph safe)

Implemented by `RcclLLPrepareAndFinalize`. Entered only for
`dp_uniform_decode=True` steps. Constraint: constant tensor shapes across
capture/replay and **no host sync** on routing tensors (`topk_ids` never leaves
device).

**Per-rank batch sizes differ across DP ranks.** Under DP+EP each DP rank runs
attention on its own local decode batch, but the MoE all2all crosses all EP
ranks, so the send/recv capacity cannot be sized from the *local* batch. ATOM
already exchanges per-rank token counts **before the step** via
`DPMetadata.num_tokens_across_dp` (an `all_reduce` on the DP cpu group) and
`.item()`s `max_tokens_across_dp` at metadata-build time — outside any graph.
In `dp_uniform_decode` mode this is exactly why `ModelRunner._preprocess` sets
`num_input_tokens = max_tokens` and `ForwardMode.decide` makes `moe_pad_bs` /
`context.graph_bs` **cross-DP unified**: every rank pads to the same captured
size. The RCCL backend builds on this — it does not add a new collective.

Fixed per-rank capacity: `C = graph_bs * num_experts_per_token`, where
`context.graph_bs` is the **cross-DP-unified** padded batch (the captured size ≥
`max_tokens_across_dp`, picked in `ForwardMode.decide`). Because every DP/EP rank
sees the identical `graph_bs`, `C` is the same constant on all ranks and across
capture/replay — so the equal-split `all_to_all_single` never shape-mismatches.
`C` is a Python int resolved on host before capture (from the pre-step count
exchange), never an in-graph `.item()`.

Non-uniform decode (`dp_uniform_decode=False`) is forced eager today
(`use_cudagraph=False`) and MoE uses the variable-length path; the RCCL backend
handles it with the prefill-style variable-split `all_to_all_single` (§4), since
host sync is permitted when not capturing. Only `dp_uniform_decode=True` uses the
fixed-capacity graph path below.

`prepare()` when `not is_prefill` and `dp_uniform_decode`:

0. Read `C` from `context.graph_bs` (cross-DP-unified, resolved on host before
   capture from the pre-step `num_tokens_across_dp` exchange — §5 intro). No
   collective and no `.item()` happen here inside the graph.
1. `topk_ids` stays on device.
2. **Device pack.** A device operation (Triton or `torch.scatter_`-based) builds
   a fixed `[world_size, C, hidden]` send buffer: for each local (token, expert)
   pair, compute `dest_rank = expert // num_local_experts` on device and write
   the token into `send[dest_rank, slot]`, where `slot` comes from a device
   per-destination running counter. Unused slots are **zero-filled** (mirrors
   `pad_for_all_gather`, which zeros pad rows to keep garbage out of the expert
   GEMM). A parallel `[world_size, C]` index buffer records the source
   (token, expert) so finalize can invert without host state.
3. **all_to_all_single (equal fixed split = C).** Static shapes → CUDA-graph
   capturable. Exchange hidden_states, and (FP8) scales, and the per-slot
   expert_ids/weights.
4. **Recv buffer** `[world_size * C, hidden]`. A device `expert_num_tokens`
   tensor (count of valid rows per local expert) feeds `fused_moe`'s
   `num_local_tokens`, which is already device-driven — no `.item()`.
5. Overflow (more than `C` pairs to one destination) is dropped and logged once
   via a device-side saturating counter checked outside the graph; under uniform
   decode with correctly sized `C` this never triggers.

`finalize()` (decode): inverse fixed `all_to_all_single` (still equal split `C`),
then a device scatter-add using the saved index buffer, weighted by
`topk_weights`, producing `[graph_bs, hidden]`; slice `[:num_token]`.

The existing `_maybe_trim_dispatch_output` graph_bs pattern in
`modular_kernel.py` is the precedent for keeping trimmed shapes consistent across
capture/replay; the decode path here follows the same constant-shape discipline.

## 6. FP8 dispatch

- Reuse `aiter.get_hip_quant(quant_type)` exactly as `MoriPrepareAndFinalize`
  does. `quant_type` is `per_1x128` (block) or `per_Token` (per-act-token),
  selected from `quant_config` in `moe.py`.
- The scale tensor is exchanged with a second `all_to_all_single` that mirrors
  the hidden_states split sizes (prefill) or the fixed `C` split (decode).
- `scale_dim == 0` (no-scale) short-circuits the scale exchange.
- `topk_indices_dtype()` returns `torch.int32`; `output_is_reduced()` returns
  `True`.

## 7. Edge cases

- **world_size == 1:** bypass all collectives; dispatch is identity, finalize is
  a weighted local combine.
- **Zero local tokens on a rank:** allocate correctly-shaped empty buffers (the
  mori async path already handles this shape convention; match it).
- **Overflow in decode:** saturating device counter, logged once; never trims
  below `num_local_tokens`.
- **`apply_router_weight_on_input`:** asserted False (matches mori), so finalize
  always applies `topk_weights`.

## 8. Testing

- **Layout/round-trip unit test (no GPU):** using ATOM's mocked-dist test style
  (mocks `torch.distributed` collectives with a local reference), assert
  dispatch→combine is identity on hidden_states for a known routing, and that
  the dispatched buffer is grouped by local expert with the correct
  `expert_num_tokens`.
- **Numerical parity test (GPU + mori):** both `RcclHTPrepareAndFinalize`
  (prefill) and `RcclLLPrepareAndFinalize` (decode) vs. `MoriPrepareAndFinalize`
  on the same routing/inputs; assert max abs diff within fp tolerance for bf16
  and fp8 dispatch. Also test the `RcclPrepareAndFinalize` dispatcher routes to
  the correct impl per forward_context.
- **CUDA-graph smoke test:** capture the decode `prepare`+`finalize` under a
  graph and replay with different (but ≤ graph_bs) token counts; assert no host
  sync (no `.item()`/`.cpu()` on routing tensors) and correct output.
- **Differing per-DP-rank batch test:** simulate DP ranks with unequal local
  decode batches; assert the fixed capacity `C` derived from the cross-DP-unified
  `graph_bs` is identical on every rank and the equal-split `all_to_all_single`
  does not shape-mismatch. Complements the smoke test by exercising the exact
  bug this revision fixes.
- **world_size==1 test:** bypass path returns the reference weighted combine.

## 9. Files touched

- **New:** `atom/model_ops/fused_moe/rccl_prepare_finalize.py`
  (`RcclAll2AllBase`, `RcclHTPrepareAndFinalize`, `RcclLLPrepareAndFinalize`,
  `RcclPrepareAndFinalize` dispatcher, device pack/unpack helpers).
- **New:** `tests/model_ops/test_rccl_all2all.py` (layout + parity + graph tests).
- **Edit:** `atom/utils/envs.py` — add `ATOM_ALL2ALL_BACKEND`.
- **Edit:** `atom/model_ops/moe.py` — backend branch in
  `_maybe_make_prepare_finalize`; relax `use_all2all_kernels` gate.
- **Docs:** note the env var in `docs/environment_variables.md` and the
  distributed guide.

## 10. Open risks

- Device pack/unpack kernels (§5 step 2) are the highest-risk piece; a
  `torch.scatter_`/`index_copy_`-based implementation is the correctness
  baseline, with a Triton kernel as a later optimization.
- Byte-exact grouping for `fused_moe` — validated by the parity test against
  MoRI, which is the source of truth for the expected layout.
