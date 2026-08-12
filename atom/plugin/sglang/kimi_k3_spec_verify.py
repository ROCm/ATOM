# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 DSpark ``TARGET_VERIFY`` bridge for the KDA (GDN) linear layers.

Background
----------
K3's KDA layers run through ATOM's ``KimiKDAAttention`` (``atom/models/kimi_k3.py``),
which already owns a native speculative-decode branch (``num_spec_decodes > 0``):

* conv : ``causal_conv1d_update`` with a *wide* rolling window
  (``conv_state_indices = spec_state_indices[:, 0]``, ``num_accepted_tokens``,
  ``max_query_len = 1 + num_spec``).  The wide window per request holds
  ``[history ... , draft_0 ... draft_{T-1}]`` so acceptance can resume from any
  step.
* ssm : ``fused_sigmoid_gating_delta_rule_update`` with a 2-D
  ``ssm_state_indices`` [bs, 1 + num_spec]: it reads the resume state from
  ``slot[num_accepted - 1]`` and writes one snapshot per draft token.

SGLang's ``MambaPool.SpeculativeState`` uses a *different* layout: a single
committed conv/temporal slot per request plus a separate per-draft-token scratch
(``intermediate_ssm`` / ``intermediate_conv_window``), committed centrally after
verify by ``update_mamba_state_after_mtp_verify``.

To keep the *ops identical to ATOM core* we feed ``KimiKDAAttention``'s existing
spec branch the ATOM-native layout, backed by the sglang scratch:

* ssm  : bind ``ssm_state`` to ``intermediate_ssm[layer]`` flattened to
  ``[(spec+1) * T, HV, V, K]``; ``spec_state_indices[i] = arange(i*T, (i+1)*T)``;
  ``num_accepted = 1`` so the kernel resumes from column 0.  We pre-copy the
  committed ``temporal[cache_idx[i]]`` into ``intermediate_ssm[i, 0]`` (the kernel
  reads column 0 as the resume state *before* overwriting it with the post-token-0
  snapshot).  Result: ``intermediate_ssm[i, t] = state-after-token-t`` — exactly
  what sglang's central ssm scatter commits.
* conv : bind ``conv_state`` to a *wide* per-verify scratch pre-seeded from the
  committed narrow window.  ATOM rolls it in place; after verify we commit the
  accepted narrow window back to the committed conv pool ourselves (sglang's
  conv scatter, which reads ``intermediate_conv_window``, is bypassed for K3).

Only the K3 path is affected; every hook is guarded by ``_atom_k3_spec`` being
present on the pool, so other hybrid-linear models keep their native commit.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from atom.config import KVCacheTensor

logger = logging.getLogger(__name__)

_POOL_SPEC_ATTR = "_atom_k3_spec"
# Single stacked [num_mamba_layers, max_bs*T, wide, conv_dim] conv scratch. When
# present, every per-layer entry of ``pool._k3_conv_scratch`` is a view into it,
# which is what lets the post-verify conv commit run as ONE fused kernel over all
# layers instead of a per-layer Python loop.
_POOL_STACKED_ATTR = "_k3_conv_scratch_stacked"


def _conv_commit_mode() -> str:
    """Which post-verify conv commit implementation to use.

    ``ATOM_K3_FUSED_CONV_COMMIT``:

    * unset / ``1``  -> ``fused``  : one triton kernel for all KDA layers
    * ``0``          -> ``legacy`` : original per-layer boolean-mask loop
    * ``check``      -> run BOTH on the live tensors and compare (debug)
    """
    raw = os.environ.get("ATOM_K3_FUSED_CONV_COMMIT", "1").lower()
    if raw in ("0", "false", "no"):
        return "legacy"
    if raw == "check":
        return "check"
    return "fused"


# ---------------------------------------------------------------------------
# CUDA-graph capture keep-alive.
#
# sglang captures the per-bs verify/decode graphs in DESCENDING bs order into a
# single shared graph mempool. Every tensor we allocate *inside* a metadata build
# that runs during capture (index tensors, kv_indptr/kv_indices, block tables,
# the pre-seed scratch cursors, ...) is written by a captured kernel. If such a
# tensor is freed once the build returns, its memory goes back to the mempool and
# the NEXT (smaller-bs) capture reuses that exact address -- silently clobbering
# the first-captured (max-bs) graph's buffers on replay (observed as correct
# output for bs < max but garbage for bs == max).
#
# We do NOT need sglang-style in-place refresh: our build kernels are themselves
# captured, so they re-run on every replay reading sglang's persistent, refreshed
# input buffers (seq_lens / out_cache_loc / state_indices_list). The only
# requirement is that our transient OUTPUT tensors keep a stable address, i.e.
# are never freed. Stashing them here (module lifetime) guarantees that. The
# amount is tiny and bounded (a handful of small tensors per captured bucket).
_GRAPH_KEEPALIVE: list = []


def keepalive_if_capturing(*objs: Any):
    """Pin ``objs`` for the process lifetime iff a CUDA graph is being captured.

    Returns the single object when called with one arg (so it can wrap an
    allocation inline), else the tuple. Outside capture it is a no-op.
    """
    try:
        capturing = torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001 - non-CUDA / no active stream
        capturing = False
    if capturing:
        _GRAPH_KEEPALIVE.extend(objs)
    return objs[0] if len(objs) == 1 else objs


def is_target_verify(forward_batch: Any) -> bool:
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None:
        return False
    fn = getattr(mode, "is_target_verify", None)
    try:
        return bool(fn()) if callable(fn) else False
    except Exception:  # noqa: BLE001
        return False


def _draft_token_num(forward_batch: Any) -> int:
    spec_info = getattr(forward_batch, "spec_info", None)
    dtn = getattr(spec_info, "draft_token_num", None)
    if dtn is None:
        raise RuntimeError("Kimi-K3 target-verify: spec_info.draft_token_num missing")
    return int(dtn)


def _committed_indices(
    forward_batch: Any, linear_backend: Any, bs: int
) -> torch.Tensor:
    """Committed mamba slot per request (the same tensor sglang commits into)."""
    fm = getattr(linear_backend, "forward_metadata", None)
    idx = getattr(fm, "mamba_cache_indices", None)
    if idx is None:
        from atom.plugin.sglang.attention_backend.backend_resolver import (
            reconstruct_linear_metadata,
        )

        reconstructed = reconstruct_linear_metadata(forward_batch, linear_backend)
        if reconstructed is None:
            raise RuntimeError("Kimi-K3 target-verify: cannot resolve mamba indices")
        _, idx = reconstructed
    return idx[:bs].to(dtype=torch.int32)


@dataclass
class _SpecPlan:
    bs: int
    draft_token_num: int  # T = 1 + num_spec
    num_spec: int
    device: torch.device
    committed_indices: torch.Tensor  # [bs] int32 committed slot per request
    spec_state_indices: torch.Tensor  # [bs, T] int32 -> flattened scratch rows
    spec_query_start_loc: torch.Tensor  # [bs + 1] int32
    num_accepted_tokens: torch.Tensor  # [bs] int32 (ones during verify forward)
    spec_sequence_masks: torch.Tensor  # [bs] bool ones
    spec_token_indx: torch.Tensor  # [bs * T] int32 arange
    # Populated by build_spec_cache_tensors (per KDA layer wide conv scratch).
    conv_scratch: dict[int, torch.Tensor] = field(default_factory=dict)
    conv_kernel_minus_1: int = 0


def build_spec_plan(forward_batch: Any, linear_backend: Any) -> _SpecPlan:
    """Build (and cache on the forward_batch) the K3 verify state plan."""
    cached = getattr(forward_batch, "_atom_k3_spec_plan", None)
    if cached is not None:
        return cached

    bs = int(forward_batch.batch_size)
    T = _draft_token_num(forward_batch)
    committed = _committed_indices(forward_batch, linear_backend, bs)
    device = committed.device

    row_base = torch.arange(bs, dtype=torch.int32, device=device).view(bs, 1) * T
    cols = torch.arange(T, dtype=torch.int32, device=device).view(1, T)
    spec_state_indices = (row_base + cols).contiguous()  # [bs, T]

    spec_query_start_loc = torch.arange(
        0, (bs + 1) * T, T, dtype=torch.int32, device=device
    )
    num_accepted_tokens = torch.ones(bs, dtype=torch.int32, device=device)
    spec_sequence_masks = torch.ones(bs, dtype=torch.bool, device=device)
    spec_token_indx = torch.arange(bs * T, dtype=torch.int32, device=device)

    plan = _SpecPlan(
        bs=bs,
        draft_token_num=T,
        num_spec=T - 1,
        device=device,
        committed_indices=committed,
        spec_state_indices=spec_state_indices,
        spec_query_start_loc=spec_query_start_loc,
        num_accepted_tokens=num_accepted_tokens,
        spec_sequence_masks=spec_sequence_masks,
        spec_token_indx=spec_token_indx,
    )
    forward_batch._atom_k3_spec_plan = plan
    # During capture the plan's index/state tensors are read by captured GDN
    # kernels; pin them so a later (smaller-bs) capture cannot reuse their memory.
    keepalive_if_capturing(
        committed,
        spec_state_indices,
        spec_query_start_loc,
        num_accepted_tokens,
        spec_sequence_masks,
        spec_token_indx,
    )
    return plan


def build_spec_gdn_metadata(plan: _SpecPlan):
    """Assemble the ATOM ``GDNAttentionMetadata`` spec fields for verify."""
    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadata

    bs = plan.bs
    T = plan.draft_token_num
    return GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=bs,
        num_spec_decode_tokens=bs * T,
        num_actual_tokens=bs * T,
        has_initial_state=None,
        spec_query_start_loc=plan.spec_query_start_loc,
        non_spec_query_start_loc=None,
        spec_state_indices_tensor=plan.spec_state_indices,
        non_spec_state_indices_tensor=None,
        spec_sequence_masks=plan.spec_sequence_masks,
        spec_token_indx=plan.spec_token_indx,
        non_spec_token_indx=None,
        num_accepted_tokens=plan.num_accepted_tokens,
        nums_dict=None,
        batch_ptr=None,
        token_chunk_offset_ptr=None,
    )


def _max_graph_bs(linear_backend: Any, bs: int) -> int:
    lst = getattr(linear_backend, "state_indices_list", None)
    if lst:
        return len(lst)
    return bs


def preallocate_k3_verify_scratch(pool: Any, max_bs: int, draft_token_num: int) -> None:
    """Pre-allocate the per-KDA-layer conv verify scratch in *normal* memory.

    This MUST run outside CUDA-graph capture (e.g. from the backend's
    ``init_cuda_graph_state``). If the scratch were first allocated during capture
    it would land in the graph memory pool, whose regions the allocator may reuse
    for other captured graphs' activations -- silently clobbering the tail rows
    that only the max-bs graph touches (observed as correct output for bs<max but
    garbage for bs==max). Allocating here keeps the buffer in the normal pool where
    it is never reused. ``build_spec_cache_tensors`` then simply finds and reuses
    it, so no allocation happens during capture.
    """
    if draft_token_num <= 0 or max_bs <= 0:
        return
    mamba_map = getattr(pool, "mamba_map", None)
    if mamba_map is None:
        return
    shared = getattr(pool, "_k3_conv_scratch", None)
    if shared is None:
        shared = {}
        pool._k3_conv_scratch = shared
    T = int(draft_token_num)
    num_spec = T - 1
    rows = max_bs * T

    layers: list[tuple[int, torch.Tensor]] = []
    for layer_id in mamba_map:
        try:
            layer_cache = pool.mamba2_layer_cache(layer_id)
        except Exception:  # noqa: BLE001,S112 - non-mamba layers
            continue
        conv = getattr(layer_cache, "conv", None)
        if conv is None:
            continue
        layers.append((layer_id, conv[0]))
    if not layers:
        return

    # Fused commit needs ONE buffer with a layer stride, so it only applies when
    # every KDA layer shares the same conv geometry (they do for K3).
    geoms = {(int(c.shape[-2]), int(c.shape[-1]), c.dtype, c.device) for _, c in layers}
    if (
        _conv_commit_mode() != "legacy"
        and len(geoms) == 1
        and isinstance(mamba_map, dict)
    ):
        km1, conv_dim, dtype, device = next(iter(geoms))
        wide = km1 + num_spec
        num_layers = len(mamba_map)
        stacked = getattr(pool, _POOL_STACKED_ATTR, None)
        if (
            stacked is None
            or stacked.shape[0] != num_layers
            or stacked.shape[1] < rows
            or stacked.shape[2] != wide
            or stacked.shape[3] != conv_dim
        ):
            stacked = torch.zeros(
                (num_layers, rows, wide, conv_dim), dtype=dtype, device=device
            )
            setattr(pool, _POOL_STACKED_ATTR, stacked)
        # Hand each layer a view; the ordinal must match the layer axis of the
        # stacked conv pool (``mamba_map`` maps layer_id -> that ordinal).
        for layer_id, _ in layers:
            shared[layer_id] = stacked[int(mamba_map[layer_id])]
        return

    setattr(pool, _POOL_STACKED_ATTR, None)
    for layer_id, conv0 in layers:
        km1 = int(conv0.shape[-2])
        conv_dim = int(conv0.shape[-1])
        wide = km1 + num_spec
        existing = shared.get(layer_id)
        if (
            existing is not None
            and existing.shape[0] >= rows
            and existing.shape[1] == wide
            and existing.shape[2] == conv_dim
        ):
            continue
        shared[layer_id] = torch.zeros(
            (rows, wide, conv_dim), dtype=conv0.dtype, device=conv0.device
        )


def build_spec_cache_tensors(
    forward_batch: Any,
    linear_backend: Any,
    pool: Any,
    mamba_map: Any,
) -> dict[str, KVCacheTensor]:
    """Bind per-KDA-layer verify state (wide conv scratch + intermediate_ssm view)
    and pre-seed the resume state from the committed pool.

    CUDA-graph notes:
    * The conv scratch is a SINGLE max-bs buffer per layer (stable address across
      all captured bs graphs). Each build writes only its ``[:bs*T]`` slice, and
      real requests always occupy the front rows, so the post-verify commit can
      read ``[:real_bs*T]`` regardless of which (possibly padded) bs graph ran.
    * ``mamba_cache_indices`` may contain ``-1`` padding sentinels under a padded
      graph batch; the pre-seed gather clamps them to 0 (padded rows are garbage
      but never committed).
    """
    plan = build_spec_plan(forward_batch, linear_backend)
    bs = plan.bs
    T = plan.draft_token_num
    device = plan.device
    max_bs = _max_graph_bs(linear_backend, bs)
    # Eager (non-graph) verify forwards run the EXTEND path and can have a real
    # bs LARGER than the captured-graph bucket count (max_bs) -- up to the
    # scheduler's request-pool capacity. Size the scratch for whichever is larger
    # so the pre-seed / commit never index past the buffer (otherwise a bs>max_bs
    # eager verify triggers an out-of-bounds HIP illegal access). The backend
    # pre-allocates for the pool capacity, so this max() is mainly a safety net.
    scratch_bs = max(max_bs, bs)
    # int32 live slots (graph-stable view; may hold -1 padding). Mirror into a
    # local int64 buffer (captured copy -> refreshed on replay) and clamp padding
    # so the gather never indexes out of bounds.
    committed_i32 = plan.committed_indices
    committed_long = torch.empty(bs, dtype=torch.long, device=device)
    committed_long.copy_(committed_i32)
    gather_idx = committed_long.clamp(min=0)
    scratch_rows = (torch.arange(bs, device=device) * T).to(dtype=torch.long)
    # These index tensors drive the captured pre-seed scatter; pin them so a
    # later (smaller-bs) capture cannot reuse their memory and clobber the
    # first-captured (max-bs) verify graph.
    keepalive_if_capturing(committed_long, gather_idx, scratch_rows)

    shared = getattr(pool, "_k3_conv_scratch", None)
    if shared is None:
        shared = {}
        pool._k3_conv_scratch = shared

    out: dict[str, KVCacheTensor] = {}
    km1 = 0
    for layer_id in mamba_map:
        layer_cache = pool.mamba2_layer_cache(layer_id)
        conv0 = layer_cache.conv[0]  # [num_slots, K-1, conv_dim]
        temporal = layer_cache.temporal  # [num_slots, HV, V, K]
        intermediate_ssm = getattr(layer_cache, "intermediate_ssm", None)
        if intermediate_ssm is None:
            raise RuntimeError(
                "Kimi-K3 target-verify requires MambaPool.SpeculativeState "
                "(intermediate_ssm); none found."
            )

        km1 = int(conv0.shape[-2])
        conv_dim = int(conv0.shape[-1])
        wide = km1 + plan.num_spec  # (K-1) + num_spec rolling window

        # One shared scratch per layer sized for the largest verify bs (graph
        # bucket max or eager pool capacity), so its address stays stable across
        # every (including padded) bs graph and covers eager bs>max_bs verifies.
        scratch_full = shared.get(layer_id)
        if (
            scratch_full is None
            or scratch_full.shape[0] < scratch_bs * T
            or scratch_full.shape[1] != wide
            or scratch_full.shape[2] != conv_dim
        ):
            scratch_full = torch.zeros(
                (scratch_bs * T, wide, conv_dim), dtype=conv0.dtype, device=conv0.device
            )
            shared[layer_id] = scratch_full
            # This layer no longer aliases the stacked buffer, so the fused
            # all-layer commit is no longer valid -- drop back to the loop.
            setattr(pool, _POOL_STACKED_ATTR, None)
        scratch = scratch_full[: bs * T]
        scratch.zero_()
        # Pre-seed the resume window (columns [0:K-1]) from the committed conv.
        scratch[scratch_rows, :km1, :] = conv0[gather_idx]

        # ssm scratch: flatten [spec+1, T, ...] -> [(spec+1)*T, ...] so that
        # spec_state_indices[i, t] = i*T + t addresses intermediate_ssm[i, t].
        ssm_view = intermediate_ssm.flatten(0, 1)
        # Pre-seed the resume state (column 0) from the committed temporal.
        intermediate_ssm[:bs, 0] = temporal[gather_idx]

        layer_name = f"layer_{layer_id}"
        out[layer_name] = KVCacheTensor(
            layer_num=layer_id,
            k_cache=scratch,
            v_cache=ssm_view,
            k_scale=None,
            v_scale=None,
        )

    # Store only bs-independent metadata (verify-active marker + shared dims). The
    # commit fetches live real-batch indices itself and reads the shared scratch.
    setattr(
        pool,
        _POOL_SPEC_ATTR,
        {
            "T": T,
            "num_spec": plan.num_spec,
            "km1": km1,
            "mamba_map": list(mamba_map),
        },
    )
    return out


def _commit_conv_windows_fused(
    pool: Any,
    spec: dict,
    committed_i32: torch.Tensor,
    last_correct_step_indices: torch.Tensor,
    bs: int,
) -> bool:
    """Commit every KDA conv window with a SINGLE fused triton kernel.

    ``fused_conv_window_scatter_with_mask`` was written upstream for exactly this
    shape of problem: it reads the source through explicit strides (so an
    *overlapping* rolling-window view is fine), applies the ``step < 0`` mask
    inside the kernel, and covers all layers in one launch
    (``grid = (requests, layers, blocks)``).

    That removes both costs of the legacy path: the ~69-iteration Python loop and
    the per-layer ``aten::nonzero`` device sync that boolean-mask indexing lowers
    to.

    Returns ``False`` when the fused path cannot be used, so the caller can fall
    back to :func:`_commit_conv_windows_legacy`.
    """
    stacked = getattr(pool, _POOL_STACKED_ATTR, None)
    if stacked is None:
        return False
    try:
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            fused_conv_window_scatter_with_mask,
        )
    except ImportError:
        return False

    mamba_caches = pool.get_speculative_mamba2_params_all_layers()
    conv_all = getattr(mamba_caches, "conv", None)
    if not conv_all:
        return False
    conv_states = conv_all[0]  # [num_layers, num_slots, K-1, conv_dim]

    T = int(spec["T"])
    km1 = int(spec["km1"])
    num_layers, rows, _, conv_dim = stacked.shape
    if (
        conv_states.ndim != 4
        or conv_states.shape[0] != num_layers
        or tuple(conv_states.shape[2:]) != (km1, conv_dim)
        or not conv_states.is_contiguous()
        or bs * T > rows
    ):
        return False

    # src[l, i, t, k, :] == stacked[l, i * T, t + k, :]
    #
    # The step axis and the in-window axis deliberately SHARE a stride: that
    # overlap *is* the wide rolling window -- step ``t``'s committed window is
    # ``stacked[l, i * T, t : t + km1]``. Read-only, so aliasing is safe.
    step_stride = stacked.stride(2)
    src = stacked.as_strided(
        (num_layers, bs, T, km1, conv_dim),
        (
            stacked.stride(0),
            T * stacked.stride(1),
            step_stride,
            step_stride,
            stacked.stride(3),
        ),
    )
    fused_conv_window_scatter_with_mask(
        conv_states, src, committed_i32, last_correct_step_indices[:bs]
    )
    return True


def _commit_conv_windows_legacy(
    pool: Any,
    spec: dict,
    shared: dict,
    committed: torch.Tensor,
    step: torch.Tensor,
    bs: int,
) -> None:
    """Original per-layer conv commit, kept for A/B comparison.

    Selected with ``ATOM_K3_FUSED_CONV_COMMIT=0`` (or automatically when the
    stacked scratch is unavailable).

    Known cost: ``window[valid]`` is boolean-mask indexing, which PyTorch lowers
    to ``aten::nonzero``. ``nonzero`` must read the hit count back to the host,
    so this synchronises the device once per KDA layer (~69 syncs per decode
    step) even though ``valid`` is the same tiny [bs] mask every time.
    """
    km1 = int(spec["km1"])
    T = int(spec["T"])
    valid = step >= 0
    if not bool(valid.any()):
        return
    device = step.device
    rows = (torch.arange(bs, device=device) * T).to(dtype=torch.long)
    win = torch.arange(km1, device=device).view(1, km1)  # [1, K-1]
    step_clamped = step.clamp(min=0)
    wide_idx = step_clamped.view(bs, 1) + win  # [bs, K-1]
    valid_rows = committed[valid]
    for layer_id in spec["mamba_map"]:
        scratch = shared.get(layer_id)
        if scratch is None:
            continue
        layer_cache = pool.mamba2_layer_cache(layer_id)
        conv0 = layer_cache.conv[0]
        # Gather the accepted narrow window from the front rows: [bs, K-1, conv_dim].
        window = scratch[rows.view(bs, 1), wide_idx, :]
        conv0[valid_rows] = window[valid].to(conv0.dtype)


_CHECK_STATE = {"steps": 0, "mismatches": 0}


def _check_conv_commit_paths(
    pool: Any,
    spec: dict,
    shared: dict,
    committed_i32: torch.Tensor,
    last_correct_step_indices: torch.Tensor,
    bs: int,
) -> None:
    """Run BOTH conv commit paths on the live tensors and compare (debug mode).

    Enabled with ``ATOM_K3_FUSED_CONV_COMMIT=check``. Unlike an offline test this
    exercises the real pool layout, real acceptance patterns and padded CUDA-graph
    batches. The legacy result is what stays in the pool, so the run itself is
    unaffected by the fused path.
    """
    conv_states = pool.get_speculative_mamba2_params_all_layers().conv[0]
    before = conv_states.clone()
    used_fused = _commit_conv_windows_fused(
        pool, spec, committed_i32, last_correct_step_indices, bs
    )
    fused_out = conv_states.clone()
    conv_states.copy_(before)
    _commit_conv_windows_legacy(
        pool,
        spec,
        shared,
        committed_i32.to(dtype=torch.long),
        last_correct_step_indices[:bs].to(dtype=torch.long),
        bs,
    )
    _CHECK_STATE["steps"] += 1
    if used_fused and not torch.equal(fused_out, conv_states):
        _CHECK_STATE["mismatches"] += 1
        logger.error(
            "Kimi-K3 conv commit MISMATCH at check step %d (%d total), bs=%d, "
            "accept steps=%s",
            _CHECK_STATE["steps"],
            _CHECK_STATE["mismatches"],
            bs,
            last_correct_step_indices[:bs].tolist(),
        )
    elif _CHECK_STATE["steps"] % 20 == 0:
        logger.info(
            "Kimi-K3 conv commit check: %d steps, %d mismatches (fused_used=%s)",
            _CHECK_STATE["steps"],
            _CHECK_STATE["mismatches"],
            used_fused,
        )


def _commit_k3_spec_state(
    pool: Any,
    committed_i32: torch.Tensor,
    last_correct_step_indices: torch.Tensor,
) -> None:
    """Commit accepted verify state: ssm via sglang's fused scatter (reading the
    ATOM-populated ``intermediate_ssm``), conv via the accepted narrow window of
    the wide scratch.

    ``committed_i32`` are the *live* real-request mamba slots for this step
    (fetched from the pool via the verify batch's ``req_pool_indices``), so this
    is correct under a padded CUDA-graph batch too: real requests occupy the front
    rows of the shared scratch / intermediate_ssm, and padded lanes are never
    referenced here.
    """
    spec = getattr(pool, _POOL_SPEC_ATTR, None)
    if spec is None:
        return
    bs = int(committed_i32.shape[0])
    if bs == 0:
        return
    shared = getattr(pool, "_k3_conv_scratch", None)
    if shared is None:
        return

    # ---- ssm commit: intermediate_ssm[i, step] -> temporal[cache_idx[i]] ----
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_mamba_state_scatter_with_mask,
    )

    mamba_caches = pool.get_speculative_mamba2_params_all_layers()
    if mamba_caches.temporal.numel() > 0:
        fused_mamba_state_scatter_with_mask(
            mamba_caches.temporal,
            mamba_caches.intermediate_ssm,
            committed_i32,
            last_correct_step_indices[:bs],
        )

    # ---- conv commit: committed window = scratch[i*T, step:step+(K-1)] --------
    mode = _conv_commit_mode()
    if mode == "check":
        _check_conv_commit_paths(
            pool, spec, shared, committed_i32, last_correct_step_indices, bs
        )
        return
    if mode == "fused" and _commit_conv_windows_fused(
        pool, spec, committed_i32, last_correct_step_indices, bs
    ):
        return
    _commit_conv_windows_legacy(
        pool,
        spec,
        shared,
        committed_i32.to(dtype=torch.long),
        last_correct_step_indices[:bs].to(dtype=torch.long),
        bs,
    )


# ---------------------------------------------------------------------------
# Hook installation: intercept the DSpark post-verify commit for K3.
#
# The DSpark worker (sglang dspark_components) does NOT call the generic
# ``update_mamba_state_after_mtp_verify``; it commits accepted tokens through
# ``TargetVerifyExecutor.commit_hidden``. For hybrid KDA targets the linear
# (mamba) state must be advanced there too, so we piggy-back the KDA commit on
# that call. ``commit_lens = correct_len + 1`` (accepted drafts + bonus), so the
# per-request last accepted verify step is ``commit_lens - 1``.
# ---------------------------------------------------------------------------
_HOOK_INSTALLED = False


def install_k3_verify_commit_hook() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    try:
        from sglang.srt.speculative.dspark_components.dspark_verify import (
            TargetVerifyExecutor,
        )
    except Exception:  # noqa: BLE001 - DSpark components optional
        return

    original = TargetVerifyExecutor.commit_hidden

    def _patched_commit_hidden(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        commit_lens = kwargs.get("commit_lens")
        bs = kwargs.get("bs")
        batch = kwargs.get("batch")
        try:
            pool = getattr(self.model_runner, "req_to_token_pool", None)
            if (
                pool is not None
                and commit_lens is not None
                and batch is not None
                and getattr(pool, _POOL_SPEC_ATTR, None) is not None
            ):
                if bs is None:
                    bs = int(commit_lens.shape[0])
                last_correct_step = commit_lens[:bs].to(torch.int64) - 1
                # Live real-request mamba slots for this step (front rows of any
                # padded graph batch). Resolve exactly like the linear backend so
                # the physical ids match temporal/conv indexing.
                req_pool_indices = batch.req_pool_indices[:bs]
                committed_i32 = pool.get_mamba_indices(req_pool_indices)
                translate = getattr(pool, "translate_mamba_indices", None)
                if translate is not None:
                    committed_i32 = translate(committed_i32)
                committed_i32 = committed_i32[:bs].to(torch.int32)
                _commit_k3_spec_state(pool, committed_i32, last_correct_step)
                # NB: do NOT clear pool._atom_k3_spec here. Under CUDA graph the
                # spec cache is (re)built only once at capture -- replay skips the
                # Python forward -- so clearing it would make every step after the
                # first skip the KDA commit, freezing the recurrent state (drift /
                # degenerate repetition). The cache now references persistent /
                # auto-refreshed buffers (conv scratch + the live int32 mamba-index
                # view), so it stays valid across replays. commit_hidden only fires
                # after a verify, so committing every time is correct.
        except Exception:
            logger.exception("Kimi-K3 DSpark KDA commit failed")
            raise
        return result

    TargetVerifyExecutor.commit_hidden = _patched_commit_hidden
    _HOOK_INSTALLED = True
    logger.info("Kimi-K3: installed DSpark KDA target-verify commit hook.")
