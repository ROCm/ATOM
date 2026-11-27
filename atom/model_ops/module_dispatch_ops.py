# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Custom ops dispatching to module methods via `static_forward_context`.

Each op below takes a `layer_name: str` and looks up the owning module from
`compilation_config.static_forward_context[layer_name]`. The module's
internal state (sub-modules, persistent buffers, per-fwd metadata fetched
through `get_forward_context()`) stays out of the op's argument list, which:

  - hides dynamic-shape internals from Dynamo / fake-tensor mode (no graph
    break, no per-shape recompile)
  - bypasses functionalization for buffer mutations — the buffers aren't in
    the op's signature, so the pass needn't insert defensive clones
  - leaves CUDAGraph capture transparent — the op still launches the same
    CUDA work on the stream; only the Python dispatch is opaque to compile

Caller contract (per op): the module registered at `layer_name` must expose
the methods listed in each op's docstring.

Currently registered:
  - torch.ops.aiter.maybe_dual_stream_forward  — V2/V3.2/V4 MoE (summed)
  - torch.ops.aiter.maybe_dual_stream_split_forward — K3 MoE (routed/shared unsummed)
  - torch.ops.aiter.indexer_score_topk         — V4 sparse indexer

and, taking no `layer_name`, the TBO-aware TP collectives that
`communication_op` routes pure TP + TBO through:
  - torch.ops.aiter.tbo_all_reduce
  - torch.ops.aiter.tbo_fused_allreduce_rmsnorm[_quant]
"""

import torch

from atom.config import get_current_atom_config
from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.forward_context import (
    get_forward_context,
    graph_records_single_stream,
    plugin_graph_is_piecewise,
)

# ---------------------------------------------------------------------------
# Dual-stream MoE dispatch (V2 / V3.2 / V4 / K3)
# ---------------------------------------------------------------------------
#
# Caller contract (the MoE module looked up by `layer_name`):
#   - `_use_dual_stream: bool`
#   - `_dual_stream_token_threshold: int` -- the largest row count that forks
#     (`envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD`)
#   - `_dual_stream_decode_token_threshold: int` -- the same on decode steps
#     (equal to the above unless the model sets its own decode default)
#   - `single_stream_moe_forward(hidden_states) -> Tensor`
#   - `dual_stream_moe_forward(hidden_states) -> Tensor`
#
# Per-call row gating: a call with more rows than its step's bound
# (`_dual_stream_row_bound`) stays single-stream.


def _dual_stream_row_bound(self) -> int:
    """The row bound for this call: the decode bound on a decode step, the
    generic one on a prefill step or where no step context is set."""
    context = get_forward_context().context
    if context is not None and not context.is_prefill:
        return self._dual_stream_decode_token_threshold
    return self._dual_stream_token_threshold


def _dual_stream_is_active(self, num_tokens: int) -> bool:
    """Whether this call should fork the shared branch onto ``alt_stream``.

    Shared by the single-tensor op and the split-return op below so the two
    can never drift into disagreeing about when dual-stream engages.
    """
    # Under TBO the two micro-batches already overlap on separate threads
    from atom.utils.tbo.ubatching import tbo_active

    # Eager, FULL capture and a token-graph segment all hold the fork/join; a
    # plugin frontend's own PIECEWISE graphs are left single-stream, and so is
    # a graph window ATOM_GRAPH_SIDE_STREAMS keeps on one stream, its warmup
    # forward included (`graph_records_single_stream`).
    return (
        self._use_dual_stream
        and 0 < num_tokens <= _dual_stream_row_bound(self)
        and not tbo_active()
        and not plugin_graph_is_piecewise()
        and not graph_records_single_stream()
    )


def maybe_dual_stream_forward(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    self = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    if _dual_stream_is_active(self, hidden_states.shape[0]):
        return self.dual_stream_moe_forward(hidden_states)
    return self.single_stream_moe_forward(hidden_states)


def _maybe_dual_stream_forward_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="maybe_dual_stream_forward",
    op_func=maybe_dual_stream_forward,
    # Op returns a fresh tensor; never writes into `hidden_states`. Declaring
    # `mutates_args=["hidden_states"]` (the V2 original) misleads the
    # functionalization pass into inserting defensive input clones.
    mutates_args=(),
    fake_impl=_maybe_dual_stream_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


# ---------------------------------------------------------------------------
# Dual-stream MoE dispatch, routed/shared returned UNSUMMED (K3)
# ---------------------------------------------------------------------------
#
# Caller contract (the MoE module looked up by `layer_name`):
#   - `_use_dual_stream: bool`
#   - `_dual_stream_token_threshold: int`,
#     `_dual_stream_decode_token_threshold: int` (as above)
#   - `single_stream_split_moe_forward(hidden_states) -> (Tensor, Tensor)`
#   - `dual_stream_split_moe_forward(hidden_states) -> (Tensor, Tensor)`
#
# Same gating as `maybe_dual_stream_forward`; the only difference is the return.
# The op above must sum internally because torch's schema inference has no
# representation for an optional tensor in a tuple return -- so a model that can
# legally defer the routed + shared add across the layer boundary (K3, whose
# next attn_res folds both addends into its on-load) cannot use it without
# paying the [T, H] elementwise kernel it is trying to skip.
#
# This op returns both branches instead. Both returns are always real tensors,
# which the schema does allow; the caller decides whether deferring is legal
# from its own static config, NOT from anything about these tensors. A model
# whose branches must be summed before a collective keeps using the op above.
#
# Deliberately a separate op rather than a widened signature on
# `maybe_dual_stream_forward`: that one is shared with DeepSeek V2/V3.2/V4,
# none of which defer the add, and all of which live in @support_torch_compile
# files.


def maybe_dual_stream_split_forward(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    self = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    if _dual_stream_is_active(self, hidden_states.shape[0]):
        return self.dual_stream_split_moe_forward(hidden_states)
    return self.single_stream_split_moe_forward(hidden_states)


def _maybe_dual_stream_split_forward_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(hidden_states), torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="maybe_dual_stream_split_forward",
    op_func=maybe_dual_stream_split_forward,
    mutates_args=(),
    fake_impl=_maybe_dual_stream_split_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


# ---------------------------------------------------------------------------
# Sparse indexer score + top-k (V2/V3.2/V4)
# ---------------------------------------------------------------------------
#
# Caller contract (the Indexer module looked up by `layer_name`):
#   - `indexer_score_topk(q_quant, weights, q_scale, topk) -> Tensor`  — real impl,
#     (q_quant is FP8, or packed FP4 uint8 when the FP4 indexer is on; q_scale is
#     the paired e8m0 Q scale on the FP4 path, None for FP8)
#     must return `[total_tokens, topk] int32` indices. Under a striped row
#     split (`attn_metadata.index_rows.split.stripe`, V4 TP prefill) the rows
#     are rank-major and the view keeps every slot behind it: read them
#     through the split (`csa_translate_pack(..., stripe=)`), never as row i.
#
# `topk` is on the op signature (not derived from the module) so the fake
# impl can size the output without any module lookup.
#
# Other inputs (block_tables, KV cache, per-fwd metadata) are read by the
# module from `self` or `get_forward_context().attn_metadata`.
#
# Why opaque (same rationale for V2 and V4):
#   - prefill paths allocate scratch tensors with shapes that depend on
#     per-fwd `total_committed` / `total_kv` — Dynamo's fake-tensor pass
#     can't size them without a graph break.
#   - decode paths write into module-owned buffers; keeping the buffers
#     out of the op signature avoids functionalization clones.


def indexer_score_topk(
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    q_scale: torch.Tensor | None,
    layer_name: str,
    topk: int,
) -> torch.Tensor:
    indexer = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    return indexer.indexer_score_topk(q_quant, weights, q_scale, topk)


def _indexer_score_topk_fake(
    q_quant: torch.Tensor,
    weights: torch.Tensor,
    q_scale: torch.Tensor | None,
    layer_name: str,
    topk: int,
) -> torch.Tensor:
    return torch.empty(
        (q_quant.shape[0], topk),
        dtype=torch.int32,
        device=q_quant.device,
    )


direct_register_custom_op(
    op_name="indexer_score_topk",
    op_func=indexer_score_topk,
    # Output is a fresh tensor (per module contract). Internal buffer
    # mutations on the module are looked up via `layer_name`, not passed
    # in, so functionalization stays unaware and skips defensive clones.
    mutates_args=(),
    fake_impl=_indexer_score_topk_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


# ---------------------------------------------------------------------------
# TBO-aware TP collectives (pure TP + TBO)
# ---------------------------------------------------------------------------
#
# Reached through `communication_op` on the pure-TP+TBO path. Whether a step
# is split is known only inside the micro-batch threads, so each op decides
# that at run time: in a micro-batch of `ATOM_TBO_TP_AR_MODE=overlap` (the
# default) the all-reduce runs on the comm stream, overlapping the partner
# micro-batch's compute; otherwise -- unsplit steps, decode, "inline" -- the
# op runs aiter's collective as without TBO, on the compute stream.


def _tbo_overlaps() -> bool:
    from atom.utils.tbo.ubatching import tbo_active

    return tbo_active() and envs.ATOM_TBO_TP_AR_MODE == "overlap"


def _tbo_tp_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """`x` all-reduced on this micro-batch's TP communicator and the comm
    stream, into a tensor allocated beforehand on the compute stream.

    Allocated on the comm stream, the output would go back to that stream's
    pool once its readers drop it, and the partner's next hand-off could be
    given that memory and write it while this micro-batch still reads it: a
    hand-off waits only on compute up to the partner's yield. Memory of the
    compute stream is reused only by work ordered after the reads. The comm
    phase returns nothing, so nothing is held either (`tbo_on_comm`).

    A disabled communicator returns without writing `out`, so it is refused
    instead of handing back uninitialized memory."""
    from atom.utils.tbo.ubatching import (
        tbo_current_ubatch_id,
        tbo_get_comm_stream,
        tbo_get_ubatch_tp_comm,
        tbo_on_comm,
    )

    comm = tbo_get_ubatch_tp_comm(tbo_current_ubatch_id())
    if comm is None:
        # world_size == 1: nothing to reduce.
        return x
    if comm.disabled:
        raise RuntimeError(
            "TBO TP all-reduce: this micro-batch's TP communicator is "
            "disabled; set ATOM_TBO_TP_AR_MODE=inline"
        )
    out = torch.empty_like(x)

    def all_reduce():
        comm.all_reduce(x, out_tensor=out, stream=tbo_get_comm_stream())

    tbo_on_comm(all_reduce, hold=False)
    return out


def tbo_all_reduce(x: torch.Tensor) -> torch.Tensor:
    if _tbo_overlaps():
        return _tbo_tp_all_reduce(x)
    from aiter.dist.communication_op import tensor_model_parallel_all_reduce

    return tensor_model_parallel_all_reduce(x)


def _tbo_all_reduce_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="tbo_all_reduce",
    op_func=tbo_all_reduce,
    mutates_args=(),
    fake_impl=_tbo_all_reduce_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def tbo_fused_allreduce_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """aiter's fused TP all-reduce + residual add + RMSNorm, `(out,
    residual_out)`. An overlapping micro-batch runs the all-reduce as a
    hand-off (`_tbo_tp_all_reduce`) and the norm after it
    (`layernorm.add_rms_norm`) instead."""
    if _tbo_overlaps():
        from atom.model_ops.layernorm import add_rms_norm

        return add_rms_norm(_tbo_tp_all_reduce(x), residual, weight, eps, gemma_norm)
    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
    )

    return tensor_model_parallel_fused_allreduce_rmsnorm(
        x, residual, weight, eps, gemma_norm=gemma_norm
    )


def _tbo_fused_allreduce_rmsnorm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(x)


direct_register_custom_op(
    op_name="tbo_fused_allreduce_rmsnorm",
    op_func=tbo_fused_allreduce_rmsnorm,
    mutates_args=(),
    fake_impl=_tbo_fused_allreduce_rmsnorm_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def tbo_fused_allreduce_rmsnorm_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    quant_type: str,
    group_size: int,
    emit_bf16: bool,
    transpose_scale: bool,
    gemma_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """aiter's fused TP all-reduce + residual add + RMSNorm + FP8 quant
    (`quant_type` "per_token" or "per_group"), `(out, residual_out, scale,
    bf16)`: `bf16` the normed activation if `emit_bf16`, else empty -- the
    schema has no optional output. An overlapping micro-batch runs the
    all-reduce as a hand-off and the norm and quant after it
    (`layernorm.per_token_fp8_quant`, `layernorm.rms_norm_per_group_fp8_quant`)
    instead."""
    if _tbo_overlaps():
        from atom.model_ops.layernorm import (
            add_rms_norm,
            per_token_fp8_quant,
            rms_norm_per_group_fp8_quant,
        )

        reduced = _tbo_tp_all_reduce(x)
        if quant_type == "per_token":
            normed = add_rms_norm(reduced, residual, weight, eps, gemma_norm)
            out, scale, bf16, residual_out = per_token_fp8_quant(normed, emit_bf16)
        else:
            out, scale, bf16, residual_out = rms_norm_per_group_fp8_quant(
                reduced,
                residual,
                weight,
                eps,
                group_size=group_size,
                transpose_scale=transpose_scale,
                emit_bf16=emit_bf16,
                gemma_norm=gemma_norm,
            )
    else:
        from aiter.dist.communication_op import (
            tensor_model_parallel_fused_allreduce_rmsnorm_quant,
        )

        outs = tensor_model_parallel_fused_allreduce_rmsnorm_quant(
            x,
            residual,
            weight,
            eps,
            quant_type=quant_type,
            group_size=group_size,
            emit_bf16=emit_bf16,
            transpose_scale=transpose_scale,
            gemma_norm=gemma_norm,
        )
        out, residual_out, scale = outs[:3]
        bf16 = outs[3] if emit_bf16 else None
    return out, residual_out, scale, x.new_empty(0) if bf16 is None else bf16


def _tbo_fused_allreduce_rmsnorm_quant_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    quant_type: str,
    group_size: int,
    emit_bf16: bool,
    transpose_scale: bool,
    gemma_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.utility.dtypes import fp8

    m, n = x.shape
    if quant_type == "per_token":
        scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    elif transpose_scale:
        # column-major, stride (1, M), as aiter's fused kernel returns it
        scale = torch.empty(
            (n // group_size, m), dtype=torch.float32, device=x.device
        ).transpose(0, 1)
    else:
        scale = torch.empty((m, n // group_size), dtype=torch.float32, device=x.device)
    bf16 = torch.empty_like(x) if emit_bf16 else x.new_empty(0)
    return torch.empty_like(x, dtype=fp8), torch.empty_like(x), scale, bf16


direct_register_custom_op(
    op_name="tbo_fused_allreduce_rmsnorm_quant",
    op_func=tbo_fused_allreduce_rmsnorm_quant,
    mutates_args=(),
    fake_impl=_tbo_fused_allreduce_rmsnorm_quant_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
