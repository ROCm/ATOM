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
  - torch.ops.aiter.v4_shared_gate_up_silu_quant — V4 shared expert gate_up
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


def tbo_all_reduce(x: torch.Tensor) -> torch.Tensor:
    from aiter.dist.communication_op import tensor_model_parallel_all_reduce

    from atom.utils.tbo.ubatching import tbo_active

    if not tbo_active():
        return tensor_model_parallel_all_reduce(x)

    # "inline": the plain all-reduce on the compute stream (no overlap).
    # "overlap" (default): on the comm stream, overlapping the partner ubatch.
    if envs.ATOM_TBO_TP_AR_MODE != "overlap":
        return tensor_model_parallel_all_reduce(x)

    from atom.utils.tbo.ubatching import (
        tbo_current_ubatch_id,
        tbo_get_comm_stream,
        tbo_get_ubatch_tp_comm,
        tbo_on_comm,
    )

    ub_comm = tbo_get_ubatch_tp_comm(tbo_current_ubatch_id())
    if ub_comm is None:
        # world_size == 1: nothing to reduce.
        return x

    # Out-of-place all_reduce on this ubatch's dedicated communicator, on the
    # comm stream, overlapping the partner ubatch's compute. Not held: see
    # `tbo_on_comm`.
    return tbo_on_comm(
        lambda: ub_comm.all_reduce(x, stream=tbo_get_comm_stream()), hold=False
    )


def _tbo_all_reduce_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="tbo_all_reduce",
    op_func=tbo_all_reduce,
    mutates_args=(),
    fake_impl=_tbo_all_reduce_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


# ---------------------------------------------------------------------------
# V4 shared expert: gate_up + SwiGLU + w2's MXFP8 input quant (gfx1250)
# ---------------------------------------------------------------------------
#
# Caller contract (the `Expert` looked up by `layer_name`):
#   - `gate_up_silu_quant(x) -> (act e4m3 [m, inter], act_scale E8M0
#     [m, inter / 32])`
#   - the op also takes `gate_up_proj.weight` ([2 * inter, dim]), which only
#     sets the output shape for tracing
#
# The op body runs eagerly with the real row count, so the fused-or-unfused
# choice and the GEMM tile follow every call's m; a compiled forward replayed
# for other row counts without guard checks keeps both right. Functional
# (`mutates_args=()`): it allocates and returns both outputs, since an E8M0
# tensor written through a mutated argument leaves an auto_functionalized_v2
# Inductor does not remove.


def v4_shared_gate_up_silu_quant(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # `gate_up_weight` only gives the fake its output width.
    self = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    return self.gate_up_silu_quant(x)


def _v4_shared_gate_up_silu_quant_fake(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    inter = gate_up_weight.shape[0] // 2
    return (
        torch.empty((x.shape[0], inter), dtype=torch.float8_e4m3fn, device=x.device),
        torch.empty(
            (x.shape[0], inter // 32), dtype=torch.float8_e8m0fnu, device=x.device
        ),
    )


direct_register_custom_op(
    op_name="v4_shared_gate_up_silu_quant",
    op_func=v4_shared_gate_up_silu_quant,
    mutates_args=(),
    fake_impl=_v4_shared_gate_up_silu_quant_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
