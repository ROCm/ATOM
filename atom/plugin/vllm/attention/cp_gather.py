"""Reuse-TP-as-CP runtime attention weight gather (RFC ROCm/ATOM#196, weight-gather
revision).

Under reuse-TP-as-CP the attention projection weights (``q_b``/``kv_b``/``o_proj``
and the absorbed BMM weights ``W_K``/``W_V``) stay TP-sharded in memory -- we do NOT
keep a full-head replicated copy per layer. For prefill / mixed batches (which run
full-head token-parallel attention on this rank's 1/cp query shard) the FULL weights
for a layer are gathered on demand right before that layer's attention, used, then
released. This module owns:

  * the per-layer weight gather itself (``gather_attn_weights``), used both to
    prefetch into a slot and as the synchronous fallback;
  * the OVERLAP pipeline (intrinsic to CP -- no separate flag): a dedicated RCCL
    communicator + a background CUDA stream + an ordered CP-attn layer registry + a
    2-slot double buffer, so layer L+1's gather runs on the side stream concurrently
    with layer L's MoE compute. A dedicated communicator is REQUIRED: the MoE's
    ``cp_ffn`` all-gather / reduce-scatter runs on the TP/CP communicator, and two
    concurrent collectives on one communicator can deadlock. If the dedicated
    communicator cannot be built (e.g. single rank) the code falls back to a plain
    synchronous gather on the TP coordinator.

Gather composability (why this is bit-exact):
  * ``q_b``/``o_proj`` use per-block / per-channel weight scales (per_1x128 /
    per_1x32 / per_token) that are shard-local along ``tp_dim`` -- gathering the
    quantized bytes together with those scales reconstructs the full weight
    (:meth:`LinearBase.gather_full_weight_scale`).
  * fp8 ``W_K``/``W_V`` use a per-(whole-)tensor SCALAR scale; CP layers therefore
    quantize them at load with a SHARED cross-rank scalar (see
    ``dynamic_per_batched_tensor_quant(..., cross_rank_group=...)``) so the per-rank
    fp8 shards tile bit-exactly after an all-gather and the scalar needs no gather.
  * fp4 (mxfp4) ``W_K``/``W_V`` use per-1x32 block scales that ARE shard-local, so
    both bytes and block scale are gathered along the head dim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

__all__ = [
    "GatheredAttnWeights",
    "register_cp_attn_layer",
    "reset_cp_gather_pipeline",
    "get_gathered_weights",
    "prefetch_next_layer",
    "release_gathered_weights",
]


@dataclass
class GatheredAttnWeights:
    """Transient full-head attention weights for ONE layer, valid for one forward."""

    qb_weight: torch.Tensor
    qb_scale: Optional[torch.Tensor]
    qb_out_size: int
    wk: torch.Tensor
    wk_scale: torch.Tensor
    wv: torch.Tensor
    wv_scale: torch.Tensor
    o_weight: torch.Tensor
    o_scale: Optional[torch.Tensor]
    o_out_size: int
    num_heads_full: int
    # Overlap bookkeeping (unused on the synchronous path).
    ready_event: Optional[torch.cuda.Event] = None


# --------------------------------------------------------------------------- #
# Ordered CP-attn layer registry (execution order == construction order).
# --------------------------------------------------------------------------- #
_CP_ATTN_LAYERS: List[object] = []
_CP_ATTN_INDEX: dict = {}


def register_cp_attn_layer(op) -> int:
    """Register a CP attention op in execution order; returns its index. MTP / non-CP
    ops never call this (they run the plain sharded path)."""
    if op in _CP_ATTN_INDEX:
        return _CP_ATTN_INDEX[op]
    idx = len(_CP_ATTN_LAYERS)
    _CP_ATTN_LAYERS.append(op)
    _CP_ATTN_INDEX[op] = idx
    return idx


def _next_cp_layer(op):
    idx = _CP_ATTN_INDEX.get(op)
    if idx is None or idx + 1 >= len(_CP_ATTN_LAYERS):
        return None
    return _CP_ATTN_LAYERS[idx + 1]


# --------------------------------------------------------------------------- #
# Dedicated gather communicator + stream (overlap only).
# --------------------------------------------------------------------------- #
class _RawGroup:
    """Minimal ``all_gather(tensor, dim=...)`` shim over a raw process group so the
    dedicated gather communicator can be consumed by
    :meth:`LinearBase.gather_full_weight_scale` exactly like a vLLM GroupCoordinator.
    """

    def __init__(self, pg, world_size: int):
        self._pg = pg
        self.world_size = world_size

    def all_gather(self, t: torch.Tensor, dim: int = 0) -> torch.Tensor:
        t = t.contiguous()
        chunks = [torch.empty_like(t) for _ in range(self.world_size)]
        torch.distributed.all_gather(chunks, t, group=self._pg)
        return torch.cat(chunks, dim=dim)


_gather_group: Optional[_RawGroup] = None
_gather_stream: Optional[torch.cuda.Stream] = None
_gather_group_inited = False


def _get_gather_group() -> Optional[_RawGroup]:
    """Dedicated communicator over the TP ranks (separate from the TP/CP comm used by
    the MoE collectives, so concurrent collectives cannot deadlock)."""
    global _gather_group, _gather_group_inited
    if _gather_group_inited:
        return _gather_group
    _gather_group_inited = True
    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tp_group()
    if tp.world_size <= 1:
        _gather_group = None
        return None
    # new_group is collective across the WHOLE world; every rank must build the same
    # subgroup for its TP ranks. Fall back to the TP comm if construction fails.
    try:
        pg = torch.distributed.new_group(ranks=tp.ranks, backend="nccl")
        _gather_group = _RawGroup(pg, tp.world_size)
    except Exception:  # pragma: no cover - depends on runtime topology
        _gather_group = None
    return _gather_group


def _get_gather_stream() -> Optional[torch.cuda.Stream]:
    global _gather_stream
    if _gather_stream is None:
        _gather_stream = torch.cuda.Stream()
    return _gather_stream


# --------------------------------------------------------------------------- #
# The gather itself.
# --------------------------------------------------------------------------- #
def _gather_wk_wv(op, group):
    """Gather the absorbed BMM weights to full heads along the head dim (dim 0).

    fp8 path: only the fp8 bytes are gathered (scale is a shared global scalar).
    fp4 path: bytes AND per-1x32 block scale are gathered (block scale is
    shard-local).
    """
    hd = getattr(op, "_cp_wk_wv_head_dim", 0)

    def _ag_bytes(t):
        # RCCL can't all_gather sub-byte / fp8 dtypes directly; move raw bytes.
        return group.all_gather(t.view(torch.uint8), dim=hd).view(t.dtype)

    if getattr(op, "is_aiter_triton_fp4_bmm_enabled", False):
        wk = _ag_bytes(op.W_K)
        wv = _ag_bytes(op.W_V)
        wk_s = group.all_gather(op.W_K_scale.view(torch.uint8), dim=hd).view(
            op.W_K_scale.dtype
        )
        wv_s = group.all_gather(op.W_V_scale.view(torch.uint8), dim=hd).view(
            op.W_V_scale.dtype
        )
    else:
        wk = _ag_bytes(op.W_K)
        wv = _ag_bytes(op.W_V)
        # fp8: shared global scalar (see dynamic_per_batched_tensor_quant), no gather.
        wk_s = op.W_K_scale
        wv_s = op.W_V_scale
    return wk, wk_s, wv, wv_s


def gather_attn_weights(op, group) -> GatheredAttnWeights:
    """Reconstruct this layer's FULL-head attention weights from the TP shards."""
    num_heads_full = op._cp_num_heads_full
    qb_weight, qb_scale = op.q_proj.gather_full_weight_scale(group=group)
    o_weight, o_scale = op.o_proj.gather_full_weight_scale(group=group)
    wk, wk_s, wv, wv_s = _gather_wk_wv(op, group)
    return GatheredAttnWeights(
        qb_weight=qb_weight,
        qb_scale=qb_scale,
        qb_out_size=num_heads_full * op.qk_head_dim,
        wk=wk,
        wk_scale=wk_s,
        wv=wv,
        wv_scale=wv_s,
        o_weight=o_weight,
        o_scale=o_scale,
        o_out_size=op._cp_hidden_size,
        num_heads_full=num_heads_full,
    )


# --------------------------------------------------------------------------- #
# Double-buffered overlap pipeline (2 slots, prefetch depth 1).
# --------------------------------------------------------------------------- #
_slots: List[Optional[GatheredAttnWeights]] = [None, None]
_slot_layer: List[Optional[object]] = [None, None]
_slot_ready_ev: List[Optional[torch.cuda.Event]] = [None, None]
_slot_consumed_ev: List[Optional[torch.cuda.Event]] = [None, None]


def reset_cp_gather_pipeline() -> None:
    """Called at model entry (per forward) so a stale slot from the previous forward
    never satisfies this forward's first layer."""
    _slot_layer[0] = _slot_layer[1] = None


def _slot_of(op) -> int:
    return _CP_ATTN_INDEX[op] % 2


def _gather_into_slot(op, slot: int) -> None:
    """Run this layer's gather on the background stream into ``slot`` and record its
    ready event. Waits until the slot's previous consumer has finished so the
    transient tensors are not overwritten while still in use."""
    group = _get_gather_group()
    stream = _get_gather_stream()
    if group is None or stream is None:
        return
    stream.wait_stream(torch.cuda.current_stream())
    if _slot_consumed_ev[slot] is not None:
        stream.wait_event(_slot_consumed_ev[slot])
    with torch.cuda.stream(stream):
        gw = gather_attn_weights(op, group)
        ev = torch.cuda.Event()
        ev.record(stream)
        gw.ready_event = ev
    _slots[slot] = gw
    _slot_layer[slot] = op
    _slot_ready_ev[slot] = ev


def prefetch_next_layer(op) -> None:
    """Kick the NEXT CP layer's gather on the background stream so it overlaps this
    layer's MoE compute. No-op for the last layer or when no dedicated gather
    communicator is available (that layer then gathers synchronously)."""
    if _get_gather_group() is None:
        return
    nxt = _next_cp_layer(op)
    if nxt is None:
        return
    _gather_into_slot(nxt, _slot_of(nxt))


def get_gathered_weights(op) -> GatheredAttnWeights:
    """Return this layer's full-head weights. The overlap pipeline prefetches the
    NEXT layer's gather during this layer's MoE compute; the first CP layer (and any
    pipeline miss) is gathered synchronously into its slot here. Falls back to a
    plain synchronous gather on the TP coordinator when no dedicated gather
    communicator is available."""
    if _get_gather_group() is None:
        # No dedicated communicator (e.g. single rank): plain synchronous gather on
        # the aiter TP coordinator (the same one the projections were sharded with)
        # so the all_gather concatenation exactly reverses the shard layout.
        from aiter.dist.parallel_state import get_tp_group

        return gather_attn_weights(op, get_tp_group())

    slot = _slot_of(op)
    if _slot_layer[slot] is not op:
        # Not prefetched (first layer, or pipeline miss): gather synchronously into
        # the slot on the background stream and wait.
        _gather_into_slot(op, slot)
    gw = _slots[slot]
    ev = _slot_ready_ev[slot]
    if ev is not None:
        torch.cuda.current_stream().wait_event(ev)
    # Keep the transients alive until the compute stream is done with them.
    for t in (gw.qb_weight, gw.wk, gw.wv, gw.o_weight):
        if t is not None:
            t.record_stream(torch.cuda.current_stream())
    return gw


def release_gathered_weights(op) -> None:
    """Mark this layer's slot consumed so the background stream may reuse it for a
    later layer's gather (double-buffer hand-off)."""
    if _get_gather_group() is None:
        return
    slot = _slot_of(op)
    ev = torch.cuda.Event()
    ev.record(torch.cuda.current_stream())
    _slot_consumed_ev[slot] = ev
