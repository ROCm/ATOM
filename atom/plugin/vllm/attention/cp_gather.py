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
    ``cp_ffn`` all-gather / reduce-scatter runs on the aliased TP/CP communicator,
    and two concurrent collectives on one communicator can deadlock. CP-on therefore
    fails initialization if every rank cannot create the independent gather group;
    there is no fallback to the activation communicator.

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

import atexit
import contextvars
import weakref
from dataclasses import dataclass, field
from typing import List, Optional

import torch

__all__ = [
    "GatheredAttnWeights",
    "initialize_cp_gather_pipeline",
    "close_cp_gather_pipeline",
    "close_cp_gather_owner",
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
# Per-model ordered layer registries. Owners are atom Config object identities;
# this separates target/draft models even when their layer prefixes overlap.
# --------------------------------------------------------------------------- #
_CP_ATTN_LAYERS: dict[str, List[weakref.ReferenceType]] = {}
_CP_ATTN_INDEX: dict[str, weakref.WeakKeyDictionary] = {}
_OWNER_GENERATION: dict[str, int] = {}


def stable_cp_owner_key(atom_config, model_role: str | None = None) -> str:
    """Build a rebuild-stable key from static model identity.

    ``model_role`` distinguishes target/draft instances that intentionally share a
    checkpoint and architecture. It is assigned by the vLLM wrapper before model
    construction and therefore survives same-process unload/reload and compile-cache
    reuse without relying on process-history ordinals or raw object ids.
    """
    hf = atom_config.hf_config
    arch = ",".join(getattr(hf, "architectures", None) or [type(hf).__name__])
    model = str(getattr(atom_config, "model", ""))
    role = model_role or getattr(atom_config, "_vllm_cp_model_role", "target")
    pp_rank = int(getattr(atom_config, "pipeline_parallel_rank", 0))
    key = (
        f"role={role}|pp={pp_rank}|arch={arch}|model={model}|"
        f"layers={getattr(hf, 'num_hidden_layers', 0)}"
    )
    return key


def register_cp_attn_layer(op, owner_id: str) -> int:
    """Register or replace one layer in a stable owner-scoped registry."""
    op._cp_gather_owner_id = owner_id
    index = _CP_ATTN_INDEX.setdefault(owner_id, weakref.WeakKeyDictionary())
    if op in index:
        return index[op]
    layers = _CP_ATTN_LAYERS.setdefault(owner_id, [])
    layer_num = int(getattr(op, "layer_num", len(layers)))
    if layer_num == 0 and layers:
        # Same stable owner reconstructed in-process: invalidate its old weak layer
        # registry and any context-local prefetched slots. Compiled keys now resolve
        # exclusively to the newly-constructed model modules.
        layers.clear()
        index.clear()
        _OWNER_GENERATION[owner_id] = _OWNER_GENERATION.get(owner_id, 0) + 1
        state = _forward_slots.get()
        if state is not None and state.owner_id == owner_id:
            _forward_slots.set(None)
    while len(layers) <= layer_num:
        layers.append(lambda: None)
    old = layers[layer_num]()
    if old is not None:
        index.pop(old, None)
    layers[layer_num] = weakref.ref(op)
    index[op] = layer_num
    return layer_num


def _next_cp_layer(op):
    owner_id = op._cp_gather_owner_id
    index = _CP_ATTN_INDEX.get(owner_id, {})
    idx = index.get(op)
    layers = _CP_ATTN_LAYERS.get(owner_id, [])
    if idx is None or idx + 1 >= len(layers):
        return None
    return layers[idx + 1]()


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


def initialize_cp_gather_pipeline() -> None:
    """Collectively create the independent weight-gather communicator.

    All world ranks build every TP subgroup in deterministic order and then exchange
    a success bit. CP-on is usable only when every rank succeeded; partial success is
    torn down and raised rather than silently falling back to the activation group.
    """
    global _gather_group, _gather_group_inited
    if _gather_group_inited:
        return
    from vllm.distributed.parallel_state import get_tp_group

    tp = get_tp_group()
    if tp.world_size <= 1:
        raise RuntimeError("ATOM_VLLM_ATTN_CP requires TP world size greater than one")

    local_pg = None
    local_error = None
    try:
        world_size = torch.distributed.get_world_size()
        all_tp_ranks = [None] * world_size
        torch.distributed.all_gather_object(all_tp_ranks, tuple(tp.ranks))
        unique_tp_groups = sorted({tuple(ranks) for ranks in all_tp_ranks})
        local_ranks = tuple(tp.ranks)
        for ranks in unique_tp_groups:
            pg = torch.distributed.new_group(ranks=list(ranks), backend="nccl")
            if ranks == local_ranks:
                local_pg = pg
        if local_pg is None:
            raise RuntimeError(f"TP ranks {local_ranks} missing from world topology")
    except Exception as exc:  # pragma: no cover - runtime topology dependent
        local_error = exc

    # Use the world group so a failure on one TP island is visible everywhere.
    status_device = getattr(tp, "device", torch.device("cuda"))
    status = torch.tensor(
        0 if local_error is not None else 1,
        dtype=torch.int32,
        device=status_device,
    )
    torch.distributed.all_reduce(status, op=torch.distributed.ReduceOp.MIN)
    if int(status.item()) != 1:
        if local_pg is not None:
            torch.distributed.destroy_process_group(local_pg)
        raise RuntimeError(
            "ATOM_VLLM_ATTN_CP failed to create the independent weight-gather "
            "communicator on every rank"
        ) from local_error

    _gather_group = _RawGroup(local_pg, tp.world_size)
    _gather_group_inited = True


def close_cp_gather_owner(owner_id: str) -> None:
    """Drop one model's weak layer registry without touching shared comms."""
    _CP_ATTN_LAYERS.pop(owner_id, None)
    _CP_ATTN_INDEX.pop(owner_id, None)


def close_cp_gather_pipeline() -> None:
    """Release transient state and the raw gather process group.

    Safe to call repeatedly during model teardown or interpreter shutdown. Borrowed
    TP/CP activation groups are never destroyed here.
    """
    global _gather_group, _gather_group_inited, _gather_stream
    if _gather_stream is not None:
        _gather_stream.synchronize()
    pg = getattr(_gather_group, "_pg", None)
    if pg is not None and torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group(pg)
        except Exception:  # pragma: no cover - shutdown ordering dependent
            pass
    _gather_group = None
    _gather_group_inited = False
    _gather_stream = None
    _CP_ATTN_LAYERS.clear()
    _CP_ATTN_INDEX.clear()
    _forward_slots.set(None)


def _get_gather_group() -> Optional[_RawGroup]:
    """Return the communicator created during initialization; never create one from
    a forward path."""
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
# Owner + forward scoped double-buffered overlap pipeline.
# --------------------------------------------------------------------------- #
@dataclass
class _ForwardSlots:
    owner_id: str
    slots: List[Optional[GatheredAttnWeights]] = field(
        default_factory=lambda: [None, None]
    )
    layers: List[Optional[object]] = field(default_factory=lambda: [None, None])
    ready: List[Optional[torch.cuda.Event]] = field(default_factory=lambda: [None, None])
    consumed: List[Optional[torch.cuda.Event]] = field(
        default_factory=lambda: [None, None]
    )


_forward_slots: contextvars.ContextVar[Optional[_ForwardSlots]] = (
    contextvars.ContextVar("atom_cp_forward_slots", default=None)
)


def reset_cp_gather_pipeline(owner_id: str) -> None:
    """Create isolated double buffers for the current GPU ubatch/forward."""
    stream = _gather_stream
    if stream is not None:
        torch.cuda.current_stream().wait_stream(stream)
    _forward_slots.set(_ForwardSlots(owner_id=owner_id))


def _state_for(op) -> _ForwardSlots:
    state = _forward_slots.get()
    owner_id = op._cp_gather_owner_id
    if state is None or state.owner_id != owner_id:
        state = _ForwardSlots(owner_id=owner_id)
        _forward_slots.set(state)
    return state


def _slot_of(op) -> int:
    return _CP_ATTN_INDEX[op._cp_gather_owner_id][op] % 2


def _gather_into_slot(op, slot: int, state: _ForwardSlots) -> None:
    """Run this layer's gather on the background stream into ``slot`` and record its
    ready event. Waits until the slot's previous consumer has finished so the
    transient tensors are not overwritten while still in use."""
    group = _get_gather_group()
    stream = _get_gather_stream()
    if group is None or stream is None:
        return
    stream.wait_stream(torch.cuda.current_stream())
    if state.consumed[slot] is not None:
        stream.wait_event(state.consumed[slot])
    with torch.cuda.stream(stream):
        gw = gather_attn_weights(op, group)
        ev = torch.cuda.Event()
        ev.record(stream)
        gw.ready_event = ev
    state.slots[slot] = gw
    state.layers[slot] = op
    state.ready[slot] = ev


def prefetch_next_layer(op) -> None:
    """Gather the next owner-scoped layer on the side stream.

    Context-local slot state prevents concurrent GPU ubatches from resetting or
    consuming one another's buffers, while the dedicated stream/communicator keeps
    collective order identical on every rank.
    """
    if _get_gather_group() is None:
        return
    nxt = _next_cp_layer(op)
    if nxt is None:
        return
    state = _state_for(op)
    _gather_into_slot(nxt, _slot_of(nxt), state)


def get_gathered_weights(op) -> GatheredAttnWeights:
    """Consume prefetched weights, synchronously filling a pipeline miss."""
    if _get_gather_group() is None:
        raise RuntimeError(
            "CP weight gather used before its independent communicator was "
            "successfully initialized"
        )
    state = _state_for(op)
    slot = _slot_of(op)
    if state.layers[slot] is not op:
        _gather_into_slot(op, slot, state)
    gw = state.slots[slot]
    if gw is None:
        raise RuntimeError("CP gather slot was not populated")
    ready = state.ready[slot]
    if ready is not None:
        torch.cuda.current_stream().wait_event(ready)
    for t in (
        gw.qb_weight,
        gw.qb_scale,
        gw.wk,
        gw.wk_scale,
        gw.wv,
        gw.wv_scale,
        gw.o_weight,
        gw.o_scale,
    ):
        if isinstance(t, torch.Tensor):
            t.record_stream(torch.cuda.current_stream())
    return gw


def release_gathered_weights(op) -> None:
    """Mark the context-local slot reusable after compute consumes the weights."""
    if _get_gather_group() is None:
        return
    state = _state_for(op)
    slot = _slot_of(op)
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream())
    state.consumed[slot] = event


atexit.register(close_cp_gather_pipeline)
