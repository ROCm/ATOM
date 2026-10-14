# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CUDA-graph-safe MoE route capture indexed by physical KV slots.

Matches the vLLM ``enable_return_routed_experts`` contract: scatter logical
expert ids into ``buffer[slot, layer, :]`` during fused MoE, then gather this
step's slots back in batch token order and hand the scheduler one
``RoutedExpertsPatch`` per step. A prefix-cache hit adds the cached prefix's
slots to the first step that forwards the request.
"""

from __future__ import annotations

from collections.abc import Sequence as AbcSequence
from typing import Any, NamedTuple

import numpy as np
import torch

_INSTANCE: RoutedExpertsCapturer | None = None

# Missed captures read as this, never as a valid expert id.
UNCAPTURED = -1


class RoutedExpertsPatch(NamedTuple):
    """One step's routes for every request it forwarded.

    Segment ``j`` is request ``req_ids[j]``'s rows for token positions
    ``[starts[j], starts[j] + offsets[j+1] - offsets[j])``. Segments apply in
    order, so a prefix-hit segment precedes the same request's step segment.
    """

    req_ids: np.ndarray
    starts: np.ndarray
    offsets: np.ndarray
    rows: np.ndarray


class PendingRoutes(NamedTuple):
    """A stored step whose rows may still be copying to the host."""

    req_ids: np.ndarray
    starts: np.ndarray
    offsets: np.ndarray
    rows_cpu: torch.Tensor
    done: torch.cuda.Event | None

    def patch(self) -> RoutedExpertsPatch:
        """The step's ``RoutedExpertsPatch``, waiting on its D2H if still queued."""
        if self.done is not None:
            self.done.synchronize()
        return RoutedExpertsPatch(
            self.req_ids, self.starts, self.offsets, self.rows_cpu.numpy()
        )


def check_return_routed_experts(
    dcp_size: int,
    pcp_size: int,
    pp_size: int = 1,
    *,
    kv_transfer_config: dict | None = None,
    enable_rapidserve: bool = False,
    enable_dp_attention: bool = False,
) -> None:
    """Refuse topologies that cannot assemble a full [seq_len-1] route tensor."""
    if dcp_size != 1 or pcp_size != 1:
        raise ValueError(
            "enable_return_routed_experts requires decode_context_parallel_size "
            "== 1 and prefill_context_parallel_size == 1"
        )
    if pp_size != 1:
        raise ValueError(
            "enable_return_routed_experts requires pipeline_parallel_size == 1"
        )
    if kv_transfer_config:
        raise ValueError(
            "enable_return_routed_experts does not support KV transfer "
            "(prefill/decode disaggregation or KV offload); routed-expert "
            "metadata is not moved with KV blocks"
        )
    if enable_rapidserve:
        raise ValueError(
            "enable_return_routed_experts does not support RapidServe "
            "prefill/decode disaggregation; decode skips KV allocation and "
            "never initializes the process-local capture buffer"
        )
    if enable_dp_attention:
        raise ValueError(
            "enable_return_routed_experts does not support enable_dp_attention: "
            "select_experts runs after the DP all-gather, so captured top-k ids "
            "are aligned to the gathered token stream, not this rank's local "
            "slot_mapping; export would silently return another rank's routes"
        )


def is_fused_moe_module(module: Any) -> bool:
    """True for a real FusedMoE, including LazyMoEWrapper instances.

    ``FusedMoEDecoratorForPluginMode`` replaces ``FusedMoE`` with a subclass
    whose ``__new__`` returns an *undecorated* instance. ``isinstance(m,
    FusedMoE)`` is then False, so callers must also accept the wrapper's base.
    """
    from atom.model_ops.moe import FusedMoE

    types: tuple[type, ...] = (FusedMoE,) + tuple(
        b for b in getattr(FusedMoE, "__bases__", ()) if isinstance(b, type)
    )
    return isinstance(module, types)


def fused_moe_modules(root: Any) -> list:
    return [m for m in root.modules() if is_fused_moe_module(m)]


def prefix_slots(block_table: AbcSequence[int], lo: int, hi: int, block_size: int):
    """KV slots of token positions ``[lo, hi)`` for one request."""
    from atom.model_ops.attentions.token_layout.slots import slot_mapping

    n = int(hi) - int(lo)
    blocks = np.asarray(block_table[: (int(hi) - 1) // block_size + 1], np.int32)
    return slot_mapping(
        np.arange(lo, hi, dtype=np.int64),
        np.array([n], dtype=np.int64),
        blocks[None, :],
        block_size,
    )


def _current_slot_mapping() -> torch.Tensor | None:
    from atom.utils.forward_context import get_forward_context

    ctx = get_forward_context()
    md = getattr(ctx, "attn_metadata", None)
    if md is None:
        return None
    if isinstance(md, dict):
        for value in md.values():
            slots = getattr(value, "slot_mapping", None)
            if isinstance(slots, torch.Tensor):
                return slots
        return None
    slots = getattr(md, "slot_mapping", None)
    return slots if isinstance(slots, torch.Tensor) else None


def capture_bytes_per_kv_block(block_size: int, num_layers: int, top_k: int) -> int:
    """PAGE-pool surcharge: int32 routes for every token slot in one KV block."""
    return int(block_size) * int(num_layers) * int(top_k) * 4


def capture_pad_row_bytes(num_layers: int, top_k: int) -> int:
    """Sacrificial dummy-slot row charged once, not per KV block."""
    return int(num_layers) * int(top_k) * 4


def trim_routed_experts(routes: np.ndarray | None, num_tokens: int):
    """Keep routes for every forwarded token: ``num_tokens - 1`` rows."""
    if routes is None:
        return None
    keep = max(int(num_tokens) - 1, 0)
    if routes.shape[0] != keep:
        return routes[:keep]
    return routes


def _gather_src_indx(gather_indx) -> torch.Tensor:
    if torch.is_tensor(gather_indx):
        return gather_indx
    src = getattr(gather_indx, "src_indx", None)
    if torch.is_tensor(src):
        return src
    raise TypeError(f"unsupported gather_indx type {type(gather_indx)}")


def topk_ids_from_triton_routing(
    routing_data,
    gather_indx,
    num_tokens: int,
    topk: int,
) -> torch.Tensor:
    """Logical expert ids ``[num_tokens, topk]`` consumed by Triton fused experts.

    Packed slots are expert-major. ``gather_indx`` maps packed slot ->
    ``token * topk + k``. Invert the histogram prefix to recover expert ids.
    """
    src = _gather_src_indx(gather_indx).to(dtype=torch.long).view(-1)
    n = int(num_tokens) * int(topk)
    src = src[:n]
    packed = torch.arange(n, device=src.device, dtype=torch.long)
    offs = routing_data.expt_data.token_offs_raw.to(dtype=torch.long)
    expert_packed = torch.searchsorted(offs[1:], packed, right=True)
    tokens = torch.div(src, int(topk), rounding_mode="floor")
    slots = src % int(topk)
    ids = expert_packed.new_empty((int(num_tokens), int(topk)))
    ids[tokens, slots] = expert_packed
    return ids


class RoutedExpertsCapturer:
    """Persistent GPU buffer: ``[num_kv_slots + 1, num_layers, top_k]``.

    The extra row is a sacrificial pad target for graph dummy slots (``-1``).
    Each step's rows are gathered in batch token order by ``store_step``,
    which returns them as ``PendingRoutes`` for the caller to keep.
    """

    def __init__(
        self,
        num_slots: int,
        num_layers: int,
        top_k: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.int32,
    ):
        if num_slots <= 0 or num_layers <= 0 or top_k <= 0:
            raise ValueError(
                f"invalid capturer shape slots={num_slots} layers={num_layers} "
                f"top_k={top_k}"
            )
        self.num_slots = int(num_slots)
        self.num_layers = int(num_layers)
        self.top_k = int(top_k)
        # Last row is a sacrificial pad target. Graph dummy slots are -1;
        # mapping them to 0 and then index-putting would clobber a real write
        # to physical slot 0 in the same capture (duplicate indices).
        self._pad_slot = self.num_slots
        self.buffer = torch.full(
            (self.num_slots + 1, self.num_layers, self.top_k),
            UNCAPTURED,
            dtype=dtype,
            device=device,
        )
        # (slot key, last layer id, dest) shared by the MoE layers of one forward.
        self._dest_cache: tuple[tuple, int, torch.Tensor] | None = None

    @classmethod
    def init(
        cls,
        num_slots: int,
        num_layers: int,
        top_k: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.int32,
    ) -> RoutedExpertsCapturer:
        global _INSTANCE
        _INSTANCE = cls(num_slots, num_layers, top_k, device, dtype=dtype)
        return _INSTANCE

    @classmethod
    def get(cls) -> RoutedExpertsCapturer | None:
        return _INSTANCE

    @classmethod
    def reset(cls) -> None:
        """Drop the process-local GPU/CPU buffers. Sleep KV teardown uses this
        so wake does not size a new pool against a still-live capture tensor.
        """
        global _INSTANCE
        _INSTANCE = None

    def capture(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Scatter logical expert ids. Pad slots (``-1``) keep prior values."""
        if layer_id < 0 or layer_id >= self.num_layers:
            return
        slots = slot_mapping if slot_mapping is not None else _current_slot_mapping()
        if slots is None or slots.numel() == 0 or topk_ids.numel() == 0:
            return
        n = min(int(slots.shape[0]), int(topk_ids.shape[0]))
        k = min(int(topk_ids.shape[-1]), self.top_k)
        ids = topk_ids[:n, :k].to(device=self.buffer.device, dtype=self.buffer.dtype)
        dest = self._forward_dest_slots(layer_id, slots, n)
        if k < self.top_k:
            row = self.buffer.new_full((n, self.top_k), UNCAPTURED)
            row[:, :k] = ids
            ids = row
        self.buffer[dest, layer_id, :] = ids

    def _forward_dest_slots(
        self, layer_id: int, slots: torch.Tensor, n: int
    ) -> torch.Tensor:
        """``_dest_slots(slots[:n])`` computed once per forward, not per layer.

        The slot_mapping staging buffer is rewritten in place between forwards,
        so tensor identity alone cannot invalidate. MoE layers run in increasing
        id order within a forward; a non-increasing id starts a new forward.
        Under CUDA-graph capture the first layer records the computation and
        later layers read its output, so replay stays correct.
        """
        key = (id(slots), slots.data_ptr(), tuple(slots.shape), n)
        cached = self._dest_cache
        if cached is not None and cached[0] == key and layer_id > cached[1]:
            self._dest_cache = (key, layer_id, cached[2])
            return cached[2]
        dest = self._dest_slots(slots[:n])
        self._dest_cache = (key, layer_id, dest)
        return dest

    def _dest_slots(self, slot_mapping: torch.Tensor) -> torch.Tensor:
        slots_i = slot_mapping.reshape(-1).to(dtype=torch.long)
        valid = slots_i >= 0
        phys = slots_i.clamp(min=0, max=self.num_slots - 1)
        return torch.where(valid, phys, torch.full_like(phys, self._pad_slot))

    def store_step(
        self,
        slot_mapping: torch.Tensor | None,
        req_ids: AbcSequence[int],
        starts: AbcSequence[int],
        lens: AbcSequence[int],
        *,
        prefix: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> PendingRoutes | None:
        """Gather this step's rows in batch token order and queue their D2H.

        Request ``i`` forwarded ``lens[i]`` tokens from position ``starts[i]``.
        ``prefix`` is ``(req_ids, starts, lens, slots)`` for cached prefixes no
        earlier step returned; their rows go first. Returns None when the step
        forwarded nothing.

        The gather runs on the current stream: ``slot_mapping`` is the reused
        staging buffer, and the next forward rewrites it in order after this
        stream only. With ``stream`` the D2H is non-blocking on it.
        """
        slots = slot_mapping if slot_mapping is not None else _current_slot_mapping()
        lens = np.asarray(lens, dtype=np.int64)
        total = int(lens.sum())
        if slots is None or total == 0:
            return None
        req_ids = np.asarray(req_ids, dtype=np.int64)
        starts = np.asarray(starts, dtype=np.int64)
        captured = min(int(slots.shape[0]), total)
        dest = self._dest_slots(slots[:captured])
        if prefix is not None:
            p_req, p_starts, p_lens, p_slots = prefix
            p_dest = torch.from_numpy(np.clip(p_slots, 0, self.num_slots - 1))
            dest = torch.cat([p_dest.to(dest.device), dest])
            req_ids = np.concatenate([p_req, req_ids])
            starts = np.concatenate([p_starts, starts])
            lens = np.concatenate([p_lens, lens])
        rows = self.buffer[dest].to(dtype=torch.int16)
        if captured < total:
            # A metadata builder exposed fewer slots than the step ran.
            rows = torch.cat(
                [rows, rows.new_full((total - captured, *rows.shape[1:]), UNCAPTURED)]
            )
        offsets = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=offsets[1:])
        if stream is not None and self.buffer.is_cuda:
            stream.wait_stream(torch.cuda.current_stream(self.buffer.device))
            rows.record_stream(stream)
            with torch.cuda.stream(stream):
                rows_cpu = rows.to("cpu", non_blocking=True)
                done = torch.cuda.Event()
                done.record(stream)
        else:
            rows_cpu, done = rows.cpu(), None
        return PendingRoutes(req_ids, starts, offsets, rows_cpu, done)


def maybe_capture_routed_experts(layer: Any, topk_ids: torch.Tensor) -> None:
    """No-op when capture is off, uninitialized, or still in dummy warmup."""
    capturer = RoutedExpertsCapturer.get()
    if capturer is None or topk_ids is None:
        return
    layer_id = getattr(layer, "moe_capture_layer_id", None)
    if layer_id is None:
        layer_id = getattr(layer, "layer_id", None)
    if layer_id is None:
        return
    capturer.capture(int(layer_id), topk_ids)
