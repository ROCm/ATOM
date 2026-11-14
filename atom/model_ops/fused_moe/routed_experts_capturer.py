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

from atom.model_ops.attentions.token_layout.slots import slot_mapping
from atom.utils.forward_context import get_forward_context

# The FusedMoE attribute holding its layer's ``buffer[:, layer]``. Bound with
# the KV cache and cleared with it on sleep.
ROUTES_VIEW_ATTR = "routed_experts_view"

# Missed captures read as this, never as a valid expert id.
UNCAPTURED = -1
# The GPU capture buffer's element type; the KV pool budget is sized from it.
CAPTURE_DTYPE = torch.int32


class RoutedExpertsPatch(NamedTuple):
    """One step's routes for every request it forwarded.

    Segment ``j`` is request ``req_ids[j]``'s rows for token positions
    ``[starts[j], starts[j] + offsets[j+1] - offsets[j])``. Segments apply in
    order, so a prefix-hit segment precedes the same request's step segment.
    ``rows`` is the device gather until the token copy moves it to the host;
    the scheduler receives it as a numpy array.
    """

    req_ids: np.ndarray
    starts: np.ndarray
    offsets: np.ndarray
    rows: np.ndarray | torch.Tensor


def merge_routed_patches(patches: list[RoutedExpertsPatch]) -> RoutedExpertsPatch:
    """One host patch holding ``patches``' segments in order, oldest step first."""
    if len(patches) == 1:
        p = patches[0]
        return p._replace(rows=np.asarray(p.rows))
    offsets = [patches[0].offsets]
    for p in patches[1:]:
        offsets.append(p.offsets[1:] + offsets[-1][-1])
    return RoutedExpertsPatch(
        np.concatenate([p.req_ids for p in patches]),
        np.concatenate([p.starts for p in patches]),
        np.concatenate(offsets),
        np.concatenate([np.asarray(p.rows) for p in patches]),
    )


def is_fused_moe_module(module: Any) -> bool:
    """True for a real FusedMoE, including LazyMoEWrapper instances.

    ``FusedMoEDecoratorForPluginMode`` replaces ``FusedMoE`` with a subclass
    whose ``__new__`` returns an *undecorated* instance. ``isinstance(m,
    FusedMoE)`` is then False, so callers must also accept the wrapper's base.
    """
    from atom.model_ops.moe import FusedMoE

    # An undecorated FusedMoE's base is nn.Module, which every module matches.
    types: tuple[type, ...] = (FusedMoE,) + tuple(
        b
        for b in getattr(FusedMoE, "__bases__", ())
        if isinstance(b, type) and b is not torch.nn.Module
    )
    return isinstance(module, types)


def fused_moe_modules(root: Any) -> list:
    return [m for m in root.modules() if is_fused_moe_module(m)]


def prefix_slots(block_table: AbcSequence[int], lo: int, hi: int, block_size: int):
    """KV slots of token positions ``[lo, hi)`` for one request."""
    n = int(hi) - int(lo)
    blocks = np.asarray(block_table[: (int(hi) - 1) // block_size + 1], np.int32)
    return slot_mapping(
        np.arange(lo, hi, dtype=np.int64),
        np.array([n], dtype=np.int64),
        blocks[None, :],
        block_size,
    )


def batch_prefix_slots(batch: Any, block_size: int):
    """``(req_ids, starts, lens, slots)`` of a batch's ``routed_prefix_ranges``,
    or None when it has none. Ranges name requests by batch position."""
    ranges = getattr(batch, "routed_prefix_ranges", None)
    if not ranges:
        return None
    if len(batch.block_tables) != len(batch.req_ids):
        # `ScheduledBatch` drops empty tables, so row i would be another
        # request's blocks and the prefix would read its routes.
        raise RuntimeError(
            f"{len(batch.block_tables)} block tables for "
            f"{len(batch.req_ids)} requests; cached-prefix routes are "
            "looked up by batch position"
        )
    req_ids = np.asarray([batch.req_ids[i] for i, _, _ in ranges], np.int64)
    starts = np.asarray([lo for _, lo, _ in ranges], np.int64)
    lens = np.asarray([hi - lo for _, lo, hi in ranges], np.int64)
    slots = np.concatenate(
        [
            prefix_slots(batch.block_tables[i], lo, hi, block_size)
            for i, lo, hi in ranges
        ]
    )
    return req_ids, starts, lens, slots


def current_slot_mapping() -> torch.Tensor:
    """This forward's per-token KV slots, from the target builder's metadata.

    A per-layer metadata dict (plugin backends) has no single slot space to
    pick, and a forward without slots would silently drop its routes; both
    raise instead.
    """
    md = getattr(get_forward_context(), "attn_metadata", None)
    slots = None if isinstance(md, dict) else getattr(md, "slot_mapping", None)
    if isinstance(slots, torch.Tensor):
        return slots
    raise RuntimeError(
        "enable_return_routed_experts: this forward's attention metadata "
        f"({type(md).__name__}) carries no single per-token slot_mapping"
    )


def capture_row_bytes(num_layers: int, top_k: int) -> int:
    """One token slot's routes in the capture buffer."""
    return int(num_layers) * int(top_k) * CAPTURE_DTYPE.itemsize


def capture_bytes_per_kv_block(block_size: int, num_layers: int, top_k: int) -> int:
    """PAGE-pool surcharge: routes for every token slot in one KV block."""
    return int(block_size) * capture_row_bytes(num_layers, top_k)


class RoutedExpertsCapturer:
    """Persistent GPU buffer: ``[num_kv_slots + 1, num_layers, top_k]``.

    The extra row is a sacrificial pad target: slots index the buffer directly,
    so a graph dummy slot (``PAD_SLOT_ID``, -1) wraps to it. ``bind`` hands
    each target FusedMoE its layer's view; each step's rows are gathered in
    batch token order by ``gather_step``, which leaves them on the device for
    the step's token copy to carry to the host.
    """

    def __init__(
        self,
        num_slots: int,
        num_layers: int,
        top_k: int,
        device: torch.device | str,
    ):
        if num_slots <= 0 or num_layers <= 0 or top_k <= 0:
            raise ValueError(
                f"invalid capturer shape slots={num_slots} layers={num_layers} "
                f"top_k={top_k}"
            )
        self.num_slots = int(num_slots)
        self.num_layers = int(num_layers)
        self.top_k = int(top_k)
        # Index -1 is this last row, so pad slots never alias physical slot 0.
        # Any other negative slot would land on a real row.
        self.buffer = torch.full(
            (self.num_slots + 1, self.num_layers, self.top_k),
            UNCAPTURED,
            dtype=CAPTURE_DTYPE,
            device=device,
        )
        charged = (self.num_slots + 1) * capture_row_bytes(self.num_layers, self.top_k)
        if self.buffer.nbytes != charged:
            raise RuntimeError(
                f"routed-expert capture buffer is {self.buffer.nbytes} B but the "
                f"KV pool budget charged {charged} B"
            )

    def bind(self, layer: Any, layer_idx: int) -> None:
        """Give ``layer``'s MoE hook its ``buffer[:, layer_idx]`` to write."""
        if not 0 <= layer_idx < self.num_layers:
            raise RuntimeError(
                f"routed-expert capture: layer {layer_idx} is outside the "
                f"buffer's {self.num_layers} layers"
            )
        setattr(layer, ROUTES_VIEW_ATTR, self.buffer[:, layer_idx])

    def gather_step(
        self,
        slot_mapping: torch.Tensor | None,
        req_ids: AbcSequence[int],
        starts: AbcSequence[int],
        lens: AbcSequence[int],
        *,
        prefix: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> RoutedExpertsPatch | None:
        """This step's rows in batch token order, still on the device.

        Request ``i`` forwarded ``lens[i]`` tokens from position ``starts[i]``.
        ``prefix`` is ``(req_ids, starts, lens, slots)`` for cached prefixes no
        earlier step returned; their rows go first. Returns None when the step
        forwarded nothing.

        The gather runs on the current stream: ``slot_mapping`` is the reused
        staging buffer, and the next forward rewrites it in order after this
        stream only.
        """
        slots = slot_mapping if slot_mapping is not None else current_slot_mapping()
        lens = np.asarray(lens, dtype=np.int64)
        total = int(lens.sum())
        if total == 0:
            return None
        req_ids = np.asarray(req_ids, dtype=np.int64)
        starts = np.asarray(starts, dtype=np.int64)
        if int(slots.shape[0]) < total:
            raise RuntimeError(
                f"routed-expert capture: the step forwarded {total} tokens but "
                f"its slot_mapping has {int(slots.shape[0])} slots"
            )
        # Slots past `total` are graph padding.
        rows_at = slots.reshape(-1)[:total]
        if prefix is not None:
            p_req, p_starts, p_lens, p_slots = prefix
            p_at = torch.from_numpy(np.asarray(p_slots))
            if rows_at.is_cuda:
                p_at = p_at.pin_memory()
            p_at = p_at.to(rows_at.device, rows_at.dtype, non_blocking=True)
            rows_at = torch.cat([p_at, rows_at])
            req_ids = np.concatenate([p_req, req_ids])
            starts = np.concatenate([p_starts, starts])
            lens = np.concatenate([p_lens, lens])
        rows = self.buffer[rows_at].to(dtype=torch.int16)
        offsets = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=offsets[1:])
        return RoutedExpertsPatch(req_ids, starts, offsets, rows)


def maybe_capture_routed_experts(layer: Any, topk_ids: torch.Tensor) -> None:
    """Scatter ``layer``'s logical expert ids at this forward's slots.

    A no-op on a layer with no bound view: capture off, a TP rank other than
    0, a drafter's MoE (which must not write over the target's routes), or
    before the KV cache is bound. Pad slots (``-1``) write the pad row. Only
    the layer's first ``top_k`` columns are routes; the rest are fused shared
    experts.
    """
    view = getattr(layer, ROUTES_VIEW_ATTR, None)
    if view is None or topk_ids is None:
        return
    name = getattr(layer, "prefix", type(layer).__name__)
    slots = current_slot_mapping()
    n = int(topk_ids.shape[0])
    if int(slots.shape[0]) != n:
        raise RuntimeError(
            f"routed-expert capture: {name} routed {n} tokens but this "
            f"forward's slot_mapping has {int(slots.shape[0])} slots"
        )
    width = int(view.shape[-1])
    k = int(layer.top_k)
    if not 0 < k <= width or int(topk_ids.shape[-1]) < k:
        raise RuntimeError(
            f"routed-expert capture: {name} top_k={k} with "
            f"{int(topk_ids.shape[-1])} top-k columns does not fit the "
            f"buffer's top_k={width}"
        )
    if n == 0:
        return
    ids = topk_ids[:, :k].to(device=view.device, dtype=view.dtype)
    if k < width:
        row = view.new_full((n, width), UNCAPTURED)
        row[:, :k] = ids
        ids = row
    view[slots.reshape(-1)] = ids
