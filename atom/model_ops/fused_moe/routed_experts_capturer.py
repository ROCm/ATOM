# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CUDA-graph-safe MoE route capture indexed by physical KV slots.

Matches the vLLM ``enable_return_routed_experts`` contract: scatter logical
expert ids into ``buffer[slot, layer, :]`` during fused MoE, then gather a
per-request ``[seq_len - 1, num_layers, top_k]`` int16 tensor from the
sequence's block table.
"""

from __future__ import annotations

from collections.abc import Sequence as AbcSequence
from typing import Any

import numpy as np
import torch

_INSTANCE: RoutedExpertsCapturer | None = None


def check_return_routed_experts(
    dcp_size: int, pcp_size: int, pp_size: int = 1
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


def kv_slots_from_block_table(
    block_table: AbcSequence[int],
    num_tokens: int,
    block_size: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Physical KV slots for the first ``num_tokens`` positions."""
    if num_tokens <= 0:
        return torch.empty(0, dtype=dtype, device=device)
    pos = torch.arange(num_tokens, device=device, dtype=dtype)
    blocks = torch.as_tensor(list(block_table), device=device, dtype=dtype)
    return blocks[pos // block_size] * int(block_size) + (pos % block_size)


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


def capture_bytes_per_kv_block(
    block_size: int, num_layers: int, top_k: int
) -> int:
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


class RoutedExpertsCapturer:
    """Persistent GPU buffer: ``[num_kv_slots + 1, num_layers, top_k]``.

    The extra row is a sacrificial pad target for graph dummy slots (``-1``).
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
        self.buffer = torch.zeros(
            (self.num_slots + 1, self.num_layers, self.top_k),
            dtype=dtype,
            device=device,
        )

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
        slots = slots[:n]
        k = min(int(topk_ids.shape[-1]), self.top_k)
        ids = topk_ids[:n, :k].to(device=self.buffer.device, dtype=self.buffer.dtype)
        slots_i = slots.to(dtype=torch.long)
        valid = slots_i >= 0
        phys = slots_i.clamp(min=0, max=self.num_slots - 1)
        dest = torch.where(valid, phys, torch.full_like(phys, self._pad_slot))
        if k < self.top_k:
            row = self.buffer.new_zeros((n, self.top_k))
            row[:, :k] = ids
            ids = row
        self.buffer[dest, layer_id, :] = ids

    def export_batch(
        self,
        req_ids: AbcSequence[int],
        block_tables: AbcSequence[AbcSequence[int]],
        num_tokens_list: AbcSequence[int],
        block_size: int,
    ) -> dict[int, np.ndarray]:
        """D2H gather: ``[num_tokens, num_layers, top_k]`` int16 per request."""
        out: dict[int, np.ndarray] = {}
        for req_id, block_table, ntok in zip(
            req_ids, block_tables, num_tokens_list, strict=False
        ):
            n = int(ntok)
            if n <= 0 or not block_table:
                continue
            slots = kv_slots_from_block_table(
                block_table,
                n,
                block_size,
                device=self.buffer.device,
            )
            slots = slots.clamp(min=0, max=self.num_slots - 1)
            routes = self.buffer[slots].to(dtype=torch.int16).detach().cpu().numpy()
            out[int(req_id)] = routes
        return out


def maybe_capture_routed_experts(
    layer: Any, topk_ids: torch.Tensor
) -> None:
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
