# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CUDA-graph-safe MoE route capture indexed by physical KV slots.

Matches the vLLM ``enable_return_routed_experts`` contract: scatter logical
expert ids into ``buffer[slot, layer, :]`` during fused MoE, D2H this step's
``slot_mapping`` rows into a CPU slot buffer, then numpy-gather a per-request
``[seq_len - 1, num_layers, top_k]`` int16 tensor from the block table.
"""

from __future__ import annotations

from collections.abc import Sequence as AbcSequence
from typing import Any

import numpy as np
import torch

_INSTANCE: RoutedExpertsCapturer | None = None


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


def kv_slots_from_block_table(
    block_table: AbcSequence[int],
    num_tokens: int,
    block_size: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.long,
) -> np.ndarray:
    """Physical KV slots for the first ``num_tokens`` positions (host numpy)."""
    del device, dtype
    n = int(num_tokens)
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    blocks = np.asarray(list(block_table), dtype=np.int64)
    pos = np.arange(n, dtype=np.int64)
    bs = np.int64(block_size)
    return blocks[pos // bs] * bs + (pos % bs)


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
    A host ``int16`` twin is filled incrementally by ``store_step`` so finish
    export is a numpy gather, not a GPU ``as_tensor(block_table)``.
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
        self.cpu_buffer = np.zeros(
            (self.num_slots + 1, self.num_layers, self.top_k),
            dtype=np.int16,
        )
        self._pending: list[tuple[torch.Tensor, torch.Tensor]] = []

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
        dest = self._dest_slots(slots)
        if k < self.top_k:
            row = self.buffer.new_zeros((n, self.top_k))
            row[:, :k] = ids
            ids = row
        self.buffer[dest, layer_id, :] = ids

    def _dest_slots(self, slot_mapping: torch.Tensor) -> torch.Tensor:
        slots_i = slot_mapping.reshape(-1).to(dtype=torch.long)
        valid = slots_i >= 0
        phys = slots_i.clamp(min=0, max=self.num_slots - 1)
        return torch.where(valid, phys, torch.full_like(phys, self._pad_slot))

    def _apply_cpu_store(self, dest: np.ndarray, rows: np.ndarray) -> None:
        dest = np.asarray(dest, dtype=np.int64).reshape(-1)
        rows = np.asarray(rows, dtype=np.int16).reshape(
            -1, self.num_layers, self.top_k
        )
        mask = (dest >= 0) & (dest < self.num_slots)
        if not np.any(mask):
            return
        self.cpu_buffer[dest[mask]] = rows[mask]

    def commit_pending(self, *, keep_last: bool = False) -> None:
        """Write host rows whose D2H is already covered by token ``copy_done``.

        ``keep_last`` leaves the in-flight current step (queued before
        ``send_to_cpu_async``) until the next ``recv_async_output``.
        """
        if not self._pending:
            return
        if keep_last and len(self._pending) <= 1:
            return
        ready, self._pending = (
            (self._pending[:-1], self._pending[-1:])
            if keep_last
            else (self._pending, [])
        )
        for dest, rows in ready:
            self._apply_cpu_store(dest.numpy(), rows.numpy())

    def store_step(
        self,
        slot_mapping: torch.Tensor | None = None,
        *,
        stream: torch.cuda.Stream | None = None,
        wait_event: torch.cuda.Event | None = None,
    ) -> bool:
        """D2H this step's ``buffer[slot_mapping]`` into the CPU slot buffer.

        Snapshot ``dest``/``rows`` on the default stream (private from the next
        capture), then enqueue non-blocking D2H on the token
        ``async_copy_stream`` after ``wait_event`` so it lands before the
        caller's ``copy_done``. Do not record a private event: the existing
        ``recv_async_output`` wait covers this memcpy.

        Returns True when a new memcpy was queued (or applied inline). False
        means this step had nothing to store; the caller must then
        ``commit_pending(keep_last=False)`` so the previous in-flight decode
        is visible to ``export_batch`` (last deferred flush has no new copy).
        """
        slots = slot_mapping if slot_mapping is not None else _current_slot_mapping()
        if slots is None or slots.numel() == 0:
            return False
        dest = self._dest_slots(slots)
        rows = self.buffer[dest].to(dtype=torch.int16)
        use_async = (
            stream is not None
            and self.buffer.is_cuda
            and torch.cuda.is_available()
        )
        if not use_async:
            self.commit_pending()
            self._apply_cpu_store(
                dest.detach().cpu().numpy(),
                rows.detach().cpu().numpy(),
            )
            return True
        with torch.cuda.stream(stream):
            if wait_event is not None:
                wait_event.wait(stream)
            dest_cpu = dest.to(dtype=torch.int32).to("cpu", non_blocking=True)
            rows_cpu = rows.to("cpu", non_blocking=True)
        self._pending.append((dest_cpu, rows_cpu))
        return True

    def export_batch(
        self,
        req_ids: AbcSequence[int],
        block_tables: AbcSequence[AbcSequence[int]],
        num_tokens_list: AbcSequence[int],
        block_size: int,
    ) -> dict[int, np.ndarray]:
        """CPU gather: ``[num_tokens, num_layers, top_k]`` int16 per request."""
        out: dict[int, np.ndarray] = {}
        for req_id, block_table, ntok in zip(
            req_ids, block_tables, num_tokens_list, strict=False
        ):
            n = int(ntok)
            if n <= 0 or not block_table:
                continue
            slots = kv_slots_from_block_table(block_table, n, block_size)
            slots = np.clip(slots, 0, self.num_slots - 1)
            out[int(req_id)] = self.cpu_buffer[slots].copy()
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
