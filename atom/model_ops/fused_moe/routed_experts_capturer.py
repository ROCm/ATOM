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

import weakref
from collections.abc import Sequence as AbcSequence
from typing import Any, NamedTuple

import numpy as np
import torch

_INSTANCE: RoutedExpertsCapturer | None = None

# Missed captures read as this, never as a valid expert id.
UNCAPTURED = -1
# The GPU capture buffer's element type; the KV pool budget is sized from it.
CAPTURE_DTYPE = torch.int32


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
    plugin_mode: bool = False,
) -> None:
    """Refuse topologies that cannot assemble a full [seq_len-1] route tensor."""
    if plugin_mode:
        raise ValueError(
            "enable_return_routed_experts is not supported in plugin mode: "
            "capture lives in ATOM's ModelRunner, which a host framework does "
            "not run"
        )
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


def check_return_routed_experts_config(
    config: Any, *, rapidserve_runner: bool = False
) -> None:
    """``check_return_routed_experts`` over an ATOM ``Config``.

    The one place the effective topology is read from a config, for every
    caller: ``Config`` at construction, the scheduler and the RapidServe
    runner, which see the config again in their own processes.
    ``rapidserve_runner`` is the runner's own knowledge that it is one.
    """
    if not getattr(config, "enable_return_routed_experts", False):
        return
    from atom.plugin import is_plugin_mode

    check_return_routed_experts(
        getattr(config, "decode_context_parallel_size", 1),
        getattr(config, "prefill_context_parallel_size", 1),
        getattr(config, "pipeline_parallel_size", 1),
        kv_transfer_config=getattr(config, "kv_transfer_config", None),
        enable_rapidserve=bool(
            rapidserve_runner
            or getattr(config, "enable_rapidserve", False)
            or getattr(config, "disagg_is_decode", False)
            or "RapidServeModelRunner" in str(getattr(config, "runner_qualname", ""))
        ),
        enable_dp_attention=bool(getattr(config, "enable_dp_attention", False)),
        plugin_mode=bool(
            getattr(getattr(config, "plugin_config", None), "is_plugin_mode", False)
            or is_plugin_mode()
        ),
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
    from atom.model_ops.attentions.token_layout.slots import slot_mapping

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
    from atom.utils.forward_context import get_forward_context

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


def capture_pad_row_bytes(num_layers: int, top_k: int) -> int:
    """Sacrificial dummy-slot row charged once, not per KV block."""
    return capture_row_bytes(num_layers, top_k)


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

    The extra row is a sacrificial pad target: slots index the buffer directly,
    so a graph dummy slot (``PAD_SLOT_ID``, -1) wraps to it. Each step's rows are gathered in batch token order by ``store_step``,
    which returns them as ``PendingRoutes`` for the caller to keep.
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
        charged = self.num_slots * capture_row_bytes(
            self.num_layers, self.top_k
        ) + capture_pad_row_bytes(self.num_layers, self.top_k)
        if self.buffer.nbytes != charged:
            raise RuntimeError(
                f"routed-expert capture buffer is {self.buffer.nbytes} B but the "
                f"KV pool budget charged {charged} B"
            )

    @classmethod
    def init(
        cls,
        num_slots: int,
        num_layers: int,
        top_k: int,
        device: torch.device | str,
        owner: object | None = None,
    ) -> RoutedExpertsCapturer:
        """Install the process's capturer.

        The MoE hook finds it through this global, so one process holds one.
        ``owner`` (the runner) is recorded; another live owner's ``init``
        raises rather than redirecting the first runner's routes.
        """
        global _INSTANCE
        held = _INSTANCE._live_owner() if _INSTANCE is not None else None
        if held is not None and held is not owner:
            raise RuntimeError(
                "enable_return_routed_experts: the routed-expert capturer is "
                f"held by {type(held).__name__} {id(held):#x}; one process "
                "supports one capturing runner"
            )
        capturer = cls(num_slots, num_layers, top_k, device)
        capturer._owner = weakref.ref(owner) if owner is not None else None
        _INSTANCE = capturer
        return capturer

    _owner: weakref.ref | None = None

    def _live_owner(self) -> object | None:
        return self._owner() if self._owner is not None else None

    @classmethod
    def get(cls) -> RoutedExpertsCapturer | None:
        return _INSTANCE

    @classmethod
    def reset(cls, owner: object | None = None) -> None:
        """Drop the process-local GPU/CPU buffers. Sleep KV teardown uses this
        so wake does not size a new pool against a still-live capture tensor.

        With ``owner``, only that owner's capturer is dropped.
        """
        global _INSTANCE
        if owner is not None and _INSTANCE is not None:
            held = _INSTANCE._live_owner()
            if held is not None and held is not owner:
                return
        _INSTANCE = None

    def capture(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
        slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Scatter logical expert ids. Pad slots (``-1``) write the pad row."""
        if layer_id < 0 or layer_id >= self.num_layers:
            return
        slots = slot_mapping if slot_mapping is not None else current_slot_mapping()
        if slots.numel() == 0 or topk_ids.numel() == 0:
            return
        n = min(int(slots.shape[0]), int(topk_ids.shape[0]))
        k = min(int(topk_ids.shape[-1]), self.top_k)
        ids = topk_ids[:n, :k].to(device=self.buffer.device, dtype=self.buffer.dtype)
        if k < self.top_k:
            row = self.buffer.new_full((n, self.top_k), UNCAPTURED)
            row[:, :k] = ids
            ids = row
        self.buffer[slots.reshape(-1)[:n], layer_id, :] = ids

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
        slots = slot_mapping if slot_mapping is not None else current_slot_mapping()
        lens = np.asarray(lens, dtype=np.int64)
        total = int(lens.sum())
        if total == 0:
            return None
        req_ids = np.asarray(req_ids, dtype=np.int64)
        starts = np.asarray(starts, dtype=np.int64)
        captured = min(int(slots.shape[0]), total)
        rows_at = slots.reshape(-1)[:captured]
        if prefix is not None:
            p_req, p_starts, p_lens, p_slots = prefix
            p_at = torch.from_numpy(np.asarray(p_slots)).to(
                rows_at.device, rows_at.dtype
            )
            rows_at = torch.cat([p_at, rows_at])
            req_ids = np.concatenate([p_req, req_ids])
            starts = np.concatenate([p_starts, starts])
            lens = np.concatenate([p_lens, lens])
        rows = self.buffer[rows_at].to(dtype=torch.int16)
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
    """No-op when capture is off, uninitialized, or still in dummy warmup.

    Only the target model's FusedMoE layers carry ``moe_capture_layer_id``; a
    drafter's MoE has none and must not write over the target's routes.
    """
    capturer = RoutedExpertsCapturer.get()
    if capturer is None or topk_ids is None:
        return
    layer_id = getattr(layer, "moe_capture_layer_id", None)
    if layer_id is None:
        return
    capturer.capture(int(layer_id), topk_ids)
