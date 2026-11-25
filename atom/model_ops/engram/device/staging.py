# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Graph-replayable Engram lookup and TP reassembly beside early layers.

A private collective instance isolates side-stream synchronization/scratch
from model collectives. Storage and IPC registration precede graph capture.
Every forward (including warmup) records the same kernels, fork and joins on
fixed addresses (`EngramStep`): a replay reads its own step.
"""

import logging
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from atom.model_ops.engram.device.hashing import (
    EngramBatch,
    engram_cursor_rows,
    engram_snapshot,
)
from atom.model_ops.engram.device.uva import UvaGather

logger = logging.getLogger(__name__)
# the caching allocator's settings, either spelling (HIP reads both)
ALLOC_CONF_ENVS = ("PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")


def _expandable_segments() -> bool:
    """Whether the caching allocator maps expandable segments: memory no IPC
    handle exports."""
    return any(
        "expandable_segments:True" in os.environ.get(name, "")
        for name in ALLOC_CONF_ENVS
    )


@dataclass(frozen=True)
class EngramStep:
    """One forward's Engram inputs at fixed addresses, its full width.

    `live`: the live leading tokens, on the device (0: nothing is read or
    written). `dead`: a token's no-own-id flag. `candidates`: a verify step's
    every accepted prefix (`commit_tentative` picks one); None advances
    `cursor` in place. `served_by_round`: the forward's
    `AttentionMetaData.served_by_round` (the model's decode round runs every
    layer and looks the rows up itself): `start` only advances the cursor.
    """

    input_ids: torch.Tensor
    live: torch.Tensor
    dead: torch.Tensor
    batch_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    positions: torch.Tensor
    cursor: torch.Tensor
    history_index: torch.Tensor
    candidates: torch.Tensor | None
    served_by_round: bool = False


class EngramStaging:
    def __init__(self, uva):
        self.uva = uva
        self.host = host = uva.host
        self.stream = torch.cuda.Stream(host.device)
        self.done = {layer: torch.cuda.Event() for layer in host.layer_ids}
        self.snapshot = torch.empty(
            host.max_num_tokens,
            uva.hash_tables.ngram,
            dtype=torch.int64,
            device=host.device,
        )
        self.compressed = torch.empty(
            host.max_num_tokens, dtype=torch.int32, device=host.device
        )
        self.step = None
        # the step `start` left to the model (`EngramStep.served_by_round`)
        self._served = False
        # the step `start` looked up on the compute stream (no fork to close)
        self._inline = False
        widths = {host.prefetcher._tables[layer].head_dim for layer in host.layer_ids}
        if len(widths) != 1:
            raise ValueError(f"Engram layers of different head dims {sorted(widths)}")
        # a rank's looked-up row of a layer (its heads)
        self.local_width = host.local_heads * widths.pop()
        dtype = host.buffers[host.layer_ids[0]].gpu.dtype
        # every layer's looked-up rows of a step in one block from its start,
        # [layer, row, local_width] (`_rows`): the one all-gather's input
        self.lookup = torch.empty(
            len(host.layer_ids) * host.max_num_tokens * self.local_width,
            dtype=dtype,
            device=host.device,
        )
        self.gather = UvaGather(
            [host.prefetcher._tables[layer] for layer in host.layer_ids], host.device
        )
        # the all-gather's output, alike [layer, row, embed_width] (`_gathered`)
        self.gathered = None
        self.collective = None
        self._init_collective()
        self.fallback_done = (
            {layer: torch.cuda.Event() for layer in host.layer_ids}
            if host._tp_group is not None and self.collective is None
            else {}
        )

    def _init_collective(self):
        host = self.host
        group = host._tp_group
        if group is None:
            return
        # The last-dimension kernel supports these TP sizes and 16-byte packs;
        # its output is a layer's rows in place only when the ranks' heads
        # tile the embedding exactly. Keep the original main-stream path for
        # other configurations.
        itemsize = self.lookup.element_size()
        if (
            group.world_size not in (2, 4, 8)
            or self.local_width * itemsize % 16
            or self.local_width * group.world_size != host.embed_width
        ):
            logger.warning(
                "engram: side-stream TP gather unsupported; using main stream"
            )
            return
        from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

        # Sharing TP GPU state would race barrier flags and scratch against
        # main-stream all-reduce. Only reuse the CPU bootstrap group. Its input
        # is `lookup`, registered once (peers read it in place: no copy into
        # the communicator's own pool, which stays at its smallest).
        collective = CustomAllreduce(group.cpu_group, host.device, max_size=1 << 20)
        if collective.disabled:
            logger.warning("engram: private TP gather unavailable; using main stream")
            return
        # only IPC registers a buffer peers read in place: VMM registers a
        # copy made now, symm_mem registers nothing; and IPC exports no
        # expandable-segment allocation (ROCm/aiter#4921), which every rank's
        # environment says alike (an export failing on one rank alone would
        # leave the others waiting in the handle exchange)
        if collective._use_vmm or collective._use_symm_mem or _expandable_segments():
            logger.warning(
                "engram: private TP gather needs the IPC transport and a "
                "non-expandable allocation; using main stream"
            )
            return
        collective.register_input_buffer(self.lookup)
        self.collective = collective
        self.gathered = torch.empty(
            len(host.layer_ids) * host.max_num_tokens * host.embed_width,
            dtype=self.lookup.dtype,
            device=host.device,
        )
        logger.info(
            "engram: hash/UVA/TP gather on one side stream with private IPC state"
        )

    def _rows(self, width):
        """Every layer's looked-up rows of a ``width``-row step, [layer, row,
        local_width]: a block from `lookup`'s start, whatever the width (the
        address the collective registered)."""
        n = len(self.host.layer_ids)
        return self.lookup[: n * width * self.local_width].view(
            n, width, self.local_width
        )

    def _gathered(self, width):
        """Every layer's gathered rows of a ``width``-row step, [layer, row,
        embed_width] (`_rows` gathered along the last dim)."""
        n, embed = len(self.host.layer_ids), self.host.embed_width
        return self.gathered[: n * width * embed].view(n, width, embed)

    def embeddings(self, width):
        """Each layer's embedding rows of a ``width``-row step, where the
        lookup leaves them: the all-gather's output under the private
        collective, else the host buffers."""
        if self.collective is not None:
            gathered = self._gathered(width)
            return {layer: gathered[i] for i, layer in enumerate(self.host.layer_ids)}
        return {layer: buf.gpu[:width] for layer, buf in self.host.buffers.items()}

    def _lookup(self, width, stream=None):
        """Every layer's rows looked up from their hashed ids (`_snapshot`'s),
        on the current stream, ``stream`` (default: the side stream; the
        compute stream when `start` does not fork), then reassembled over TP
        by one all-gather where the private collective serves; each layer's
        `done` recorded once its rows are complete."""
        host = self.host
        stream = self.stream if stream is None else stream
        rows = self._rows(width)
        self.gather(
            self.uva.row_ids[:, :width],
            rows.view(len(host.layer_ids), width, host.local_heads, -1),
            head_start=host.head_start,
            local_heads=host.local_heads,
            total_heads=host.total_heads,
        )
        if self.collective is None:
            for i, layer in enumerate(host.layer_ids):
                if host._tp_group is None:
                    host.buffers[layer].gpu[:width].copy_(rows[i])
                self.done[layer].record(stream)
        if self.collective is not None:
            self.collective.all_gather_reg(
                rows, out=self._gathered(width), dim=rows.dim() - 1
            )
            for layer in host.layer_ids:
                self.done[layer].record(stream)

    def prepare(self, step, width):
        """Name the forward's inputs (`EngramStep`); `start` launches on them."""
        if not 0 <= width <= self.host.max_num_tokens:
            raise ValueError("Engram staging exceeds capacity")
        self.step = step
        return EngramStagedRows(self, width)

    def _snapshot(self, width):
        """One launch: the compressed ids, the lookbacks (and a verify step's
        cursor candidates) before the cursor advance, they alias, and every
        layer's row ids hashed from them (`uva.row_ids`, `_lookup`'s)."""
        step, tables = self.step, self.uva.hash_tables
        batch = EngramBatch(
            compressed=self.compressed[:width],
            batch_ids=step.batch_ids[:width],
            cu_seqlens=step.cu_seqlens,
            history=step.cursor[:, 1:],
            history_index=step.history_index,
        )
        engram_snapshot(
            tables,
            batch,
            self.snapshot[:width],
            cursor_positions=step.positions,
            cursor_out=step.candidates,
            live=step.live,
            compress=(step.input_ids[:width], step.dead[:width]),
            rows=(self.host.layer_ids, self.uva.row_ids[:, :width]),
        )

    def _advance_cursor(self, width):
        """A non-verify step's cursor, committed in place."""
        step = self.step
        if step.candidates is None:
            engram_cursor_rows(
                self.uva.hash_tables,
                self.compressed[:width],
                step.cu_seqlens,
                step.positions,
                step.cursor[:, 1:],
                step.history_index,
                step.cursor,
                live=step.live,
            )

    def start(self, width):
        from atom.model_ops.module_dispatch_ops import graph_records_single_stream

        # A step whose model looks the rows up itself: only the cursor advance
        # (no snapshot, no side stream), and `join` waits on nothing.
        self._served = self.step.served_by_round
        self._inline = False
        if self._served:
            self._advance_cursor(width)
            return
        host = self.host
        self._snapshot(width)
        compute = torch.cuda.current_stream(host.device)
        if graph_records_single_stream():
            # A graph recorded on one stream (ATOM_GRAPH_SIDE_STREAMS=0): the
            # lookup runs inline after the cursor advance, which reads the
            # snapshot too, and `join` has no fork to close.
            self._inline = True
            self._advance_cursor(width)
            self._lookup(width, compute)
            return
        # Get the parent BEFORE entering the guard: waiting on ourselves
        # would neither order the inputs nor join the graph capture.
        self.stream.wait_stream(compute)
        # past the fork: the side stream reads the snapshot, not the cursor
        self._advance_cursor(width)
        with torch.cuda.stream(self.stream):
            self._lookup(width)

    def consume(self, layer, width):
        host = self.host
        torch.cuda.current_stream(host.device).wait_event(self.done[layer])
        if host._tp_group is not None and self.collective is None:
            out = self._rows(width)[host.layer_ids.index(layer)]
            out = host._tp_group.all_gather(out, use_custom=True, dim=1)
            out = out[:, : host.embed_width]
            host.buffers[layer].gpu[:width].copy_(out)

    def join(self):
        if self._served or self._inline:
            return
        # Close the fork even if a partial forward did not consume every layer.
        torch.cuda.current_stream(self.host.device).wait_stream(self.stream)


class EngramStagedRows(dict):
    """A width-bound view, not a consumable pending queue.

    Capture calls stage/get/join after warmup without another prepare.
    Replay uses the frozen addresses; the step's own metadata fills them.
    """

    def __init__(self, staging, width):
        super().__init__(
            (layer, rows.unsqueeze(0))
            for layer, rows in staging.embeddings(width).items()
        )
        self.staging, self.width = staging, width
        self._fallback_ready = set()

    def stage(self, *, tbo=False):
        self._fallback_ready.clear()
        self.staging.start(self.width)
        # Only TBO must serialize shared-communicator gathers before workers
        # start. Ordinary TP consumes each layer just before it is needed.
        if tbo:
            for layer in self.staging.fallback_done:
                self.get(layer)

    def get(self, layer, default=None):
        if layer in self:
            done = self.staging.fallback_done.get(layer)
            if done is None or layer not in self._fallback_ready:
                self.staging.consume(layer, self.width)
                if done is not None:
                    done.record()
                    self._fallback_ready.add(layer)
            elif done is not None:
                torch.cuda.current_stream(self.staging.host.device).wait_event(done)
        return super().get(layer, default)

    def join(self):
        self.staging.join()

    def slice(self, token_slice):
        return EngramRowsView(self, token_slice)


class EngramRowsView(Mapping):
    """Read-only token slices; every value access waits for the parent lookup.

    Mapping routes get/items/values/dict copies through __getitem__, so none
    can expose a device row without establishing its consumer dependency.
    """

    def __init__(self, parent, token_slice):
        self._rows = {layer: rows[:, token_slice] for layer, rows in parent.items()}
        self.parent = parent

    def __getitem__(self, layer):
        rows = self._rows[layer]
        self.parent.get(layer)
        return rows

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)


@contextmanager
def engram_staging(rows: Mapping[int, torch.Tensor], *, tbo=False):
    """Own the parent fork/join; token views never restart its lookup."""
    if not isinstance(rows, EngramStagedRows):
        yield
        return
    try:
        rows.stage(tbo=tbo)
        yield
    finally:
        rows.join()
