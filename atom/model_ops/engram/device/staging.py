# SPDX-License-Identifier: MIT
"""Graph-replayable Engram lookup and TP reassembly beside early layers.

A private collective instance isolates side-stream synchronization/scratch
from model collectives. Storage and IPC registration precede graph capture.
Every forward (including warmup) records the same fork and per-layer joins.
"""

import logging
import os
from unittest.mock import patch

import torch

from atom.model_ops.engram.device.hashing import (
    engram_snapshot,
    engram_snapshot_indices,
)
from atom.model_ops.engram.device.uva import uva_gather_into

logger = logging.getLogger(__name__)


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
        self.flat = {
            layer: torch.empty(
                host.max_num_tokens,
                host.local_heads * host.prefetcher._tables[layer].head_dim,
                dtype=host.buffers[layer].gpu.dtype,
                device=host.device,
            )
            for layer in host.layer_ids
        }

        self.collective = None
        self.gathered = {}
        self._init_collective()

    def _init_collective(self):
        host = self.host
        group = host._tp_group
        if group is None:
            return
        # The last-dimension kernel supports these TP sizes and 16-byte packs.
        # Keep the original main-stream path for other configurations.
        first = host.layer_ids[0]
        local_width = host.local_heads * host.prefetcher._tables[first].head_dim
        itemsize = host.buffers[first].gpu.element_size()
        if group.world_size not in (2, 4, 8) or local_width * itemsize % 16:
            logger.warning(
                "engram: side-stream TP gather unsupported; using main stream"
            )
            return
        from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

        # Sharing TP GPU state would race barrier flags and scratch against
        # main-stream all-reduce. Only reuse the CPU bootstrap group.
        # AITER exports pool handles with offset zero. Use its raw hipMalloc
        # pool so even small allocations have an exportable allocation base.
        # Scope this to initialization; do not change the model communicator.
        with patch.dict(os.environ, {"AITER_CUSTOM_AR_RAW_INPUT_POOL": "1"}):
            collective = CustomAllreduce(
                group.cpu_group,
                host.device,
                max_size=max(1024 * 1024, host.max_num_tokens * local_width * itemsize),
            )
        if collective.disabled:
            logger.warning("engram: private TP gather unavailable; using main stream")
            return
        self.collective = collective
        for layer in host.layer_ids:
            full_width = local_width * group.world_size
            self.gathered[layer] = (
                host.buffers[layer].gpu
                if full_width == host.embed_width
                else torch.empty(
                    host.max_num_tokens,
                    full_width,
                    dtype=host.buffers[layer].gpu.dtype,
                    device=host.device,
                )
            )
        logger.info(
            "engram: hash/UVA/TP gather on one side stream with private IPC state"
        )

    def prepare(self, batch, width, *, cursor_positions=None, cursor_out=None):
        if not 0 <= width <= self.host.max_num_tokens:
            raise ValueError("Engram staging exceeds capacity")
        snapshot = self.snapshot[:width]
        if batch is None:
            # Capture uses serving's kernels without accessing synthetic state.
            snapshot.fill_(-2)
        else:
            engram_snapshot(
                self.uva.hash_tables,
                batch,
                snapshot,
                cursor_positions=cursor_positions,
                cursor_out=cursor_out,
            )
        return EngramStagedRows(self, width)

    def start(self, width):
        host = self.host
        compute = torch.cuda.current_stream(host.device)
        # Get the parent BEFORE entering the guard: waiting on ourselves
        # would neither order the inputs nor join the graph capture.
        self.stream.wait_stream(compute)
        with torch.cuda.stream(self.stream):
            for layer in host.layer_ids:
                ids = engram_snapshot_indices(
                    self.uva.hash_tables,
                    layer,
                    self.snapshot[:width],
                    self.uva.row_ids[:width],
                )
                table = host.prefetcher._tables[layer]
                uva_gather_into(
                    table,
                    ids,
                    self.flat[layer][:width].view(
                        width, host.local_heads, table.head_dim
                    ),
                    head_start=host.head_start,
                    local_heads=host.local_heads,
                    total_heads=host.total_heads,
                )
                if self.collective is not None:
                    out = self.gathered[layer][:width]
                    # Use the communicator's pre-registered input pool. Direct
                    # external registration in AITER assumes a zero IPC offset,
                    # which is invalid for caching-allocator suballocations.
                    self.collective.all_gather_unreg(
                        self.flat[layer][:width], out=out, dim=1
                    )
                    if self.gathered[layer] is not host.buffers[layer].gpu:
                        host.buffers[layer].gpu[:width].copy_(
                            out[:, : host.embed_width]
                        )
                elif host._tp_group is None:
                    host.buffers[layer].gpu[:width].copy_(self.flat[layer][:width])
                # Ready means the complete embedding, including TP reassembly.
                self.done[layer].record(self.stream)

    def consume(self, layer, width):
        host = self.host
        torch.cuda.current_stream(host.device).wait_event(self.done[layer])
        if host._tp_group is not None and self.collective is None:
            out = self.flat[layer][:width]
            out = host._tp_group.all_gather(out, use_custom=True, dim=1)
            out = out[:, : host.embed_width]
            host.buffers[layer].gpu[:width].copy_(out)

    def join(self):
        # Close the fork even if a partial forward did not consume every layer.
        torch.cuda.current_stream(self.host.device).wait_stream(self.stream)


class EngramStagedRows(dict):
    """A width-bound view, not a consumable pending queue.

    Capture calls stage/get/join after warmup without another prepare.
    Replay uses the frozen addresses; prepare updates their contents.
    """

    def __init__(self, staging, width):
        super().__init__(
            (layer, buffer.gpu[:width].unsqueeze(0))
            for layer, buffer in staging.host.buffers.items()
        )
        self.staging, self.width = staging, width

    def stage(self):
        self.staging.start(self.width)

    def get(self, layer, default=None):
        if layer in self:
            self.staging.consume(layer, self.width)
        return super().get(layer, default)

    def join(self):
        self.staging.join()
