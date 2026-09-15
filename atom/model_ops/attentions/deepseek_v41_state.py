# SPDX-License-Identifier: MIT
"""Eager CSA2 cache ownership and inputs for the unmodified V4 attention kernels."""

from dataclasses import dataclass, field

import torch

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import WindowParams
from atom.model_ops.deepseek_v41.compressor import CompressorTail
from atom.model_ops.deepseek_v41.paged_indices import build_sparse_indices
from atom.model_ops.v4_kernels.state_writes import swa_write
from atom.models.deepseek_v41.config import AttentionMode


@dataclass
class AttentionStep:
    position: int
    length: int
    positions: torch.Tensor
    cu_seqlens_q: torch.Tensor
    indices: dict[int, torch.Tensor] = field(default_factory=dict)
    candidates: dict[int, torch.Tensor] = field(default_factory=dict)

    @property
    def decode(self):
        return self.length == 1


class EagerAttentionCache:
    packed = False

    """Fixed-batch storage with private SWA rings and one global region per owner.

    The pool uses V4's row-addressed BF16 ABI. It is allocated for one offline
    request batch; scheduler paging and request migration are separate work.
    """

    def __init__(self, config, topology, batch_size, max_length, device):
        if batch_size < 1 or not 1 <= max_length <= config.max_position_embeddings:
            raise ValueError("Invalid eager CSA2 cache capacity")
        self.batch_size, self.max_length = batch_size, max_length
        self.position = 0
        self.tails: dict[int, CompressorTail | None] = {}
        self.main, self.index, self.main_offsets = {}, {}, {}
        rows = len(topology) * config.sliding_window
        for spec in topology:
            if spec.mode == AttentionMode.FULL:
                self.main_offsets[spec.layer_id] = rows
                rows += max_length // spec.ratio
        self.pool = torch.empty(
            batch_size * rows, config.head_dim, dtype=torch.bfloat16, device=device
        )
        view = self.pool.view(batch_size, rows, config.head_dim)
        self.slots = torch.arange(batch_size, dtype=torch.int32, device=device)
        self.window = {
            spec.layer_id: WindowParams(
                ring_start=i * config.sliding_window,
                slot_rows=rows,
                ring_slots=config.sliding_window,
                ring_stride=config.sliding_window,
                run_rows=config.sliding_window,
            )
            for i, spec in enumerate(topology)
        }
        for spec in topology:
            if spec.mode == AttentionMode.FULL:
                count = max_length // spec.ratio
                start = self.main_offsets[spec.layer_id]
                self.main[spec.layer_id] = view[:, start : start + count]
                self.index[spec.layer_id] = torch.empty(
                    batch_size,
                    count,
                    config.index_head_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )

    def rope_positions(self, step):
        return step.positions[: step.length]

    def requests(self, step):
        yield self, step, slice(None)

    def read_tail(self, owner, position):
        return self.tails.get(owner)

    def write_tail(self, owner, tail, *, rows=None):
        self.tails[owner] = tail

    def write_global(self, owner, begin, main, index):
        end = begin + main.shape[1]
        self.main[owner][:, begin:end] = main
        self.index[owner][:, begin:end] = index

    def index_keys(self, owner, count):
        return self.index[owner][:, :count]

    def begin_step(self, position, length, batch_size):
        if position != self.position or batch_size != self.batch_size or length < 1:
            raise ValueError(
                "Eager cache requires the next contiguous step of its fixed batch"
            )
        if position + length > self.max_length:
            raise ValueError("Eager cache capacity exceeded")
        positions = torch.arange(
            position, position + length, dtype=torch.int32, device=self.pool.device
        ).repeat(batch_size)
        cu_seqlens = (
            torch.arange(batch_size + 1, dtype=torch.int32, device=self.pool.device)
            * length
        )
        return AttentionStep(position, length, positions, cu_seqlens)

    def finish_step(self, step):
        if step.position != self.position:
            raise ValueError("Attention step does not match the cache position")
        self.position += step.length

    def write_window(self, layer_id, kv, step):
        window = self.window[layer_id]
        swa_write(
            kv.flatten(0, 1),
            step.positions,
            step.cu_seqlens_q,
            self.slots,
            self.pool,
            window,
            min(step.length, window.ring_slots),
        )

    def attention_indices(self, spec, step):
        return build_sparse_indices(
            step.indices[spec.topk_owner] if spec.ratio else None,
            position=step.position,
            length=step.length,
            batch_size=self.batch_size,
            window=self.window[spec.layer_id],
            global_start=self.main_offsets[spec.kv_owner] if spec.ratio else 0,
            device=self.pool.device,
        )
