# SPDX-License-Identifier: MIT
"""Request spans shared by CSA2 paging, compression and Engram staging."""

from dataclasses import dataclass, field

import numpy as np
import torch

from atom.model_ops.attentions.deepseek_v41_state import AttentionStep
from atom.model_ops.attentions.token_layout.batch_ids import build_batch_ids
from atom.model_ops.attentions.token_layout.prefill import prefill_positions
from atom.utils import CpuGpuBuffer, pack_rows


@dataclass(frozen=True)
class RequestSpan:
    request_id: int
    position: int
    offset: int
    length: int
    slot: int
    block_ids: tuple[int, ...]

    @property
    def end(self):
        return self.position + self.length

    @property
    def token_slice(self):
        return slice(self.offset, self.offset + self.length)


@dataclass
class BatchStep:
    requests: tuple[RequestSpan, ...]
    positions: torch.Tensor
    cu_seqlens_q: torch.Tensor
    slots: torch.Tensor
    batch_ids: torch.Tensor
    block_tables: torch.Tensor
    request_steps: tuple[AttentionStep, ...]
    selected: dict[int, torch.Tensor] = field(default_factory=dict)
    # ratio -> CompressPlan. One per distinct compression ratio in the model,
    # built once per forward and read by every owner that shares that ratio.
    plans: dict[int, object] = field(default_factory=dict)
    tentative: bool = False

    @property
    def length(self):
        return self.positions.numel()

    @property
    def decode(self):
        # Verification has ring slack for the entire tentative block. All rows
        # can use the same causal paged-decode kernel as autoregressive decode.
        return self.tentative or all(request.length == 1 for request in self.requests)

    @property
    def max_length(self):
        return max((request.length for request in self.requests), default=0)


def prepare_batch_step(
    requests,
    device,
    *,
    tentative=False,
    buffers=None,
    running_bs=None,
    running_tokens=None,
    state_slot_out=None,
):
    """Stage request metadata using the same persistent buffers/layout as V4.

    The serving builder owns buffers; isolated cache callers may allocate private
    ones. CPU request spans remain available for Engram and state lifecycle work.
    Existing eager consumers see active views; the backing token map uses V4's
    -1 padding sentinel and block-table stride stays fixed across steps.
    """
    scheduled_bs = len(requests)
    lengths = np.asarray([span.length for span in requests], dtype=np.int32)
    scheduled_tokens = int(lengths.sum())
    running_bs = scheduled_bs if running_bs is None else running_bs
    running_tokens = scheduled_tokens if running_tokens is None else running_tokens
    if running_bs < scheduled_bs or running_tokens < scheduled_tokens:
        raise ValueError("Request metadata exceeds the declared batch/token capacity")
    if buffers is None:
        width = max((len(span.block_ids) for span in requests), default=0)
        shapes = {
            "positions": (running_tokens,),
            "cu_seqlens_q": (running_bs + 1,),
            "batch_id_per_q_token": (running_tokens,),
            "block_tables": (running_bs, width),
        }
        buffers = {
            name: CpuGpuBuffer(
                *shape,
                dtype=torch.int32,
                device=device,
                pin_memory=torch.device(device).type != "cpu",
            )
            for name, shape in shapes.items()
        }
    required = {
        "positions": running_tokens,
        "cu_seqlens_q": running_bs + 1,
        "batch_id_per_q_token": running_tokens,
        "block_tables": running_bs,
    }
    for name, count in required.items():
        if count > buffers[name].np.shape[0]:
            raise ValueError(f"{name} metadata buffer cannot hold {count} rows")
    cu = buffers["cu_seqlens_q"]
    cu.np[0] = 0
    np.cumsum(lengths, out=cu.np[1 : scheduled_bs + 1])
    cu.np[scheduled_bs + 1 : running_bs + 1] = scheduled_tokens
    positions = buffers["positions"]
    prefill_positions(
        np.arange(scheduled_tokens, dtype=positions.np.dtype),
        np.asarray([span.position for span in requests], dtype=positions.np.dtype),
        cu.np[: scheduled_bs + 1],
        lengths,
        out=positions.np[:scheduled_tokens],
    )
    positions.np[scheduled_tokens:running_tokens] = 0
    batches = buffers["batch_id_per_q_token"]
    build_batch_ids(lengths, pad_to=running_tokens, out=batches.np)
    if state_slot_out is None:
        # Isolated eager cache callers have no metadata builder. Serving passes
        # the already-published V4 state_slot_out view; it is never restaged here.
        state_slot_out = torch.tensor(
            [span.slot for span in requests], dtype=torch.int32, device=device
        )
    tables = buffers["block_tables"]
    if scheduled_bs:
        pack_rows(
            tables.np, [np.asarray(span.block_ids, dtype=np.int32) for span in requests]
        )
    tables.np[scheduled_bs:running_bs] = 0
    published = {
        name: buffers[name].copy_to_gpu(count) for name, count in required.items()
    }
    pos = published["positions"][:scheduled_tokens]
    ptr = published["cu_seqlens_q"][: scheduled_bs + 1]
    local_steps = tuple(
        AttentionStep(
            span.position,
            span.length,
            pos[span.token_slice],
            ptr[i : i + 2] - span.offset,
        )
        for i, span in enumerate(requests)
    )
    return BatchStep(
        requests,
        pos,
        ptr,
        state_slot_out[:scheduled_bs],
        published["batch_id_per_q_token"][:scheduled_tokens],
        published["block_tables"][:scheduled_bs],
        local_steps,
        tentative=tentative,
    )
