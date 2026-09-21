# SPDX-License-Identifier: MIT
"""Request spans shared by CSA2 paging, compression and Engram staging."""

from dataclasses import dataclass, field

import numpy as np
import torch

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
    """One forward's shape, in rows the kernels run rather than tokens owned.

    Every tensor here spans the forward's own width -- `running_tokens` rows
    and `running_bs` requests -- and not the scheduled batch, because a
    captured graph replays the width it was captured at whatever the batch
    turns out to be. The tail past `scheduled` is padding: a token there
    carries batch id -1, which is what the scatters bail on, and a request
    there is zero-length in `cu_seqlens_q`, which is what the per-request
    kernels bail on.
    """

    requests: tuple[RequestSpan, ...]
    positions: torch.Tensor
    cu_seqlens_q: torch.Tensor
    slots: torch.Tensor
    batch_ids: torch.Tensor
    block_tables: torch.Tensor
    # Tokens the requests own, against `width` rows the forward runs.
    scheduled: int = 0
    # Rows per request this forward runs -- the CUDAGraph query bucket, not
    # the longest request in the batch. A ragged verify step whose longest
    # request is shorter still replays the bucket's graph, so every shape
    # derived from it has to be the bucket's.
    max_q_len: int = 0
    # Everything below is one forward's, not one layer's. `visible` and
    # `indptrs` are filled into fixed addresses before any layer runs; the
    # rest are filled by the layer that gets there first and dropped by
    # `begin_forward`.
    selected: dict[int, torch.Tensor] = field(default_factory=dict)
    candidates: dict[int, torch.Tensor] = field(default_factory=dict)
    tiles: dict[int, torch.Tensor] = field(default_factory=dict)
    indptrs: dict[int, tuple] = field(default_factory=dict)
    # ratio -> [width] int32: compressed rows each query row may see. Worked
    # out on the host, where `positions` is staged and where RoPE takes its
    # own, so this adds no second source of truth.
    visible: dict[int, torch.Tensor] = field(default_factory=dict)
    # ratio -> CompressPlan. One per distinct compression ratio in the model,
    # built once per forward and read by every owner that shares that ratio.
    plans: dict[int, object] = field(default_factory=dict)
    tentative: bool = False
    # Where each request starts, on the host. Built for `prefill_positions`
    # anyway, and published so the state lifecycle compares against the same
    # array rather than walking the spans again per forward.
    request_positions: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )

    def begin_forward(self):
        """Drop what the last forward over this step worked out.

        A capture runs the model twice on one step, so a table surviving into
        the recorded pass is a kernel that pass skips -- absent from the graph,
        and read at capture-time values on every replay.
        """
        self.selected.clear()
        self.candidates.clear()
        self.tiles.clear()

    @property
    def width(self):
        return self.positions.numel()

    @property
    def scheduled_bs(self):
        return len(self.requests)

    @property
    def decode(self):
        # Verification has ring slack for the entire tentative block. All rows
        # can use the same causal paged-decode kernel as autoregressive decode.
        return self.tentative or all(request.length == 1 for request in self.requests)


def visible_buffer_name(ratio):
    """One spelling for the buffer the builder declares and this module fills."""
    return f"v41_index_visible_{ratio}"


def _publish_block_tables(tables, requests, running_bs):
    """Reuse the fixed device page table until its rows or padding change.

    Every V4.1 publication, including dummy/capture batches, goes through here.
    DSpark borrows padding entries only temporarily and restores them before
    returning. Cache the mapping, not positions: advancing within an allocated
    page does not change an address in this table.
    """
    rows = tuple(tuple(span.block_ids) for span in requests)
    key = (running_bs, rows)
    previous = getattr(tables, "_v41_published_rows", None)
    if previous is not None and previous[0] is tables.gpu and previous[1] == key:
        return tables.gpu[:running_bs]
    if rows:
        pack_rows(tables.np, [np.asarray(row, dtype=np.int32) for row in rows])
    tables.np[len(rows) : running_bs] = 0
    published = tables.copy_to_gpu(running_bs)
    tables._v41_published_rows = (tables.gpu, key)
    return published


def prepare_batch_step(
    requests,
    device,
    *,
    tentative=False,
    buffers=None,
    running_bs=None,
    running_tokens=None,
    max_q_len=None,
    state_slot_out=None,
    ratios=(),
):
    """Stage request metadata using the same persistent buffers/layout as V4.

    The serving builder owns buffers; isolated cache callers may allocate private
    ones. CPU request spans remain available for Engram and state lifecycle work.
    The published views span the forward's full width, padding included, since
    that is the width its kernels run; the backing token map uses V4's -1
    padding sentinel and block-table stride stays fixed across steps.
    """
    scheduled_bs = len(requests)
    lengths = np.asarray([span.length for span in requests], dtype=np.int32)
    scheduled_tokens = int(lengths.sum())
    running_bs = scheduled_bs if running_bs is None else running_bs
    running_tokens = scheduled_tokens if running_tokens is None else running_tokens
    if running_bs < scheduled_bs or running_tokens < scheduled_tokens:
        raise ValueError("Request metadata exceeds the declared batch/token capacity")
    if max_q_len is not None and lengths.size and max_q_len < int(lengths.max()):
        raise ValueError("A request is longer than the query width this forward runs")
    if buffers is None:
        width = max((len(span.block_ids) for span in requests), default=0)
        shapes = {
            "positions": (running_tokens,),
            "cu_seqlens_q": (running_bs + 1,),
            "batch_id_per_q_token": (running_tokens,),
            "block_tables": (running_bs, width),
            **{visible_buffer_name(ratio): (running_tokens,) for ratio in ratios},
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
        **{visible_buffer_name(ratio): running_tokens for ratio in ratios},
    }
    for name, count in required.items():
        if count > buffers[name].np.shape[0]:
            raise ValueError(f"{name} metadata buffer cannot hold {count} rows")
    cu = buffers["cu_seqlens_q"]
    cu.np[0] = 0
    np.cumsum(lengths, out=cu.np[1 : scheduled_bs + 1])
    cu.np[scheduled_bs + 1 : running_bs + 1] = scheduled_tokens
    positions = buffers["positions"]
    starts = np.asarray([span.position for span in requests], dtype=positions.np.dtype)
    prefill_positions(
        np.arange(scheduled_tokens, dtype=positions.np.dtype),
        starts,
        cu.np[: scheduled_bs + 1],
        lengths,
        out=positions.np[:scheduled_tokens],
    )
    positions.np[scheduled_tokens:running_tokens] = 0
    for ratio in ratios:
        # Padding rows hold position 0 and earn that position's count; the
        # scorer reaches them with no visible columns either way. Written
        # through the destination, which narrows serving's int64 positions to
        # the int32 the scorer reads and spares an expression's temporaries.
        visible = buffers[visible_buffer_name(ratio)].np[:running_tokens]
        np.add(positions.np[:running_tokens], 1, out=visible)
        np.floor_divide(visible, ratio, out=visible)
    batches = buffers["batch_id_per_q_token"]
    build_batch_ids(lengths, pad_to=running_tokens, out=batches.np)
    if state_slot_out is None:
        # Isolated eager cache callers have no metadata builder. Serving passes
        # the already-published V4 state_slot_out view; it is never restaged here.
        # Padded to the same width serving publishes, so a caller that asks for
        # a wider forward than its batch gets the shape the kernels will see.
        state_slot_out = torch.tensor(
            [span.slot for span in requests] + [0] * (running_bs - scheduled_bs),
            dtype=torch.int32,
            device=device,
        )
    published = {
        name: buffers[name].copy_to_gpu(count)
        for name, count in required.items()
        if name != "block_tables"
    }
    published["block_tables"] = _publish_block_tables(
        buffers["block_tables"], requests, running_bs
    )
    return BatchStep(
        requests,
        published["positions"],
        published["cu_seqlens_q"],
        state_slot_out[:running_bs],
        published["batch_id_per_q_token"],
        published["block_tables"],
        scheduled=scheduled_tokens,
        max_q_len=(
            max((span.length for span in requests), default=0)
            if max_q_len is None
            else max_q_len
        ),
        tentative=tentative,
        visible={ratio: published[visible_buffer_name(ratio)] for ratio in ratios},
        request_positions=starts,
    )
