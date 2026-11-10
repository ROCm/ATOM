# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Row chunking for the dense ``fp8_mqa_logits`` indexer logits buffer.

Every sparse-MLA indexer prefill path (native DeepSeek V3.2/V4, GLM-5.x, and the
vLLM plugin) scores queries against the committed KV through a dense
``[rows, row_width]`` fp32 logits matrix. ``row_width`` is the sum of all
co-scheduled prefill contexts, which ``max_num_batched_tokens`` does not bound,
so a burst of long-context requests can push one allocation to tens of GiB
(issue #1376). Those paths therefore split the Q rows into chunks; this module
owns the one rule they all need to agree on.
"""

import logging

import numpy as np
import torch

from atom.utils import envs

logger = logging.getLogger("atom")

# A buffer resource descriptor addresses at most 2^31 bytes, so aiter's
# fp8_mqa_logits drops to plain global load/store once the logits tensor reaches
# 2 GiB. On gfx950 that USE_BUFFER_STORE=False specialization of the gluon kernel
# does not survive codegen -- the AMDGCN backend trips
# `llvm/ADT/Sequence.h: Assertion 'Begin <= End'` and abort()s the process, with
# no Python traceback and every TP rank dying at once. Landing exactly on the cap
# is easy to do by accident: 16384 rows x 32768 committed tokens x 4 B is 2 GiB
# to the byte. Keep the buffer STRICTLY under the cap so the buffer-store
# specialization is the only one ever compiled.
_BUFFER_DESCRIPTOR_LIMIT_BYTES = 2 * 1024 * 1024 * 1024

_LOGIT_BYTES = 4  # fp32
# The kernels tile rows by 128; keep chunk sizes on that grid.
_ROW_TILE = 128


def sparse_indexer_row_chunk(total_rows: int, row_width: int, budget_mb: int) -> int:
    """Rows to score per ``fp8_mqa_logits`` call for a ``[rows, row_width]`` buffer.

    ``budget_mb`` is the caller's soft byte budget (``0`` = no soft budget); the
    2 GiB buffer-descriptor cap applies either way, because exceeding it is a
    hard crash rather than a memory-pressure trade-off.

    Returns ``total_rows`` when one shot already fits. Otherwise the budget-derived
    row count rounded DOWN: to a multiple of 128 in the normal regime, avoiding
    coarse power-of-2 doubling; below 128 rows (extreme ``row_width``) to a
    power-of-2 floor, so it degrades 64/32/.../1 instead of collapsing to 1.
    """
    if total_rows <= 0 or row_width <= 0:
        return max(total_rows, 1)

    budget_bytes = budget_mb * 1024 * 1024
    cap_bytes = _BUFFER_DESCRIPTOR_LIMIT_BYTES
    if budget_bytes > 0:
        cap_bytes = min(budget_bytes, cap_bytes)

    # Exclusive bound: the kernel needs bytes < cap, not <=.
    max_rows = (cap_bytes - 1) // (row_width * _LOGIT_BYTES)
    if max_rows >= total_rows:
        return total_rows
    if max_rows >= _ROW_TILE:
        return (max_rows // _ROW_TILE) * _ROW_TILE
    return 1 << (max(1, max_rows).bit_length() - 1)


def plane_rows(width):
    """Query rows a `width`-column logits plane may hold at once.

    Its readers reach a row as `row * stride` in int32, so a plane past 2**31
    elements wraps to a negative address. `width` follows the model-length cap
    and not the live context, so only a long-context configuration can get
    there; wherever a batch already fits, this leaves it in one piece.
    """
    return max(1, (2**31 - 1) // width)


def logits_rows(rows, width):
    """One logits band's rows, shared by allocation and execution."""
    return min(
        plane_rows(width),
        sparse_indexer_row_chunk(
            rows, width, envs.ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB
        ),
    )


def capped_logits_rows(rows, width, capacity, min_rows=1):
    """`logits_rows` within a workspace of `capacity` rows of `width`; fails
    when it cannot hold `min_rows`."""
    if capacity < min_rows:
        raise ValueError(
            f"logits workspace holds {capacity} rows of {width} columns, "
            f"fewer than {min_rows}"
        )
    return min(logits_rows(rows, width), capacity)


def decode_band_rows(rows, width, next_n, capacity=None):
    """Rows per decode band: `logits_rows` rounded down to whole requests of
    `next_n` rows, never below one request, and within `capacity` rows when
    given. A band that split a request would mis-slice its query and lengths.
    No rows still gives a positive band, so a loop over them is empty."""
    if rows <= 0:
        return next_n
    band = (
        logits_rows(rows, width)
        if capacity is None
        else capped_logits_rows(rows, width, capacity, next_n)
    )
    return min(rows, max(next_n, band // next_n * next_n))


def flat_view(flat, rows, width):
    """`[rows, width]` view of the front of a flat buffer; fails rather than
    growing past it."""
    if rows * width > flat.numel():
        raise ValueError(
            f"a [{rows}, {width}] scorer temporary exceeds its workspace of "
            f"{flat.numel()} elements"
        )
    return flat[: rows * width].view(rows, width)


def qsplit_rows(num_rows: int, rank: int, world_size: int) -> tuple[int, int, int, int]:
    """This rank's slot in the gathered buffer and the rows it scores into it.

    Under DCP every rank already all-gathers the whole index cache, so having
    all ``world_size`` of them score every row recomputes the same top-k that
    many times. Each rank scores ``[row_lo, row_hi)`` instead, against the full
    key set, and the selections are all-gathered back. (V4's plain-TP split
    deals its rows round-robin instead: `RowSplit.striped`.)

    The stride is uniform -- ``ceil(num_rows / world_size)`` for every rank --
    so a rank-major concatenation of stride-sized buffers lands row ``i`` back
    at row ``i``, with no index exchange. The price is that the last ranks'
    buffers can have tails of unscored rows: the gathered payload is
    ``world_size * ceil(num_rows / world_size)`` rows, at most
    ``world_size - 1`` more than ``num_rows``. Those tails sit at gathered
    indices ``>= num_rows`` and are never read back.

    ``slot_lo`` is where this rank's stride sits in the gathered buffer, and it
    is deliberately NOT clamped, while ``row_lo``/``row_hi`` -- the rows to
    score -- are. The in-place all-gather requires its input to alias the
    CALLING rank's own output slot; a rank with no rows clamps to ``num_rows``,
    which lands inside a LOWER rank's slot, and that overlap is not a layout
    RCCL supports. So the two have to be separate numbers rather than one
    serving both.

    ``row_lo == row_hi`` when a rank has no rows, which is exactly when
    ``slot_lo >= num_rows``. That is not the same as ``num_rows < world_size``:
    at ``num_rows=12, world_size=8`` the stride is 2 and the last two ranks are
    already empty. At ``world_size == 1`` the range is the whole of it.
    """
    rows_per_rank = -(-num_rows // world_size)
    slot_lo = rank * rows_per_rank
    row_lo = min(slot_lo, num_rows)
    row_hi = min(slot_lo + rows_per_rank, num_rows)
    return rows_per_rank, slot_lo, row_lo, row_hi


def indexer_qsplit_world(
    tp: int,
    *,
    physical_tp: int,
    enabled: bool,
    gfx1250_opus: bool,
    dp_attention: bool,
    tbo: bool,
    pcp: int,
    dcp: int,
    kv_consumer: bool,
) -> int:
    """TP width the DeepSeek-V4 prefill indexer splits its query rows over
    (`RowSplit` with `group="tp"`); 1 means no split.

    The V4 indexer is replicated across TP, so every rank would score the
    same rows. The split needs every TP rank to hold the same rows, which DP
    attention, PCP and DCP each break; a prefill TBO's ubatch rows do not
    take it yet. Simulated TP (`--fake-eplb` on fewer devices) reports a
    width the device group does not have. gfx1250 OPUS scores through its own
    plans. A P/D decode instance (`kv_consumer`) prefills only to recompute,
    too rarely to pay for the split's startup gather. Within this width a
    prefill step still splits only from `ATOM_V4_INDEXER_QSPLIT_MIN_LOGITS`
    index logits (the V4 builder's `_prefill_row_split`).
    """
    if (
        not enabled
        or tp <= 1
        or physical_tp != tp
        or gfx1250_opus
        or dp_attention
        or tbo
        or pcp > 1
        or dcp > 1
        or kv_consumer
    ):
        return 1
    return tp


class LogitsWorkspace:
    """`capacity` fp32 elements every indexer of a model scores its bands
    into, one layer after another: the logits plane, held for the server's
    life.

    A plane is as wide as a request may be long (`max_model_len // ratio`),
    not as its live context. Allocated per forward it is multi-GiB at a height
    that varies with the batch: the memory profile (a prefill) never sees it,
    a token graph's pool keeps the largest bucket's, and the caching allocator
    is left holding segments too small for the next long prefill. Sized from
    the configuration, before the memory profile, it is counted in it and
    never reallocated; scoring runs in bands of it."""

    def __init__(self, capacity: int, device):
        self.capacity = capacity
        self._logits = torch.empty(capacity, dtype=torch.float32, device=device)

    @classmethod
    def for_rows(cls, rows: int, width: int, device) -> "LogitsWorkspace":
        """A plane for `rows` query rows of `width` columns, in as many bands
        as the logits budget asks."""
        return cls(logits_rows(rows, width) * width, device)

    @property
    def nbytes(self) -> int:
        return self._logits.numel() * self._logits.element_size()

    def logits_rows(self, rows: int, width: int, group: int = 1) -> int:
        """Rows of a `width`-column band: within the byte budget and this
        allocation, whichever is smaller, in whole groups of `group` rows."""
        return decode_band_rows(rows, width, group, self._logits.numel() // width)

    def logits(self, rows: int, width: int) -> torch.Tensor:
        """`[rows, width]` fp32, one band of the plane; fails past it."""
        return flat_view(self._logits, rows, width)


# aiter allocates a dense prefill plane with rows padded to this many columns;
# a plane carved from a workspace keeps the same row pitch.
CONCAT_PLANE_ROW_ALIGN = 256


def concat_plane_width(keys: int) -> int:
    """Row pitch of an FP8 prefill (concat) logits band over `keys` gathered
    keys: `keys` rounded up to `CONCAT_PLANE_ROW_ALIGN`."""
    align = CONCAT_PLANE_ROW_ALIGN
    return max(align, -(-keys // align) * align)


def concat_window_bands(first, end, row_cu, key_cu, plane: LogitsWorkspace):
    """Bands of rows `[first, end)` in `plane`, each scored against a window
    `(lo, hi)` of just its own requests' keys (`row_cu`/`key_cu` per request)."""
    bands, windows = [], []
    start = first
    while start < end:
        request = int(np.searchsorted(row_cu, start, side="right")) - 1
        lo = int(key_cu[request])
        stop, hi = start, lo
        while stop < end:
            next_stop = min(int(row_cu[request + 1]), end)
            next_hi = int(key_cu[request + 1])
            rows = next_stop - start
            width = concat_plane_width(next_hi - lo)
            fit = plane.logits_rows(rows, width) if width <= plane.capacity else 0
            if fit < rows:
                if stop == start:
                    if fit < 1:
                        raise ValueError(
                            f"the logits plane holds no row of request {request}'s "
                            f"{next_hi - lo} keys"
                        )
                    stop, hi = start + fit, next_hi
                break
            stop, hi = next_stop, next_hi
            request += 1
        bands.append((start, stop))
        windows.append((lo, hi))
        start = stop
    return tuple(bands), tuple(windows)


class ConcatKeyWorkspace:
    """The FP8 prefill indexer's gathered keys (`ConcatKeys`): every scheduled
    request's committed keys end to end, as FP8 rows and their fp32 scales,
    and under DCP the all-gathered rank shards they are de-interleaved from.
    Held for the server's life, like `LogitsWorkspace`.

    Allocated per step these are as long as the step's context, so each chunk
    of a long prefill asks for a slightly larger block than the last one: the
    caching allocator never finds one to reuse and reserves another, chunk
    after chunk, until HBM runs out. Sized from the configuration before the
    memory profile, they are counted in it. A step whose keys outgrow a buffer
    doubles it -- the only allocation after startup, and a rare one: one
    request is at most the model length, a step past it is several long
    requests scheduled together."""

    def __init__(
        self, capacity: int, dim: int, dtype, device, gathered_capacity: int = 0
    ):
        self.dim, self.dtype, self.device = dim, dtype, device
        self._keys = self._buffers(capacity)
        self._gathered = self._buffers(gathered_capacity)

    def _buffers(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty((rows, self.dim), dtype=self.dtype, device=self.device),
            torch.empty((rows, 1), dtype=torch.float32, device=self.device),
        )

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self._keys, *self._gathered))

    def _grown(self, buffers, rows: int, what: str):
        held = buffers[0].shape[0]
        if rows <= held:
            return buffers
        grown = max(rows, 2 * held)
        logger.warning(
            "FP8 prefill indexer %s workspace: a step of %d keys outgrew %d; "
            "growing it to %d",
            what,
            rows,
            held,
            grown,
        )
        return self._buffers(grown)

    def keys(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`[rows, dim]` FP8 keys and `[rows, 1]` fp32 scales the scorer reads."""
        self._keys = self._grown(self._keys, rows, "key")
        values, scales = self._keys
        return values[:rows], scales[:rows]

    def gathered(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`[rows, dim]` FP8 and `[rows, 1]` fp32: DCP's all-gathered shards,
        rank-major, which `keys` is de-interleaved from."""
        self._gathered = self._grown(self._gathered, rows, "gathered-shard")
        values, scales = self._gathered
        return values[:rows], scales[:rows]
