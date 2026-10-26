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

import torch

from atom.utils import envs

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


def dcp_qsplit_rows(
    num_rows: int, rank: int, world_size: int
) -> tuple[int, int, int, int]:
    """This rank's slot in the gathered buffer and the rows it scores into it.

    Under DCP every rank already all-gathers the whole index cache, so having
    all ``world_size`` of them score every row recomputes the same top-k that
    many times. Each rank scores ``[row_lo, row_hi)`` instead, against the full
    key set, and the selections are all-gathered back.

    The stride is uniform -- ``ceil(num_rows / world_size)`` for every rank --
    so a rank-major concatenation of stride-sized buffers lands row ``i`` back
    at row ``i``, with no index exchange. The price is that the last rank's
    buffer has a tail of unwritten rows: the gathered payload is
    ``world_size * ceil(num_rows / world_size)`` rows, at most
    ``world_size - 1`` more than ``num_rows``. That tail sits at gathered
    indices ``>= num_rows`` and is never read back.

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
