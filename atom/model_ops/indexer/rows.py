# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the indexer scores against, and the step's query rows as it scores
them. The attention builder binds the first and publishes the second once a
step (`attn_metadata.index_rows`); a model reads both and computes neither."""

from typing import Any, NamedTuple

import torch
from aiter.ops.opus.pa_mqa_logits_mxfp4 import (
    pa_mqa_logits_mxfp4_plan,
    pa_mqa_logits_mxfp4_plan_buffers,
)

from atom.model_ops.fp4_mqa_ragged_metadata import Fp4MqaRaggedMetadata
from atom.model_ops.sparse_indexer_chunk import (
    dcp_qsplit_rows,
    decode_band_rows,
    logits_rows,
)

PRESHUFFLE = "preshuffle"
# gfx1250: FP4 rows in natural order, scored by the OPUS kernels.
NATURAL = "natural"


class IndexCache(NamedTuple):
    """A layer's index keys: FP8 `values` with their scales inline, or FP4
    `values` and `scales` planes, paged at `rows_per_block` index rows and
    laid out as `layout` says."""

    values: torch.Tensor
    scales: torch.Tensor | None
    rows_per_block: int
    layout: str = PRESHUFFLE

    @property
    def fp4(self) -> bool:
        return self.scales is not None


class ConcatKeys(NamedTuple):
    """FP8 prefill: the step's requests' committed keys gathered end to end
    (`cu_committed` [requests + 1], `total` rows) and scored densely, row r
    over `[row_starts[r], row_starts[r] + row_ends[r])` of them."""

    block_tables: torch.Tensor
    cu_committed: torch.Tensor
    total: int
    row_starts: torch.Tensor


class DcpSplit(NamedTuple):
    """Rows under DCP, where every rank scores against the whole key set: this
    rank scores `[first, end)` into its slot of `stride` rows at `slot`, and
    the `world` slots are all-gathered back in place, row i landing at row i
    (`dcp_qsplit_rows`). Without DCP, a world of one: every row, nothing to
    gather."""

    first: int
    end: int
    slot: int
    stride: int
    world: int

    @classmethod
    def of(cls, rows: int, rank: int = 0, world: int = 1) -> "DcpSplit":
        """This rank's split of `rows`."""
        stride, slot, first, end = dcp_qsplit_rows(rows, rank, world)
        return cls(first, end, slot, stride, world)


class IndexRows(NamedTuple):
    """A step's index query rows. Row r sees columns `[0, row_ends[r])` of its
    request's keys and is scored in one of `bands` (row ranges) into a
    `width`-column plane.

    Paged rows: FP8 reads a row's keys off its own `row_tables` row -- or,
    with `next_n` > 1, a request's `next_n` consecutive rows read their
    request's keys together, `row_ends` and `row_tables` then a request each
    and the bands whole requests; FP4 scores a request's rows together, band i
    launched as `fp4_bands[i]` holds it. `packed_offsets`: FP4 rows packed
    into the workspace's flat logits, each at its offset from its band's
    start, a band holding as many as fit. `concat`: FP8 prefill rows against
    their requests' gathered keys (`ConcatKeys`), `width` then their total.
    `split`: the bands are this DCP rank's rows alone (`DcpSplit`).
    `workspace`: the plane the bands were cut to (`LogitsWorkspace`, or one
    that also packs); None, a plane allocated as the rows are scored."""

    row_ends: torch.Tensor
    width: int
    bands: tuple[tuple[int, int], ...]
    split: DcpSplit
    row_tables: torch.Tensor | None = None
    fp4_bands: tuple | None = None
    packed_offsets: torch.Tensor | None = None
    concat: ConcatKeys | None = None
    next_n: int = 1
    workspace: Any = None

    @property
    def out_rows(self) -> int:
        """Rows of the selection: every rank's whole slot, under DCP the last
        one's past the rows."""
        return self.split.stride * self.split.world


class Fp4LaunchStorage:
    """Fixed homes for a recorded decode's per-step FP4 launch arguments: a
    captured graph replays the addresses it recorded, so each band's query
    starts and OPUS plan are rewritten in place every step rather than
    allocated afresh (`fp4_rows`). One per set of rows live at once -- a TBO
    ubatch has its own."""

    def __init__(self, device, rows: int, sequences: int):
        self.device, self.rows, self.sequences = device, rows, sequences
        self._starts: dict[int, torch.Tensor] = {}
        self._plans: dict[tuple[int, str], Any] = {}

    def starts(self, band: int) -> torch.Tensor:
        """Band `band`'s query starts, `[sequences + 1]`."""
        if band not in self._starts:
            self._starts[band] = torch.empty(
                self.sequences + 1, dtype=torch.int32, device=self.device
            )
        return self._starts[band]

    def plan_buffers(self, band: int, max_qlen: int):
        """Band `band`'s OPUS plan, the variant a sequence of `max_qlen` rows
        launches."""
        variant = "qlen1_kv64" if max_qlen == 1 else "qlen4_kv64"
        if (band, variant) not in self._plans:
            self._plans[band, variant] = pa_mqa_logits_mxfp4_plan_buffers(
                self.device, self.rows, self.sequences, variant=variant
            )
        return self._plans[band, variant]


class Fp4LaunchStorages(dict):
    """Each set of live rows' own `Fp4LaunchStorage` -- keyed by its ubatch
    prefix, "" for the whole step -- made on first use."""

    def __init__(self, device, rows: int, sequences: int):
        super().__init__()
        self.device, self.rows, self.sequences = device, rows, sequences

    def __missing__(self, ubatch: str) -> Fp4LaunchStorage:
        storage = self[ubatch] = Fp4LaunchStorage(
            self.device, self.rows, self.sequences
        )
        return storage


def _uniform_bands(first, end, band):
    return tuple((start, min(start + band, end)) for start in range(first, end, band))


def _plane_bands(first, end, width, workspace, group=1):
    """Bands of rows `[first, end)` in a plane, whole groups of `group` rows:
    as many as the workspace's plane holds (the logits budget's, without one)
    -- a count that follows the rows and the width alone, so a token graph
    bucket fixes it."""
    rows = end - first
    band = (
        decode_band_rows(rows, width, group)
        if workspace is None
        else workspace.logits_rows(rows, width, group)
    )
    return _uniform_bands(first, end, band)


def fp8_rows(row_ends, width, workspace, row_tables, *, next_n=1) -> IndexRows:
    """FP8 paged rows: row r's keys are listed by `row_tables[r]`, or with
    `next_n` > 1 request b's `next_n` rows' by `row_tables[b]`."""
    rows = row_ends.shape[0] * next_n
    return IndexRows(
        row_ends,
        width,
        _plane_bands(0, rows, width, workspace, next_n),
        DcpSplit.of(rows),
        row_tables=row_tables,
        next_n=next_n,
        workspace=workspace,
    )


def concat_rows(row_ends, keys: ConcatKeys, split: DcpSplit | None = None):
    """FP8 prefill rows against their requests' gathered keys, banded by the
    logits budget alone: a prefill's temporaries are allocated as it runs.
    This DCP rank's rows of them, `split` (every row by default)."""
    split = split or DcpSplit.of(row_ends.shape[0])
    band = max(1, logits_rows(split.end - split.first, keys.total))
    return IndexRows(
        row_ends,
        keys.total,
        _uniform_bands(split.first, split.end, band),
        split,
        concat=keys,
    )


def fp4_rows(
    row_ends,
    width,
    workspace,
    sequences: Fp4MqaRaggedMetadata | None,
    *,
    heads: int,
    page_size: int,
    layout: str = PRESHUFFLE,
    row_to_batch: torch.Tensor | None = None,
    row_tables: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    bands: tuple[tuple[int, int], ...] | None = None,
    packed_offsets: torch.Tensor | None = None,
    split: DcpSplit | None = None,
    storage: Fp4LaunchStorage | None = None,
) -> IndexRows:
    """FP4 rows, a sequence's together (`sequences`), each band's launch
    planned here, once: the row-group plan (preshuffle) or the OPUS plan
    (natural, which also takes each row's sequence, `row_to_batch`).
    `page_size` is the cache's index rows a block. Without `sequences` each
    row is a sequence of its own on its own `row_tables` row, a band's
    starting at `row_starts` (`[rows + 1]`, 0 .. rows). Rows a graph records
    keep each launch's per-step arguments in `storage`.

    Banded over the workspace's plane (this DCP rank's rows of it, `split`,
    every row by default) unless `bands` are given -- with `packed_offsets`,
    where each band's rows were packed."""
    rows = row_ends.shape[0]
    split = split or DcpSplit.of(rows)
    if bands is None:
        bands = _plane_bands(split.first, split.end, width, workspace)
    launches = []
    for index, (first, end) in enumerate(bands):
        ends = row_ends[first:end]
        band = (
            Fp4MqaRaggedMetadata(
                row_starts[: end - first + 1], 1, row_tables[first:end]
            )
            if sequences is None
            else sequences.band(
                first,
                end - first,
                rows,
                out=None if storage is None else storage.starts(index),
            )
        )
        if layout == NATURAL:
            plan = pa_mqa_logits_mxfp4_plan(
                band.query_start_loc,
                ends,
                buffers=(
                    None
                    if storage is None
                    else storage.plan_buffers(index, band.max_qlen)
                ),
                total_q=end - first,
                row_to_batch=row_to_batch[first:end],
            )
            launches.append({"block_tables": band.block_tables, "plan": plan})
        else:
            launches.append(
                band.kernel_args(
                    ends, heads=heads, page_size=page_size, max_seq_len=width
                )
            )
    return IndexRows(
        row_ends,
        width,
        bands,
        split,
        fp4_bands=tuple(launches),
        packed_offsets=packed_offsets,
        workspace=workspace,
    )
