# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What the indexer scores against, and the step's query rows as it scores
them. The attention builder binds the first and publishes the second once a
step (`attn_metadata.index_rows`); a model reads both and computes neither."""

from typing import Any, Literal, NamedTuple

import torch
from aiter.ops.opus.pa_mqa_logits_mxfp4 import (
    mqa_logits_mxfp4_config,
    mqa_logits_mxfp4_schedule,
)

from atom.model_ops.fp4_mqa_ragged_metadata import Fp4MqaRaggedMetadata
from atom.model_ops.sparse_indexer_chunk import (
    decode_band_rows,
    logits_rows,
    qsplit_rows,
)

PRESHUFFLE = "preshuffle"
# gfx1250: FP4 rows in natural order, scored by the OPUS kernels.
NATURAL = "natural"
# gfx950 OPUS: MFMA-permuted scales, distinct from FlyDSL's preshuffle.
MFMA = "mfma"

# ATOM shares a TABLE schedule across V4 layers only for short decode groups.
# Longer groups keep the config-selected mode; this is not a prefill detector.
_V4_SHARED_SCHEDULE_MAX_QLEN = 8


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


def _check_split_group(group: str) -> None:
    if group not in ("dcp", "tp"):
        raise ValueError(f"RowSplit group must be 'dcp' or 'tp', got {group!r}")


class RowSplit(NamedTuple):
    """Rows split across ranks that each score against the whole key set:
    this rank scores `[first, end)` into its slot of `stride` rows at `slot`,
    and the `world` slots are all-gathered back in place, row i landing at
    row i (`qsplit_rows`). `group` names the ranks: "dcp" (GLM under DCP,
    which all-gathers the keys) or "tp" (a TP-replicated indexer, V4 under
    plain TP). Without a split, a world of one: every row, nothing to
    gather.

    `stripe`: the rows are dealt round-robin instead (`striped`), and the
    rows' metadata holds this rank's alone, `[first, end)` all of them."""

    first: int
    end: int
    slot: int
    stride: int
    world: int
    group: Literal["dcp", "tp"] = "dcp"
    stripe: bool = False

    @classmethod
    def of(
        cls,
        rows: int,
        rank: int = 0,
        world: int = 1,
        *,
        group: Literal["dcp", "tp"] = "dcp",
    ) -> "RowSplit":
        """This rank's split of `rows`."""
        _check_split_group(group)
        stride, slot, first, end = qsplit_rows(rows, rank, world)
        return cls(first, end, slot, stride, world, group)

    @classmethod
    def striped(
        cls, rows: int, rank: int, world: int, *, group: Literal["dcp", "tp"] = "tp"
    ) -> "RowSplit":
        """This rank's rows of `rows` dealt round-robin: rows rank,
        rank + world, ... A prefill row's cost grows with its position, so
        contiguous shares leave the last rank the most work; striped ones
        even it out. The gathered slots stay rank-major (`score_topk`)."""
        _check_split_group(group)
        stride = -(-rows // world)
        count = len(range(rank, rows, world))
        return cls(0, count, rank * stride, stride, world, group, stripe=True)

    @property
    def rank(self) -> int:
        """This rank's place in the group (0 for a step of no rows)."""
        return self.slot // self.stride if self.stride else 0


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
    `split`: the bands are this rank's rows alone (`RowSplit`).
    `workspace`: the plane the bands were cut to (`LogitsWorkspace`, or one
    that also packs); None, a plane allocated as the rows are scored."""

    row_ends: torch.Tensor
    width: int
    bands: tuple[tuple[int, int], ...]
    split: RowSplit
    row_tables: torch.Tensor | None = None
    fp4_bands: tuple | None = None
    packed_offsets: torch.Tensor | None = None
    concat: ConcatKeys | None = None
    next_n: int = 1
    workspace: Any = None

    @property
    def out_rows(self) -> int:
        """Rows of the selection: every rank's whole slot. Under a split those
        past the step's rows are slots' unscored tails, never used --
        between slots, under a striped split, whose rows stay rank-major."""
        return self.split.stride * self.split.world


class Fp4LaunchStorage:
    """Fixed homes for a recorded decode's per-step FP4 launch arguments: a
    captured graph replays the addresses it recorded, so each band's query
    starts and OPUS schedule are rewritten in place every step rather than
    allocated afresh (`fp4_rows`). One per set of rows live at once -- a TBO
    ubatch has its own."""

    def __init__(self, device, rows: int, sequences: int):
        self.device, self.rows, self.sequences = device, rows, sequences
        self._starts: dict[int, torch.Tensor] = {}
        self._schedules: dict[tuple[int, Any], Any] = {}

    def starts(self, band: int) -> torch.Tensor:
        """Band `band`'s query starts, `[sequences + 1]`."""
        if band not in self._starts:
            self._starts[band] = torch.empty(
                self.sequences + 1, dtype=torch.int32, device=self.device
            )
        return self._starts[band]

    def schedule_buffers(self, band: int, cfg):
        """One band's buffers for this config; never shared between ubatches."""
        key = (band, cfg)
        if key not in self._schedules:
            self._schedules[key] = cfg.schedule_buffers()
        return self._schedules[key]


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
        RowSplit.of(rows),
        row_tables=row_tables,
        next_n=next_n,
        workspace=workspace,
    )


def concat_rows(row_ends, keys: ConcatKeys, split: RowSplit | None = None):
    """FP8 prefill rows against their requests' gathered keys, banded by the
    logits budget alone: a prefill's temporaries are allocated as it runs.
    This rank's rows of them, `split` (every row by default)."""
    split = split or RowSplit.of(row_ends.shape[0])
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
    is_decode: bool = False,
    row_tables: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    bands: tuple[tuple[int, int], ...] | None = None,
    packed_offsets: torch.Tensor | None = None,
    split: RowSplit | None = None,
    storage: Fp4LaunchStorage | None = None,
) -> IndexRows:
    """FP4 rows, a sequence's together (`sequences`), each band's launch
    prepared here, once: the row-group plan (preshuffle) or the OPUS config
    (natural/MFMA). `is_decode` enables a shared schedule for short V4 decode
    groups; prefill (including short chunks) and V4.1 keep the config's mode.
    `page_size` is the cache's index rows a block. Without `sequences` each
    row is a sequence of its own on its own `row_tables` row, a band's
    starting at `row_starts` (`[rows + 1]`, 0 .. rows). Rows a graph records
    keep each launch's per-step arguments in `storage`.

    Banded over the workspace's plane (this rank's rows of it, `split`,
    every row by default) unless `bands` are given -- with `packed_offsets`,
    where each band's rows were packed."""
    rows = row_ends.shape[0]
    split = split or RowSplit.of(rows)
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
        if layout in (NATURAL, MFMA):
            cfg = mqa_logits_mxfp4_config(
                row_ends.device,
                heads=heads,
                page_size=page_size,
                pages_per_block=band.pages_per_block,
                max_rows=end - first,
                max_seqs=band.query_start_loc.numel() - 1,
                max_qlen=band.max_qlen,
                max_seq_len=width,
            )
            # V4 amortizes one decode schedule across its indexer layers.
            # Prefill and V4.1 use the config-selected schedule-free mode.
            schedule = None
            if (
                is_decode
                and heads == 64
                and band.max_qlen <= _V4_SHARED_SCHEDULE_MAX_QLEN
            ):
                schedule = mqa_logits_mxfp4_schedule(
                    cfg,
                    band.query_start_loc,
                    ends,
                    buffers=(
                        None
                        if storage is None
                        else storage.schedule_buffers(index, cfg)
                    ),
                )
            table = band.block_tables
            if table.shape[1] < cfg.block_table_width:
                table = torch.nn.functional.pad(
                    table, (0, cfg.block_table_width - table.shape[1])
                )
            launches.append(
                {
                    "cfg": cfg,
                    "schedule": schedule,
                    "block_tables": table,
                    "query_start_loc": band.query_start_loc,
                    "row_ends": ends,
                }
            )
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
