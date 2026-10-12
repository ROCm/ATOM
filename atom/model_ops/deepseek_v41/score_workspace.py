# SPDX-License-Identifier: MIT
"""The paged index scorer's largest temporaries, held for the server's life.

Both are as wide as a request may be long (`plane_rows`), not as its live
context. Allocated per call they were multi-GiB, at a height that varies with
the batch, so the caching allocator was left holding segments too small for
the next long prefill, which then mapped new ones past the memory budget.
Sized here from the configuration, before the memory profile, they are counted
in it and never reallocated.

On the FP4 plane the logits are packed instead: each row as long as its own
visibility, at a start of its own (`PackedRows`), so the buffer follows the
batch's live context rather than its rows times the model length.
"""

from typing import NamedTuple

import numpy as np
import torch

# A packed row starts on a 256-byte boundary, which the row-group scorer's
# stores need to run at speed, and on a whole candidate block, so its block
# maxima start at its logits' start over the block.
ROW_ALIGN = 64


def plane_rows(width):
    """Query rows a `width`-column logits plane may hold at once.

    Its readers reach a row as `row * stride` in int32, so a plane past 2**31
    elements wraps to a negative address. `width` follows the model-length cap
    and not the live context, so only a long-context configuration can get
    there; wherever a batch already fits, this leaves it in one piece.
    """
    return max(1, (2**31 - 1) // width)


def packed_capacity(geometry, pages, max_tokens, plane_elements):
    """Elements of the packed logits buffer: every decode row a PAGE pool of
    `pages` can hold seeing all of it, its rows aligned, and never more than
    the plane it replaces (`plane_elements`).

    A decode row sees at most its request's context, and the requests'
    contexts are in the pool, so a step's rows need at most `1 + drafts` rows
    a pooled token at the widest ratio. Shared prefixes are counted once in
    the pool and once a request here: a step past this is scored eagerly in
    bands (`band_offsets`), never in a captured graph.
    """
    widest = geometry.rows_per_page(min(ratio for _, ratio in geometry.owners))
    rows_per_token = 1 + geometry.speculative_tokens
    live = pages * widest * rows_per_token + max_tokens * ROW_ALIGN
    return min(live, plane_elements)


def aligned_spans(visible):
    """Each row's packed length: its visibility rounded up to `ROW_ALIGN`."""
    return (visible + ROW_ALIGN - 1) // ROW_ALIGN * ROW_ALIGN


def band_offsets(visible, capacity, out):
    """Pack rows `visible` [rows] into bands of at most `capacity` elements.

    Writes each row's start, counted from its band's, into `out` and returns
    the bands as `(first, end)` row ranges in order. A row longer than the
    capacity is refused: no band could hold it.
    """
    spans = aligned_spans(visible.astype(np.int64))
    if spans.size and int(spans.max()) > capacity:
        raise ValueError(
            f"a {int(spans.max())}-element row exceeds the {capacity}-element "
            "packed logits buffer"
        )
    ends = np.cumsum(spans)
    bands, first, base = [], 0, 0
    while first < len(spans):
        end = int(np.searchsorted(ends, base + capacity, side="right"))
        out[first:end] = ends[first:end] - spans[first:end] - base
        bands.append((first, end))
        base, first = int(ends[end - 1]), end
    return tuple(bands)


class PackedRows(NamedTuple):
    """A step's rows at one ratio laid out in the packed logits: each row's
    visibility, where its logits and its block maxima start, counted from its
    band's start, and the bands' rows (`band_offsets`)."""

    visible: torch.Tensor
    offsets: torch.Tensor
    block_offsets: torch.Tensor
    bands: tuple[tuple[int, int], ...]


class ScoreWorkspace:
    """The scorer's logits band and each ratio's tile table, at their largest,
    and the row starts of one-row sequences. The FP4 plane has no tile tables:
    its scorers read the PAGE table itself, and its logits are packed
    (`capacity` elements, `packed_logits`, laid out by `pack`) with the
    candidate pick's block maxima packed beside them.

    `max_tokens` bounds a forward's query rows and `columns` a block table's
    width: the two dimensions every later request is checked against.
    `pages` bounds the PAGE pool the FP4 plane's packed rows can see.
    """

    def __init__(self, geometry, max_tokens, columns, device, pages=None):
        ratios = sorted({ratio for _, ratio in geometry.owners})
        self._tiles = {
            ratio: torch.empty(
                max_tokens * columns * geometry.index_blocks_per_page(ratio),
                dtype=torch.int32,
                device=device,
            )
            for ratio in ([] if geometry.index_fp4 else ratios)
        }
        widths = [columns * geometry.rows_per_page(ratio) for ratio in ratios]
        plane = max((min(max_tokens, plane_rows(w)) * w for w in widths), default=0)
        self.packed = geometry.index_fp4
        if self.packed and pages is None:
            raise ValueError("the FP4 plane packs its logits: it needs the PAGEs")
        self.capacity = (
            packed_capacity(geometry, pages, max_tokens, plane)
            if self.packed
            else plane
        )
        # A step's rows see the most at the widest ratio; what they may fill is
        # what is left by the padding rows a captured graph adds, each at
        # position 0 and so seeing (0 + 1) // ratio rows.
        self._ratio = min(ratios, default=1)
        self._rows_capacity = self.capacity - max_tokens * aligned_spans(
            1 // self._ratio
        )
        self._logits = torch.empty(self.capacity, dtype=torch.float32, device=device)
        self.block_rows = geometry.index_block_rows
        self._maxima = (
            torch.empty(
                self.capacity // self.block_rows, dtype=torch.float32, device=device
            )
            if self.packed
            else None
        )
        self._row_starts = torch.arange(
            max_tokens + 1, dtype=torch.int32, device=device
        )

    def unit_table(self, ratio, tokens, width):
        """`[tokens, width]` int32 for `unit_table`'s output at this ratio."""
        if ratio not in self._tiles:
            raise ValueError("the FP4 index plane's scorers read no tile table")
        return _view(self._tiles[ratio], tokens, width)

    def logits(self, rows, width):
        """`[rows, width]` fp32, one band of the scorer's logits plane."""
        return _view(self._logits, rows, width)

    def fits(self, rows, contexts):
        """Whether a step of requests with `rows` query rows each, every row
        seeing at most its request's `contexts` tokens, packs in one band --
        all a captured graph scores. A plane always does."""
        if not self.packed:
            return True
        spans = aligned_spans(np.asarray(contexts, np.int64) // self._ratio)
        return int(np.dot(np.asarray(rows, np.int64), spans)) <= self._rows_capacity

    def band_rows(self, width):
        """Rows of a `width`-column plane one band holds."""
        return max(1, min(plane_rows(width), self.capacity // width))

    def pack(self, visible, offsets, block_offsets):
        """Lay host rows `visible` out in the packed logits: each row's logits
        start into `offsets` and its block maxima's into `block_offsets`, both
        from its band's start. Returns the bands' rows (`band_offsets`)."""
        rows = len(visible)
        cuts = band_offsets(visible, self.capacity, offsets)
        np.floor_divide(offsets[:rows], self.block_rows, out=block_offsets[:rows])
        return cuts

    def packed_logits(self):
        """The packed logits, flat: a row at its `PackedRows` offset."""
        if not self.packed:
            raise ValueError("only the FP4 plane packs its logits")
        return self._logits

    def packed_maxima(self):
        """The packed block maxima, flat: a row's at its `PackedRows` block
        offset."""
        if not self.packed:
            raise ValueError("only the FP4 plane packs its logits")
        return self._maxima

    def row_starts(self, rows):
        """`[rows + 1]` int32 0 .. rows: each of `rows` rows its own sequence."""
        return self._row_starts[: rows + 1]


def _view(flat, rows, width):
    if rows * width > flat.numel():
        raise ValueError(
            f"a [{rows}, {width}] scorer temporary exceeds its workspace of "
            f"{flat.numel()} elements"
        )
    return flat[: rows * width].view(rows, width)
