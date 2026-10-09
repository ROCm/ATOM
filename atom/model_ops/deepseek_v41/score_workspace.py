# SPDX-License-Identifier: MIT
"""The paged index scorer's largest temporaries, held for the server's life.

Both are as wide as a request may be long (`plane_rows`), not as its live
context. Allocated per call they were multi-GiB, at a height that varies with
the batch, so the caching allocator was left holding segments too small for
the next long prefill, which then mapped new ones past the memory budget.
Sized here from the configuration, before the memory profile, they are counted
in it and never reallocated.
"""

import torch


def plane_rows(width):
    """Query rows a `width`-column logits plane may hold at once.

    Its readers reach a row as `row * stride` in int32, so a plane past 2**31
    elements wraps to a negative address. `width` follows the model-length cap
    and not the live context, so only a long-context configuration can get
    there; wherever a batch already fits, this leaves it in one piece.
    """
    return max(1, (2**31 - 1) // width)


class ScoreWorkspace:
    """The scorer's logits band and each ratio's tile table, at their largest,
    and the row starts of one-row sequences. The FP4 plane has no tile tables:
    its scorers read the PAGE table itself.

    `max_tokens` bounds a forward's query rows and `columns` a block table's
    width: the two dimensions every later request is checked against.
    """

    def __init__(self, geometry, max_tokens, columns, device):
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
        self._logits = torch.empty(
            max((min(max_tokens, plane_rows(w)) * w for w in widths), default=0),
            dtype=torch.float32,
            device=device,
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
