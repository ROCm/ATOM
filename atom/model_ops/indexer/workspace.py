# SPDX-License-Identifier: MIT
"""Logits storage and packed row layouts shared by DSA indexers.

Model builders decide the capacity before memory profiling. Scorers consume
views of that storage, either rectangular bands or rows packed by visibility.
PAGE-pool sizing and candidate-block storage remain with their model builder.
"""

from typing import NamedTuple

import numpy as np
import torch

from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace

__all__ = [
    "ROW_ALIGN",
    "LogitsWorkspace",
    "PackedRows",
    "aligned_spans",
    "band_offsets",
]

# FP4 row-group stores need 256-byte alignment: 64 fp32 logits.
ROW_ALIGN = 64


def aligned_spans(visible):
    """Each packed row's visibility rounded up to its store alignment."""
    return (visible + ROW_ALIGN - 1) // ROW_ALIGN * ROW_ALIGN


def band_offsets(visible, capacity, out):
    """Pack visible rows into fixed-capacity bands and publish their offsets.

    Offsets are relative to the band's start. A row that cannot fit alone is
    refused instead of growing the workspace during serving.
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
    """Visible lengths and logits/block-maxima offsets within each band."""

    visible: torch.Tensor
    offsets: torch.Tensor
    block_offsets: torch.Tensor
    bands: tuple[tuple[int, int], ...]
