# SPDX-License-Identifier: MIT
"""The packed row layout DSA prefill scorers share with V4.1, and the FP8
prefill indexer's gathered keys with the passes they are gathered in.

Builders size the storage from the configuration before memory profiling, so
the KV budget counts it and nothing is allocated per step (the logits plane
is `LogitsWorkspace`). PAGE-pool sizing stays with V4.1's builder.
"""

from typing import NamedTuple

import numpy as np
import torch

from atom.model_ops.sparse_indexer_chunk import cdiv, qsplit_rows

# Packed rows (V4.1's FP4, FP8 prefill) start 256-byte aligned: 64 fp32 logits.
ROW_ALIGN = 64


def aligned_spans(visible):
    """Each packed row's visibility rounded up to its store alignment."""
    return (visible + ROW_ALIGN - 1) // ROW_ALIGN * ROW_ALIGN


def band_offsets(visible, capacity, out):
    """Pack visible rows into bands of `capacity`, each row's offset from its
    band's start into `out`. A row too long for any band is refused."""
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


class ConcatKeyWorkspace:
    """FP8 prefill keys gathered end to end (`ConcatKeys`) and, under DCP, the
    rank shards they come from. Never grows: a larger step goes in passes."""

    def __init__(
        self, capacity: int, dim: int, dtype, device, gathered_capacity: int = 0
    ):
        self.capacity, self.gathered_capacity = capacity, gathered_capacity
        self._keys = _key_buffers(capacity, dim, dtype, device)
        self._gathered = _key_buffers(gathered_capacity, dim, dtype, device)

    @classmethod
    def for_config(
        cls,
        max_model_len: int,
        max_batched_tokens: int,
        max_requests: int,
        dim: int,
        dtype,
        device,
        dcp_world: int = 1,
        interleave: int = 1,
    ) -> "ConcatKeyWorkspace":
        """One request of the model length plus a step's tokens; under DCP also
        every rank's shard of a pass, each request padded to its super-block."""
        keys = max_model_len + max_batched_tokens
        gathered = 0
        if dcp_world > 1:
            shard = cdiv(keys, dcp_world) + max_requests * interleave
            gathered = dcp_world * shard
        return cls(keys, dim, dtype, device, gathered)

    @property
    def nbytes(self) -> int:
        tensors = (*self._keys, *self._gathered)
        return sum(t.numel() * t.element_size() for t in tensors)

    def keys(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`[rows, dim]` FP8 keys and `[rows, 1]` fp32 scales."""
        return _rows(self._keys, rows, "key")

    def gathered(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """DCP's all-gathered shards, rank-major, as `keys`."""
        return _rows(self._gathered, rows, "gathered-shard")


class ConcatPass(NamedTuple):
    """Requests `[first, end)` gathered together: keys `[key_base, +keys)` of the
    step's run; rows from `row_base`, `stride` a slot of `world`, this rank's
    `[lo, hi)` in `bands`. `dcp`: `(local_total, local_cu, gather_index)`."""

    first: int
    end: int
    key_base: int
    keys: int
    row_base: int
    stride: int
    rank: int
    world: int
    lo: int
    hi: int
    bands: tuple[tuple[int, int], ...]
    dcp: tuple[int, torch.Tensor, torch.Tensor] | None = None


def concat_passes(
    row_cu, key_cu, row_ends, offsets, key_capacity, plane_capacity, rank=0, world=1
):
    """Consecutive requests whose keys (`key_cu`) fit `key_capacity`, their rows
    (`row_cu`) split over the `world` ranks; this rank's packed into bands of
    `plane_capacity`, offsets into `offsets`. Every DCP rank plans alike."""
    passes, first, requests = [], 0, len(key_cu) - 1
    while first < requests:
        end = first + 1
        while end < requests and key_cu[end + 1] - key_cu[first] <= key_capacity:
            end += 1
        keys = int(key_cu[end] - key_cu[first])
        if keys > key_capacity:
            raise ValueError(
                f"request {first}'s {keys} keys exceed the {key_capacity}-key "
                "FP8 prefill indexer key workspace"
            )
        row_base = int(row_cu[first])
        stride, _, lo, hi = qsplit_rows(int(row_cu[end]) - row_base, rank, world)
        lo, hi = lo + row_base, hi + row_base
        cuts = band_offsets(row_ends[lo:hi], plane_capacity, offsets[lo:hi])
        bands = tuple((a + lo, b + lo) for a, b in cuts)
        passes.append(
            ConcatPass(
                first,
                end,
                int(key_cu[first]),
                keys,
                row_base,
                stride,
                rank,
                world,
                lo,
                hi,
                bands,
            )
        )
        first = end
    return tuple(passes)


def _key_buffers(rows, dim, dtype, device):
    return (
        torch.empty((rows, dim), dtype=dtype, device=device),
        torch.empty((rows, 1), dtype=torch.float32, device=device),
    )


def _rows(buffers, rows, what):
    values, scales = buffers
    if rows > values.shape[0]:
        raise ValueError(
            f"{rows} keys exceed the {values.shape[0]}-key FP8 prefill indexer "
            f"{what} workspace"
        )
    return values[:rows], scales[:rows]
