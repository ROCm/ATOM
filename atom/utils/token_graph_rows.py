# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The real rows of a prefill padded to a token-graph bucket.

A backend whose eager regions plan their launches from the step's real rows
(Kimi-K3's KDA and MLA) builds its prefill metadata for those rows only and
marks the step with how many there are (`AttentionMetaData.prefill_real_rows`).
A region then cuts its bucket-high inputs back to the real rows, and hands its
output back at bucket height, the shape the next recorded segment reads
(`atom/utils/region_graph.py`). Outside such a step each helper is the
identity, so an unpadded prefill and every decode run as before.
"""

import torch

from atom.utils import forward_context


def padded_prefill_rows() -> int | None:
    """The real rows of the running step if it is a marked padded prefill, else None."""
    metadata = getattr(forward_context.get_forward_context(), "attn_metadata", None)
    return getattr(metadata, "prefill_real_rows", None)


def real_rows(
    t: torch.Tensor | None, real: int, rows: int, groups_major: bool = False
) -> torch.Tensor | None:
    """`t` cut to its first `real` rows when it is a `rows`-high token-major tensor.

    A 2-D per-group scale written group-major (`transpose_scale`: row r of group
    g at g * rows + r in a (rows, groups) buffer) is rebuilt group-major at the
    real height, which is what an unpadded producer lays out.
    """
    if t is None or t.dim() == 0 or t.shape[0] != rows:
        return t
    if groups_major and t.dim() == 2 and t.shape[1] > 1:
        groups = t.shape[1]
        return t.reshape(groups, rows)[:, :real].contiguous().reshape(real, groups)
    return t[:real]


def to_rows(out: torch.Tensor, rows: int) -> torch.Tensor:
    """`out` at `rows` rows: its own rows, then zeros. Identity at `rows` already."""
    if out.shape[0] == rows:
        return out
    if out.shape[0] > rows:
        raise ValueError(f"a region returned {out.shape[0]} rows for a {rows}-row step")
    padded = out.new_zeros((rows, *out.shape[1:]))
    padded[: out.shape[0]] = out
    return padded
