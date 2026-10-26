# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DSA indexer's scoring, shared by every model that has one.

A model owns its query projection, how it writes index keys, and what it does
with the selection; the attention builder publishes the step's rows once,
cut to the logits plane it owns (`IndexRows.workspace`). Scoring and top-k --
split across DCP or TP ranks, each rank's share of the rows and their
all-gather -- are here."""

from .query import fold_query_scale, quantize_query, quantize_query_rows
from .rows import (
    NATURAL,
    PRESHUFFLE,
    ConcatKeys,
    Fp4LaunchStorages,
    IndexCache,
    IndexRows,
    RowSplit,
    concat_rows,
    fp4_rows,
    fp8_rows,
)
from .score import score_topk

__all__ = [
    "NATURAL",
    "PRESHUFFLE",
    "ConcatKeys",
    "Fp4LaunchStorages",
    "IndexCache",
    "IndexRows",
    "RowSplit",
    "concat_rows",
    "fold_query_scale",
    "fp4_rows",
    "fp8_rows",
    "quantize_query",
    "quantize_query_rows",
    "score_topk",
]
