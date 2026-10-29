# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DSA indexer's scoring, shared by every model that has one.

A model owns its query projection, how it writes index keys, and what it does
with the selection; the attention builder publishes the step's rows once,
cut to the logits plane it owns (`IndexRows.workspace`). Scoring and top-k --
under DCP, each rank's share of the rows and their all-gather -- are here.

The names below are loaded on first use, so importing a submodule that needs
no kernels (``schedule``) does not import AITER."""

import importlib

_EXPORTS = {
    "NATURAL": ".rows",
    "PRESHUFFLE": ".rows",
    "ConcatKeys": ".rows",
    "DcpSplit": ".rows",
    "Fp4LaunchStorages": ".rows",
    "IndexCache": ".rows",
    "IndexRows": ".rows",
    "concat_rows": ".rows",
    "fp4_rows": ".rows",
    "fp8_rows": ".rows",
    "fold_query_scale": ".query",
    "quantize_query": ".query",
    "quantize_query_rows": ".query",
    "score_topk": ".score",
}

__all__ = [
    "NATURAL",
    "PRESHUFFLE",
    "ConcatKeys",
    "DcpSplit",
    "Fp4LaunchStorages",
    "IndexCache",
    "IndexRows",
    "concat_rows",
    "fold_query_scale",
    "fp4_rows",
    "fp8_rows",
    "quantize_query",
    "quantize_query_rows",
    "score_topk",
]


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value
