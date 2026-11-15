# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""DCP prefill's FP4 logits plane is one workspace at `plane_capacity`.

Allocated per call at the step's longest row, the plane grows with the live
context and the caching allocator keeps every smaller one: about 15 GiB a
rank on AgentX c16 (wuhuikx/LLM-Booster#564). The workspace must cut the
same bands as the per-call plane did, so scoring is unchanged.
"""

import pytest
import torch

from atom.model_ops.indexer.rows import _plane_bands
from atom.model_ops.sparse_indexer_chunk import (
    _BUFFER_DESCRIPTOR_LIMIT_BYTES,
    LogitsWorkspace,
    plane_capacity,
)
from atom.utils import envs

# Widths of the prefills that grew rank 0's cache in the #564 run, plus the
# warmup's and a 1M-token context.
_WIDTHS = [8192, 16384, 206592, 222976, 239360, 255744, 316928, 523904, 1048576]


@pytest.fixture
def budget(monkeypatch):
    def set_mb(mb):
        monkeypatch.setattr(envs, "ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB", mb)

    return set_mb


@pytest.mark.parametrize("mb", [2048, 0, 4096, 512])
def test_capacity_stays_strictly_under_the_byte_cap(budget, mb):
    budget(mb)
    cap = _BUFFER_DESCRIPTOR_LIMIT_BYTES
    if mb > 0:
        cap = min(cap, mb << 20)
    assert plane_capacity() * 4 < cap
    assert plane_capacity() * 4 >= cap - 4


@pytest.mark.parametrize("mb", [2048, 512])
@pytest.mark.parametrize("rows", [2048, 1024, 130, 1])
@pytest.mark.parametrize("width", _WIDTHS)
def test_the_workspace_cuts_the_per_call_planes_bands(budget, mb, rows, width):
    budget(mb)
    ws = LogitsWorkspace(plane_capacity(), "meta")
    assert _plane_bands(0, rows, width, ws) == _plane_bands(0, rows, width, None)
    # A rank's share starting mid-batch (DCP's RowSplit) bands the same way.
    assert _plane_bands(rows, 2 * rows, width, ws) == _plane_bands(
        rows, 2 * rows, width, None
    )


@pytest.mark.parametrize("width", _WIDTHS)
def test_every_band_fits_the_workspace(budget, width):
    budget(2048)
    ws = LogitsWorkspace(plane_capacity(), "meta")
    for first, end in _plane_bands(0, 2048, width, ws):
        view = ws.logits(end - first, width)
        assert view.shape == (end - first, width)
        assert view.dtype == torch.float32
