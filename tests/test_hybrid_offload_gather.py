# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""GPU round-trip tests for the DSV4 offload-unit region gather/scatter."""

from __future__ import annotations

import pytest
import torch

from atom.kv_transfer.offload.triton_offload_gather import (
    GatherRegion,
    gather_regions,
    scatter_regions,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="gather/scatter kernels require GPU"
)


def _blocks(num_blocks: int, unit_bytes: int) -> torch.Tensor:
    return torch.randint(
        0, 256, (num_blocks * unit_bytes,), dtype=torch.uint8, device="cuda"
    )


def test_gather_heterogeneous_regions_matches_reference():
    # Two sources with different per-block strides and different id lists.
    comp = _blocks(8, 512)  # "compressed" region
    swa = _blocks(6, 384)  # "swa" region
    comp_ids = [3, 1, 4, 0]
    swa_ids = [5, 2]

    layout_comp_off = 0
    layout_swa_off = len(comp_ids) * 512
    total = layout_swa_off + len(swa_ids) * 384
    buf = torch.zeros(total, dtype=torch.uint8, device="cuda")

    regions = [
        GatherRegion(comp.data_ptr(), 512, comp_ids, layout_comp_off),
        GatherRegion(swa.data_ptr(), 384, swa_ids, layout_swa_off),
    ]
    gather_regions(buf, regions)
    torch.cuda.synchronize()

    # Reference: contiguous concat of the selected blocks.
    for i, bid in enumerate(comp_ids):
        got = buf[layout_comp_off + i * 512 : layout_comp_off + (i + 1) * 512]
        assert torch.equal(got, comp[bid * 512 : (bid + 1) * 512])
    for i, bid in enumerate(swa_ids):
        got = buf[layout_swa_off + i * 384 : layout_swa_off + (i + 1) * 384]
        assert torch.equal(got, swa[bid * 384 : (bid + 1) * 384])


def test_scatter_is_inverse_of_gather():
    comp = _blocks(8, 512)
    swa = _blocks(6, 384)
    comp_ids = [3, 1, 4, 0]  # distinct: no scatter write collisions
    swa_ids = [5, 2]

    swa_off = len(comp_ids) * 512
    total = swa_off + len(swa_ids) * 384
    buf = torch.zeros(total, dtype=torch.uint8, device="cuda")
    gather_regions(
        buf,
        [
            GatherRegion(comp.data_ptr(), 512, comp_ids, 0),
            GatherRegion(swa.data_ptr(), 384, swa_ids, swa_off),
        ],
    )
    torch.cuda.synchronize()

    # Scatter into fresh destinations and confirm the gathered blocks round-trip.
    comp2 = torch.zeros_like(comp)
    swa2 = torch.zeros_like(swa)
    scatter_regions(
        buf,
        [
            GatherRegion(comp2.data_ptr(), 512, comp_ids, 0),
            GatherRegion(swa2.data_ptr(), 384, swa_ids, swa_off),
        ],
    )
    torch.cuda.synchronize()

    for bid in comp_ids:
        assert torch.equal(
            comp2[bid * 512 : (bid + 1) * 512], comp[bid * 512 : (bid + 1) * 512]
        )
    for bid in swa_ids:
        assert torch.equal(
            swa2[bid * 384 : (bid + 1) * 384], swa[bid * 384 : (bid + 1) * 384]
        )


def test_negative_ids_are_skipped():
    comp = _blocks(4, 256)
    ids = [2, -1, 0]  # -1 = window-freed, must be skipped
    buf = torch.full((3 * 256,), 0xAB, dtype=torch.uint8, device="cuda")
    gather_regions(buf, [GatherRegion(comp.data_ptr(), 256, ids, 0)])
    torch.cuda.synchronize()
    # slot 0 <- block 2, slot 2 <- block 0, slot 1 untouched (still 0xAB).
    assert torch.equal(buf[0:256], comp[2 * 256 : 3 * 256])
    assert torch.equal(buf[512:768], comp[0:256])
    assert torch.all(buf[256:512] == 0xAB)


def test_empty_regions_noop():
    buf = torch.zeros(16, dtype=torch.uint8, device="cuda")
    gather_regions(buf, [])
    gather_regions(buf, [GatherRegion(buf.data_ptr(), 8, [], 0)])
    torch.cuda.synchronize()
    assert torch.all(buf == 0)
