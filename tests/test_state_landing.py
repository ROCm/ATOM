# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One KDA state slot scatters into the decode rank's slot, whole units."""

import ctypes

import msgpack
import torch

from atom.kv_transfer.disaggregation.landing_scatter import segment_table_capacity
from atom.kv_transfer.disaggregation.mooncake.landing import (
    MSG_LANDING_CREDIT,
    LandingReceiver,
)

# Three regions of one KDA layer, 4-byte aligned, as a decode TP rank stores
# them. The packed image is conv (12) then SSM (8) then the cursor (4).
_WIDTHS = (12, 8, 4)
_DST_SLOT = 3


def test_one_slot_scatters_each_region_into_its_state_slot():
    regions = [torch.zeros(8, width, dtype=torch.uint8) for width in _WIDTHS]
    sent = []
    finished = []
    recv = LandingReceiver(
        device=None,
        pool_slots=2,
        slot_bytes=64,
        units_per_page=1,
        consumer_key="consumer:1",
        region_bases=[int(region.data_ptr()) for region in regions],
        region_block_bytes=list(_WIDTHS),
        region_indices=[0, 1, 2],
        send=lambda addr, parts: sent.append((addr, parts)),
        finish=lambda req, failed: finished.append((req, failed)),
        table_segments=segment_table_capacity(1024),
        credit_pool="state",
    )
    offered = recv.advertise("stage0", 8)
    assert len(offered["slots"]) == 1  # 2 // 8, and one slot still lands
    slot = offered["slots"][0]
    packed = torch.arange(sum(_WIDTHS), dtype=torch.uint8)
    recv.pool[slot, : packed.numel()].copy_(packed)
    pool = recv.pool

    def copy(src, dst, nbytes, _events):
        for s, d, n in zip(src, dst, nbytes):
            ctypes.memmove(recv._dst_origin + int(d), pool.data_ptr() + int(s), int(n))

    recv._copy_segments = copy
    recv.begin("r", 5, [_DST_SLOT], {0: "stage0"})
    offset = 0
    items = []
    for region, width in enumerate(_WIDTHS):
        items.append([region, 0, 1, offset])
        offset += width
    recv.on_ready(
        {
            "request_id": "r",
            "write_nonce": 5,
            "pp_rank": 0,
            "seq": 0,
            "slot": slot,
            "items": items,
        }
    )
    assert recv.stage_done("r", 0, 5, True, [slot])
    assert finished == []
    recv._process(recv._drain(block=False))
    assert finished == [("r", False)]
    offset = 0
    for region, width in zip(regions, _WIDTHS):
        assert torch.equal(region[_DST_SLOT], packed[offset : offset + width])
        offset += width
    addr, (kind, payload) = sent[0]
    assert addr == "stage0" and kind == MSG_LANDING_CREDIT
    body = msgpack.loads(payload)
    assert body["pool"] == "state"
    assert body["slots"] == [slot]
