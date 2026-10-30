# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""M3 PAGE geometry and shared startup validation, without a GPU runtime."""

import sys
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.disaggregation.page_region import page_region
from atom.kv_transfer.disaggregation.types import KVTransferTensors, PageRegion
from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec
from atom.kv_transfer.offload.hybrid.m3 import connector as m3_connector
from atom.kv_transfer.offload.hybrid.m3.codec import build_m3_page_codec
from atom.kv_transfer.offload.mooncake_store.worker import MooncakeStoreOffloadConnector


@pytest.fixture
def transfer():
    tensors = KVTransferTensors(
        pages=[
            page_region(torch.zeros((4, width), dtype=torch.uint8), semantic_role=role)
            for role, width in (("m3.k", 32), ("m3.index", 16), ("m3.draft.k", 8))
        ]
    )
    tensors.set_block_count(4)
    return tensors


@pytest.mark.parametrize("num_blocks, expected_blocks", [(None, 4), (2, 2)])
def test_page_codec_preserves_region_order_and_block_count(
    transfer, monkeypatch, num_blocks, expected_blocks
):
    # Keep the real codec's geometry validation while isolating GPU capability.
    monkeypatch.setattr(
        DSV4PageSlotCodec, "has_fused_chunk_major_staging", property(lambda _: True)
    )
    codec = build_m3_page_codec(transfer, num_blocks)
    assert codec.num_blocks == expected_blocks
    assert codec.num_slots == 0
    assert codec.slot_regions == ()
    assert codec.device == transfer.block_tensor_views[0].device
    assert codec.bytes_per_block == 56
    assert [region.semantic_role for region in codec.page_regions] == [
        "m3.k",
        "m3.index",
        "m3.draft.k",
    ]
    assert [region.base_addr for region in codec.page_regions] == [
        region.base_addr for region in transfer.block_regions
    ]


@pytest.mark.parametrize("missing", [None, KVTransferTensors()])
def test_page_codec_rejects_missing_pages(missing):
    with pytest.raises(ValueError, match="complete PAGE"):
        build_m3_page_codec(missing, 4)


def test_page_codec_rejects_slot_state(transfer):
    transfer.slot_regions.append(transfer.block_regions[0])
    with pytest.raises(ValueError, match="no SLOT state"):
        build_m3_page_codec(transfer)


def test_page_codec_rejects_a_region_without_its_view(transfer):
    transfer.pages[-1] = PageRegion(region=transfer.block_regions[-1])
    with pytest.raises(ValueError, match="tensor view for every PAGE"):
        build_m3_page_codec(transfer)


@pytest.mark.parametrize("num_blocks", [None, 0])
def test_page_codec_rejects_missing_or_unset_block_count(transfer, num_blocks):
    transfer.num_blocks = num_blocks
    with pytest.raises(ValueError, match="num_blocks"):
        build_m3_page_codec(transfer)


@pytest.mark.parametrize("backend", ["lmcache", "mooncake_store"])
def test_connectors_reject_unavailable_page_staging(transfer, monkeypatch, backend):
    # CPU PAGEs make the real codec report no Triton staging. Both workers
    # must refuse before constructing any offload engine, pool or Store client.
    config = SimpleNamespace(hf_config=SimpleNamespace(model_type="minimax_m3"))
    if backend == "mooncake_store":
        connector = object.__new__(MooncakeStoreOffloadConnector)
        connector._config = config
        with pytest.raises(RuntimeError, match="Triton.*PAGE staging"):
            connector._make_codec({}, transfer, 4)
    else:
        monkeypatch.setitem(
            sys.modules,
            "aiter.dist.parallel_state",
            SimpleNamespace(get_tp_group=lambda: None),
        )
        monkeypatch.setattr(m3_connector, "pp_aware_rank_and_world", lambda *_: (0, 1))
        connector = object.__new__(m3_connector.M3OffloadConnector)
        connector._config = config
        with pytest.raises(RuntimeError, match="Triton.*PAGE staging"):
            connector.register_kv_caches({}, transfer, 4)
