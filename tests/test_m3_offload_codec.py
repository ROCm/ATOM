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
from atom.kv_transfer.offload.mooncake_store.layouts import build_mooncake_codec
from atom.kv_transfer.offload.page_codec import pure_page_codec


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
    codec = pure_page_codec(transfer, num_blocks)
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


def test_page_regions_round_trip_index_and_draft_bytes(transfer, monkeypatch):
    """The packed block image keeps the NSA index and the draft KV.

    The Store round trip that copies these bytes on a GPU is skipped without a
    running Store. This follows the codec's own region list, which is the order
    and the width the staging kernel packs.
    """
    monkeypatch.setattr(
        DSV4PageSlotCodec, "has_fused_chunk_major_staging", property(lambda _: True)
    )
    codec = pure_page_codec(transfer)
    planes = {
        region.semantic_role: transfer.block_tensor_views[index][:, 0, :]
        for index, region in enumerate(codec.page_regions)
    }
    for role, width, fill in (
        ("m3.k", 32, 1),
        ("m3.index", 16, 2),
        ("m3.draft.k", 8, 3),
    ):
        planes[role].fill_(fill)
        assert planes[role].shape[1] == width
    packed = b"".join(
        planes[region.semantic_role][0].numpy().tobytes()
        for region in codec.page_regions
    )
    assert len(packed) == codec.bytes_per_block
    index_at = planes["m3.k"].shape[1]
    draft_at = index_at + planes["m3.index"].shape[1]
    assert packed[index_at:draft_at] == bytes([2]) * planes["m3.index"].shape[1]
    assert packed[draft_at:] == bytes([3]) * planes["m3.draft.k"].shape[1]
    for plane in planes.values():
        plane.zero_()
    offset = 0
    for region in codec.page_regions:
        width = region.unit_bytes
        restored = torch.tensor(
            list(packed[offset : offset + width]), dtype=torch.uint8
        )
        planes[region.semantic_role][0].copy_(restored)
        offset += width
    assert torch.equal(planes["m3.index"][0], torch.full((16,), 2, dtype=torch.uint8))
    assert torch.equal(planes["m3.draft.k"][0], torch.full((8,), 3, dtype=torch.uint8))
    assert int(planes["m3.k"][0, 0]) == 1


@pytest.mark.parametrize("missing", [None, KVTransferTensors()])
def test_page_codec_rejects_missing_pages(missing):
    with pytest.raises(ValueError, match="non-empty block_regions"):
        pure_page_codec(missing, 4)


def test_page_codec_rejects_slot_state(transfer):
    transfer.slot_regions.append(transfer.block_regions[0])
    with pytest.raises(ValueError, match="empty SLOT"):
        pure_page_codec(transfer)


def test_page_codec_rejects_a_region_without_its_view(transfer):
    transfer.pages[-1] = PageRegion(region=transfer.block_regions[-1])
    with pytest.raises(ValueError, match="tensor view for every PAGE"):
        pure_page_codec(transfer)


@pytest.mark.parametrize("num_blocks", [None, 0])
def test_page_codec_rejects_missing_or_unset_block_count(transfer, num_blocks):
    transfer.num_blocks = num_blocks
    with pytest.raises(ValueError, match="positive num_blocks"):
        pure_page_codec(transfer)


@pytest.mark.parametrize("backend", ["lmcache", "mooncake_store"])
def test_connectors_reject_unavailable_page_staging(transfer, monkeypatch, backend):
    # CPU PAGEs make the real codec report no Triton staging. Mooncake refuses
    # inside the codec ``register_kv_caches`` builds; LMCache refuses inside
    # ``register_kv_caches``, before either constructs an engine, pool, or client.
    config = SimpleNamespace(hf_config=SimpleNamespace(model_type="minimax_m3"))
    if backend == "mooncake_store":
        with pytest.raises(RuntimeError, match="Triton.*PAGE staging"):
            build_mooncake_codec(
                config, {}, transfer, 4, permit_per_request_state=False
            )
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
