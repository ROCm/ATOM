# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""M3 PAGE geometry and shared startup validation, without a GPU runtime."""

import subprocess
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


def test_store_layout_table_does_not_import_the_lmcache_m3_connector():
    """The codec id is a leaf module; loading it must not build the LMCache worker."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import atom.kv_transfer.offload.mooncake_store.layouts; "
                "assert 'atom.kv_transfer.offload.hybrid.m3.connector' "
                "not in sys.modules; "
                "from atom.kv_transfer.offload.hybrid.m3 import M3OffloadConnector; "
                "assert M3OffloadConnector.__name__ == 'M3OffloadConnector'; "
                "assert 'atom.kv_transfer.offload.hybrid.m3.connector' in sys.modules"
            ),
        ],
        check=True,
        timeout=30,
    )


def test_split_codecs_partition_regions_by_draft_namespace(monkeypatch):
    from atom.kv_transfer.disaggregation.types import PageUnitSpec
    from atom.kv_transfer.offload.kv_store.layouts import split_draft_pages
    from atom.kv_transfer.offload.mooncake_store.layouts import (
        build_mooncake_draft_codec,
        mooncake_codec_id,
    )

    monkeypatch.setattr(
        DSV4PageSlotCodec, "has_fused_chunk_major_staging", property(lambda _: True)
    )
    # A last-stage layout: the target's regions, then the draft's merged after
    # them, with one target region after a draft one to pin the order.
    widths = [("mha", "k", 32), ("mha", "index", 16), ("draft", "k", 8)]
    widths += [("draft", "v", 8), ("mha", "v", 24)]
    transfer = KVTransferTensors(
        pages=[
            page_region(
                torch.zeros((4, width), dtype=torch.uint8),
                semantic_role=f"{ns}.{field}",
                unit_spec=PageUnitSpec(ns, field, 0, width, "torch.uint8"),
            )
            for ns, field, width in widths
        ]
    )
    transfer.set_block_count(4)
    config = SimpleNamespace(hf_config=SimpleNamespace(model_type="minimax_m3"))
    full = build_mooncake_codec(config, {}, transfer, 4, permit_per_request_state=False)
    assert build_mooncake_draft_codec(config, transfer, 4) is None  # off
    assert mooncake_codec_id(config) == "m3-page-regions"

    monkeypatch.setenv("ATOM_MOONCAKE_STORE_DRAFT_WINDOW", "1")
    assert mooncake_codec_id(config) == "m3-page-regions-draft-window-v1"
    target = build_mooncake_codec(
        config, {}, transfer, 4, permit_per_request_state=False
    )
    draft = build_mooncake_draft_codec(config, transfer, 4)
    roles = [region.semantic_role for region in full.page_regions]
    assert [r.semantic_role for r in target.page_regions] == [
        role for role in roles if not role.startswith("draft.")
    ]
    assert [r.semantic_role for r in draft.page_regions] == ["draft.k", "draft.v"]
    assert target.bytes_per_block + draft.bytes_per_block == full.bytes_per_block
    assert (target.bytes_per_block, draft.bytes_per_block) == (72, 16)
    # The shared descriptor keeps every region; the copies do not alias it.
    split_target, split_draft = split_draft_pages(transfer)
    assert len(transfer.pages) == 5 and split_target.num_blocks == 4
    assert split_target.pages is not transfer.pages
    assert len(split_target.pages) + len(split_draft.pages) == 5
    # A stage without the draft builds no draft codec.
    stage = KVTransferTensors(pages=[transfer.pages[0], transfer.pages[1]])
    stage.set_block_count(4)
    assert split_draft_pages(stage) == (stage, None)
    assert build_mooncake_draft_codec(config, stage, 4) is None
    # Only M3 has a split.
    dense = SimpleNamespace(hf_config=SimpleNamespace(model_type="llama"))
    with pytest.raises(ValueError, match="splits the m3 layout only"):
        mooncake_codec_id(dense)
