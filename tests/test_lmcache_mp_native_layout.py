# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU byte contracts for direct native checkpoint registration."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.disaggregation.types import KVTransferRegion, KVTransferTensors
from atom.kv_transfer.offload.mp.native_state_layout import (
    build_native_state_mp_layout,
)
from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec


def _transfer(*, widths=(8, 8, 2), image_bytes=39, num_blocks=7):
    views = []
    for index, width in enumerate(widths):
        # Nonzero allocation offsets exercise the actual shared-arena views.
        raw = torch.arange(num_blocks * width + 16, dtype=torch.uint8)[8:-8]
        raw.add_(index * 40)
        dtype = torch.bfloat16 if index == 0 and width % 2 == 0 else torch.uint8
        views.append(raw.view(dtype).view(num_blocks, 1, -1))
    spec = PagedStateCheckpointSpec(
        page_unit_bytes=sum(widths),
        slot_bytes=max(image_bytes, 128),
        image_bytes=image_bytes,
        layout_id="native-test-v1",
    )
    transfer = KVTransferTensors(
        block_regions=[
            KVTransferRegion(
                view.data_ptr(),
                num_blocks * width,
                width,
                semantic_role=f"region.{i}",
            )
            for i, (view, width) in enumerate(zip(views, widths, strict=True))
        ],
        slot_regions=[],
        block_tensor_views=views,
        paged_state_checkpoint_spec=spec,
        execute_paged_state_copies=lambda stores, restores: None,
    )
    transfer.set_block_count(num_blocks)
    return transfer


def _layout(transfer):
    return build_native_state_mp_layout(transfer, block_size=4, chunk_size=16)


def _gather(layout, ids):
    image = torch.empty(layout.checkpoint_spec.image_bytes, dtype=torch.uint8)
    for span in layout.image_plan(ids):
        image[span.image_offset : span.image_offset + span.nbytes].copy_(
            layout.tensors[span.tensor_index][span.block_id].flatten()
        )
    return image


def test_native_image_round_trip_uses_arbitrary_unit_ids_and_valid_page_zero():
    source = _transfer()
    source_layout = _layout(source)
    source_ids = [4, 0, 3]
    actual = _gather(source_layout, source_ids)
    expected = torch.cat(
        [
            view[unit_id].view(torch.uint8).flatten()
            for unit_id in source_ids
            for view in source.block_tensor_views
        ]
    )[:39]
    assert torch.equal(actual, expected)

    destination = _transfer()
    for view in destination.block_tensor_views:
        view.view(torch.uint8).fill_(0xCD)
    destination_layout = _layout(destination)
    destination_ids = [1, 5, 2]
    for span in destination_layout.image_plan(destination_ids):
        destination_layout.tensors[span.tensor_index][span.block_id].flatten().copy_(
            actual[span.image_offset : span.image_offset + span.nbytes]
        )
    assert torch.equal(_gather(destination_layout, destination_ids), expected)
    # The partial final unit owns only the first three bytes of region zero.
    assert torch.all(
        destination.block_tensor_views[0][2].view(torch.uint8).flatten()[3:] == 0xCD
    )
    assert torch.all(destination.block_tensor_views[1][2].view(torch.uint8) == 0xCD)
    for view in destination.block_tensor_views:
        assert torch.all(view[0].view(torch.uint8) == 0xCD)
        assert torch.all(view[6].view(torch.uint8) == 0xCD)


def test_layout_coalesces_equal_shapes_inside_ordinal_and_preserves_trim_stride():
    transfer = _transfer()
    layout = _layout(transfer)
    assert layout.units_per_checkpoint == 3
    assert layout.bytes_per_block == 18
    assert layout.page_region_count == 3
    assert len(layout.tensors) == 10
    assert len(layout.kernel_groups) == 8
    assert layout.layer_groups == ((0,), (1,), (2,), (3, 4), (5,), (6, 7), (8,), (9,))
    assert [group.engine_group_id for group in layout.kernel_groups] == [
        0,
        0,
        0,
        1,
        1,
        2,
        2,
        3,
    ]
    for group in layout.kernel_groups:
        if group.engine_group_id:
            assert group.tokens_per_block == group.sw_size_tokens == 16
            assert group.recurrent_state is True
            assert group.extra_object_group_tag == 0
        else:
            assert group.tokens_per_block == 4
            assert group.sw_size_tokens == -1
            assert group.recurrent_state is False
    assert [
        (r.unit_ordinal, r.region_index, r.image_offset, r.nbytes)
        for r in layout.state_regions
    ] == [
        (0, 0, 0, 8),
        (0, 1, 8, 8),
        (0, 2, 16, 2),
        (1, 0, 18, 8),
        (1, 1, 26, 8),
        (1, 2, 34, 2),
        (2, 0, 36, 3),
    ]
    tail = layout.tensors[-1]
    assert tuple(tail.shape) == (7, 1, 3)
    assert tail.stride(0) == 8
    assert not tail.is_contiguous()
    for region in layout.state_regions:
        tensor = layout.tensors[region.tensor_index]
        owner = transfer.block_tensor_views[region.region_index]
        assert tensor.untyped_storage().data_ptr() == owner.untyped_storage().data_ptr()
        assert tensor.data_ptr() == owner.data_ptr()
        assert tensor.storage_offset() == owner.storage_offset() * owner.element_size()
        assert tensor.stride(0) == owner.stride(0) * owner.element_size()


@pytest.mark.parametrize("image_bytes", [1, 8, 16, 18, 19, 36, 39, 54])
def test_image_trims_at_region_and_unit_boundaries(image_bytes):
    layout = _layout(_transfer(image_bytes=image_bytes))
    assert sum(region.nbytes for region in layout.state_regions) == image_bytes
    tail = layout.state_regions[-1]
    assert tail.image_offset + tail.nbytes == image_bytes
    assert {region.unit_ordinal for region in layout.state_regions} == set(
        range(layout.units_per_checkpoint)
    )


@pytest.mark.parametrize("ids", [[1], [0, -1, 2], [0, 7, 2], [0, 0, 2], [True, 2, 3]])
def test_image_plan_rejects_missing_null_out_of_range_and_duplicate_units(ids):
    with pytest.raises(ValueError, match="unit ID"):
        _layout(_transfer()).image_plan(ids)


def test_equal_shape_unequal_stride_is_rejected_before_registration():
    # Last ordinal contains 4 bytes from each region: the second region has
    # stride 8, so LMCache cannot coalesce it with the first region's stride 4.
    with pytest.raises(ValueError, match="different block strides"):
        _layout(_transfer(widths=(4, 8), image_bytes=20))


def test_registration_rejects_mismatched_native_geometry_and_missing_restore():
    transfer = _transfer()
    transfer.execute_paged_state_copies = None
    with pytest.raises(TypeError, match="execute_paged_state_copies"):
        _layout(transfer)
    transfer = _transfer()
    transfer.paged_state_checkpoint_spec = replace(
        transfer.paged_state_checkpoint_spec, page_unit_bytes=17
    )
    with pytest.raises(ValueError, match="do not cover"):
        _layout(transfer)
    transfer = _transfer()
    transfer.block_tensor_views[0] = transfer.block_tensor_views[0].clone()
    with pytest.raises(ValueError, match="does not alias"):
        _layout(transfer)


@pytest.fixture
def export_method():
    # Execute the shipped exporter without importing attention kernels. Its
    # global dependencies are only torch and its local transfer-types import;
    # model construction and AITER imports are outside this CPU contract.
    source = Path(__file__).parents[1] / "atom/model_ops/attentions/deepseek_v4_attn.py"
    module = ast.parse(source.read_text())
    builder = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DeepseekV4AttentionMetadataBuilder"
    )
    method = next(
        node
        for node in builder.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_kv_transfer_tensors"
    )
    namespace = {"torch": torch}
    exec(  # noqa: S102 -- execute the local exporter under the CPU fixture
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["get_kv_transfer_tensors"]


@pytest.mark.parametrize("indexer_fp4", [False, True])
def test_attention_export_retains_page_owners_native_spec_and_restore_callback(
    export_method, indexer_fp4
):
    num_blocks, num_slots, envelope_rows = 5, 2, 3
    planes = [
        torch.arange(
            (num_blocks * envelope_rows + num_slots * 2) * 4, dtype=torch.uint8
        ).view(-1, 4),
        torch.zeros(
            num_blocks * envelope_rows + num_slots * 2, 2, dtype=torch.bfloat16
        ),
    ]
    index_data = torch.zeros(2, num_blocks, 2, 4, dtype=torch.uint8)
    index_scale = torch.zeros(2, num_blocks, 2, 1, dtype=torch.uint8)
    pools = [(index_data, "dsv4.indexer.data")]
    if indexer_fp4:
        pools.append((index_scale, "dsv4.indexer.scale"))
    page_bytes = 24 + 16 + (4 if indexer_fp4 else 0)
    spec = PagedStateCheckpointSpec(page_bytes, 128, "dsv4-paged-state-v3:test", 45)
    restore = lambda stores, restores: None
    geo = SimpleNamespace(
        envelope_rows=envelope_rows,
        block_bytes=lambda row_bytes: envelope_rows * row_bytes,
        slot_bytes=lambda row_bytes: 2 * row_bytes,
        physical_slot=lambda slot: slot,
        slot_span=lambda slot: (num_blocks * envelope_rows, 0),
    )
    builder = SimpleNamespace(
        model_runner=SimpleNamespace(
            v4_unified_kv=[],
            config=SimpleNamespace(kv_transfer_config={"kv_connector": "lmcache_mp"}),
            state_runtime=SimpleNamespace(checkpoint_spec=spec),
        ),
        _indexer_fp4=indexer_fp4,
        num_state_slots=num_slots,
        num_blocks=num_blocks,
        pool_geometry=geo,
        _plane_fields=[SimpleNamespace(name="main"), SimpleNamespace(name="rope")],
        _kv_planes=lambda: planes,
        _plane_row_widths=lambda: [4, 4],
        _indexer_page_pools=lambda: pools,
        csa_layers=[2, 7],
        execute_paged_state_copies=restore,
    )
    export_method.__globals__["_uses_pd_staging"] = lambda config: False
    export_method.__globals__["_validate_fp4_indexer_transfer"] = lambda config: None
    transfer = export_method(builder)
    transfer.set_block_count(num_blocks)
    assert transfer.paged_state_checkpoint_spec is spec
    assert transfer.execute_paged_state_copies is restore
    assert len(transfer.block_tensor_views) == 4 + (2 if indexer_fp4 else 0)
    for region, view in zip(
        transfer.block_regions, transfer.block_tensor_views, strict=True
    ):
        assert view.data_ptr() == region.base_addr
        assert view.shape[:2] == (num_blocks, 1)
        assert view.numel() * view.element_size() == region.total_bytes
        assert view.stride(0) * view.element_size() == region.unit_bytes
    transfer.block_tensor_views[0][4].fill_(0xAB)
    assert torch.all(planes[0][12:15] == 0xAB)
    assert torch.any(planes[0][15:] != 0xAB)
    _layout(transfer)


def test_actual_lmcache_registration_preserves_native_aliases_and_stride():
    pytest.importorskip("lmcache.lmcache_native", exc_type=ImportError)
    from lmcache.utils import EngineType
    from lmcache.v1.gpu_connector.utils import (
        normalize_and_discover_per_layer_formats,
    )
    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

    layout = _layout(_transfer())
    groups = layout.engine_group_infos()
    normalized, formats = normalize_and_discover_per_layer_formats(
        list(layout.tensors), layout.layer_groups, EngineType.ATOM
    )
    manager = KVLayerGroupsManager(
        normalized,
        formats,
        groups,
        16,
        separate_object_groups=True,
    )
    assert [tuple(group.layer_indices) for group in manager.kernel_groups] == list(
        layout.layer_groups
    )
    assert manager.kernel_groups[-1].shape_desc.block_stride_elems == 8
    assert manager.kernel_groups[-1].shape_desc.hs == 3
    assert len(manager.object_groups) == 2
    assert manager.object_groups[-1].sw_size_chunks == 1
