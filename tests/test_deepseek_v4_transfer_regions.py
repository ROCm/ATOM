# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.disaggregation.types import KVTransferTensors
from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec
from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.model_engine.state_runtime import StateRuntime, StateTransfer
from atom.model_ops.attentions.pool_layout.v4_pool_geometry import UnifiedPoolGeometry

_MISSING = object()
_HEAD_DIM = 16
_ROPE_HEAD_DIM = 4
_CSA_ROWS_PER_BLOCK = 64
_INDEX_ROW_BYTES = 132
_CSA_REGION_COUNT = 2
_FP4_K_TILES = 1


def test_generic_transfer_tensors_default_to_no_full_slot_expectation():
    transfer = KVTransferTensors(block_regions=[], slot_regions=[], num_blocks=1)

    assert transfer.expected_full_slot_region_count is None


@contextmanager
def _stub_v4_runtime_imports():
    """Supply only unavailable GPU imports while loading the real builder."""
    aiter = types.ModuleType("aiter")
    aiter.__path__ = []
    aiter.dtypes = SimpleNamespace(fp8=torch.uint8)

    aiter_jit = types.ModuleType("aiter.jit")
    aiter_jit.__path__ = []
    aiter_jit_utils = types.ModuleType("aiter.jit.utils")
    aiter_jit_utils.__path__ = []
    chip_info = types.ModuleType("aiter.jit.utils.chip_info")
    chip_info.get_gfx = lambda: "gfx950"

    pcp_utils = types.ModuleType("atom.distributed.pcp_utils")
    for name in (
        "get_pcp_world_size",
        "pcp_is_enabled",
        "pcp_pad_dense",
        "pcp_pad_indptr",
        "pcp_pad_len",
        "pcp_reindex_ragged",
        "pcp_round_robin_query_indices",
    ):
        setattr(pcp_utils, name, lambda *args, **kwargs: None)

    backends = types.ModuleType("atom.model_ops.attentions.backends")
    for name in (
        "AttentionBackend",
        "AttentionMetadataBuilder",
        "CommonAttentionBuilder",
    ):
        setattr(backends, name, type(name, (), {}))

    kernels = types.ModuleType("atom.model_ops.v4_kernels")
    kernels.FP4_MQA_BLOCK_K = 128
    kernels.FP4_MQA_PARALLEL_UNIT_NUM = 1
    for name in (
        "build_v4_paged_decode_indptr",
        "fp4_indexer_enabled",
        "hca_compress_paged_offsets",
        "plan_context_lens",
        "write_v4_paged_decode_indices",
        "write_v4_paged_prefill_indices",
    ):
        setattr(kernels, name, lambda *args, **kwargs: None)

    replacements = {
        "aiter": aiter,
        "aiter.jit": aiter_jit,
        "aiter.jit.utils": aiter_jit_utils,
        "aiter.jit.utils.chip_info": chip_info,
        "atom.distributed.pcp_utils": pcp_utils,
        "atom.model_ops.attentions.backends": backends,
        "atom.model_ops.v4_kernels": kernels,
    }
    previous = {name: sys.modules.get(name, _MISSING) for name in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture(scope="module")
def v4_builder_cls():
    module_name = "_atom_test_deepseek_v4_attn"
    module_path = (
        Path(__file__).parents[1]
        / "atom"
        / "model_ops"
        / "attentions"
        / "deepseek_v4_attn.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with _stub_v4_runtime_imports():
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module.DeepseekV4AttentionMetadataBuilder


def _transfer_builder(
    builder_cls,
    *,
    kv_dtype: str,
    transfer_config: dict | None = None,
    indexer_fp4: bool = False,
    pipeline_parallel_size: int = 1,
):
    ratios = [4, 128, 4, 0]
    num_blocks = 3
    num_slots = 2
    geo = UnifiedPoolGeometry(
        ratios,
        num_blocks=num_blocks,
        num_slots=num_slots,
        ring_slots=7,
        block_size=256,
        arena_rows=10,
    )

    builder = builder_cls.__new__(builder_cls)
    builder._kv_fp8 = kv_dtype == "fp8"
    builder._indexer_fp4 = indexer_fp4
    builder._classical_dtype = torch.uint8 if builder._kv_fp8 else torch.bfloat16
    builder._rope_dtype = torch.bfloat16
    builder.head_dim = _HEAD_DIM
    builder.rope_head_dim = _ROPE_HEAD_DIM
    builder.csa_layers = [0, 2]
    builder.csa_rows_per_block = _CSA_ROWS_PER_BLOCK
    builder._index_row_bytes = _INDEX_ROW_BYTES
    builder._idx_k_tiles = _FP4_K_TILES
    builder.pool_geometry = geo
    builder.compress_ratios = ratios
    builder._checkpoint_range_cache = None
    builder._page_unit_region_cache = None
    builder._page_unit_region_owners = ()

    if kv_dtype == "fp8":
        row_widths = [
            _HEAD_DIM * torch.uint8.itemsize,
            _ROPE_HEAD_DIM * torch.bfloat16.itemsize,
        ]
    else:
        row_widths = [_HEAD_DIM * torch.bfloat16.itemsize]
    planes = [
        torch.empty(geo.plane_bytes(row_bytes), dtype=torch.uint8)
        for row_bytes in row_widths
    ]
    if indexer_fp4:
        indexers = torch.empty(
            (
                len(builder.csa_layers),
                num_blocks,
                builder._idx_k_tiles,
                4,
                builder.csa_rows_per_block,
                16,
            ),
            dtype=torch.uint8,
        )
        indexer_scales = torch.empty(
            (
                len(builder.csa_layers),
                num_blocks,
                builder._idx_k_tiles,
                4,
                builder.csa_rows_per_block,
            ),
            dtype=torch.uint8,
        )
        indexer_pools = [indexers, indexer_scales]
    else:
        indexers = torch.empty(
            (
                len(builder.csa_layers),
                num_blocks,
                builder.csa_rows_per_block,
                builder._index_row_bytes,
            ),
            dtype=torch.uint8,
        )
        indexer_scales = None
        indexer_pools = [indexers]
    config = SimpleNamespace(
        kv_transfer_config=transfer_config or {},
        pipeline_parallel_size=pipeline_parallel_size,
        kv_cache_block_size=256,
        decode_context_parallel_size=1,
    )
    # This fixture bypasses the builder constructor, so provide the minimal
    # arena layout used by the checkpoint image planner. The transfer-region
    # tests do not model compressor fields; their image consists of the live
    # window rows in each KV plane.
    builder._arena_planes = [[] for _ in row_widths]
    builder.model_runner = SimpleNamespace(config=config)
    layout_id = "test.dsv4.unified-page-slot"
    state_runtime = StateRuntime(
        transfer=StateTransfer.copy(layout_id),
        checkpoint_spec=PagedStateCheckpointSpec(
            page_unit_bytes=(
                sum(geo.block_bytes(width) for width in row_widths)
                + sum(
                    len(builder.csa_layers) * pool.stride(1) * pool.element_size()
                    for pool in indexer_pools
                )
            ),
            slot_bytes=sum(geo.slot_bytes(width) for width in row_widths),
            image_bytes=builder.checkpoint_image_bytes(),
            layout_id=layout_id,
        ),
    )
    runner_values = {
        "config": config,
        "state_runtime": state_runtime,
        "num_physical_kvcache_blocks": num_blocks,
        "pool_plan": SimpleNamespace(entries={STATE_SLOT_CLASS: num_slots}),
        "v4_unified_kv": planes[0],
        "v4_kv_plane": planes[0],
        "v4_kv_plane_rope": planes[1] if len(planes) == 2 else None,
        "v4_csa_idx_kv": indexers,
    }
    if indexer_scales is not None:
        runner_values["v4_csa_idx_kv_scale"] = indexer_scales
    builder.model_runner = SimpleNamespace(**runner_values)
    return builder, geo, planes, row_widths


def _assert_plane_region_geometry(
    transfer,
    geo: UnifiedPoolGeometry,
    planes: list[torch.Tensor],
    row_widths: list[int],
):
    for plane, row_bytes, page_region, slot_region in zip(
        planes,
        row_widths,
        transfer.block_regions[: len(planes)],
        transfer.swa_block_regions,
        strict=True,
    ):
        assert page_region.base_addr == plane.data_ptr()
        assert page_region.unit_bytes == geo.envelope_rows * row_bytes
        assert slot_region.unit_bytes == geo.slot_rows * row_bytes
        assert slot_region.unit_bytes > geo.arena_rows * row_bytes
        for group in range(transfer.num_slots):
            slot_start, _ = geo.slot_span(geo.physical_slot(group))
            assert slot_region.unit_addr(group) == (
                plane.data_ptr() + slot_start * row_bytes
            )


def _assert_transfer_geometry(
    builder,
    geo: UnifiedPoolGeometry,
    planes: list[torch.Tensor],
    row_widths: list[int],
):
    transfer = builder.get_kv_transfer_tensors()
    assert transfer is not None
    assert transfer.num_blocks == geo.num_blocks
    assert transfer.num_slots == geo.num_slots
    assert transfer.expected_full_slot_region_count == len(planes)

    unified_codec = DSV4PageSlotCodec(
        transfer.block_regions,
        transfer.swa_block_regions,
        num_blocks=transfer.num_blocks,
        num_slots=transfer.num_slots,
        device="cpu",
    )
    assert sum(region.unit_bytes for region in transfer.block_regions) == (
        unified_codec.bytes_per_block
    )
    assert sum(region.unit_bytes for region in transfer.swa_block_regions) == (
        unified_codec.slot_bytes
    )

    indexer_pool_count = 2 if builder._indexer_fp4 else 1
    assert len(transfer.block_regions) == (
        len(planes) + indexer_pool_count * _CSA_REGION_COUNT
    )
    assert len(transfer.swa_block_regions) == len(planes)
    assert transfer.slot_regions == []
    assert all(not region.reverse_indexed for region in transfer.block_regions)
    assert all(region.reverse_indexed for region in transfer.swa_block_regions)
    assert all(
        region.total_bytes == transfer.num_blocks * region.unit_bytes
        for region in transfer.block_regions
    )
    assert all(
        region.total_bytes == transfer.num_slots * region.unit_bytes
        for region in transfer.swa_block_regions
    )

    _assert_plane_region_geometry(transfer, geo, planes, row_widths)

    indexer_regions = transfer.block_regions[len(planes) :]
    expected_indexer_regions = [
        (pool[pos], f"{role_prefix}.layer_{layer_id}")
        for pool, role_prefix in builder._indexer_page_pools()
        for pos, layer_id in enumerate(builder.csa_layers)
    ]
    for region, (tensor, role) in zip(
        indexer_regions,
        expected_indexer_regions,
        strict=True,
    ):
        assert region.base_addr == tensor.data_ptr()
        assert region.unit_bytes == tensor.stride(0) * tensor.element_size()
        assert region.total_bytes == tensor.numel() * tensor.element_size()
        assert region.semantic_role == role

    checkpoint_bases, checkpoint_strides = builder._page_unit_regions()
    assert checkpoint_bases.tolist() == [
        region.base_addr for region in transfer.block_regions
    ]
    assert checkpoint_strides.tolist() == [
        region.unit_bytes for region in transfer.block_regions
    ]


def test_bf16_transfer_regions_cover_page_and_full_slot_geometry(v4_builder_cls):
    builder, geo, planes, row_widths = _transfer_builder(
        v4_builder_cls,
        kv_dtype="bf16",
        transfer_config={"kv_connector": "LMCacheOffloadConnector"},
    )

    _assert_transfer_geometry(builder, geo, planes, row_widths)

    assert row_widths == [_HEAD_DIM * torch.bfloat16.itemsize]
    assert len(planes) == 1
    assert len(builder.get_kv_transfer_tensors().swa_block_regions) == 1


def test_fp8_transfer_regions_cover_both_pages_and_full_slots(v4_builder_cls):
    builder, geo, planes, row_widths = _transfer_builder(
        v4_builder_cls,
        kv_dtype="fp8",
        transfer_config={"kv_connector": "LMCacheOffloadConnector"},
    )

    _assert_transfer_geometry(builder, geo, planes, row_widths)

    assert row_widths == [
        _HEAD_DIM * torch.uint8.itemsize,
        _ROPE_HEAD_DIM * torch.bfloat16.itemsize,
    ]
    assert len(planes) == 2
    assert len(builder.get_kv_transfer_tensors().swa_block_regions) == 2


def test_page_region_sizing_rejects_mutated_builder_helper(v4_builder_cls):
    builder, _, _, row_widths = _transfer_builder(
        v4_builder_cls,
        kv_dtype="bf16",
        transfer_config={"kv_connector": "LMCacheOffloadConnector"},
    )
    builder._plane_row_widths = lambda: [row_widths[0] + 1]

    with pytest.raises(RuntimeError, match="do not cover the sized PAGE unit"):
        builder.get_kv_transfer_tensors()


def test_fp4_indexer_offload_covers_data_and_scale_page_regions(v4_builder_cls):
    builder, geo, planes, row_widths = _transfer_builder(
        v4_builder_cls,
        kv_dtype="fp8",
        transfer_config={
            "kv_connector": "LMCacheOffloadConnector",
            "kv_role": "offload",
        },
        indexer_fp4=True,
    )

    _assert_transfer_geometry(builder, geo, planes, row_widths)

    transfer = builder.get_kv_transfer_tensors()
    assert transfer is not None
    data_bytes = _FP4_K_TILES * 4 * _CSA_ROWS_PER_BLOCK * 16
    scale_bytes = _FP4_K_TILES * 4 * _CSA_ROWS_PER_BLOCK
    assert [region.unit_bytes for region in transfer.block_regions[-4:]] == [
        data_bytes,
        data_bytes,
        scale_bytes,
        scale_bytes,
    ]
    assert [region.semantic_role for region in transfer.block_regions[-4:]] == [
        "dsv4.csa_indexer.fp4_data.layer_0",
        "dsv4.csa_indexer.fp4_data.layer_2",
        "dsv4.csa_indexer.fp4_scale.layer_0",
        "dsv4.csa_indexer.fp4_scale.layer_2",
    ]
    assert len(transfer.swa_block_regions) == len(planes)


def test_fp4_indexer_pd_transfer_remains_unsupported(v4_builder_cls):
    builder, *_ = _transfer_builder(
        v4_builder_cls,
        kv_dtype="fp8",
        transfer_config={"kv_connector": "mooncake", "kv_role": "kv_both"},
        indexer_fp4=True,
    )

    with pytest.raises(
        NotImplementedError,
        match=r"PD transfer.*index_cache_dtype fp4.*unsupported",
    ):
        builder.get_kv_transfer_tensors()


def test_page_region_registration_rejects_incomplete_geometry(v4_builder_cls):
    builder, *_ = _transfer_builder(
        v4_builder_cls,
        kv_dtype="fp8",
        transfer_config={"kv_connector": "LMCacheOffloadConnector"},
        indexer_fp4=True,
    )
    runtime = builder.model_runner.state_runtime
    spec = runtime.checkpoint_spec
    builder.model_runner.state_runtime = replace(
        runtime,
        checkpoint_spec=replace(spec, page_unit_bytes=spec.page_unit_bytes + 1),
    )

    with pytest.raises(RuntimeError, match="do not cover the sized PAGE unit"):
        builder.get_kv_transfer_tensors()


def test_fp4_indexer_without_transfer_keeps_single_node_behavior(v4_builder_cls):
    builder, *_ = _transfer_builder(
        v4_builder_cls,
        kv_dtype="fp8",
        transfer_config={},
        indexer_fp4=True,
    )

    assert builder.get_kv_transfer_tensors() is None


@pytest.mark.parametrize(
    ("transfer_config", "expected"),
    [
        (None, False),
        ({}, False),
        ({"kv_connector": "lmcache_offload", "kv_role": "offload"}, False),
        ({"kv_connector": " LMCache_Offload "}, False),
        ({"kv_connector": "LMCacheOffloadConnector"}, False),
        ({"kv_connector": "LMCacheConnectorV1"}, False),
        ({"kv_connector": "mooncake"}, True),
        ({"kv_connector": "moriio"}, True),
    ],
)
def test_transfer_topology_allocates_pd_staging_only_when_needed(
    v4_builder_cls,
    transfer_config,
    expected,
):
    uses_pd_staging = v4_builder_cls.allocate_per_req_cache.__globals__[
        "_uses_pd_staging"
    ]

    assert uses_pd_staging(transfer_config) is expected


@pytest.mark.parametrize(
    ("transfer_config", "error_type", "message"),
    [
        ([], TypeError, "kv_transfer_config must be a dict or None"),
        (
            {"kv_role": "offload"},
            ValueError,
            "requires a non-empty 'kv_connector'",
        ),
        ({"kv_connector": "unknown"}, ValueError, "unknown KV connector"),
    ],
)
def test_invalid_transfer_topology_fails_before_staging_allocation(
    v4_builder_cls,
    transfer_config,
    error_type,
    message,
):
    uses_pd_staging = v4_builder_cls.allocate_per_req_cache.__globals__[
        "_uses_pd_staging"
    ]

    with pytest.raises(error_type, match=message):
        uses_pd_staging(transfer_config)


def _prepare_cpu_allocation(builder, row_widths, monkeypatch):
    module_globals = builder.allocate_per_req_cache.__globals__

    class _Arena:
        entry_bytes = 64

        def view(self, name):
            return None

    monkeypatch.setitem(module_globals, "StateArena", lambda *args, **kwargs: object())
    monkeypatch.setitem(module_globals, "SplitStateArena", lambda arenas: _Arena())
    builder._swa_dtype = torch.bfloat16
    builder._state_dtype = torch.float32
    builder._field_window_layers = set()
    builder._arena_planes = [[]]
    builder.num_layers = 3
    builder._plane_row_widths = lambda: row_widths
    builder.model_runner.device = torch.device("cpu")


@pytest.mark.parametrize(
    ("transfer_config", "expected_pool_size"),
    [
        ({"kv_connector": "lmcache_offload"}, 0),
        ({"kv_connector": "mooncake"}, 32),
        ({"kv_connector": "moriio"}, 32),
    ],
)
def test_allocate_per_req_cache_sizes_pd_staging_for_transfer_topology(
    v4_builder_cls,
    monkeypatch,
    transfer_config,
    expected_pool_size,
):
    builder, _, _, row_widths = _transfer_builder(
        v4_builder_cls,
        kv_dtype="bf16",
        transfer_config=transfer_config,
    )
    _prepare_cpu_allocation(builder, row_widths, monkeypatch)

    allocated = builder.allocate_per_req_cache({STATE_SLOT_CLASS: 2})

    assert allocated["v4_state_pool_size"] == expected_pool_size
    assert allocated["v4_state_pool"].numel() == expected_pool_size * 16


def test_v4_sidecar_offload_rejects_pipeline_parallelism_before_registration(
    v4_builder_cls,
):
    builder, *_ = _transfer_builder(
        v4_builder_cls,
        kv_dtype="fp8",
        transfer_config={"kv_connector": "LMCacheOffloadConnector"},
        pipeline_parallel_size=2,
    )

    with pytest.raises(
        NotImplementedError,
        match=(
            r"DeepSeek-V4 KV transfer/PD and sidecar offload.*"
            r"pipeline parallelism.*unsupported"
        ),
    ):
        builder.get_kv_transfer_tensors()
