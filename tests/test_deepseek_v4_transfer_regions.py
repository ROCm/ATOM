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
from typing import ClassVar

import numpy as np
import pytest
import torch

from atom.kv_transfer.disaggregation.types import (
    INDEX_CACHE_FP4_PREFIX,
    KVTransferRegion,
    KVTransferTensors,
    PageRegion,
)
from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec
from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.model_engine.state_runtime import StateRuntime, StateTransfer
from atom.model_ops.attentions.pool_layout.v4_pool_fields import (
    fp4_indexer_block_fields,
    fp8_indexer_block_fields,
    indexer_block_regions,
    main_kv_plane_fields,
)
from atom.model_ops.attentions.pool_layout.v4_pool_geometry import UnifiedPoolGeometry

_MISSING = object()
_HEAD_DIM = 16
_ROPE_HEAD_DIM = 4
_CSA_ROWS_PER_BLOCK = 64
_INDEX_HEAD_DIM = 128
_INDEX_ROW_BYTES = _INDEX_HEAD_DIM + 4
_CSA_REGION_COUNT = 2
_FP4_K_TILES = _INDEX_HEAD_DIM // 128


def test_generic_transfer_tensors_default_to_no_full_slot_expectation():
    transfer = KVTransferTensors()

    assert transfer.expected_full_slot_region_count is None


def _page_region(blocks: int, unit_bytes: int, role: str) -> KVTransferRegion:
    return KVTransferRegion(
        base_addr=0x1000,
        total_bytes=blocks * unit_bytes,
        unit_bytes=unit_bytes,
        semantic_role=role,
    )


class TestTheBlockIdSpacePageRegionsAreAddressedIn:
    """`num_blocks` is the scheduler's count, and every region has to hold it.

    Three backends used to state it themselves, in two different units -- one
    in scheduler blocks and one in its own page. Where a backend's page and the
    scheduler's block are the same *size* the two spell the same number, so the
    disagreement is invisible until a config makes them differ, at which point a
    peer computes `base + block_id * unit_bytes` against a stride the other end
    does not share. Same number is not same unit; the count comes from the one
    caller holding the scheduler's, and every region is checked against it.
    """

    def test_regions_in_the_scheduler_block_are_accepted(self):
        transfer = KVTransferTensors(
            pages=[
                PageRegion(_page_region(8, 2080, "kv.layer_0")),
                PageRegion(_page_region(8, 132, "index.layer_0")),
            ]
        )

        transfer.set_block_count(8)

        assert transfer.num_blocks == 8

    def test_a_region_registered_in_another_unit_is_refused(self):
        """MLA pages at one token: registering per row rather than per
        scheduler block gives `block_ratio` times as many units of
        `1/block_ratio` the bytes -- the same total, which is why no byte count
        catches it."""
        transfer = KVTransferTensors(
            pages=[
                PageRegion(_page_region(8, 2080, "kv.layer_0")),
                PageRegion(_page_region(8 * 16, 2080 // 16, "kv.layer_1")),
            ]
        )

        with pytest.raises(ValueError, match="kv.layer_1 holds 128 blocks"):
            transfer.set_block_count(8)

    def test_a_region_that_does_not_divide_into_whole_blocks_is_refused(self):
        transfer = KVTransferTensors(
            pages=[
                PageRegion(KVTransferRegion(0x1000, 100, 32, semantic_role="ragged"))
            ]
        )

        with pytest.raises(ValueError, match="does not divide into whole blocks"):
            transfer.set_block_count(3)

    def test_the_count_is_not_something_a_backend_can_pass(self):
        """`init=False` is the enforcement: a backend cannot state a count that
        its own regions contradict, because it cannot state one at all."""
        with pytest.raises(TypeError):
            KVTransferTensors(num_blocks=8)


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
    # Replaced, not reused: other test modules leave their own, narrower
    # parallel_state stub in sys.modules.
    aiter_dist = types.ModuleType("aiter.dist")
    aiter_dist.__path__ = []
    parallel_state = types.ModuleType("aiter.dist.parallel_state")
    parallel_state.get_tensor_model_parallel_world_size = lambda: 1
    parallel_state.get_tp_group = lambda: None

    pcp_utils = types.ModuleType("atom.distributed.pcp_utils")
    for name in (
        "get_pcp_rank",
        "get_pcp_world_size",
        "pcp_is_enabled",
        "pcp_owned_rows_by_request",
        "pcp_pad_dense",
        "pcp_pad_indptr",
        "pcp_pad_len",
        "pcp_reindex_ragged",
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
    for name in (
        "build_v4_paged_decode_indptr",
        "fp4_indexer_enabled",
        "hca_compress_paged_offsets",
        "plan_context_lens",
        "v4_decode_split_plan",
        "v4_uniform_split_table",
        "write_v4_paged_decode_indices",
        "write_v4_paged_prefill_indices",
    ):
        setattr(kernels, name, lambda *args, **kwargs: None)
    # No aiter here, so the persistent HCA decode is never usable.
    kernels.hca_persist = SimpleNamespace(
        unusable_reason=lambda **kwargs: "no aiter in this test",
        prepare=lambda *args, **kwargs: None,
    )

    # The indexer's scorer needs aiter's kernels; the builder only names them.
    indexer = types.ModuleType("atom.model_ops.indexer")
    indexer.NATURAL, indexer.PRESHUFFLE = "natural", "preshuffle"
    for name in ("ConcatKeys", "RowSplit", "IndexRows"):
        setattr(indexer, name, type(name, (), {}))
    for name in ("Fp4LaunchStorages", "concat_rows", "fp4_rows", "fp8_rows"):
        setattr(indexer, name, lambda *args, **kwargs: None)
    # Its sequences are aiter's FP4 MQA plan; the builder only names the type.
    ragged = types.ModuleType("atom.model_ops.fp4_mqa_ragged_metadata")
    ragged.Fp4MqaRaggedMetadata = type("Fp4MqaRaggedMetadata", (), {})

    replacements = {
        "aiter": aiter,
        "aiter.jit": aiter_jit,
        "aiter.jit.utils": aiter_jit_utils,
        "aiter.jit.utils.chip_info": chip_info,
        "aiter.dist": aiter_dist,
        "aiter.dist.parallel_state": parallel_state,
        "atom.distributed.pcp_utils": pcp_utils,
        "atom.model_ops.attentions.backends": backends,
        "atom.model_ops.fp4_mqa_ragged_metadata": ragged,
        "atom.model_ops.indexer": indexer,
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
    # `_stub_v4_runtime_imports` replaces `CommonAttentionBuilder` with an empty
    # class, so nothing the real base sets exists here. This is V4's own block
    # count -- the unit `UnifiedPoolGeometry` above was built in, which is why
    # it is the same `num_blocks` handed to both.
    builder.num_blocks = num_blocks
    builder._kv_fp8 = kv_dtype == "fp8"
    builder._indexer_fp4 = indexer_fp4
    builder.indexer_quant_mode = "fp4" if indexer_fp4 else "per_row_fp8"
    builder.indexer_layout = "fp4-gfx950-preshuffle" if indexer_fp4 else "fp8"
    builder._classical_dtype = torch.uint8 if builder._kv_fp8 else torch.bfloat16
    builder.head_dim = _HEAD_DIM
    builder.rope_head_dim = _ROPE_HEAD_DIM
    builder.csa_layers = [0, 2]
    builder.csa_rows_per_block = _CSA_ROWS_PER_BLOCK
    builder.index_head_dim = _INDEX_HEAD_DIM
    # The two declarations `__init__` would have made, from the same functions
    # the backend calls: this fixture is after the transfer regions, and a
    # hand-written layout here would be one more copy of what they collapse.
    builder._plane_fields = main_kv_plane_fields(
        _HEAD_DIM,
        builder._classical_dtype,
        (_ROPE_HEAD_DIM, torch.bfloat16) if builder._kv_fp8 else None,
    )
    builder._indexer_fields = (
        fp4_indexer_block_fields(_CSA_ROWS_PER_BLOCK, _INDEX_HEAD_DIM)
        if indexer_fp4
        else fp8_indexer_block_fields(_CSA_ROWS_PER_BLOCK, _INDEX_HEAD_DIM, torch.uint8)
    )
    builder._indexer_regions, builder._indexer_block_bytes = indexer_block_regions(
        builder._indexer_fields
    )
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
                _FP4_K_TILES,
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
                _FP4_K_TILES,
                4,
                builder.csa_rows_per_block,
            ),
            dtype=torch.uint8,
        )
        indexer_pools = [indexers, indexer_scales]
    else:
        # Row width spelled out rather than taken off the builder, which no
        # longer knows one. Spelling it independently is what makes the
        # geometry assertions a check on the declaration: change the block and
        # this pool stops covering the sized PAGE unit.
        indexers = torch.empty(
            (
                len(builder.csa_layers),
                num_blocks,
                builder.csa_rows_per_block,
                _INDEX_ROW_BYTES,
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
    # ModelRunner's job in production, and it is the gate as well as the
    # assignment: it refuses a PAGE region that does not divide into exactly
    # this many blocks. Passing the geometry's count here is the claim that
    # V4's PAGE regions are registered per scheduler block.
    transfer.set_block_count(geo.num_blocks)
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


@pytest.mark.parametrize("kv_dtype", ["fp8", "bf16"])
@pytest.mark.parametrize("composite", [False, True])
def test_fp4_indexer_pd_covers_data_and_scales(v4_builder_cls, kv_dtype, composite):
    config = {"kv_connector": "mooncake", "kv_role": "kv_both"}
    if composite:
        config = {
            "kv_connector": "multi",
            "connectors": [
                config,
                {"kv_connector": "lmcache_offload", "kv_role": "offload"},
            ],
        }
    builder, geo, planes, widths = _transfer_builder(
        v4_builder_cls,
        kv_dtype=kv_dtype,
        transfer_config=config,
        indexer_fp4=True,
    )
    _assert_transfer_geometry(builder, geo, planes, widths)


@pytest.mark.parametrize("composite", [False, True])
def test_fp4_indexer_moriio_remains_unsupported(v4_builder_cls, composite):
    config = {"kv_connector": "moriio"}
    if composite:
        config = {"kv_connector": "multi", "connectors": [config]}
    builder, *_ = _transfer_builder(
        v4_builder_cls,
        kv_dtype="fp8",
        transfer_config=config,
        indexer_fp4=True,
    )

    with pytest.raises(
        NotImplementedError,
        match=r"FP4 index PD transfer requires Mooncake",
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

    monkeypatch.setitem(
        module_globals, "EntryMajorArena", lambda *args, **kwargs: object()
    )
    monkeypatch.setitem(module_globals, "SplitEntryMajorArena", lambda arenas: _Arena())
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


@pytest.fixture(scope="module")
def mooncake_page_writer():
    """Run the production descriptor planner with a byte-copy NIC substitute."""
    import ast
    import logging
    import time

    from atom.kv_transfer.disaggregation.port_offset import consumer_region_indices

    path = Path(__file__).parents[1] / (
        "atom/kv_transfer/disaggregation/mooncake/mooncake_connector.py"
    )
    tree = ast.parse(path.read_text())
    names = {"_execute_block_slot_transfer", "_consumer_region_map"}
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "KVTransferRegion": KVTransferRegion,
        "consumer_region_indices": consumer_region_indices,
        "logger": logging.getLogger(__name__),
        "time": time,
        "torch": torch,
    }
    exec(  # noqa: S102 - runs the production planner extracted above
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace


def _fp4_pd_pair(builder_cls, writer, kv_dtype="fp8", coalesce=False):
    from dataclasses import asdict

    builders = [
        _transfer_builder(
            builder_cls,
            kv_dtype=kv_dtype,
            transfer_config={"kv_connector": "mooncake"},
            indexer_fp4=True,
        )[0]
        for _ in range(2)
    ]
    src, dst = [b.get_kv_transfer_tensors() for b in builders]
    conn = SimpleNamespace(
        _block_regions=[(r.base_addr, r.unit_bytes) for r in src.block_regions],
        _block_region_roles=[r.semantic_role for r in src.block_regions],
        _fp4_index_layout=any(
            r.semantic_role is not None
            and r.semantic_role.startswith(INDEX_CACHE_FP4_PREFIX)
            for r in src.block_regions
        ),
        _block_region_consumer_indices=None,
        _block_mr_ends=[[r.total_bytes] for r in src.block_regions],
        _swa_block_regions=src.swa_block_regions,
        _slot_regions=[],
        _coalesce_blocks=coalesce,
        _trace_transfers=False,
        _num_local_layers=4,
        _start_layer=0,
        pp_size=1,
        tp_rank=0,
    )
    for name in ("_execute_block_slot_transfer", "_consumer_region_map"):
        setattr(conn, name, types.MethodType(writer[name], conn))
    req = {
        # FP4 consumers advertise under the FP4-only key (version fence).
        "consumer_block_base_addrs_fp4": [r.base_addr for r in dst.block_regions],
        "consumer_block_bpb": [r.unit_bytes for r in dst.block_regions],
        "consumer_region_roles": [r.semantic_role for r in dst.block_regions],
        "consumer_block_mr_ends": [[r.total_bytes] for r in dst.block_regions],
        "consumer_slot_base_addrs": [],
        "consumer_slot_bps": [],
        "consumer_swa_block_regions": [asdict(r) for r in dst.swa_block_regions],
        "dst_swa_block_ids": [0],
        "dst_slot_index": -1,
    }
    return builders, src, dst, conn, req


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("coalesce", [False, True])
@pytest.mark.parametrize("src_ids,dst_ids", [([2, 0], [0, 2]), ([0, 1], [1, 2])])
def test_fp4_pd_copies_data_scales_and_swa_without_touching_other_blocks(
    v4_builder_cls, mooncake_page_writer, kv_dtype, coalesce, src_ids, dst_ids
):
    import ctypes

    owners, src, dst, conn, req = _fp4_pd_pair(
        v4_builder_cls, mooncake_page_writer, kv_dtype, coalesce
    )
    src_regions = [*src.block_regions, *src.swa_block_regions]
    dst_regions = [*dst.block_regions, *dst.swa_block_regions]
    expected = []
    for n, (sr, dr) in enumerate(zip(src_regions, dst_regions, strict=True)):
        # Different layers and byte offsets expose missing scales, swapped
        # pools, and incorrect block strides instead of just copying zeros.
        payload = bytes((i + 31 * n) % 251 for i in range(sr.total_bytes))
        ctypes.memmove(sr.base_addr, payload, len(payload))
        ctypes.memset(dr.base_addr, 255, dr.total_bytes)
        target = bytearray([255]) * dr.total_bytes
        pairs = zip(src_ids, dst_ids) if n < len(src.block_regions) else [(1, 0)]
        for sb, db in pairs:
            so = sr.unit_addr(sb) - sr.base_addr
            do = dr.unit_addr(db) - dr.base_addr
            target[do : do + dr.unit_bytes] = payload[so : so + sr.unit_bytes]
        expected.append(bytes(target))

    calls = []

    selected_engine = object()

    def write(
        target,
        sources,
        destinations,
        sizes,
        request_id,
        kind,
        *,
        engine=None,
        deadline=None,
    ):
        assert engine is selected_engine
        calls.append(kind)
        for s, d, size in zip(sources, destinations, sizes, strict=True):
            ctypes.memmove(d, s, size)
        return True

    conn._rdma_write = write
    assert conn._execute_block_slot_transfer(
        req,
        "test",
        src_ids,
        dst_ids,
        {"slot_index": -1, "swa_block_ids": [1]},
        "fp4",
        engine=selected_engine,
    )
    assert calls == ["block"]
    for region, wanted in zip(dst_regions, expected, strict=True):
        assert ctypes.string_at(region.base_addr, region.total_bytes) == wanted
    assert owners  # Keep all tensor allocations alive through the copy.


@pytest.mark.parametrize(
    "mismatch",
    [
        "missing_roles",
        "missing_scale",
        "scale_width",
        "swapped_layers",
        "fp8_producer",
        "fp8_consumer",
    ],
)
def test_fp4_pd_layout_mismatch_fails_before_any_write(
    v4_builder_cls, mooncake_page_writer, mismatch
):
    owners, _src, _dst, conn, req = _fp4_pd_pair(v4_builder_cls, mooncake_page_writer)
    if mismatch == "missing_roles":
        req.pop("consumer_region_roles")
    elif mismatch == "missing_scale":
        for key in (
            "consumer_block_base_addrs_fp4",
            "consumer_block_bpb",
            "consumer_region_roles",
        ):
            req[key].pop()
    elif mismatch == "scale_width":
        req["consumer_block_bpb"][-1] *= 2
    elif mismatch == "swapped_layers":
        roles = req["consumer_region_roles"]
        roles[-1], roles[-2] = roles[-2], roles[-1]
    else:
        fp8_builder = _transfer_builder(
            v4_builder_cls,
            kv_dtype="fp8",
            transfer_config={"kv_connector": "mooncake"},
        )[0]
        owners.append(fp8_builder)
        regions = fp8_builder.get_kv_transfer_tensors().block_regions
        if mismatch == "fp8_consumer":
            req["consumer_block_base_addrs"] = [r.base_addr for r in regions]
            req["consumer_block_bpb"] = [r.unit_bytes for r in regions]
            req["consumer_region_roles"] = [r.semantic_role for r in regions]
        else:
            conn._block_regions = [(r.base_addr, r.unit_bytes) for r in regions]
            conn._block_region_roles = [r.semantic_role for r in regions]

    calls = []
    conn._rdma_write = lambda *args: calls.append(args) or True
    with pytest.raises(
        RuntimeError,
        match="mismatch|out of range|disagree|requires the consumer|Cannot layer-map|one-to-one",
    ):
        conn._execute_block_slot_transfer(
            req, "test", [0], [1], {"slot_index": -1, "swa_block_ids": [1]}, "fp4"
        )
    assert not calls


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


# ---- The prefill indexer's row split across TP, as the builder wires it -------


class _RecordingSplit:
    """Stands in for `RowSplit`: records how the builder asks for its split
    and deals the rows round-robin, as `RowSplit.striped` does."""

    calls: ClassVar[list] = []

    @classmethod
    def striped(cls, rows, rank, world, *, group):
        cls.calls.append(((rows, rank, world), {"group": group}))
        return SimpleNamespace(
            rank=rank, world=world, end=len(range(rank, rows, world))
        )


def _rows_by_request(cu, first, stride, owned):
    """`pcp_owned_rows_by_request`, by counting: the rows `first`,
    `first + stride`, ... below each request's start, then `owned`."""
    below = [len(range(first, start, stride)) for start in cu[:-1]]
    return np.array([*below, owned], dtype=np.int32)


def _split_builder(v4_builder_cls, monkeypatch, *, world, rank=3, fp4, bound=None):
    """A builder as TP rank `rank` of `world`, splitting from `bound` index
    logits (the default's, by default)."""
    # The fixture loads the builder module off `sys.modules`; its globals are
    # reachable through any of its functions.
    module = v4_builder_cls._prefill_row_split.__globals__
    _RecordingSplit.calls = []
    monkeypatch.setitem(module, "RowSplit", _RecordingSplit)
    monkeypatch.setitem(
        module, "get_tp_group", lambda: SimpleNamespace(rank_in_group=rank)
    )
    monkeypatch.setitem(module, "visible_csa", lambda positions: positions)
    if bound is None:
        monkeypatch.delenv("ATOM_V4_INDEXER_QSPLIT_MIN_LOGITS", raising=False)
        bound = module["envs"].ATOM_V4_INDEXER_QSPLIT_MIN_LOGITS
    builder = v4_builder_cls.__new__(v4_builder_cls)
    builder._indexer_qsplit_world = world
    builder._indexer_qsplit_min_logits = bound
    builder._indexer_fp4 = fp4
    return builder, module


def _prefill_step(builder, module, monkeypatch, positions, rows, n_committed):
    """What `builder` hands on for a prefill step of `rows` new rows per
    request: FP4's `_index_rows` arguments, or FP8's `concat_rows`'."""
    handed = {}

    def index_rows(metadata, row_ends, width, **kwargs):
        handed.update(kwargs, row_ends=row_ends, width=width)

    def concat(row_ends, keys, split):
        handed.update(row_ends=row_ends, keys=keys, split=split)

    builder._index_rows = index_rows
    builder._stage = lambda name, values: torch.as_tensor(values)
    monkeypatch.setitem(module, "concat_rows", concat)
    monkeypatch.setitem(module, "ConcatKeys", lambda **kwargs: kwargs)
    monkeypatch.setitem(module, "pcp_owned_rows_by_request", _rows_by_request)
    metadata = SimpleNamespace(
        n_committed_csa_per_seq_cpu=n_committed,
        batch_id_per_q_token=_batch_ids(rows),
        block_tables=None,
    )
    builder._build_v4_prefill_index_rows(
        metadata,
        positions,
        rows.shape[0],
        int(rows.sum()),
        "the step's starts",
        "",
        rows,
    )
    return handed


def _batch_ids(rows):
    return torch.repeat_interleave(torch.arange(rows.shape[0]), torch.from_numpy(rows))


@pytest.mark.parametrize("fp4", [True, False], ids=["fp4", "fp8"])
def test_split_prefill_rows_are_every_world_th_row(v4_builder_cls, monkeypatch, fp4):
    """Split across TP, the prefill rows a rank publishes are every `world`-th
    of the step's from its rank, each with its own bound and request: FP4
    grouped by request (a rank may hold none of one), at most the most any
    request has; FP8 each at its request's run of the gathered keys."""
    world = 4
    rows = np.array([5, 1, 9, 3], dtype=np.int32)
    starts = np.array([100, 0, 37, 2000], dtype=np.int32)
    n_committed = (starts + rows) // 4
    cu = np.concatenate(([0], np.cumsum(rows)))
    cu_committed = np.concatenate(([0], np.cumsum(n_committed)))
    total = int(cu[-1])
    positions = torch.cat([torch.arange(s, s + n) for s, n in zip(starts, rows)])
    for rank in range(world):
        builder, module = _split_builder(
            v4_builder_cls, monkeypatch, world=world, rank=rank, fp4=fp4, bound=0
        )
        handed = _prefill_step(
            builder, module, monkeypatch, positions, rows, n_committed
        )
        mine = slice(rank, total, world)
        assert _RecordingSplit.calls == [((total, rank, world), {"group": "tp"})]
        assert handed["row_ends"].tolist() == positions[mine].tolist()
        batch_ids = _batch_ids(rows)[mine]
        if fp4:
            owned = [
                len([i for i in range(cu[b], cu[b + 1]) if i % world == rank])
                for b in range(rows.shape[0])
            ]
            assert handed["query_start_loc"].tolist() == [0, *np.cumsum(owned)]
            assert handed["max_qlen"] == max(max(owned), 1)
            assert handed["row_to_batch"].tolist() == batch_ids.tolist()
        else:
            row_starts = cu_committed[batch_ids.numpy()]
            assert handed["keys"]["row_starts"].tolist() == row_starts.tolist()
            assert handed["keys"]["total"] == cu_committed[-1]
        assert handed["split"].rank == rank


@pytest.mark.parametrize("fp4", [True, False], ids=["fp4", "fp8"])
def test_split_ranks_share_a_cold_chunk_s_scoring_evenly(
    v4_builder_cls, monkeypatch, fp4
):
    """A prefill row scores the keys its position sees, so a cold chunk's last
    rows cost the most: contiguous shares would leave the last of 8 ranks
    1.9 times the mean. Every 8th row each, the ranks' keys are within 1%."""
    world, total = 8, 1 << 13
    rows = np.array([total], dtype=np.int32)
    keys = []
    for rank in range(world):
        builder, module = _split_builder(
            v4_builder_cls, monkeypatch, world=world, rank=rank, fp4=fp4, bound=0
        )
        handed = _prefill_step(
            builder, module, monkeypatch, torch.arange(total), rows, rows // 4
        )
        keys.append(int(handed["row_ends"].sum()))
    assert max(keys) / (sum(keys) / world) < 1.01


@pytest.mark.parametrize("fp4", [True, False], ids=["fp4", "fp8"])
def test_unsplit_prefill_rows_are_the_whole_step(v4_builder_cls, monkeypatch, fp4):
    """Below the bound every rank publishes every row, as without a split:
    FP4 grouped by the caller's starts at the step's longest request."""
    builder, module = _split_builder(v4_builder_cls, monkeypatch, world=8, fp4=fp4)
    rows = np.array([3, 4], dtype=np.int32)
    positions = torch.tensor([5, 6, 7, 0, 1, 2, 3])
    handed = _prefill_step(
        builder, module, monkeypatch, positions, rows, np.array([2, 1], np.int32)
    )
    assert _RecordingSplit.calls == []
    assert handed["split"] is None
    assert handed["row_ends"].tolist() == positions.tolist()
    if fp4:
        assert handed["query_start_loc"] == "the step's starts"
        assert handed["max_qlen"] is None
        assert handed["row_to_batch"].tolist() == [0, 0, 0, 1, 1, 1, 1]
    else:
        assert handed["keys"]["row_starts"].tolist() == [0, 0, 0, 2, 2, 2, 2]


class _Metadata(SimpleNamespace):
    """Stands in for the metadata `prepare_prefill` takes from its base and
    promotes to V4's: a class of its own, so `__class__` can be reassigned."""


def test_prepare_prefill_gates_the_split_by_each_request_s_new_rows(
    v4_builder_cls, monkeypatch
):
    """`prepare_prefill` hands the split each scheduled request's new rows, not
    its context: lost on the way, no prefill would ever split."""
    builder, module = _split_builder(v4_builder_cls, monkeypatch, world=8, fp4=True)
    starts, rows = [8, 16, 0], [3, 5, 3]
    cu = np.cumsum([0, *rows], dtype=np.int32)
    context = np.add(starts, rows, dtype=np.int32)
    positions = torch.cat([torch.arange(s, s + n) for s, n in zip(starts, rows)])
    metadata = _Metadata(
        block_tables=torch.zeros(3, 4, dtype=torch.int32),
        cu_seqlens_q=torch.from_numpy(cu),
        replay_start_cpu=None,
        replay_end_cpu=None,
        n_committed_csa_per_seq_cpu=context // 4,
        batch_id_per_q_token=torch.zeros(11, dtype=torch.long),
    )
    monkeypatch.setitem(module, "AttentionMetaData_DSV4", _Metadata)
    # `_stub_v4_runtime_imports` leaves the base builder an empty class.
    monkeypatch.setattr(
        v4_builder_cls.__mro__[1],
        "prepare_prefill",
        lambda self, batch, running_bs, running_tokens: (metadata, positions),
        raising=False,
    )
    builder.model_runner = SimpleNamespace(
        forward_vars={
            "positions": SimpleNamespace(np=positions.numpy()),
            "cu_seqlens_q": SimpleNamespace(np=cu),
            "context_lens": SimpleNamespace(np=context),
        }
    )
    builder.csa_layers = [0]
    builder._populate_state_slot_mappings = lambda *args, **kwargs: (
        None,
        np.arange(3, dtype=np.int32),
    )
    for name in (
        "_populate_state_slot_in",
        "_build_compress_plans",
        "_attach_v4_per_fwd_meta",
        "_build_paged_prefill_meta",
        "_attach_tbo_prefill_cpu_lens",
        "_index_rows",
    ):
        setattr(builder, name, lambda *args, **kwargs: None)
    asked = []
    builder._prefill_row_split = lambda *args: asked.append(args)
    batch = SimpleNamespace(
        total_seqs_num_prefill=3, total_tokens_num_prefill=11, is_dummy_run=False
    )

    builder.prepare_prefill(batch, 3, 11)

    [(tokens, new_rows, n_committed)] = asked
    assert tokens == 11
    assert new_rows.tolist() == rows
    # Handed through only: the metadata above set it; its setter is stubbed.
    assert n_committed.tolist() == (context // 4).tolist()


@pytest.mark.parametrize("world, given", [(1, True), (8, False)])
def test_unsplit_prefill_rows_carry_no_split(v4_builder_cls, monkeypatch, world, given):
    """No split without a split world, or for a caller with no per-request rows
    (PCP, TBO), however big the step."""
    builder, _ = _split_builder(v4_builder_cls, monkeypatch, world=world, fp4=True)
    big = np.array([1 << 14], dtype=np.int32)
    split = builder._prefill_row_split(1 << 14, big if given else None, big)
    assert split is None and _RecordingSplit.calls == []


@pytest.mark.parametrize("over", [0, 1], ids=["at_bound", "under"])
@pytest.mark.parametrize(
    "rows, n_committed, logits",
    [
        (8, 126, 1000),
        (1 << 14, 1 << 12, 1 << 25),
        (1 << 10, (1 << 16) + (1 << 8), (1 << 26) + (1 << 17)),
    ],
    ids=["8_rows", "cold_16k_chunk", "prefix_hit"],
)
def test_a_step_splits_from_the_keys_its_rows_see(
    v4_builder_cls, monkeypatch, rows, n_committed, logits, over
):
    """A step's logits are the keys its rows see: up to the committed keys,
    one fewer every 4 rows back. 8 rows at 126 keys are 1000 logits, not
    1008; a cold 16K chunk's rows see 2K keys on average, 2^25 logits, half
    of rows times keys; 1K new rows past a 256K-token prefix hit see 2^26 +
    2^17. A step splits from exactly its bound."""
    builder, _ = _split_builder(
        v4_builder_cls, monkeypatch, world=8, fp4=True, bound=logits + over
    )
    split = builder._prefill_row_split(
        rows,
        np.array([rows], dtype=np.int32),
        np.array([n_committed], dtype=np.int32),
    )
    assert (split is not None) == (over == 0)


def test_split_pairs_each_request_s_rows_with_its_own_keys(v4_builder_cls, monkeypatch):
    """A step's logits count each request's rows at its own committed keys,
    the scoring those rows do, not at the plane's width (the longest
    request's): one row of a long request beside three of a fresh one is one
    row's scoring."""
    builder, _ = _split_builder(
        v4_builder_cls, monkeypatch, world=8, fp4=True, bound=1000
    )
    rows = np.array([1, 3], dtype=np.int32)
    n_committed = np.array([999, 0], dtype=np.int32)
    assert builder._prefill_row_split(4, rows, n_committed) is None


def test_split_logits_do_not_wrap_in_int32(v4_builder_cls, monkeypatch):
    """16K rows of a 1M-token request are 2^32 logits, past int32 (the host
    metadata's dtype): wrapped, the biggest steps would stay whole."""
    builder, _ = _split_builder(v4_builder_cls, monkeypatch, world=8, fp4=True)
    rows = np.array([1 << 14], dtype=np.int32)
    n_committed = np.array([1 << 18], dtype=np.int32)
    assert builder._prefill_row_split(1 << 14, rows, n_committed) is not None


_MOONCAKE = {"kv_connector": "mooncake"}


@pytest.mark.parametrize(
    "enabled, tbo, kv, world",
    [
        ("1", False, {}, 8),
        ("1", False, {**_MOONCAKE, "kv_role": "kv_producer"}, 8),
        ("1", False, {"kv_connector": "lmcache_offload", "kv_role": "kv_consumer"}, 8),
        ("1", True, {}, 1),
        ("1", False, {**_MOONCAKE, "kv_role": "kv_consumer"}, 1),
        ("0", False, {}, 1),
    ],
    ids=[
        "splits",
        "pd_prefill",
        "offload_load_only",
        "prefill_tbo",
        "pd_decode",
        "flag_off",
    ],
)
def test_a_run_that_splits_warms_the_split_s_gather(
    v4_builder_cls, monkeypatch, enabled, tbo, kv, world
):
    """With the flag on, plain TP8 splits, a P/D prefill instance and a
    load-only offload node included; the flag off keeps the rows whole, and
    so, for now, does prefill TBO, whose ubatches' rows do not take the split
    yet, and so does a P/D decode instance, which only recomputes prefills.
    A run that splits gathers once at startup, on the TP group and from this
    rank's own slot as the split does, so what RCCL allocates on its first
    all-gather is inside the memory profile; one that does not never
    gathers."""
    module = v4_builder_cls._indexer_qsplit_world_size.__globals__
    monkeypatch.setitem(module, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setitem(
        module,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=3, device_group="tp"),
    )
    monkeypatch.setitem(module, "get_pcp_world_size", lambda: 1)
    monkeypatch.setitem(module, "is_plugin_mode", lambda: False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 8)
    gathers = []
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_into_tensor",
        lambda out, mine, group: gathers.append((out, mine, group)),
    )
    monkeypatch.setenv("ATOM_V4_INDEXER_QSPLIT", enabled)
    builder = v4_builder_cls.__new__(v4_builder_cls)
    builder.indexer_layout = "fp8"
    builder.device = "cpu"
    config = SimpleNamespace(
        enable_dp_attention=False,
        enable_tbo=tbo,
        enable_tbo_decode=False,
        decode_context_parallel_size=1,
        kv_transfer_config=kv,
    )
    assert builder._indexer_qsplit_world_size(config) == world
    if world == 1:
        assert gathers == []
        return
    [(out, mine, group)] = gathers
    assert group == "tp" and out.numel() == world
    assert mine.numel() == 1 and mine.data_ptr() == out[3:].data_ptr()
