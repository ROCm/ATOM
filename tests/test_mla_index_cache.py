# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Compact MLA index-cache tests.

CPU layout helpers live in ``mla_cache_layout`` so Pre Checkin can exercise
IndexShare compaction without AITER. Builder / ModelRunner cases skip when
the AITER MLA backend is unavailable.
"""

from types import SimpleNamespace

import pytest

from atom.model_ops.attentions.mla_cache_layout import (
    _aligned_index_cache_dim,
    _global_index_cache_layer_ids,
    _mla_kv_cache_dim,
)
from atom.models.utils import get_pp_indices


def test_global_index_cache_layout_excludes_shared_and_keeps_mtp():
    assert _global_index_cache_layer_ids(
        ("full", "shared", "shared", "full"), 4, 2
    ) == (0, 3, 4, 5)


def test_global_index_cache_layout_without_schedule_is_unchanged():
    assert _global_index_cache_layer_ids(None, 4, 1) == (0, 1, 2, 3, 4)


def test_global_index_cache_layout_includes_real_stack_draft_layers():
    """Standalone DSpark MLA drafts share the target pool as N extra rows."""
    assert _global_index_cache_layer_ids(None, 61, 5) == tuple(range(61 + 5))


def test_cache_dimensions_are_derived_and_index_rows_are_aligned():
    hf_config = SimpleNamespace(kv_lora_rank=480, qk_rope_head_dim=32)

    assert _mla_kv_cache_dim(hf_config) == 512
    assert _aligned_index_cache_dim(111) == 128
    assert _aligned_index_cache_dim(124) == 128
    assert _aligned_index_cache_dim(125) == 144


def test_local_total_layers_adds_mtp_only_on_drafter_stage():
    """Mirror ModelRunner._get_local_total_num_layers without importing it.

    ModelRunner pulls AITER at import time; the PP/MTP accounting itself is
    just get_pp_indices + optional draft depth on the last stage.
    """
    num_hidden = 6
    num_draft = 2

    start, end = get_pp_indices(num_hidden, 0, 2)
    assert end - start == 3

    start, end = get_pp_indices(num_hidden, 1, 2)
    assert (end - start) + num_draft == 5


def _require_aiter_mla():
    try:
        import aiter  # noqa: F401
        from atom.model_ops.attentions import aiter_mla
        from atom.model_ops.attentions.aiter_mla import AiterMLAMetadataBuilder
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"aiter MLA backend unavailable: {exc}")
    return aiter_mla, AiterMLAMetadataBuilder


def _mock_pp(monkeypatch, rank: int, world_size: int) -> None:
    from aiter.dist import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=rank, world_size=world_size),
    )


def _builder(
    AiterMLAMetadataBuilder,
    indexer_types,
    total_local_layers: int,
    *,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    index_head_dim: int = 128,
):
    hf_config = SimpleNamespace(
        num_hidden_layers=len(indexer_types) if indexer_types is not None else 6,
        indexer_types=indexer_types,
        index_head_dim=index_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    )
    runner = SimpleNamespace(
        config=SimpleNamespace(
            hf_config=hf_config,
            kv_cache_dtype="fp8",
            speculative_config=SimpleNamespace(
                draft_model_hf_config=SimpleNamespace(
                    num_nextn_predict_layers=1,
                ),
                use_dspark_with_draft=lambda: False,
            ),
        ),
        block_size=16,
        is_deepseek_v32=True,
        _get_local_total_num_layers=lambda: total_local_layers,
        _get_local_num_target_layers=lambda: 3,
        # Global draft-layer count (ModelRunner._num_draft_kv_layers), not the
        # PP-local row count (_get_local_total_num_layers - target_layers).
        _num_draft_kv_layers=lambda: 1,
    )
    builder = object.__new__(AiterMLAMetadataBuilder)
    builder.model_runner = runner
    return builder, runner


def test_model_runner_local_total_layers_adds_mtp_only_on_drafter_stage(
    monkeypatch,
):
    _require_aiter_mla()
    from atom.model_engine import model_runner
    from atom.model_engine.model_runner import ModelRunner

    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        hf_config=SimpleNamespace(num_hidden_layers=6),
        speculative_config=SimpleNamespace(
            draft_model_hf_config=SimpleNamespace(num_nextn_predict_layers=2),
            use_dspark_with_draft=lambda: False,
        ),
    )

    monkeypatch.setattr(
        model_runner,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=2),
    )
    assert runner._get_local_total_num_layers() == 3

    runner.drafter = object()
    monkeypatch.setattr(
        model_runner,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=1, world_size=2),
    )
    assert runner._get_local_total_num_layers() == 5


def test_num_draft_kv_layers_counts_integrated_dspark_stages():
    _require_aiter_mla()
    from atom.model_engine.model_runner import ModelRunner

    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        model="/data/DeepSeek-V4-Pro-DSpark",
        hf_config=SimpleNamespace(num_hidden_layers=61),
        speculative_config=SimpleNamespace(
            draft_model_hf_config=SimpleNamespace(
                dspark_block_size=8,
                dspark_num_layers=3,
                num_nextn_predict_layers=1,
            ),
            use_dspark_with_draft=lambda: False,
            use_dspark=lambda: True,
        ),
    )
    assert runner._num_draft_kv_layers() == 3


def test_pp_index_cache_layout_uses_global_layer_ids(monkeypatch):
    _, AiterMLAMetadataBuilder = _require_aiter_mla()
    non_draft_builder, _ = _builder(
        AiterMLAMetadataBuilder,
        ("full", "shared", "shared", "full", "shared", "full"),
        total_local_layers=3,
    )
    _mock_pp(monkeypatch, rank=0, world_size=2)
    local_layer_ids, global_layer_ids = non_draft_builder._index_cache_layout()

    assert global_layer_ids == (0, 3, 5, 6)
    assert local_layer_ids == (0,)

    draft_builder, _ = _builder(
        AiterMLAMetadataBuilder,
        ("full", "shared", "shared", "full", "shared", "full"),
        total_local_layers=4,
    )
    _mock_pp(monkeypatch, rank=1, world_size=2)
    local_layer_ids, global_layer_ids = draft_builder._index_cache_layout()

    assert global_layer_ids == (0, 3, 5, 6)
    assert local_layer_ids == (3, 5, 6)


def test_sub_pool_entry_bytes_uses_compact_index_layer_count(monkeypatch):
    aiter_mla, AiterMLAMetadataBuilder = _require_aiter_mla()
    builder, _ = _builder(
        AiterMLAMetadataBuilder,
        ("full", "shared", "shared", "full", "shared", "full"),
        total_local_layers=4,
    )
    _mock_pp(monkeypatch, rank=1, world_size=2)
    fake_fp8 = SimpleNamespace(itemsize=1)
    monkeypatch.setattr(
        aiter_mla,
        "dtypes",
        SimpleNamespace(d_dtypes={"fp8": fake_fp8}, fp8=fake_fp8),
    )
    hf_config = builder.model_runner.config.hf_config
    mla_cache_dim = _mla_kv_cache_dim(hf_config)
    aligned_index_dim = _aligned_index_cache_dim(hf_config.index_head_dim)

    assert builder.sub_pool_specs()[0].entry_bytes == 16 * (
        4 * mla_cache_dim + 3 * aligned_index_dim
    )


def test_sub_pool_entry_bytes_support_nonstandard_dimensions(monkeypatch):
    aiter_mla, AiterMLAMetadataBuilder = _require_aiter_mla()
    builder, _ = _builder(
        AiterMLAMetadataBuilder,
        ("full", "shared", "shared", "full", "shared", "full"),
        total_local_layers=4,
        kv_lora_rank=480,
        qk_rope_head_dim=32,
        index_head_dim=111,
    )
    _mock_pp(monkeypatch, rank=1, world_size=2)
    fake_fp8 = SimpleNamespace(itemsize=1)
    monkeypatch.setattr(
        aiter_mla,
        "dtypes",
        SimpleNamespace(d_dtypes={"fp8": fake_fp8}, fp8=fake_fp8),
    )

    assert builder.sub_pool_specs()[0].entry_bytes == 16 * (4 * 512 + 3 * 128)


def test_compact_layout_uses_fewer_bytes_than_full_layout(monkeypatch):
    aiter_mla, AiterMLAMetadataBuilder = _require_aiter_mla()
    compact, _ = _builder(
        AiterMLAMetadataBuilder,
        ("full", "shared", "shared", "full", "shared", "full"),
        total_local_layers=4,
    )
    full, _ = _builder(AiterMLAMetadataBuilder, None, total_local_layers=4)
    _mock_pp(monkeypatch, rank=1, world_size=2)
    fake_fp8 = SimpleNamespace(itemsize=1)
    monkeypatch.setattr(
        aiter_mla,
        "dtypes",
        SimpleNamespace(d_dtypes={"fp8": fake_fp8}, fp8=fake_fp8),
    )

    full_entry_bytes = full.sub_pool_specs()[0].entry_bytes
    compact_entry_bytes = compact.sub_pool_specs()[0].entry_bytes
    assert full_entry_bytes - compact_entry_bytes == 16 * 144


def test_allocate_index_cache_uses_compact_shape_and_map(monkeypatch):
    aiter_mla, AiterMLAMetadataBuilder = _require_aiter_mla()
    builder, runner = _builder(
        AiterMLAMetadataBuilder,
        ("full", "shared", "shared", "full", "shared", "full"),
        total_local_layers=4,
    )
    _mock_pp(monkeypatch, rank=1, world_size=2)
    fake_fp8 = SimpleNamespace(itemsize=1)
    monkeypatch.setattr(
        aiter_mla,
        "dtypes",
        SimpleNamespace(d_dtypes={"fp8": fake_fp8}, fp8=fake_fp8),
    )
    runner.num_physical_kvcache_blocks = 8
    runner.physical_block_size = 1
    allocations = []

    def fake_zeros(*shape, **kwargs):
        allocations.append((shape, kwargs))
        return SimpleNamespace(shape=shape)

    monkeypatch.setattr(aiter_mla.torch, "zeros", fake_zeros)

    out = builder.allocate_kv_cache_tensors(num_kv_heads=1, num_draft_layers=1)

    assert out["kv_cache"].shape == (4, 8, 1, 576)
    assert out["index_cache"].shape == (3, 8, 1, 144)
    assert out["index_cache_layer_ids"] == (3, 5, 6)
    assert out["index_cache_layer_map"] == {3: 0, 5: 1, 6: 2}
    assert len(allocations) == 2


class _FakeCache:
    def __init__(self, prefix):
        self.prefix = prefix

    def __getitem__(self, index):
        return _FakeCacheSlice((self.prefix, index))


class _FakeCacheSlice:
    def __init__(self, identity):
        self.identity = identity

    def view(self, *shape):
        return self.identity, shape


class _FakeTransferTensor:
    def __init__(self, address):
        self._address = address

    def stride(self, dim):
        assert dim == 0
        return 1

    def element_size(self):
        return 1

    def numel(self):
        return 8

    def data_ptr(self):
        return self._address


class _FakeTransferStack:
    def __init__(self, num_layers, address_base):
        self.shape = (num_layers,)
        self._layers = [
            _FakeTransferTensor(address_base + layer_id)
            for layer_id in range(num_layers)
        ]

    def __getitem__(self, index):
        return self._layers[index]


def test_build_kv_cache_tensor_binds_compact_index_slice():
    _, AiterMLAMetadataBuilder = _require_aiter_mla()
    builder = object.__new__(AiterMLAMetadataBuilder)
    runner = SimpleNamespace(
        kv_cache=_FakeCache("kv"),
        index_cache=_FakeCache("index"),
        index_cache_layer_map={3: 0, 5: 1},
        is_deepseek_v32=True,
        num_physical_kvcache_blocks=8,
        physical_block_size=1,
        aligned_index_dim=144,
        config=SimpleNamespace(
            max_model_len=1024,
            hf_config=SimpleNamespace(kv_lora_rank=480, qk_rope_head_dim=32),
        ),
    )
    builder.model_runner = runner
    module = SimpleNamespace(
        base_attention=object(),
        use_mla=True,
        layer_num=5,
        indexer=SimpleNamespace(
            k_cache=SimpleNamespace(kv_cache=[None]),
        ),
    )

    cache_tensor = builder.build_kv_cache_tensor(layer_id=2, module=module)

    assert module.kv_cache == (("kv", 2), (8, 1, 512))
    assert module.indexer.k_cache.kv_cache[0][0] == ("index", 1)
    assert cache_tensor.layer_num == 2
    assert cache_tensor.index_cache.identity == ("index", 1)


def test_build_shared_layer_keeps_main_kv_without_index_slice():
    _, AiterMLAMetadataBuilder = _require_aiter_mla()
    builder = object.__new__(AiterMLAMetadataBuilder)
    runner = SimpleNamespace(
        kv_cache=_FakeCache("kv"),
        index_cache=_FakeCache("index"),
        index_cache_layer_map={0: 0},
        is_deepseek_v32=True,
        num_physical_kvcache_blocks=8,
        physical_block_size=1,
        aligned_index_dim=144,
        config=SimpleNamespace(
            max_model_len=1024,
            hf_config=SimpleNamespace(kv_lora_rank=480, qk_rope_head_dim=32),
        ),
    )
    builder.model_runner = runner
    module = SimpleNamespace(
        base_attention=object(),
        use_mla=True,
        layer_num=1,
        indexer=None,
    )

    cache_tensor = builder.build_kv_cache_tensor(layer_id=1, module=module)

    assert module.kv_cache == (("kv", 1), (8, 1, 512))
    assert cache_tensor.index_cache is None


def test_transfer_regions_use_explicit_compact_consumer_map(monkeypatch):
    _, AiterMLAMetadataBuilder = _require_aiter_mla()
    builder, runner = _builder(
        AiterMLAMetadataBuilder,
        ("full", "shared", "shared", "full", "shared", "full"),
        total_local_layers=4,
    )
    _mock_pp(monkeypatch, rank=1, world_size=2)
    builder.block_ratio = 1
    runner.kv_cache = _FakeTransferStack(4, 100)
    runner.index_cache = _FakeTransferStack(3, 200)
    runner.index_cache_layer_ids = (3, 5, 6)
    runner.config.num_kvcache_blocks = 8

    transfer_tensors = builder.get_kv_transfer_tensors()

    assert len(transfer_tensors.block_regions) == 7
    assert transfer_tensors.block_region_consumer_indices == [
        3,
        4,
        5,
        6,
        8,
        9,
        10,
    ]
