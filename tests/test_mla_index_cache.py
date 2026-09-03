# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Compact MLA index-cache tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.models.utils import get_pp_indices
from atom.utils.forward_context import get_published_dcp_local_context_lens

try:
    import aiter  # noqa: F401

    from atom.model_ops.attentions import aiter_mla
    from atom.model_ops.attentions.aiter_mla import (
        AiterMLAMetadataBuilder,
        _pad_prefill_mla_draft_tail,
    )
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"aiter MLA backend unavailable: {exc}", allow_module_level=True)


def test_global_index_cache_layout_excludes_shared_and_keeps_mtp():
    assert aiter_mla._global_index_cache_layer_ids(
        ("full", "shared", "shared", "full"), 4, 2
    ) == (0, 3, 4, 5)


def test_global_index_cache_layout_without_schedule_is_unchanged():
    assert aiter_mla._global_index_cache_layer_ids(None, 4, 1) == (0, 1, 2, 3, 4)


def test_padded_prefill_mla_rows_have_empty_initialized_kv_ranges():
    kv_indptr = torch.tensor([0, 3, 7, 101, 202], dtype=torch.int32)
    kv_last_page_lens = np.array([3, 4, 9, 9], dtype=np.int32)
    block_tables = np.full((4, 3), 17, dtype=np.int32)

    _pad_prefill_mla_draft_tail(
        kv_indptr,
        kv_last_page_lens,
        block_tables,
        scheduled_bs=2,
        running_bs=4,
    )

    assert kv_indptr.tolist() == [0, 3, 7, 7, 7]
    assert kv_last_page_lens.tolist() == [3, 4, 0, 0]
    assert block_tables[:2].tolist() == [[17, 17, 17], [17, 17, 17]]
    assert not block_tables[2:].any()


def test_empty_dp_rank_initializes_every_padded_prefill_mla_row():
    kv_indptr = torch.tensor([0, 101, 202], dtype=torch.int32)
    kv_last_page_lens = np.full(2, 9, dtype=np.int32)
    block_tables = np.full((2, 3), 17, dtype=np.int32)

    _pad_prefill_mla_draft_tail(
        kv_indptr,
        kv_last_page_lens,
        block_tables,
        scheduled_bs=0,
        running_bs=2,
    )

    assert kv_indptr.tolist() == [0, 0, 0]
    assert not kv_last_page_lens.any()
    assert not block_tables.any()


def test_global_index_cache_layout_includes_real_stack_draft_layers():
    """Standalone DSpark MLA drafts share the target pool as N extra rows."""
    assert aiter_mla._global_index_cache_layer_ids(None, 61, 5) == tuple(range(61 + 5))


def test_local_total_layers_adds_mtp_only_on_drafter_stage():
    """Mirror ModelRunner._get_total_num_layers without importing it.

    ModelRunner pulls AITER at import time; the PP/MTP accounting itself is
    just get_pp_indices + optional draft depth on the last stage.
    """
    num_hidden = 6
    num_draft = 2

    start, end = get_pp_indices(num_hidden, 0, 2)
    assert end - start == 3

    start, end = get_pp_indices(num_hidden, 1, 2)
    assert (end - start) + num_draft == 5


def _mock_pp(monkeypatch, rank: int, world_size: int) -> None:
    from aiter.dist import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=rank, world_size=world_size),
    )


def _builder(
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
        _get_total_num_layers=lambda: total_local_layers,
    )
    builder = object.__new__(AiterMLAMetadataBuilder)
    builder.model_runner = runner
    return builder, runner


def test_model_runner_local_total_layers_adds_mtp_only_on_drafter_stage(
    monkeypatch,
):
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
    assert runner._get_total_num_layers() == 3

    runner.drafter = object()
    monkeypatch.setattr(
        model_runner,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=1, world_size=2),
    )
    assert runner._get_total_num_layers() == 5


def test_pp_shared_indexer_uses_the_producer_buffer_width(monkeypatch):
    from atom.model_engine import model_runner
    from atom.model_engine.model_runner import ModelRunner

    runner = object.__new__(ModelRunner)
    runner._pp_share_indexer_ready = False
    runner.is_deepseek_v32 = True
    runner.model = SimpleNamespace(start_layer=0, end_layer=2)
    runner.config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=4,
            indexer_types=("full", "shared", "shared", "full"),
            index_topk=2048,
            index_topk_freq=1,
            index_skip_topk_offset=1,
        )
    )
    runner.attn_metadata_builder = SimpleNamespace(index_topk_out=2176)
    runner.rank_name = "test"
    monkeypatch.setattr(
        model_runner,
        "get_pp_group",
        lambda: SimpleNamespace(
            world_size=2,
            is_first_rank=True,
            is_last_rank=False,
        ),
    )

    runner._setup_pp_shared_indexer()

    assert runner._pp_send_needs_sparse
    assert runner._pp_index_topk == 2176


def test_pp_index_cache_layout_uses_global_layer_ids(monkeypatch):
    non_draft_builder, _ = _builder(
        ("full", "shared", "shared", "full", "shared", "full"),
        total_local_layers=3,
    )
    _mock_pp(monkeypatch, rank=0, world_size=2)
    local_layer_ids, global_layer_ids = non_draft_builder._index_cache_layout()

    assert global_layer_ids == (0, 3, 5, 6)
    assert local_layer_ids == (0,)

    draft_builder, _ = _builder(
        ("full", "shared", "shared", "full", "shared", "full"),
        total_local_layers=4,
    )
    _mock_pp(monkeypatch, rank=1, world_size=2)
    local_layer_ids, global_layer_ids = draft_builder._index_cache_layout()

    assert global_layer_ids == (0, 3, 5, 6)
    assert local_layer_ids == (3, 5, 6)


def test_sub_pool_entry_bytes_uses_compact_index_layer_count(monkeypatch):
    builder, _ = _builder(
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
    index_dim = hf_config.index_head_dim + 4
    aligned_index_dim = ((index_dim + 15) // 16) * 16

    assert builder.sub_pool_specs()[0].entry_bytes == 16 * (
        4 * 576 + 3 * aligned_index_dim
    )


def test_compact_layout_uses_fewer_bytes_than_full_layout(monkeypatch):
    compact, _ = _builder(
        ("full", "shared", "shared", "full", "shared", "full"),
        total_local_layers=4,
    )
    full, _ = _builder(None, total_local_layers=4)
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
    builder, runner = _builder(
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

    assert module.kv_cache == (("kv", 2), (8, 1, 576))
    assert module.indexer.k_cache.kv_cache[0][0] == ("index", 1)
    assert cache_tensor.layer_num == 2
    assert cache_tensor.index_cache.identity == ("index", 1)


def test_build_shared_layer_keeps_main_kv_without_index_slice():
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

    assert module.kv_cache == (("kv", 1), (8, 1, 576))
    assert cache_tensor.index_cache is None


def test_transfer_regions_use_explicit_compact_consumer_map(monkeypatch):
    builder, runner = _builder(
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


def test_hybrid_transfer_regions_compact_both_kv_and_index_rows(monkeypatch):
    builder, runner = _builder(("full",) * 6, total_local_layers=3)
    runner.config.speculative_config = None
    runner.config.hf_config.layer_types = [
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention",
    ]
    runner.full_attention_layers = [1, 3, 5]
    _mock_pp(monkeypatch, rank=1, world_size=2)
    builder.block_ratio = 1
    runner.kv_cache = _FakeTransferStack(2, 100)
    runner.index_cache = _FakeTransferStack(2, 200)
    runner.index_cache_layer_ids = (3, 5)
    runner.config.num_kvcache_blocks = 8

    transfer_tensors = builder.get_kv_transfer_tensors()

    # Global consumer layout: 3 compact MLA rows, then 3 compact index rows.
    # This PP rank owns MLA rows 1/2 and index rows 1/2.
    assert transfer_tensors.block_region_consumer_indices == [1, 2, 4, 5]
    assert len(set(transfer_tensors.block_region_consumer_indices)) == 4


class _FakeMetadataBuffer:
    def __init__(self, size):
        self.np = np.zeros(size, dtype=np.int32)
        self.gpu = torch.zeros(size, dtype=torch.int32)
        self.copy_sizes = []

    def copy_to_gpu(self, size):
        self.copy_sizes.append(size)
        self.gpu[:size].copy_(torch.from_numpy(self.np[:size]))
        return self.gpu[:size]


def _capture_metadata_builder(dcp_world_size, *, is_sparse):
    bs = 4
    var = {
        "slot_mapping": _FakeMetadataBuffer(bs),
        "context_lens": _FakeMetadataBuffer(bs),
        "block_tables": _FakeMetadataBuffer(bs),
        "cu_seqlens_q": _FakeMetadataBuffer(bs + 1),
        "kv_indptr": _FakeMetadataBuffer(bs + 1),
        "kv_indices": _FakeMetadataBuffer(bs),
        "kv_last_page_lens": _FakeMetadataBuffer(bs),
        "positions": _FakeMetadataBuffer(bs),
        "g_kv_indptr": _FakeMetadataBuffer(bs + 1),
    }
    if is_sparse:
        var["sparse_kv_indptr"] = _FakeMetadataBuffer(bs + 1)
        var["sparse_kv_last_page_lens"] = _FakeMetadataBuffer(bs)
    if is_sparse and dcp_world_size > 1:
        var["dcp_local_context_lens"] = _FakeMetadataBuffer(bs)

    builder = object.__new__(AiterMLAMetadataBuilder)
    builder.model_runner = SimpleNamespace(forward_vars=var)
    builder.block_size = 1
    builder.is_sparse = is_sparse
    builder.dcp_world_size = dcp_world_size
    builder._publishes_dcp_local_lens = is_sparse and dcp_world_size > 1
    builder._tbo_full_running_bs = 0
    builder.dtype_q = None
    builder.set_mla_persistent_worker_buffers = lambda *args, **kwargs: {}
    return builder, var


def test_cudagraph_capture_publishes_dcp_local_context_lens():
    builder, var = _capture_metadata_builder(dcp_world_size=4, is_sparse=True)

    metadata, _ = builder.build_for_cudagraph_capture(bs=4)

    torch.testing.assert_close(
        metadata.dcp_local_context_lens, torch.ones(4, dtype=torch.int32)
    )
    assert var["dcp_local_context_lens"].copy_sizes == [4]
    assert (
        get_published_dcp_local_context_lens(metadata, 4)
        is metadata.dcp_local_context_lens
    )


@pytest.mark.parametrize(
    ("dcp_world_size", "is_sparse"),
    [(1, True), (4, False)],
)
def test_cudagraph_capture_keeps_non_sparse_dcp_paths_independent(
    dcp_world_size, is_sparse
):
    builder, var = _capture_metadata_builder(
        dcp_world_size=dcp_world_size, is_sparse=is_sparse
    )
    assert "dcp_local_context_lens" not in var

    metadata, _ = builder.build_for_cudagraph_capture(bs=4)

    assert metadata.dcp_local_context_lens is None


def test_ubatch_metadata_views_full_dcp_local_context_buffer():
    builder, var = _capture_metadata_builder(dcp_world_size=4, is_sparse=True)
    builder._tbo_full_running_bs = 4
    builder._set_ubatch_mla_buffers = lambda *args, **kwargs: None
    var["dcp_local_context_lens"].gpu.copy_(
        torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    )
    p = "ub1_"
    for name, size in (
        ("slot_mapping", 2),
        ("context_lens", 2),
        ("block_tables", 2),
        ("cu_seqlens_q", 3),
        ("kv_indptr", 3),
        ("kv_indices", 2),
        ("kv_last_page_lens", 2),
        ("sparse_kv_indptr", 3),
        ("g_kv_indptr", 3),
    ):
        var[f"{p}{name}"] = _FakeMetadataBuffer(size)
    for name in (
        "work_meta_data",
        "work_info_set",
        "work_indptr",
        "reduce_indptr",
        "reduce_final_map",
        "reduce_partial_map",
    ):
        var[f"{p}{name}"] = torch.empty(1)

    metadata = builder.build_ubatch_metadata(ubatch_idx=1, running_bs=2)

    torch.testing.assert_close(
        metadata.dcp_local_context_lens,
        torch.tensor([13, 14], dtype=torch.int32),
    )
    assert "ub1_dcp_local_context_lens" not in var
