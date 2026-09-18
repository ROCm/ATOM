# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contracts for attention-builder views consumed by LMCache MP."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.offload.mp.backend import _build_cache_views
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.mla_kv_pool import MlaKvPool
from atom.model_ops.attentions.pool_layout.entry_arena import EntryField

_MISSING = object()
_REPO_ROOT = Path(__file__).parents[1]


def _module(name: str, **attributes):
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


@contextmanager
def _temporary_modules(replacements):
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


def _load_source_module(module_name: str, relative_path: str, replacements):
    path = _REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with _temporary_modules(replacements):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


def _builder_bases():
    class AttentionBackend:
        pass

    class CommonAttentionBuilder:
        pass

    return _module(
        "atom.model_ops.attentions.backends",
        AttentionBackend=AttentionBackend,
        CommonAttentionBuilder=CommonAttentionBuilder,
    )


def _sub_pool_module():
    return _module(
        "atom.model_ops.attentions.pool_layout.sub_pool_spec",
        SubPoolSpec=type("SubPoolSpec", (), {}),
        page_pool=lambda size: size,
    )


@pytest.fixture(scope="module")
def mha_export_method():
    # Execute only the shipped exporter: importing the full backend requires a
    # GPU AITER build, while this PAGE-view contract is plain torch geometry.
    source = _REPO_ROOT / "atom/model_ops/attentions/aiter_attention.py"
    module = ast.parse(source.read_text())
    builder = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "AiterAttentionMetadataBuilder"
    )
    method = next(
        node
        for node in builder.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_kv_transfer_tensors"
    )
    namespace = {}
    exec(  # noqa: S102 -- execute the local exporter under the CPU fixture
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["get_kv_transfer_tensors"]


@pytest.fixture(scope="module")
def mla_builder_cls():
    def noop(*args, **kwargs):
        return None

    aiter = _module(
        "aiter",
        decode_update_mla_metadata_v1=noop,
        get_mla_metadata_info_v1=noop,
        get_mla_metadata_v1=noop,
        dtypes=SimpleNamespace(d_dtypes={"fp8": torch.uint8}),
    )
    aiter.__path__ = []
    replacements = {
        "aiter": aiter,
        "triton": _module("triton", next_power_of_2=lambda value: value),
        "atom.distributed.dcp_utils": _module(
            "atom.distributed.dcp_utils",
            dcp_persistent_supported=noop,
            get_dcp_rank=lambda: 0,
            get_dcp_world_size=lambda: 1,
        ),
        "atom.distributed.pcp_utils": _module(
            "atom.distributed.pcp_utils",
            get_pcp_world_size=lambda: 1,
            pcp_is_enabled=lambda: False,
            pcp_pad_dense=noop,
            pcp_pad_len=noop,
            pcp_round_robin_query_indices=noop,
        ),
        "atom.model_engine.scheduler": _module(
            "atom.model_engine.scheduler", ScheduledBatch=type("ScheduledBatch", (), {})
        ),
        "atom.model_ops.attention_mla": _module(
            "atom.model_ops.attention_mla",
            _MLA_MIN_HEADS=16,
            _MLA_SPLIT_BUDGET_AUTO=-1,
            MLAAttention=type("MLAAttention", (), {}),
            mla_dcp_decode_is_persistent=lambda *args, **kwargs: False,
            mla_dcp_kernel_num_heads=noop,
        ),
        "atom.model_ops.glm5_next.geometry": _module(
            "atom.model_ops.glm5_next.geometry",
            effective_kpool_size=noop,
            topk_output_width=noop,
        ),
        "atom.utils": _module(
            "atom.utils",
            CpuGpuBuffer=type("CpuGpuBuffer", (), {}),
            envs=SimpleNamespace(
                ATOM_MLA_PAGE_SIZE=1,
                ATOM_USE_TRITON_MLA=False,
                ATOM_USE_TRITON_MLA_SHUFFLE_KV=False,
            ),
            upload_numpy=noop,
        ),
        "atom.utils.block_convert": _module(
            "atom.utils.block_convert",
            kv_indices_generate_triton=noop,
            mtp_prepare_decode_mla_kernel=noop,
        ),
        "atom.utils.forward_context": _module(
            "atom.utils.forward_context",
            AttentionMetaData=type("AttentionMetaData", (), {}),
            Context=type("Context", (), {}),
        ),
        "atom.model_ops.attentions.backends": _builder_bases(),
        "atom.model_ops.attentions.pool_layout.sub_pool_spec": _sub_pool_module(),
    }
    module = _load_source_module(
        "atom.model_ops.attentions._test_lmcache_mp_aiter_mla",
        "atom/model_ops/attentions/aiter_mla.py",
        replacements,
    )
    return module.AiterMLAMetadataBuilder


def _assert_region_view_geometry(transfer):
    assert len(transfer.block_regions) == len(transfer.block_tensor_views)
    for region, view in zip(
        transfer.block_regions, transfer.block_tensor_views, strict=True
    ):
        assert view.ndim == 3
        assert view.is_contiguous()
        assert view.data_ptr() == region.base_addr
        assert view[0].numel() * view.element_size() == region.unit_bytes
        assert view.numel() * view.element_size() == region.total_bytes


def test_minimax_m3_builder_publishes_gqa_and_index_views_without_tp_collapse(
    mha_export_method,
):
    num_blocks = 3
    pool = MhaKvPool(
        layers=2,
        block_size=128,
        num_kv_heads=1,
        head_dim=128,
        kv_dtype=torch.float8_e4m3fnuz,
        extra_fields=(EntryField("index", 1, (128 * 128,), torch.uint8),),
    )
    pool.allocate(num_blocks, "cpu")
    builder = SimpleNamespace(kv_pools={"gqa4": pool})

    transfer = mha_export_method(builder)
    transfer.set_block_count(num_blocks)

    assert transfer.tp_replication_factor == 1
    assert len(transfer.block_regions) == 4 * 2 + 1
    assert all(
        region.semantic_role.startswith("mha.gqa4.")
        for region in transfer.block_regions
    )
    assert all(view.shape[:2] == (num_blocks, 1) for view in transfer.block_tensor_views)
    _assert_region_view_geometry(transfer)
    cache_views = _build_cache_views(transfer, num_blocks=num_blocks)
    assert len(cache_views.tensors) == len(transfer.block_regions)

    transfer.block_tensor_views[-1][1].fill_(7)
    assert torch.all(pool.field_view("index", 0, torch.uint8, (num_blocks, -1))[1] == 7)


@pytest.mark.parametrize("index_layers,index_rows", [(0, 0), (2, 2), (1, 1)])
def test_mla_builder_publishes_latent_and_index_views(
    mla_builder_cls, index_layers, index_rows
):
    # A real pool covers dense, sparse, and compact/pooled index layouts. The
    # runner no longer owns kv_cache/index_cache after the pool refactor.
    runner = SimpleNamespace(
        config=SimpleNamespace(tensor_parallel_size=8),
    )
    pool = MlaKvPool(
        layers=2,
        block_size=2,
        entry_dim=7,
        kv_dtype=torch.float16,
        index_layers=index_layers,
        index_rows_per_block=index_rows,
        index_dim=3,
        index_dtype=torch.uint8,
    )
    pool.allocate(2, "cpu")
    builder = mla_builder_cls.__new__(mla_builder_cls)
    builder.model_runner = runner
    builder.kv_pool = pool

    transfer = builder.get_kv_transfer_tensors()
    # ModelRunner fixes the scheduler id space after all contributors publish.
    assert transfer.num_blocks == 0
    transfer.set_block_count(2)

    assert [tuple(view.shape) for view in transfer.block_tensor_views] == [
        (2, 2, 7),
        (2, 2, 7),
    ] + [(2, index_rows, 3)] * index_layers
    # Main's DCP transfer contract intentionally collapses physical rows into
    # transport roles. MP still keeps them distinct through the plane index in
    # each tensor key (``page.<index>.<role>``).
    assert [region.semantic_role for region in transfer.block_regions] == [
        "mla.kv",
        "mla.kv",
    ] + ["dsa.index_cache"] * index_layers
    assert transfer.block_region_consumer_indices is None
    assert transfer.tp_replication_factor == 8
    _assert_region_view_geometry(transfer)
    cache_views = _build_cache_views(transfer, num_blocks=2)
    # Arena alignment padding is allocated but is not cache payload.
    assert cache_views.bytes_per_block == 2 * 2 * 7 * 2 + index_layers * index_rows * 3
    # Transfer writes must update the allocation used by attention kernels.
    transfer.block_tensor_views[0][1].fill_(7)
    assert torch.all(pool.layer("kv", 0)[1] == 7)
