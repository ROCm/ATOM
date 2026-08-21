# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contracts for DeepSeek-V4's FP8 paged-prefill indexer metadata.

The scorer runs on ROCm, but the page-table expansion is ordinary PyTorch and
must remain testable without AITER or a GPU.  Load the real V4 metadata builder
behind small import stubs, then exercise the same rebuild entry points used by
plain prefill, PCP, and TBO.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_MISSING = object()
_ROWS_PER_PAGE = 64


@contextmanager
def _stub_v4_runtime_imports():
    """Stub only unavailable GPU imports while loading the real builder."""

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
        "fp4_indexer_enabled",
        "fp4_mqa_prefill_parallel_unit_num",
        "hca_compress_paged_offsets",
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
def v4_module():
    module_name = "_atom_test_v4_indexer_paged_metadata"
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
    return module


def _builder(v4_module, *, backend: str = "auto"):
    builder_cls = v4_module.DeepseekV4AttentionMetadataBuilder
    builder = builder_cls.__new__(builder_cls)
    builder._indexer_fp4 = False
    builder._fp8_indexer_prefill_backend = backend
    builder.csa_rows_per_block = _ROWS_PER_PAGE
    # Production stages the legacy committed cumsum through a CpuGpuBuffer.
    # A fresh CPU tensor preserves its value and, importantly, its per-build
    # ownership without requiring pinned memory or a device.
    builder._stage = lambda _name, value: torch.from_numpy(value.copy())
    return builder


def _metadata(v4_module, *, counts, batch_ids, block_tables):
    md = v4_module.AttentionMetaData_DSV4()
    counts_np = np.asarray(counts, dtype=np.int32)
    md.n_committed_csa_per_seq_cpu = counts_np
    md.n_committed_csa_per_seq = torch.tensor(counts_np, dtype=torch.int32)
    md.batch_id_per_token = torch.tensor(batch_ids, dtype=torch.int32)
    md.block_tables = torch.tensor(block_tables, dtype=torch.int32)
    md.state = v4_module.AttnState.PREFILL_PREFIX
    return md


def _build(builder, md, positions, *, prefix: str = ""):
    return builder._build_v4_indexer_meta(
        attn_metadata=md,
        positions_gpu=torch.tensor(positions, dtype=torch.int32),
        scheduled_bs=len(md.n_committed_csa_per_seq_cpu),
        total_tokens=len(positions),
        device=torch.device("cpu"),
        buf_prefix_ubatch=prefix,
    )


def test_page_table_expansion_covers_zero_and_64_row_boundaries(v4_module):
    """0/1/63/64/65 committed rows require 0/1/1/1/2 live pages."""

    builder = _builder(v4_module)
    counts = [0, 1, 63, 64, 65]
    # Unique, deliberately non-monotonic physical pages make a wrong row or a
    # synthesized contiguous table immediately visible.
    block_tables = [
        [41, 7, 90],
        [13, 55, 91],
        [29, 3, 92],
        [61, 17, 93],
        [5, 47, 94],
    ]
    md = _metadata(
        v4_module,
        counts=counts,
        batch_ids=[0, 1, 2, 3, 4],
        block_tables=block_tables,
    )

    # (position + 1) // 4 gives exactly the boundary vector above.
    meta = _build(builder, md, [0, 3, 251, 255, 259])

    assert meta["visible_end_gpu"].tolist() == counts
    assert meta["paged_prefill_max_seq_len"] == 65
    pages = meta["paged_prefill_block_tables_per_token"]
    assert pages.shape == (5, 2)
    assert pages.is_contiguous()
    assert torch.equal(pages, md.block_tables[:, :2])


@pytest.mark.parametrize(
    ("backend", "table_width"),
    [
        pytest.param("legacy", 2, id="legacy-never-publishes"),
        pytest.param("auto", 1, id="auto-missing-live-page"),
        pytest.param("paged", 1, id="forced-paged-missing-live-page"),
    ],
)
def test_unusable_paged_metadata_is_not_published(v4_module, backend, table_width):
    builder = _builder(v4_module, backend=backend)
    rows = [[11, 12][:table_width], [21, 22][:table_width]]
    md = _metadata(
        v4_module,
        counts=[64, 65],
        batch_ids=[0, 1],
        block_tables=rows,
    )

    meta = _build(builder, md, [255, 259])

    assert "paged_prefill_block_tables_per_token" not in meta
    assert "paged_prefill_max_seq_len" not in meta


def test_pcp_reindex_rebuilds_pages_for_owned_rows_and_safe_pad(v4_module, monkeypatch):
    """PCP rank 1 owns rows 1/3 plus a padded query with batch id -1."""

    builder = _builder(v4_module)
    md = _metadata(
        v4_module,
        counts=[2, 64, 65],
        batch_ids=[0, 0, 1, 1, 2],
        block_tables=[[31, 32], [41, 42], [51, 52]],
    )
    md.indexer_meta = {"stale": True}
    positions = torch.tensor([3, 7, 251, 255, 259], dtype=torch.int32)

    globals_ = builder._apply_pcp_reindex.__globals__
    monkeypatch.setitem(globals_, "get_pcp_world_size", lambda: 2)
    monkeypatch.setitem(
        globals_,
        "pcp_pad_len",
        lambda n, world: ((n + world - 1) // world) * world,
    )
    # Model rank 1's round-robin stripe: the final index is the padded query.
    monkeypatch.setitem(
        globals_,
        "pcp_round_robin_query_indices",
        lambda padded, world: torch.arange(1, padded, world, dtype=torch.int64),
    )

    local_positions = builder._apply_pcp_reindex(
        md, positions, scheduled_bs=3, total_tokens=5
    )

    assert local_positions.tolist() == [7, 255, 0]
    assert md.batch_id_per_token.tolist() == [0, 1, -1]
    rebuilt = md.indexer_meta
    assert rebuilt["visible_end_gpu"].tolist() == [2, 64, 0]
    # The -1 dummy is unread (visible_end=0), but its page row must still be a
    # legal index_select result. Production maps it to sequence 0.
    assert rebuilt["paged_prefill_block_tables_per_token"].tolist() == [
        [31, 32],
        [41, 42],
        [31, 32],
    ]
    assert rebuilt["paged_prefill_block_tables_per_token"].is_contiguous()


def test_tbo_rebuilds_own_non_aliasing_page_tables(v4_module):
    """Two ubatch builds must not share a page-table allocation or its source."""

    builder = _builder(v4_module)
    block_tables = [[71, 72], [81, 82]]
    ub0_md = _metadata(
        v4_module,
        counts=[63, 65],
        batch_ids=[0, 1],
        block_tables=block_tables,
    )
    ub1_md = _metadata(
        v4_module,
        counts=[63, 65],
        batch_ids=[1, 0],
        block_tables=block_tables,
    )

    ub0 = _build(builder, ub0_md, [251, 259], prefix="ub0_")
    ub0_pages = ub0["paged_prefill_block_tables_per_token"]
    ub0_snapshot = ub0_pages.clone()
    ub1 = _build(builder, ub1_md, [259, 251], prefix="ub1_")
    ub1_pages = ub1["paged_prefill_block_tables_per_token"]

    assert ub0_pages.data_ptr() != ub1_pages.data_ptr()
    assert ub0_pages.data_ptr() != ub0_md.block_tables.data_ptr()
    assert ub1_pages.data_ptr() != ub1_md.block_tables.data_ptr()
    assert ub0_pages.tolist() == [[71, 72], [81, 82]]
    assert ub1_pages.tolist() == [[81, 82], [71, 72]]

    # Simulate the later ubatch being reused/overwritten. The first ubatch and
    # the sequence-level source tables must remain intact until both forwards
    # have consumed them.
    ub1_pages.fill_(-7)
    assert torch.equal(ub0_pages, ub0_snapshot)
    assert ub0_md.block_tables.tolist() == block_tables
    assert ub1_md.block_tables.tolist() == block_tables
