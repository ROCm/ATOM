# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""BF16 sparse prefill plans each row whole: the granularity helper, and that
the metadata builder and the DCP rebuild hand it to `get_mla_metadata_v1`.

Against the stand-in AITER, so Pre Checkin (no AITER) runs this file."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_under_stubs

with atom_under_stubs() as load:
    attention_mla = load("atom.model_ops.attention_mla")
    aiter_mla = load("atom.model_ops.attentions.aiter_mla")
    head_groups = load("atom.model_ops.attentions.sparse_mla_head_groups")

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
granularity = head_groups.sparse_prefill_kv_granularity
WORK = head_groups.PERSISTENT_WORK_BUFFERS


@pytest.mark.parametrize(
    "dtype, row_width, page_size, expected",
    [
        (BF16, 2048, 1, 2048),
        # GLM-5.3 pooled rows: top-k plus the unscored tail pool.
        (BF16, 2176, 1, 4096),
        (BF16, 2048, 64, 2048),
        (BF16, 8, 1, 16),
        (FP8, 2048, 1, 16),
        (FP8, 2048, 64, 64),
    ],
)
def test_granularity(dtype, row_width, page_size, expected):
    assert granularity(dtype, dtype, row_width, page_size) == expected


def test_granularity_rounds_to_the_next_power_of_two():
    power = 1
    for row_width in range(1, 70000):
        while power < row_width:
            power *= 2
        assert granularity(BF16, BF16, row_width) == max(power, 16)


@pytest.mark.parametrize(
    "row_width, applies",
    [(2048, True), (1025, True), (1024, False), (2176, False)],
)
def test_head_groups_take_rows_that_round_to_one_region(row_width, applies):
    q = torch.empty(4, 128, 576, dtype=BF16)
    o = torch.empty(4, 128, 512, dtype=BF16)
    kv = torch.empty(8, 576, dtype=BF16)
    assert head_groups.SparsePrefillHeadGroups.applies(q, o, kv, row_width) is applies


def _capture(monkeypatch, module):
    seen = {}

    def fake_get_mla_metadata_v1(*args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(module, "get_mla_metadata_v1", fake_get_mla_metadata_v1)
    return seen


def test_builder_plans_with_its_granularity(monkeypatch):
    seen = _capture(monkeypatch, aiter_mla)
    builder = object.__new__(aiter_mla.AiterMLAMetadataBuilder)
    builder.dcp_sparse_prefill_nonps = False
    builder.block_size = 1
    builder.dtype_q = builder.dtype_kv = BF16
    builder.sparse_prefill_kv_granularity = 4096
    work = {name: torch.zeros(1) for name in WORK}
    builder._sparse_prefill_work_views = lambda prefix, heads: work
    md = SimpleNamespace(
        sparse_cu_seqlens_q=torch.arange(3, dtype=torch.int32),
        sparse_kv_indptr=torch.arange(3, dtype=torch.int32),
    )

    builder._sparse_prefill_work_plan(md, "", 16, torch.ones(2, dtype=torch.int32))

    assert seen["kv_granularity"] == 4096
    for name in WORK:
        assert getattr(md, f"sparse_prefill_{name}") is work[name]


@pytest.mark.parametrize(
    "work_prefix, kv_dtype, expected",
    [("sparse_prefill_", BF16, 4096), ("sparse_prefill_", FP8, 16), ("", BF16, 16)],
)
def test_dcp_rebuild_plans_sparse_prefill_rows_whole(
    monkeypatch, work_prefix, kv_dtype, expected
):
    seen = _capture(monkeypatch, attention_mla)
    layer = attention_mla.MLAAttention.__new__(attention_mla.MLAAttention)
    layer._sparse_prefill_row_width = 2176
    md = SimpleNamespace(
        max_seqlen_q=1,
        **{f"{work_prefix}{name}": torch.zeros(1) for name in WORK},
    )
    heads = 16
    rows = torch.arange(3, dtype=torch.int32)

    layer._rebuild_sparse_dcp_persistent_metadata(
        md,
        torch.empty(2, heads, 576, dtype=BF16),
        torch.empty(8, 576, dtype=kv_dtype),
        rows,
        rows,
        torch.ones(2, dtype=torch.int32),
        heads,
        work_prefix=work_prefix,
    )

    assert seen["kv_granularity"] == expected
