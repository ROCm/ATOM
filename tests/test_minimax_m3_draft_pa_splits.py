# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""``sparse_pa_per_row(max_context_length=)``: a caller's bound on every row's
length (the DSpark draft's sliding window) sizes the split-KV partitions, the
plan's cap included; without one the legacy split stays. On gfx1250 the
window-sized split gives the legacy split's attention up to the merge order."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def m():
    with atom_without_aiter() as load:
        base_attention = load("atom.model_ops.base_attention")
        yield SimpleNamespace(
            dtypes=base_attention.dtypes,
            base_attention=base_attention,
            sparse_attn=load("atom.model_ops.minimax_m3.sparse_attn"),
        )


@pytest.mark.parametrize("bound", [None, 2071])
def test_the_bound_sizes_the_split_and_the_plan_cap(monkeypatch, m, bound):
    seen = {}

    def splits(num_seqs, num_kv_heads, split_kv_blocks=1, **kw):
        seen["bound"] = kw.get("max_context_length")
        return 17 if kw.get("max_context_length") else 8

    fake = ModuleType("aiter.ops.triton.gluon.pa_decode_gluon")
    fake.get_recommended_splits = splits
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(m.sparse_attn, "_sparse_pa_ran_on_asm", lambda *a: False)
    monkeypatch.setattr(m.base_attention, "run_pa_decode", lambda **kw: seen.update(kw))
    rows, pages = 7, 4
    kv = torch.zeros(pages, 1, 8, 16, 16, dtype=torch.bfloat16)
    q = torch.zeros(rows, 8, 128, dtype=torch.bfloat16)
    bt = torch.zeros(rows, pages, dtype=torch.int32)
    lens = torch.ones(rows, dtype=torch.int32)
    owner = SimpleNamespace()
    m.sparse_attn.sparse_pa_per_row(
        q, kv, kv, bt, lens, 1, 1.0, torch.empty_like(q), None, None,
        plan_step_owner=owner, max_context_length=bound,
    )  # fmt: skip
    assert seen["bound"] == bound
    assert seen["max_context_partition_num"] == (17 if bound else 8)
    assert seen["plan_partition_cap"] == seen["max_context_partition_num"]
    assert seen["plan_step_owner"] is owner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("rows", [7, 14])
def test_window_split_is_the_legacy_attention_up_to_merge_order(monkeypatch, m, rows):
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("the draft's FlyDSL PA decode runs on gfx1250")
    monkeypatch.setattr(m.base_attention.envs, "ATOM_PA_FLYDSL", True)
    torch.manual_seed(rows)
    dev, pages, window = "cuda", 4096, 2048
    q = torch.randn(rows, 8, 128, device=dev, dtype=torch.bfloat16)
    k = (torch.randn(pages, 1, 8, 16, 16, device=dev) / 4).to(m.dtypes.fp8)
    v = (torch.randn(pages, 1, 1, 128, 16, device=dev) / 4).to(m.dtypes.fp8)
    scale = torch.rand(pages, 1, 16, device=dev) + 0.5
    width = (window + 16 + 7 + 15) // 16
    bt = torch.randint(0, pages, (rows, width), dtype=torch.int32, device=dev)
    lens = torch.randint(
        window - 15, window + 8, (rows,), dtype=torch.int32, device=dev
    )
    outs = []
    for bound in (None, window + 16 + 7):
        out = torch.empty_like(q)
        m.sparse_attn.sparse_pa_per_row(
            q, k, v, bt, lens, 1, 128**-0.5, out, scale, scale,
            max_context_length=bound,
        )  # fmt: skip
        outs.append(out.float())
    legacy, split = outs
    assert torch.isfinite(split).all()
    # bf16 outputs of the same attention summed in another order
    assert (split - legacy).abs().max() <= 2e-2 * legacy.abs().max()
