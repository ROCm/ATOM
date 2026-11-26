# SPDX-License-Identifier: MIT
"""MiniMax-M3 sparse paged attention past one FlyDSL plan's sequences.

A sparse row plans one sequence a (token, local KV head), and aiter's planner
keeps one LDS int32 a sequence: 81792 on gfx1250. PP x TP1 keeps all 4 KV heads
a rank, so a 32768-token prefill (131072 sequences) needs two plans. These run
on the CPU: the planner and the paged decode are stubs that record what they
were given, so the chunk slices, output offsets and plan refreshes are checked
without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="the sparse attention imports aiter")

import aiter.ops.flydsl.pa_decode as flydsl_pa
from aiter.ops.triton.gluon import pa_decode_gluon

from atom.model_ops import base_attention as ba
from atom.model_ops.minimax_m3 import sparse_attn, sparse_attn_nvfp4

GFX1250_PLAN = 81792


@pytest.fixture(autouse=True)
def tiny_plan_floor(monkeypatch):
    """Rows within 4096 sequences skip the cap lookup; the CPU cases use tiny caps."""
    monkeypatch.setattr(ba, "_FLYDSL_PLAN_MAX_BATCH", 1)


@pytest.mark.parametrize(
    "rows, kv_heads, want",
    [
        (32768, 4, [(0, 16384), (16384, 32768)]),  # PP4 x TP1 at 32K: 2 plans
        (20448, 4, [(0, 20448)]),  # the old ceiling: one plan
        (20449, 4, [(0, 10225), (10225, 20449)]),
        (32768, 1, [(0, 32768)]),  # TP4: one KV head a rank, unchanged
        (65536, 4, [(0, 16384), (16384, 32768), (32768, 49152), (49152, 65536)]),
        (0, 4, [(0, 0)]),
    ],
)
def test_row_chunks(monkeypatch, rows, kv_heads, want):
    monkeypatch.setattr(ba, "_flydsl_plan_max_batch", lambda: GFX1250_PLAN)
    chunks = ba.flydsl_row_chunks(rows, kv_heads)
    assert chunks == want
    assert all((end - start) * kv_heads <= GFX1250_PLAN for start, end in chunks)
    assert [s for s, _ in chunks[1:]] == [e for _, e in chunks[:-1]]


@pytest.fixture
def planner(monkeypatch):
    """The real `_flydsl_prepare_explicit_plan` over a stub planner: every
    plan records the lengths it was last refreshed with."""
    calls = []

    def plan_pa_decode(lengths, num_kv_heads, *, plan=None, **_kw):
        if plan is None:
            plan = SimpleNamespace(
                max_partitions=1,
                capacity=1,
                num_kv_heads=num_kv_heads,
                reduce_info=torch.empty(lengths.numel(), 2),
            )
        plan.lengths = lengths.clone()
        calls.append(plan)
        return plan

    monkeypatch.setattr(flydsl_pa, "plan_pa_decode", plan_pa_decode)
    monkeypatch.setattr(ba, "_FLYDSL_RUNTIME_PLANS", {})
    monkeypatch.setattr(ba, "_FLYDSL_PLAN_SCRATCH", {})
    monkeypatch.setattr(
        ba.torch.cuda,
        "current_stream",
        lambda device=None: SimpleNamespace(cuda_stream=0),
    )
    monkeypatch.setattr(ba.torch.cuda, "is_current_stream_capturing", lambda: False)
    return calls


def _nvfp4_inputs(rows, kv_heads, group=2, head_dim=8, pages=6, topk=3):
    q = torch.randn(rows, kv_heads * group, head_dim, dtype=torch.bfloat16)
    kp = torch.zeros(pages, kv_heads, 4, head_dim // 2, dtype=torch.uint8)
    ks = torch.zeros(pages, kv_heads, 4, 1, dtype=torch.uint8)
    seqs = rows * kv_heads
    table = torch.arange(seqs * topk, dtype=torch.int32).view(seqs, topk)
    # Distinct lengths per sequence, so a plan refreshed for another chunk shows.
    ctx = torch.arange(1, seqs + 1, dtype=torch.int32)
    return q, kp, ks, table, ctx


def _record_pa_decode(monkeypatch):
    """pa_decode stub: out = q + ctx (per sequence); records each launch."""
    launches = []

    def pa_decode(out, q, k, v, ctx, table, scale, ql, *, work_plan, **kw):
        assert torch.equal(work_plan.lengths, ctx), "launched on a stale plan"
        assert ctx.numel() == q.shape[0] == table.shape[0] <= work_plan_cap[0]
        out.copy_(q + ctx.view(-1, 1, 1).to(q.dtype))
        launches.append((q.data_ptr(), q.shape[0], table, kw))

    work_plan_cap = [GFX1250_PLAN]
    monkeypatch.setattr(flydsl_pa, "pa_decode", pa_decode)
    monkeypatch.setattr(pa_decode_gluon, "get_recommended_splits", lambda *_a, **_k: 1)
    return launches, work_plan_cap


def _run_nvfp4(q, kp, ks, table, ctx, step_owner, cache_owner, **kw):
    return sparse_attn_nvfp4._flydsl_attention(
        q, kp, ks, kp, ks, table, ctx, 1.0, step_owner, cache_owner, **kw
    )


@pytest.mark.parametrize("rows, kv_heads, cap", [(10, 4, 16), (9, 4, 16), (7, 2, 4)])
def test_nvfp4_chunks_write_every_row_once(monkeypatch, planner, rows, kv_heads, cap):
    """Chunked rows land at their own offsets, each chunk on a plan refreshed
    for its own lengths in every layer of the step (no stale step plan)."""
    monkeypatch.setattr(ba, "_flydsl_plan_max_batch", lambda: cap)
    launches, plan_cap = _record_pa_decode(monkeypatch)
    plan_cap[0] = cap
    q, kp, ks, table, ctx = _nvfp4_inputs(rows, kv_heads)
    group = q.shape[1] // kv_heads
    want = q.view(-1, group, q.shape[-1]) + ctx.view(-1, 1, 1).to(q.dtype)
    md = SimpleNamespace()
    for _layer in range(3):  # later layers of the step reuse the plans
        out = _run_nvfp4(q, kp, ks, table, ctx, md, md)
        assert torch.equal(out.view_as(want), want)
    chunks = ba.flydsl_row_chunks(rows, kv_heads)
    assert len(chunks) > 1
    assert len(launches) == 3 * len(chunks)
    seq_bytes = group * q.shape[-1] * q.element_size()
    for (ptr, n, tab, _), (start, end) in zip(launches, chunks * 3):
        assert ptr == q.data_ptr() + start * kv_heads * seq_bytes
        assert n == (end - start) * kv_heads
        assert torch.equal(tab, table[start * kv_heads : end * kv_heads])


def test_nvfp4_one_plan_keeps_the_step_plan(monkeypatch, planner):
    """Within one plan (TP4 at 32K, PP4 at 20352) the call is unchanged: one
    launch on the whole tensors, the plan refreshed once a step."""
    monkeypatch.setattr(ba, "_flydsl_plan_max_batch", lambda: 64)
    launches, plan_cap = _record_pa_decode(monkeypatch)
    plan_cap[0] = 64
    q, kp, ks, table, ctx = _nvfp4_inputs(16, 4)
    md = SimpleNamespace()
    for _layer in range(3):
        _run_nvfp4(q, kp, ks, table, ctx, md, md)
    assert len(launches) == 3
    for ptr, n, tab, _ in launches:
        assert (ptr, n) == (q.data_ptr(), 64)
        assert tab.data_ptr() == table.data_ptr() and tab.shape == table.shape
    assert len(planner) == 1  # created once, never refreshed again this step


def test_nvfp4_chunked_quantized_output_slices(monkeypatch, planner):
    """A per-token FP8 output (one KV head) splits with its rows; a padded
    bucket's rows go to the last chunk only."""
    monkeypatch.setattr(ba, "_flydsl_plan_max_batch", lambda: 4)
    launches, plan_cap = _record_pa_decode(monkeypatch)
    plan_cap[0] = 4
    rows, pad = 7, 3
    q, kp, ks, table, ctx = _nvfp4_inputs(rows, 1)
    quant = torch.zeros(rows + pad, *q.shape[1:], dtype=torch.uint8)
    scale = torch.zeros(rows + pad, 1)
    md = SimpleNamespace()
    _run_nvfp4(
        q, kp, ks, table, ctx, md, md, quantized_output=quant, output_scale=scale
    )
    got = [(kw["quantized_output"].data_ptr(), kw["quantized_output"].shape[0],
            kw["output_scale"].shape[0], kw["quantized_pad_rows"]) for *_, kw in launches]  # fmt: skip
    row_bytes = quant[0].numel()
    assert got == [
        (quant.data_ptr(), 4, 4, 0),
        (quant.data_ptr() + 4 * row_bytes, 3 + pad, 3 + pad, pad),
    ]


def _fp8_inputs(rows, kv_heads, group=2, head_dim=128):
    q = torch.randn(rows, kv_heads * group, head_dim, dtype=torch.bfloat16)
    k = torch.zeros(3, kv_heads, 16, 8, 16, dtype=torch.uint8)
    bt = torch.arange(rows * kv_heads * 2, dtype=torch.int32).view(-1, 2)
    ctx = torch.arange(1, rows * kv_heads + 1, dtype=torch.int32)
    return q, k, bt, ctx


@pytest.mark.parametrize("gluon", [False, True])
def test_fp8_rows_chunk_only_without_gluon(monkeypatch, gluon):
    """The FP8 path chunks where FlyDSL is the only paged decode (gfx1250);
    gfx950's gluon takes any row count, so it keeps one call there."""
    monkeypatch.setattr(ba, "_flydsl_plan_max_batch", lambda: 8)
    monkeypatch.setattr(sparse_attn, "pa_decode_gluon_supports_arch", lambda: gluon)
    monkeypatch.setattr(sparse_attn.envs, "ATOM_PA_FLYDSL", True)
    monkeypatch.setattr(sparse_attn, "_sparse_pa_ran_on_asm", lambda *_: False)
    monkeypatch.setattr(pa_decode_gluon, "get_recommended_splits", lambda *_a, **_k: 1)
    calls = []
    monkeypatch.setattr(ba, "run_pa_decode", lambda **kw: calls.append(kw))
    rows, kv = 5, 4
    q, k, bt, ctx = _fp8_inputs(rows, kv)
    out = torch.empty_like(q)
    md = SimpleNamespace()
    sparse_attn.sparse_pa_per_row(
        q, k, k, bt, ctx, kv, 1.0, out, None, None,
        plan_step_owner=md, plan_cache_owner=md,
    )  # fmt: skip
    if gluon:
        assert len(calls) == 1 and calls[0]["plan_step_owner"] is md
        assert calls[0]["q"].shape[0] == rows * kv
        return
    chunks = ba.flydsl_row_chunks(rows, kv)
    assert chunks == [(0, 2), (2, 4), (4, 5)]
    assert len(calls) == len(chunks)
    seq_bytes = q.shape[1] // kv * q.shape[2] * q.element_size()
    for kw, (start, end) in zip(calls, chunks):
        assert kw["plan_step_owner"] is None and kw["plan_cache_owner"] is md
        assert kw["q"].data_ptr() == q.data_ptr() + start * kv * seq_bytes
        assert kw["output"].data_ptr() == out.data_ptr() + start * kv * seq_bytes
        assert torch.equal(kw["context_lens"], ctx[start * kv : end * kv])
        assert torch.equal(kw["block_tables"], bt[start * kv : end * kv])
