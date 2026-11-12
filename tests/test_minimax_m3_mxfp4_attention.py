# SPDX-License-Identifier: MIT
"""`sparse_pa_mxfp4` (gfx950): one sparse selection per row over the MXFP4 KV pool,
against attention over the dequantized pages a row selected, at its exact context."""

import types

import pytest
import torch

from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES, page_views
from atom.mono.weights.formats import dequantize_mxfp4

aiter = pytest.importorskip("aiter")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    or not hasattr(aiter.ops, "flydsl"),
    reason="needs gfx950 and aiter's FlyDSL",
)
GROUP, TOPK, SCALE = 16, 16, 128**-0.5


def make(rows, hkv, seed, pages=48):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def ri(lo, hi, *shape):
        return torch.randint(lo, hi, shape, generator=g, device="cuda")

    def pool():
        p = ri(0, 256, pages, hkv, PAGE_BYTES).to(torch.uint8)
        p[:, :, 8192:] = ri(125, 130, pages, hkv, 512).to(torch.uint8)  # scales near 1
        return p

    bt, ctx = (
        torch.zeros(rows * hkv, TOPK).int().cuda(),
        torch.zeros(rows * hkv).int().cuda(),
    )
    for r in range(rows * hkv):
        n = int(ri(1, TOPK + 1, 1))
        bt[r, :n] = (
            torch.randperm(pages, generator=g, device="cuda")[:n].int() * hkv + r % hkv
        )
        ctx[r] = (n - 1) * 128 + int(ri(1, 129, 1))
    q = torch.randn(rows, GROUP * hkv, 128, generator=g, device="cuda") * 0.7
    return pool(), pool(), bt, ctx, q.to(torch.float8_e4m3fn)


def check(out, k, v, bt, ctx, q, hkv):
    kd = dequantize_mxfp4(*page_views(k.view(-1, PAGE_BYTES)))
    vd = dequantize_mxfp4(*page_views(v.view(-1, PAGE_BYTES)))
    ref = torch.empty(q.shape[0], GROUP * hkv, 128, device="cuda")
    for r in range(q.shape[0]):
        for h in range(hkv):
            i, n = r * hkv + h, int(ctx[r * hkv + h])
            pg, hs = bt[i, : -(-n // 128)].long(), slice(h * GROUP, (h + 1) * GROUP)
            p = torch.softmax(
                q[r, hs].float() @ kd[pg].reshape(-1, 128)[:n].T * SCALE, -1
            )
            ref[r, hs] = p @ vd[pg].reshape(-1, 128)[:n]
    assert ((out.float() - ref).norm() / ref.norm()).item() < 3e-3


def attend(k, v, bt, ctx, q, hkv, owner=None, **kw):
    from atom.model_ops.minimax_m3.sparse_attn_mxfp4 import sparse_pa_mxfp4

    return sparse_pa_mxfp4(q, k, v, bt, ctx, hkv, SCALE, torch.bfloat16, plan_step_owner=owner, **kw)  # fmt: skip


@pytest.mark.parametrize(
    "planned,rows,hkv", [(True, 37, 2), (False, 1, 1), (False, 700, 2)]
)
def test_rows_attend_their_selected_pages(planned, rows, hkv):
    """Planned (decode) or plan-free (prefill, any row count), into the caller's buffer
    `out=` (a DP step's pad rows past it are left alone)."""
    k, v, bt, ctx, q = make(rows, hkv, seed=rows + hkv, pages=64)
    buf = torch.full((rows + 3, *q.shape[1:]), 7.0, dtype=torch.bfloat16, device="cuda")
    out = attend(
        k, v, bt, ctx, q, hkv, types.SimpleNamespace(), planned=planned, out=buf[:rows]
    )
    assert out.data_ptr() == buf.data_ptr() and (buf[rows:] == 7).all()
    check(out, k, v, bt, ctx, q, hkv)


def test_plans_are_refreshed_once_a_step_and_segmented(monkeypatch):
    from atom.model_ops.base_attention import _flydsl_step_refreshed
    from atom.model_ops.minimax_m3 import sparse_attn_mxfp4 as m

    monkeypatch.setattr(m, "_FLYDSL_PLAN_MAX_BATCH", 16)  # 37 rows -> 3 segments
    k, v, bt, ctx, q = make(37, 1, seed=5)
    calls, real = [], m._flydsl_prepare_explicit_plan

    def spy(**kw):
        before = len(_flydsl_step_refreshed(kw["plan_step_owner"]))
        plan = real(**kw)
        calls.append(len(_flydsl_step_refreshed(kw["plan_step_owner"])) - before)
        return plan

    monkeypatch.setattr(m, "_flydsl_prepare_explicit_plan", spy)
    step = types.SimpleNamespace()
    for _layer in range(3):
        check(attend(k, v, bt, ctx, q, 1, step, planned=True), k, v, bt, ctx, q, 1)
    assert calls == [1, 1, 1, 0, 0, 0, 0, 0, 0]  # a refresh per segment, then reuse
    ctx2 = (ctx // 2).clamp(
        min=1
    )  # the next step: other lengths, first layer refreshes
    out = attend(k, v, bt, ctx2, q, 1, types.SimpleNamespace(), planned=True)
    check(out, k, v, bt, ctx2, q, 1)


def test_tbo_ubatches_keep_their_own_plans(monkeypatch):
    """Both ubatches run on one stream with the same row count: ubatch 1's first layer
    must not rewrite the plan ubatch 0's later layers still read."""
    from atom.model_ops.minimax_m3 import sparse_attn_mxfp4 as m
    from atom.utils.tbo import ubatching

    cases, plans, now = [make(8, 1, seed=11), make(8, 1, seed=12)], {}, [0]
    owners, real = [
        types.SimpleNamespace() for _ in cases
    ], m._flydsl_prepare_explicit_plan
    monkeypatch.setattr(
        m,
        "_flydsl_prepare_explicit_plan",
        lambda **kw: plans.setdefault(now[0], real(**kw)),
    )
    monkeypatch.setattr(ubatching, "tbo_current_ubatch_id", lambda: now[0])
    for ub in (0, 1, 0, 1):  # ubatch 0 layer 0, ubatch 1 layer 0, ...
        now[0] = ub
        k, v, bt, ctx, q = cases[ub]
        check(
            attend(k, v, bt, ctx, q, 1, owners[ub], planned=True), k, v, bt, ctx, q, 1
        )
    assert plans[0] is not plans[1]
