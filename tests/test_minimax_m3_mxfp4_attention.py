# SPDX-License-Identifier: MIT
"""`sparse_pa_mxfp4`: one sparse selection per row over the MXFP4 KV pool (gfx950).

The oracle is attention over the dequantized pages the row selected, in the
order emitted with the tail last, at the row's exact context length. Pinned: the
(token, kv head) fold and its page-id encoding, a pool field that is not one
kv head, the per-step plan refresh (second layer reuses it, next step redoes it)
and the segmenting past the planner's batch limit.
"""

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

    def pool():
        p = torch.randint(
            0,
            256,
            (pages, hkv, PAGE_BYTES),
            device="cuda",
            generator=g,
            dtype=torch.uint8,
        )
        sc = torch.randint(125, 130, (pages, hkv, 512), device="cuda", generator=g)
        p[:, :, 8192:] = sc.to(torch.uint8)
        return p

    k, v = pool(), pool()
    bt = torch.zeros(rows * hkv, TOPK, dtype=torch.int32, device="cuda")
    ctx = torch.zeros(rows * hkv, dtype=torch.int32, device="cuda")
    for r in range(rows * hkv):
        n = int(torch.randint(1, TOPK + 1, (1,), device="cuda", generator=g))
        pg = torch.randperm(pages, device="cuda", generator=g)[:n]
        bt[r, :n] = (pg * hkv + r % hkv).to(torch.int32)
        tail = int(torch.randint(1, 129, (1,), device="cuda", generator=g))
        ctx[r] = (n - 1) * 128 + tail
    q = (torch.randn(rows, GROUP * hkv, 128, device="cuda", generator=g) * 0.7).to(
        torch.float8_e4m3fn
    )
    return k, v, bt, ctx, q


def reference(k, v, bt, ctx, q, hkv):
    rows = q.shape[0]
    kd = dequantize_mxfp4(
        *page_views(k.view(-1, PAGE_BYTES))
    )  # [pages * hkv, 128, 128]
    vd = dequantize_mxfp4(*page_views(v.view(-1, PAGE_BYTES)))
    out = torch.empty(rows, GROUP * hkv, 128, device="cuda")
    for r in range(rows):
        for h in range(hkv):
            i = r * hkv + h
            pages = bt[i, : -(-int(ctx[i]) // 128)].long()
            kk = kd[pages].reshape(-1, 128)[: int(ctx[i])]
            vv = vd[pages].reshape(-1, 128)[: int(ctx[i])]
            qq = q[r, h * GROUP : (h + 1) * GROUP].float()
            p = torch.softmax(qq @ kk.T * SCALE, dim=-1)
            out[r, h * GROUP : (h + 1) * GROUP] = p @ vv
    return out


def rel(a, b):
    return ((a.float() - b).norm() / b.norm()).item()


@pytest.mark.parametrize("hkv", [1, 2])
def test_rows_attend_their_selected_pages(hkv):
    from atom.model_ops.minimax_m3.sparse_attn_mxfp4 import sparse_pa_mxfp4

    k, v, bt, ctx, q = make(37, hkv, seed=hkv)
    out = sparse_pa_mxfp4(
        q, k, v, bt, ctx, hkv, SCALE, torch.bfloat16,
        planned=True, plan_step_owner=types.SimpleNamespace(),
    )  # fmt: skip
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert rel(out, reference(k, v, bt, ctx, q, hkv)) < 3e-3


def test_plans_are_refreshed_once_a_step_and_segmented(monkeypatch):
    from atom.model_ops.minimax_m3 import sparse_attn_mxfp4 as m

    monkeypatch.setattr(m, "_FLYDSL_PLAN_MAX_BATCH", 16)  # 37 rows -> 3 segments
    k, v, bt, ctx, q = make(37, 1, seed=5)
    from atom.model_ops.base_attention import _flydsl_step_refreshed

    calls = []
    real = m._flydsl_prepare_explicit_plan

    def spy(**kw):
        owner = kw["plan_step_owner"]
        before = len(_flydsl_step_refreshed(owner))
        plan = real(**kw)
        calls.append(len(_flydsl_step_refreshed(owner)) - before)  # 1 = refreshed
        return plan

    monkeypatch.setattr(m, "_flydsl_prepare_explicit_plan", spy)
    step = types.SimpleNamespace()
    ref = reference(k, v, bt, ctx, q, 1)
    for _layer in range(3):
        out = m.sparse_pa_mxfp4(
            q, k, v, bt, ctx, 1, SCALE, torch.bfloat16,
            planned=True, plan_step_owner=step,
        )  # fmt: skip
        assert rel(out, ref) < 3e-3
    assert calls == [1, 1, 1, 0, 0, 0, 0, 0, 0]
    # the next step has other lengths: its first layer refreshes, and is right
    ctx2 = (ctx // 2).clamp(min=1)
    bt2 = bt.clone()
    out = m.sparse_pa_mxfp4(
        q, k, v, bt2, ctx2, 1, SCALE, torch.bfloat16,
        planned=True, plan_step_owner=types.SimpleNamespace(),
    )  # fmt: skip
    assert rel(out, reference(k, v, bt2, ctx2, q, 1)) < 3e-3


def test_tbo_ubatches_keep_their_own_plans(monkeypatch):
    """Both ubatches run on one stream with the same row count: ubatch 1's first
    layer must not rewrite the plan ubatch 0's later layers still read."""
    from atom.model_ops.minimax_m3 import sparse_attn_mxfp4 as m
    from atom.utils.tbo import ubatching

    cases = [make(8, 1, seed=11), make(8, 1, seed=12)]
    assert not torch.equal(cases[0][3], cases[1][3])
    owners = [types.SimpleNamespace(), types.SimpleNamespace()]
    plans = {}
    real = m._flydsl_prepare_explicit_plan

    def spy(**kw):
        plan = real(**kw)
        plans.setdefault(ub_now[0], plan)
        return plan

    monkeypatch.setattr(m, "_flydsl_prepare_explicit_plan", spy)
    ub_now = [0]

    def layer(ub):
        ub_now[0] = ub
        monkeypatch.setattr(ubatching, "tbo_current_ubatch_id", lambda: ub)
        k, v, bt, ctx, q = cases[ub]
        out = m.sparse_pa_mxfp4(
            q, k, v, bt, ctx, 1, SCALE, torch.bfloat16,
            planned=True, plan_step_owner=owners[ub],
        )  # fmt: skip
        return rel(out, reference(k, v, bt, ctx, q, 1))

    # ubatch 0 layer 0, ubatch 1 layer 0, ubatch 0 layer 1, ubatch 1 layer 1
    assert all(layer(ub) < 3e-3 for ub in (0, 1, 0, 1))
    assert plans[0] is not plans[1]


def test_prefix_and_chunk_pages_are_read_alike():
    """A chunk of a prompt whose first two pages came from the prefix cache (pool
    pages 30 and 3, written by an earlier request) and whose own tokens fill page 11:
    row t of the chunk selects the prefix pages and its own page, at its causal
    length. The reader takes the pages by id, so the prefix needs nothing of its own
    -- this pins that with pages that are neither adjacent nor in order."""
    from atom.model_ops.minimax_m3.sparse_attn_mxfp4 import sparse_pa_mxfp4

    hkv, prefix, chunk = 1, 256, 40
    k, v, _, _, _ = make(1, hkv, seed=7)
    g = torch.Generator(device="cuda").manual_seed(8)
    q = torch.randn(chunk, GROUP, 128, device="cuda", generator=g) * 0.7
    q = q.to(torch.float8_e4m3fn)
    bt = torch.zeros(chunk, TOPK, dtype=torch.int32, device="cuda")
    bt[:, :3] = torch.tensor([30, 3, 11], dtype=torch.int32, device="cuda")
    ctx = (prefix + torch.arange(1, chunk + 1, device="cuda")).to(torch.int32)
    out = sparse_pa_mxfp4(q, k, v, bt, ctx, hkv, SCALE, torch.bfloat16)
    assert rel(out, reference(k, v, bt, ctx, q, hkv)) < 3e-3
    assert rel(out[-1], out[0]) > 0.01  # a row's own context decides what it sees


@pytest.mark.parametrize("hkv", [1, 2])
@pytest.mark.parametrize("rows", [1, 255, 257, 700])
def test_plan_free_rows_attend_their_selected_pages(rows, hkv):
    """The prefill form: no plan, one direct task a row, any row count."""
    from atom.model_ops.minimax_m3.sparse_attn_mxfp4 import sparse_pa_mxfp4

    k, v, bt, ctx, q = make(rows, hkv, seed=rows + hkv, pages=64)
    out = sparse_pa_mxfp4(q, k, v, bt, ctx, hkv, SCALE, torch.bfloat16)
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert rel(out, reference(k, v, bt, ctx, q, hkv)) < 3e-3


@pytest.mark.parametrize("planned", [True, False])
def test_out_is_the_allocated_result_in_the_callers_buffer(planned):
    """`out=` (a slice of a larger buffer, as a DP step's pad rows leave it) holds
    exactly what the allocated result is and the rows past it are left alone."""
    from atom.model_ops.minimax_m3.sparse_attn_mxfp4 import sparse_pa_mxfp4

    rows = 37
    k, v, bt, ctx, q = make(rows, 2, seed=11)
    args = (q, k, v, bt, ctx, 2, SCALE, torch.bfloat16)
    step = types.SimpleNamespace()
    want = sparse_pa_mxfp4(*args, planned=planned, plan_step_owner=step)
    buf = torch.full((rows + 3, *q.shape[1:]), 7.0, dtype=torch.bfloat16, device="cuda")
    got = sparse_pa_mxfp4(*args, planned=planned, plan_step_owner=step, out=buf[:rows])
    assert got.data_ptr() == buf.data_ptr()
    assert torch.equal(buf[:rows], want) and (buf[rows:] == 7).all()
