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

from atom.model_ops.minimax_m3.kv_mxfp4 import PAGE_BYTES, dequantize_pages

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
    kd = dequantize_pages(k.view(-1, PAGE_BYTES))  # [pages * hkv, 128, 128]
    vd = dequantize_pages(v.view(-1, PAGE_BYTES))
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
        q, k, v, bt, ctx, hkv, SCALE, torch.bfloat16, types.SimpleNamespace()
    )
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert rel(out, reference(k, v, bt, ctx, q, hkv)) < 3e-3


def test_plans_are_refreshed_once_a_step_and_segmented(monkeypatch):
    from atom.model_ops.minimax_m3 import sparse_attn_mxfp4 as m

    monkeypatch.setattr(m, "_FLYDSL_PLAN_MAX_BATCH", 16)  # 37 rows -> 3 segments
    k, v, bt, ctx, q = make(37, 1, seed=5)
    calls = []
    real = m._plan

    def spy(owner, context, segment, hkv):
        before = len(m._refreshed(owner))
        plan = real(owner, context, segment, hkv)
        calls.append(len(m._refreshed(owner)) - before)  # 1 = refreshed
        return plan

    monkeypatch.setattr(m, "_plan", spy)
    step = types.SimpleNamespace()
    ref = reference(k, v, bt, ctx, q, 1)
    for _layer in range(3):
        out = m.sparse_pa_mxfp4(q, k, v, bt, ctx, 1, SCALE, torch.bfloat16, step)
        assert rel(out, ref) < 3e-3
    assert calls == [1, 1, 1, 0, 0, 0, 0, 0, 0]
    # the next step has other lengths: its first layer refreshes, and is right
    ctx2 = (ctx // 2).clamp(min=1)
    bt2 = bt.clone()
    out = m.sparse_pa_mxfp4(
        q, k, v, bt2, ctx2, 1, SCALE, torch.bfloat16, types.SimpleNamespace()
    )
    assert rel(out, reference(k, v, bt2, ctx2, q, 1)) < 3e-3


@pytest.mark.parametrize("hkv", [1, 2])
@pytest.mark.parametrize("rows", [1, 7, 255, 256, 257, 511, 512, 700])
def test_plan_free_rows_attend_their_selected_pages(rows, hkv):
    """The prefill form: no plan, one direct task a row, any row count."""
    from atom.model_ops.minimax_m3.sparse_attn_mxfp4 import sparse_pa_mxfp4

    k, v, bt, ctx, q = make(rows, hkv, seed=rows + hkv, pages=64)
    out = sparse_pa_mxfp4(q, k, v, bt, ctx, hkv, SCALE, torch.bfloat16, None)
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert rel(out, reference(k, v, bt, ctx, q, hkv)) < 3e-3


def test_plan_free_is_the_planned_path_with_one_task_a_row():
    """Bit for bit the kernel run with a plan of max_partitions=1 (the splits coincide),
    and the production planned split (default partitions) agrees to rounding."""
    from aiter.ops.flydsl.pa_decode import plan_pa_decode
    from aiter.ops.flydsl.pa_decode_mxfp4 import pa_decode_mxfp4
    from atom.model_ops.minimax_m3.kv_mxfp4 import page_views
    from atom.model_ops.minimax_m3.sparse_attn_mxfp4 import sparse_pa_mxfp4

    k, v, bt, ctx, q = make(37, 1, seed=9)
    free = sparse_pa_mxfp4(q, k, v, bt, ctx, 1, SCALE, torch.bfloat16, None)
    kd, ks = page_views(k.view(-1, 1, PAGE_BYTES))
    vd, vs = page_views(v.view(-1, 1, PAGE_BYTES))
    out = torch.empty(37, GROUP, 128, dtype=torch.bfloat16, device="cuda")
    plan = plan_pa_decode(ctx, 1, max_partitions=1, query_length=1)
    pa_decode_mxfp4(
        out, q.view(37, GROUP, 128), kd, ks, vd, vs, ctx, bt, SCALE,
        work_plan=plan, max_context_length=TOPK * 128,
    )  # fmt: skip
    assert torch.equal(free, out.view(q.shape))
    planned = sparse_pa_mxfp4(
        q, k, v, bt, ctx, 1, SCALE, torch.bfloat16, types.SimpleNamespace()
    )
    assert rel(free, planned.float()) < 2e-3


def test_plan_free_builds_nothing_per_row_count():
    """A prefill chunk has a new row count nearly every forward; the planner would
    compile a kernel for each. Plan-free touches neither the planner's compile cache,
    nor the plan cache, nor the PA kernel's build cache after its first call."""
    from aiter.ops.flydsl.kernels.pa_decode_mxfp4_kernel import build_pa_decode_mxfp4
    from aiter.ops.flydsl.kernels.pa_decode_plan import compile_pa_decode_plan
    from atom.model_ops.minimax_m3 import sparse_attn_mxfp4 as m

    k, v, bt, ctx, q = make(8, 1, seed=3, pages=64)
    m.sparse_pa_mxfp4(q, k, v, bt, ctx, 1, SCALE, torch.bfloat16, None)  # first build
    state = (
        compile_pa_decode_plan.cache_info().currsize,
        build_pa_decode_mxfp4.cache_info().currsize,
        len(m._PLANS),
    )
    for rows in (1, 9, 31, 100, 255, 256, 257, 300, 511, 513, 1000, 1023):
        k, v, bt, ctx, q = make(rows, 1, seed=rows, pages=64)
        out = m.sparse_pa_mxfp4(q, k, v, bt, ctx, 1, SCALE, torch.bfloat16, None)
        assert out.shape == q.shape
    assert state == (
        compile_pa_decode_plan.cache_info().currsize,
        build_pa_decode_mxfp4.cache_info().currsize,
        len(m._PLANS),
    )
