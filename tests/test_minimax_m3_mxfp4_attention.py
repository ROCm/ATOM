# SPDX-License-Identifier: MIT
"""`sparse_pa_mxfp4`: one sparse selection per row over the MXFP4 KV pool.

Rows are tokens: a decode token, a speculative-verify token (q_len > 1: each token a
row of its own, one more context token than the one before it) or a prefill token. A
row reads at most `topk` pages by id, `page * Hkv + head`, whether a cached prefix's
or its own chunk's.

CPU (no aiter, no GPU): the module is loaded with the planner and PA kernel replaced
by a torch kernel that enforces the contract the real one validates (page/head strides
shared by data and scale views, int32 tables, a bf16 output) and attends over the
dequantized pages, against an oracle that indexes the pool by (page, head) rather
than by the emitted id. That pins the views, the id encoding, the segmenting past
the planner's batch limit and the plan-free form. The kernel's own numerics are
aiter's tests'; the GPU cases at the end (gfx950, aiter's FlyDSL) run the real one.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES, page_views
from atom.mono.weights.formats import dequantize_mxfp4

GROUP, TOPK, SCALE = 16, 16, 128**-0.5
SRC = Path(__file__).resolve().parent.parent / "atom/model_ops/minimax_m3"


def make_pool(pages, hkv, g):
    p = torch.randint(0, 256, (pages, hkv, PAGE_BYTES), generator=g, dtype=torch.uint8)
    sc = torch.randint(122, 130, (pages, hkv, 512), generator=g)  # finite e8m0
    p[:, :, 8192:] = sc.to(torch.uint8)
    return p


def make(rows, hkv, seed, pages=48, topk=TOPK):
    """Pools, the (token, kv head)-row tables as index_topk emits them, and q."""
    g = torch.Generator().manual_seed(seed)
    k, v = make_pool(pages, hkv, g), make_pool(pages, hkv, g)
    bt = torch.zeros(rows * hkv, topk, dtype=torch.int32)
    ctx = torch.zeros(rows * hkv, dtype=torch.int32)
    for r in range(rows * hkv):
        n = int(torch.randint(1, topk + 1, (1,), generator=g))
        pg = torch.randperm(pages, generator=g)[:n]
        bt[r, :n] = (pg * hkv + r % hkv).to(torch.int32)
        tail = int(torch.randint(1, 129, (1,), generator=g))
        ctx[r] = (n - 1) * 128 + tail
    q = (torch.randn(rows, GROUP * hkv, 128, generator=g) * 0.7).to(torch.float8_e4m3fn)
    return k, v, bt, ctx, q


def dequant_pool(pool):
    """[pages, Hkv, 128, 128] float, indexed by (page, head)."""
    data, scale = page_views(pool)
    return dequantize_mxfp4(data, scale)


def reference(k, v, bt, ctx, q, hkv):
    """Attention per (row, kv head) over its selected pages, indexed by (page, head)
    -- the id `page * hkv + head` is decoded here, not by flattening the pool."""
    rows = q.shape[0]
    kd, vd = dequant_pool(k), dequant_pool(v)
    out = torch.empty(rows, GROUP * hkv, 128)
    for r in range(rows):
        for h in range(hkv):
            i = r * hkv + h
            if int(ctx[i]) == 0:  # a pad row: no keys, no attention mass
                out[r, h * GROUP : (h + 1) * GROUP] = 0
                continue
            ids = bt[i, : -(-int(ctx[i]) // 128)].long()
            assert (ids % hkv == h).all() and (ids // hkv < k.shape[0]).all()
            page = ids // hkv
            kk = kd[page, h].reshape(-1, 128)[: int(ctx[i])]
            vv = vd[page, h].reshape(-1, 128)[: int(ctx[i])]
            qq = q[r, h * GROUP : (h + 1) * GROUP].float()
            p = torch.softmax(qq @ kk.T * SCALE, dim=-1)
            out[r, h * GROUP : (h + 1) * GROUP] = p @ vv
    return out


def rel(a, b):
    return ((a.float() - b).norm() / b.norm()).item()


class FakeAiter:
    """The planner and PA kernel of aiter, as the reader calls them."""

    def __init__(self):
        self.pa_calls, self.plans = [], []

    def pa_decode(
        self, out, query, k_data, v_data, ctx, bt, scale, query_length, *,
        key_scale, value_scale, work_plan=None, max_context_length=None, **kw
    ):  # fmt: skip
        n = query.shape[0]
        # what aiter's MXFP4 pa_decode validates
        assert query_length == 1 and query.dtype == torch.float8_e4m3fn
        assert out.dtype == torch.bfloat16 and out.shape == query.shape
        assert query.shape[2] == 128 and 1 <= query.shape[1] <= 16
        pages = k_data.shape[0]
        strides = k_data.stride()[:2]
        for name, t, tail in (
            ("k", k_data, 64), ("v", v_data, 64),
            ("ks", key_scale, 4), ("vs", value_scale, 4),
        ):  # fmt: skip
            assert t.dtype == torch.uint8 and t.shape == (pages, 1, 128, tail), name
            assert t.stride() == (*strides, tail, 1), name
        assert all(s % 16 == 0 for s in strides)
        assert bt.dtype == ctx.dtype == torch.int32
        assert ctx.shape[0] == bt.shape[0] == n
        assert bt.is_contiguous() and ctx.is_contiguous()
        assert (
            max_context_length is not None and max_context_length <= bt.shape[1] * 128
        )
        self.pa_calls.append(
            {"n": n, "work_plan": work_plan, "max_context_length": max_context_length}
        )
        kd = dequantize_mxfp4(k_data, key_scale)  # [pages, 1, 128, 128]
        vd = dequantize_mxfp4(v_data, value_scale)
        for r in range(n):
            c = int(ctx[r])
            ids = bt[r, : -(-c // 128)].long()
            kk = kd[ids, 0].reshape(-1, 128)[:c]
            vv = vd[ids, 0].reshape(-1, 128)[:c]
            p = torch.softmax(query[r].float() @ kk.T * scale, dim=-1)
            out[r] = (p @ vv).to(out.dtype)


@pytest.fixture
def reader(monkeypatch):
    """(module, FakeAiter): sparse_attn_mxfp4 loaded fresh over a stand-in planner."""
    fake = FakeAiter()
    base = types.ModuleType("atom.model_ops.base_attention")
    base._FLYDSL_PLAN_MAX_BATCH = 4096
    base.calls = []

    def prepare(**kw):
        plan = types.SimpleNamespace(segment=kw["plan_segment"], n=kw["num_seqs"])
        base.calls.append((kw, plan))
        fake.plans.append(plan)
        return plan

    base._flydsl_prepare_explicit_plan = prepare
    base._flydsl_plan_scratch = lambda plan, *a: (None, None, None)
    gluon = types.ModuleType("aiter.ops.triton.gluon.pa_decode_gluon")
    gluon.get_recommended_splits = lambda n, h: 4
    flydsl = types.ModuleType("aiter.ops.flydsl.pa_decode")
    flydsl.pa_decode = fake.pa_decode
    for name, mod in (
        ("atom.model_ops.base_attention", base),
        ("aiter.ops.triton.gluon.pa_decode_gluon", gluon),
        ("aiter.ops.flydsl.pa_decode", flydsl),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location(
        "_sparse_attn_mxfp4_under_test", SRC / "sparse_attn_mxfp4.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.base = base
    return m, fake


@pytest.mark.parametrize("hkv", [1, 2])
@pytest.mark.parametrize("planned", [False, True])
def test_rows_attend_the_pages_their_ids_name(reader, hkv, planned):
    m, fake = reader
    k, v, bt, ctx, q = make(9, hkv, seed=hkv)
    out = m.sparse_pa_mxfp4(
        q, k, v, bt, ctx, hkv, SCALE, torch.bfloat16,
        planned=planned, plan_step_owner=types.SimpleNamespace() if planned else None,
    )  # fmt: skip
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert rel(out, reference(k, v, bt, ctx, q, hkv)) < 1e-2
    assert (fake.pa_calls[0]["work_plan"] is not None) == planned


def test_a_page_id_is_page_times_heads_plus_head(reader):
    """Two heads, one row each reading the same page: swapping the head part of the
    id must change what is read (the K data of page 5 head 1 is not head 0's)."""
    m, _ = reader
    hkv = 2
    k, v, bt, ctx, q = make(1, hkv, seed=3, pages=8)
    bt[:] = 0
    bt[0, 0], bt[1, 0] = 5 * hkv + 0, 5 * hkv + 1
    ctx[:] = 100
    out = m.sparse_pa_mxfp4(q, k, v, bt, ctx, hkv, SCALE, torch.bfloat16)
    assert rel(out, reference(k, v, bt, ctx, q, hkv)) < 1e-2
    swapped = bt.clone()
    swapped[0, 0], swapped[1, 0] = 5 * hkv + 1, 5 * hkv + 0
    out2 = m.sparse_pa_mxfp4(q, k, v, swapped, ctx, hkv, SCALE, torch.bfloat16)
    assert rel(out2, out.float()) > 0.1


def test_prefix_and_chunk_pages_are_read_alike(reader):
    """A chunk of a prompt whose first two pages came from the prefix cache (pool
    pages 30 and 3, written by an earlier request) and whose own tokens fill page 11:
    row t of the chunk selects the prefix pages and its own page, at its causal
    length. The reader takes the pages by id, so the prefix needs nothing of its own
    -- this pins that with pages that are neither adjacent nor in order."""
    m, _ = reader
    hkv, prefix, chunk = 1, 256, 40
    k, v, _, _, _ = make(1, hkv, seed=7, pages=48)
    g = torch.Generator().manual_seed(8)
    q = (torch.randn(chunk, GROUP, 128, generator=g) * 0.7).to(torch.float8_e4m3fn)
    bt = torch.zeros(chunk, TOPK, dtype=torch.int32)
    bt[:, :3] = torch.tensor([30, 3, 11], dtype=torch.int32)
    ctx = (prefix + torch.arange(1, chunk + 1)).to(torch.int32)
    out = m.sparse_pa_mxfp4(q, k, v, bt, ctx, hkv, SCALE, torch.bfloat16)
    assert rel(out, reference(k, v, bt, ctx, q, hkv)) < 1e-2
    # a row's own context decides how many of page 11's tokens it sees
    assert rel(out[0], reference(k, v, bt[:1], ctx[:1], q[:1], hkv)[0]) < 1e-2
    assert rel(out[-1], out[0]) > 0.01


@pytest.mark.parametrize("hkv", [1, 2])
@pytest.mark.parametrize("planned", [False, True])
def test_verify_tokens_are_rows_of_their_own(reader, hkv, planned):
    """Speculative verify (q_len 4, 3 requests): 12 rows, each token's context one
    longer than the previous token's of its request, on the same selected pages."""
    m, _ = reader
    reqs, qlen = 3, 4
    k, v, bt, ctx, q = make(reqs * qlen, hkv, seed=20 + hkv)
    for b in range(reqs):
        base = bt[b * qlen * hkv : (b * qlen + 1) * hkv].clone()  # token 0's rows
        for t in range(qlen):
            rows = slice((b * qlen + t) * hkv, (b * qlen + t + 1) * hkv)
            bt[rows] = base
            ctx[rows] = ctx[b * qlen * hkv : (b * qlen + 1) * hkv] + t
    out = m.sparse_pa_mxfp4(
        q, k, v, bt, ctx, hkv, SCALE, torch.bfloat16,
        planned=planned, plan_step_owner=types.SimpleNamespace() if planned else None,
    )  # fmt: skip
    assert out.shape == (reqs * qlen, GROUP * hkv, 128)
    assert rel(out, reference(k, v, bt, ctx, q, hkv)) < 1e-2


def test_decode_runs_in_segments_each_with_its_own_plan(reader):
    """More rows than one plan takes: segments of equal size must not share a plan
    (the step-owner refresh would then keep only the last one's lengths)."""
    m, fake = reader
    m._FLYDSL_PLAN_MAX_BATCH = 4  # 11 rows -> 4 + 4 + 3
    k, v, bt, ctx, q = make(11, 1, seed=5)
    owner = types.SimpleNamespace()
    out = m.sparse_pa_mxfp4(
        q, k, v, bt, ctx, 1, SCALE, torch.bfloat16, planned=True, plan_step_owner=owner
    )
    assert rel(out, reference(k, v, bt, ctx, q, 1)) < 1e-2
    calls = m.base.calls
    assert [(c["plan_segment"], c["num_seqs"]) for c, _ in calls] == [
        (0, 4),
        (1, 4),
        (2, 3),
    ]
    assert all(
        c["plan_step_owner"] is owner and c["query_length"] == 1 for c, _ in calls
    )
    assert all(c["context_bound"] == TOPK * 128 for c, _ in calls)
    # each segment's kernel call took its own plan and its own rows
    assert [(c["n"], c["work_plan"]) for c in fake.pa_calls] == [
        (4, calls[0][1]),
        (4, calls[1][1]),
        (3, calls[2][1]),
    ]


def test_prefill_is_plan_free_whatever_the_row_count(reader):
    m, fake = reader
    for rows in (1, 9, 33):
        k, v, bt, ctx, q = make(rows, 1, seed=rows, pages=64)
        out = m.sparse_pa_mxfp4(q, k, v, bt, ctx, 1, SCALE, torch.bfloat16)
        assert out.shape == q.shape
    assert m.base.calls == []  # the planner compiles a kernel per batch size
    assert all(c["work_plan"] is None for c in fake.pa_calls)
    assert all(c["max_context_length"] == TOPK * 128 for c in fake.pa_calls)


def test_what_the_kernel_cannot_run_is_refused_before_it_is_called(reader):
    m, fake = reader
    k, v, bt, ctx, q = make(4, 1, seed=1)
    args = (k, v, bt, ctx, 1, SCALE, torch.bfloat16)
    with pytest.raises(TypeError, match="e4m3"):
        m.sparse_pa_mxfp4(q.to(torch.bfloat16), *args)
    with pytest.raises(TypeError, match="bf16"):
        m.sparse_pa_mxfp4(q, k, v, bt, ctx, 1, SCALE, torch.float16)
    with pytest.raises(ValueError, match="contiguous"):
        m.sparse_pa_mxfp4(q.transpose(0, 1).contiguous().transpose(0, 1), *args)
    with pytest.raises(ValueError, match="sparse rows"):
        m.sparse_pa_mxfp4(q, k, v, bt[:3], ctx, 1, SCALE, torch.bfloat16)
    assert fake.pa_calls == []


def test_plan_segments_of_one_size_get_plans_of_their_own(monkeypatch):
    """`plan_segment` keys the runtime plan: two equal-size segments of one call must
    not share a plan, or the step's once-only refresh leaves both on the last
    segment's lengths. Needs base_attention importable (aiter, triton); no GPU."""
    pytest.importorskip("aiter")
    pytest.importorskip("triton")
    from atom.model_ops import base_attention as ba

    made, refreshed = [], []

    def plan_pa_decode(lengths, kv_heads, *, plan=None, **kw):
        if plan is not None:
            refreshed.append(plan)
            return plan
        made.append(types.SimpleNamespace(max_partitions=1, capacity=1))
        return made[-1]

    monkeypatch.setattr(
        importlib.import_module("aiter.ops.flydsl.pa_decode"),
        "plan_pa_decode",
        plan_pa_decode,
        raising=False,
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: types.SimpleNamespace(cuda_stream=7),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    owner, cache = types.SimpleNamespace(), types.SimpleNamespace()

    def prepare(seg):
        return ba._flydsl_prepare_explicit_plan(
            q=torch.zeros(4, 16, 128), k_cache=torch.zeros(1, 1, 128, 64),
            context_lens=torch.ones(4, dtype=torch.int32), num_seqs=4, query_length=1,
            plan_partition_cap=4, context_bound=128, existing_plan=None,
            plan_step_owner=owner, plan_cache_owner=cache, plan_segment=seg,
        )  # fmt: skip

    p0, p1 = prepare(0), prepare(1)
    assert p0 is not p1 and len(made) == 2
    # the next layer of the step: same plans back, no refresh (the owner's marks)
    assert prepare(0) is p0 and prepare(1) is p1 and refreshed == []


# --- GPU (gfx950, aiter's FlyDSL): the real kernel. Not run by the CPU gate. ---------
def _real_gpu():
    if not torch.cuda.is_available():
        return False
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return False
    try:
        import aiter

        return hasattr(aiter.ops, "flydsl")
    except ImportError:
        return False


needs_gfx950 = pytest.mark.skipif(not _real_gpu(), reason="needs gfx950 and aiter")


def _to_cuda(*ts):
    return [t.cuda() for t in ts]


@needs_gfx950
@pytest.mark.parametrize("hkv", [1, 2])
@pytest.mark.parametrize("planned", [False, True])
def test_real_kernel_rows_attend_their_selected_pages(hkv, planned):
    from atom.model_ops.minimax_m3.sparse_attn_mxfp4 import sparse_pa_mxfp4

    k, v, bt, ctx, q = make(37, hkv, seed=hkv)
    want = reference(k, v, bt, ctx, q, hkv)
    k, v, bt, ctx, q = _to_cuda(k, v, bt, ctx, q)
    out = sparse_pa_mxfp4(
        q, k, v, bt, ctx, hkv, SCALE, torch.bfloat16,
        planned=planned, plan_step_owner=types.SimpleNamespace() if planned else None,
    )  # fmt: skip
    assert rel(out.cpu(), want) < 3e-3


@needs_gfx950
@pytest.mark.parametrize("rows", [1, 255, 256, 257, 700])
def test_real_kernel_plan_free_any_row_count(rows):
    from atom.model_ops.minimax_m3.sparse_attn_mxfp4 import sparse_pa_mxfp4

    k, v, bt, ctx, q = make(rows, 1, seed=rows, pages=64)
    want = reference(k, v, bt, ctx, q, 1)
    k, v, bt, ctx, q = _to_cuda(k, v, bt, ctx, q)
    out = sparse_pa_mxfp4(q, k, v, bt, ctx, 1, SCALE, torch.bfloat16)
    assert rel(out.cpu(), want) < 3e-3


@needs_gfx950
def test_real_kernel_segments_past_the_planner_limit(monkeypatch):
    from atom.model_ops.minimax_m3 import sparse_attn_mxfp4 as m

    monkeypatch.setattr(m, "_FLYDSL_PLAN_MAX_BATCH", 16)
    k, v, bt, ctx, q = make(37, 1, seed=5)
    want = reference(k, v, bt, ctx, q, 1)
    k, v, bt, ctx, q = _to_cuda(k, v, bt, ctx, q)
    step = types.SimpleNamespace()
    for _layer in range(3):  # later layers reuse the step's three plans
        out = m.sparse_pa_mxfp4(
            q, k, v, bt, ctx, 1, SCALE, torch.bfloat16,
            planned=True, plan_step_owner=step,
        )  # fmt: skip
        assert rel(out.cpu(), want) < 3e-3


@needs_gfx950
@pytest.mark.parametrize("planned", [False, True])
def test_real_kernel_pad_rows_of_length_zero_come_out_zero(planned):
    """Not run by the CPU gate. A padded decode step (a graph bucket) runs rows of
    context 0 -- pad requests -- among real ones; the reader must hand them to the
    kernel as they are and get zeros back without disturbing their neighbours. aiter's
    own MXFP4 cases have no empty row, so this is the only check of that."""
    from atom.model_ops.minimax_m3.sparse_attn_mxfp4 import sparse_pa_mxfp4

    k, v, bt, ctx, q = make(12, 1, seed=4)
    pad = torch.tensor([0, 5, 6, 11])
    ctx[pad] = 0
    bt[pad] = 0
    want = reference(k, v, bt, ctx, q, 1)  # an empty row: zeros
    live = torch.ones(12, dtype=torch.bool)
    live[pad] = False
    k, v, bt, ctx, q = _to_cuda(k, v, bt, ctx, q)
    out = sparse_pa_mxfp4(
        q, k, v, bt, ctx, 1, SCALE, torch.bfloat16,
        planned=planned, plan_step_owner=types.SimpleNamespace() if planned else None,
    )  # fmt: skip
    out = out.cpu().float()
    assert (out[pad] == 0).all()
    assert rel(out[live], want[live]) < 3e-3
