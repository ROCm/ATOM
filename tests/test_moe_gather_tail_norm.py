# SPDX-License-Identifier: MIT
"""ATOM_MOE_GATHER_TAIL_NORM: the routed experts' top-k gather-reduce folded
into the MoE tail norm (atom/model_ops/moe_tail_norm.py,
`moe_gather_tail_rmsnorm`).

CPU: the flag, which blocks defer the gather (`defer_moe_gated_sums`), what a
deferring block returns, and how GemmaRMSNorm takes a `GatheredRouted` (the
fused kernel where it can, aiter's gather-reduce first everywhere else).

GPU (skipped without one): the kernel bitwise against aiter's
moe_gather_reduce followed by the tail kernel -- padding tokens, dropped routes,
the drafter's row subset, no gate, a CUDA graph -- and end to end through
aiter's grouped MoE under `defer_gather_reduce` (gfx1250).
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

mt = pytest.importorskip(
    "atom.model_ops.moe_tail_norm",
    reason="moe_tail_norm imports aiter and triton",
    exc_type=ImportError,
)

HIDDEN, EXPERTS, TOPK, EPS = 2048, 256, 8, 1e-6


def ln():
    try:
        import atom.model_ops.layernorm as module
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"atom.model_ops.layernorm does not import here: {exc}")
    return module


def _qwen3_next():
    try:
        import atom.models.qwen3_next as module
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"atom.models.qwen3_next does not import here: {exc}")
    return module


def gather_math(g: "mt.GatheredRouted") -> torch.Tensor:
    """moe_gather_reduce in torch: fp32 `acc + w_k * row_k`, k in order,
    rounded once; negative rows contribute 0 (tokens past the count: 0)."""
    tokens, hidden = g.shape
    acc = torch.zeros(tokens, hidden, dtype=torch.float32, device=g.grouped.device)
    for k in range(g.route_rows.shape[1]):
        r = g.route_rows[:, k].long()
        x = g.grouped[r.clamp(min=0)].float() * (r >= 0)[:, None]
        acc = acc + g.weights[:, k].float()[:, None] * x
    if g.num_valid_tokens is not None:
        acc[int(g.num_valid_tokens[0]) :] = 0
    return acc.to(g.grouped.dtype)


def _gathered(tokens, dtype=torch.bfloat16, seed=0, device="cpu", nvt=None, drop=0.0):
    gen = torch.Generator(device=device).manual_seed(seed)
    n_rows = tokens * TOPK + EXPERTS * 16
    grouped = (torch.randn(n_rows, HIDDEN, generator=gen, device=device) * 0.5).to(
        dtype
    )
    perm = torch.randperm(n_rows, generator=gen, device=device)[: tokens * TOPK]
    route_rows = perm.to(torch.int32).view(tokens, TOPK)
    if drop:
        dropped = torch.rand(tokens, TOPK, generator=gen, device=device) < drop
        route_rows = torch.where(dropped, torch.full_like(route_rows, -1), route_rows)
    weights = torch.softmax(
        torch.randn(tokens, TOPK, generator=gen, device=device), -1
    ).float()
    nv = None if nvt is None else torch.tensor([nvt], dtype=torch.int32, device=device)
    return mt.GatheredRouted(grouped, route_rows, weights, nv)


def _inputs(tokens, seed=0, device="cpu", **kw):
    gen = torch.Generator(device=device).manual_seed(seed + 1)

    def rn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=device) * scale).to(
            torch.bfloat16
        )

    logits = rn(tokens, EXPERTS + 1, scale=3.0)
    return SimpleNamespace(
        g=_gathered(tokens, seed=seed, device=device, **kw),
        shared=rn(tokens, HIDDEN, scale=0.3),
        gate=logits[:, EXPERTS:],
        residual=rn(tokens, HIDDEN, scale=2.0),
        weight=rn(HIDDEN, scale=0.1),
    )


def _norm(weight):
    norm = ln().GemmaRMSNorm(weight.shape[0], eps=EPS)
    norm.weight.data = weight.clone()
    return norm


# ---------------------------------------------------------------------------
# The flag and the wiring
# ---------------------------------------------------------------------------


def test_flag_defaults_off(monkeypatch):
    from atom.utils import envs

    monkeypatch.delenv("ATOM_MOE_GATHER_TAIL_NORM", raising=False)
    assert envs.ATOM_MOE_GATHER_TAIL_NORM is False
    monkeypatch.setenv("ATOM_MOE_GATHER_TAIL_NORM", "1")
    assert envs.ATOM_MOE_GATHER_TAIL_NORM is True
    monkeypatch.delenv("ATOM_MOE_GATHER_TAIL_NORM_PIPE", raising=False)
    assert envs.ATOM_MOE_GATHER_TAIL_NORM_PIPE is False


def _moe_block(qn, *, tp=1, all2all=False, dp=1, replicate=True):
    block = object.__new__(qn.Qwen3NextSparseMoeBlock)
    nn.Module.__init__(block)
    block.n_routed_experts = EXPERTS
    block.is_rocm_aiter_fusion_shared_expert_enabled = False
    block.tp_size = tp
    block.reduce_results = True
    block._use_dual_stream = False
    block.shared_expert = lambda h: h
    block.experts = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            use_all2all_kernels=all2all,
            dp_size=dp,
            replicate_experts=replicate,
            dp_logical_ratio=1,
            pcp_merge=False,
        )
    )
    return block


def _layer(mlp):
    layer = nn.Module()
    layer.mlp = mlp
    layer.input_layernorm = ln().GemmaRMSNorm(16)
    layer.layer_scale = False
    return layer


def test_defer_marks_the_blocks_whose_rows_stay_local(monkeypatch):
    qn = _qwen3_next()
    monkeypatch.setattr(qn, "is_plugin_mode", lambda: False)
    monkeypatch.setenv("ATOM_FUSED_MOE_TAIL_NORM", "1")
    monkeypatch.delenv("ATOM_MEGA_COMBINE_TAIL", raising=False)
    final = ln().GemmaRMSNorm(16)

    def build():
        return [
            _layer(_moe_block(qn)),  # DP4 replicated experts: local rows
            _layer(_moe_block(qn, all2all=True)),  # mori EP
            _layer(_moe_block(qn, dp=4, replicate=False)),  # DP gather / scatter
            _layer(_moe_block(qn)),  # -> the final norm
        ]

    monkeypatch.delenv("ATOM_MOE_GATHER_TAIL_NORM", raising=False)
    layers = build()
    assert qn.defer_moe_gated_sums(layers, final) == 4
    assert [ly.mlp.defer_gather for ly in layers] == [False] * 4
    monkeypatch.setenv("ATOM_MOE_GATHER_TAIL_NORM", "1")
    layers = build()
    assert qn.defer_moe_gated_sums(layers, final) == 4
    assert [ly.mlp.defer_gather for ly in layers] == [True, False, False, True]


def test_norm_takes_a_deferred_gather(monkeypatch):
    """The fused kernel where `can_fuse_gather_tail` holds; aiter's reduce
    (`materialize`) and the tail kernel otherwise."""
    lnm = ln()
    calls = []
    t = _inputs(6)
    norm = _norm(t.weight)
    moe = mt.MoEGatedSum(t.g, t.shared, t.gate)

    def fused(moe, residual, weight, eps, rows=None, write_residual=True):
        calls.append(("fused", rows is None, write_residual))
        return "fused"

    monkeypatch.setattr(lnm, "moe_gather_tail_rmsnorm", fused)
    assert norm(moe, t.residual) == "fused"
    rows = torch.tensor([3, 1])
    assert norm.forward_moe_tail(moe, t.residual, rows, False) == "fused"
    assert calls == [("fused", True, True), ("fused", False, False)]

    # A route map the fused kernel does not take (int64): aiter's reduce
    # first, then the tail kernel on its output.
    monkeypatch.setattr(
        mt.GatheredRouted, "materialize", lambda self: gather_math(self)
    )

    def tail(routed, *args):
        calls.append(("tail", torch.equal(routed, gather_math(t.g))))
        return "tail"

    monkeypatch.setattr(lnm, "moe_tail_rmsnorm", tail)
    g64 = t.g._replace(route_rows=t.g.route_rows.long())
    assert norm(mt.MoEGatedSum(g64, t.shared, t.gate), t.residual) == "tail"
    assert calls[-1] == ("tail", True)


def test_can_fuse_gather_tail_refuses_what_the_kernel_does_not_take():
    t = _inputs(4)
    ok = mt.MoEGatedSum(t.g, t.shared, t.gate)
    assert mt.can_fuse_gather_tail(ok, t.residual, t.weight)
    assert not mt.can_fuse_gather_tail(
        mt.MoEGatedSum(t.shared, t.shared, t.gate), t.residual, t.weight
    )  # a plain routed tensor
    bad_rows = t.g._replace(route_rows=t.g.route_rows.long())
    assert not mt.can_fuse_gather_tail(
        ok._replace(routed=bad_rows), t.residual, t.weight
    )
    half = t.g._replace(grouped=t.g.grouped.half())
    assert not mt.can_fuse_gather_tail(ok._replace(routed=half), t.residual, t.weight)
    assert not mt.can_fuse_gather_tail(ok, t.residual.float(), t.weight)
    assert not mt.can_fuse_gather_tail(ok, t.residual, t.weight, torch.zeros(2))


# ---------------------------------------------------------------------------
# GPU: bitwise against moe_gather_reduce + the tail kernel
# ---------------------------------------------------------------------------

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _unfused(t, gate, rows=None, write_residual=True):
    routed = t.g.materialize()  # aiter's moe_gather_reduce
    if write_residual:
        return mt.moe_tail_rmsnorm(routed, t.shared, gate, t.residual, t.weight, EPS)
    return mt.moe_tail_rmsnorm_rows(
        routed, t.shared, gate, t.residual, t.weight, EPS, rows
    )


@gpu
@pytest.mark.parametrize("tokens", [1, 85, 340, 480, 4099])
@pytest.mark.parametrize("pad", [0, 5])
def test_kernel_is_gather_reduce_then_the_tail(tokens, pad):
    nvt = max(tokens - pad, 1) if pad else None
    t = _inputs(tokens, seed=tokens, device="cuda", nvt=nvt, drop=0.02)
    nv = tokens if nvt is None else nvt
    for gate in (t.gate, None):
        moe = mt.MoEGatedSum(t.g, t.shared, gate)
        ref_out, ref_res = _unfused(t, gate)
        out, res = mt.moe_gather_tail_rmsnorm(moe, t.residual, t.weight, EPS)
        assert torch.equal(out[:nv], ref_out[:nv])
        assert torch.equal(res[:nv], ref_res[:nv])
        # Padding tokens: a routed sum of 0, never stale bytes.
        assert torch.isfinite(out).all()
        rows = torch.arange(nv - 1, -1, -3, device="cuda", dtype=torch.int32)
        sub = mt.moe_gather_tail_rmsnorm(
            moe, t.residual, t.weight, EPS, rows, write_residual=False
        )
        assert torch.equal(sub, _unfused(t, gate, rows, False))
    # And the torch statement of the reduce, likewise.
    assert torch.equal(t.g.materialize()[:nv], gather_math(t.g)[:nv])


@gpu
def test_kernel_in_a_cuda_graph():
    t = _inputs(160, seed=7, device="cuda", nvt=150)
    moe = mt.MoEGatedSum(t.g, t.shared, t.gate)
    eager = mt.moe_gather_tail_rmsnorm(moe, t.residual, t.weight, EPS)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        captured = mt.moe_gather_tail_rmsnorm(moe, t.residual, t.weight, EPS)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured[0], eager[0]) and torch.equal(captured[1], eager[1])


# ---------------------------------------------------------------------------
# GPU (gfx1250): through aiter's grouped MoE
# ---------------------------------------------------------------------------


def _gfx1250():
    try:
        from aiter.jit.utils.chip_info import get_gfx
    except ImportError:
        return False
    return torch.cuda.is_available() and get_gfx() == "gfx1250"


@pytest.mark.skipif(not _gfx1250(), reason="the TDM kernel is gfx1250 only")
@pytest.mark.parametrize("tokens", [8192, 9001])
def test_pipelined_kernel_is_the_one_token_kernel(tokens, monkeypatch):
    """ATOM_MOE_GATHER_TAIL_NORM_PIPE at prefill sizes: the TDM kernel, bitwise
    the one-token kernel on every row (padding, dropped routes, no gate, a row
    subset) and so moe_gather_reduce + the tail kernel."""
    from atom.model_ops import moe_gather_tail_gluon as tdm

    calls = []
    launch = tdm.launch_gather_tail_tdm
    monkeypatch.setattr(
        tdm,
        "launch_gather_tail_tdm",
        lambda *a, **k: calls.append(1) or launch(*a, **k),
    )
    nvt = tokens - 37
    t = _inputs(tokens, seed=tokens, device="cuda", nvt=nvt, drop=0.02)
    rows = torch.randint(0, nvt, (tdm.PIPE_MIN_ROWS,), device="cuda", dtype=torch.int32)
    for gate, sub in ((t.gate, None), (None, None), (t.gate, rows)):
        moe = mt.MoEGatedSum(t.g, t.shared, gate)
        args = (moe, t.residual, t.weight, EPS, sub, sub is None)
        monkeypatch.setenv("ATOM_MOE_GATHER_TAIL_NORM_PIPE", "0")
        ref = mt.moe_gather_tail_rmsnorm(*args)
        monkeypatch.setenv("ATOM_MOE_GATHER_TAIL_NORM_PIPE", "1")
        n = len(calls)
        out = mt.moe_gather_tail_rmsnorm(*args)
        assert len(calls) == n + 1
        if sub is None:
            assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
            unfused = _unfused(t, gate)
            assert torch.equal(out[0][:nvt], unfused[0][:nvt])
        else:
            assert torch.equal(out, ref)
    # Decode sizes keep the one-token kernel.
    small = _inputs(340, seed=3, device="cuda")
    n = len(calls)
    mt.moe_gather_tail_rmsnorm(
        mt.MoEGatedSum(small.g, small.shared, small.gate),
        small.residual,
        small.weight,
        EPS,
    )
    assert len(calls) == n


@pytest.mark.skipif(not _gfx1250(), reason="aiter's grouped MoE is gfx1250 only")
@pytest.mark.parametrize("tokens", [85, 340])
@pytest.mark.parametrize("deferred_topk", [False, True])
def test_grouped_moe_deferral_is_the_reduce(tokens, deferred_topk):
    import aiter
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.grouped_moe_gfx1250 import (
        defer_gather_reduce,
        defer_topk_softmax,
    )

    inter = 512
    gen = torch.Generator(device="cuda").manual_seed(1)
    u8 = torch.uint8

    def rand_u8(lo, hi, *shape):
        return torch.randint(lo, hi, shape, dtype=u8, device="cuda", generator=gen)

    w1 = rand_u8(0, 256, EXPERTS, 2 * inter, HIDDEN // 2).view(dtypes.fp4x2)
    w2 = rand_u8(0, 256, EXPERTS, HIDDEN, inter // 2).view(dtypes.fp4x2)
    s1 = rand_u8(118, 126, EXPERTS, 2 * inter, HIDDEN // 32)
    s2 = rand_u8(118, 126, EXPERTS, HIDDEN, inter // 32)
    logits = (
        torch.randn(tokens, EXPERTS + 1, device="cuda", generator=gen) * 3
    ).bfloat16()
    hidden = torch.randn(tokens, HIDDEN, device="cuda", generator=gen).bfloat16()
    nv = tokens - 5
    nvr = torch.tensor([nv], dtype=torch.int32, device="cuda")

    def experts():
        w = torch.empty(tokens, TOPK, dtype=torch.float32, device="cuda")
        ids = torch.empty(tokens, TOPK, dtype=torch.int32, device="cuda")
        if deferred_topk:
            defer_topk_softmax(w, ids, logits[:, :EXPERTS], True, nvr)
        else:
            aiter.topk_softmax(
                w, ids, torch.empty_like(ids), logits[:, :EXPERTS], True, 0, "", nvr
            )
        return fused_moe(
            hidden,
            w1,
            w2,
            w,
            ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=s1,
            w2_scale=s2,
            num_local_tokens=nvr,
            gate_mode="interleave",
        )

    routed = experts()
    with defer_gather_reduce() as req:
        stand_in = experts()
    assert req.taken and stand_in.data_ptr() == req.placeholder.data_ptr()
    g = mt.GatheredRouted(
        req.grouped, req.route_rows, req.weights, req.num_valid_tokens
    )
    assert torch.equal(g.materialize()[:nv], routed[:nv])
    t = _inputs(tokens, seed=2, device="cuda")
    ref = mt.moe_tail_rmsnorm(
        routed, t.shared, logits[:, EXPERTS:], t.residual, t.weight, EPS
    )
    out = mt.moe_gather_tail_rmsnorm(
        mt.MoEGatedSum(g, t.shared, logits[:, EXPERTS:]), t.residual, t.weight, EPS
    )
    assert torch.equal(out[0][:nv], ref[0][:nv]) and torch.equal(
        out[1][:nv], ref[1][:nv]
    )
    # One deferral a scope: a second MoE call in it reduces as usual.
    with defer_gather_reduce() as req2:
        experts()
        again = experts()
    assert req2.taken and torch.equal(again[:nv], routed[:nv])


# ---------------------------------------------------------------------------
# `python tests/test_moe_gather_tail_norm.py [M ...]`: CUDA-graph replays of
# moe_gather_reduce + the tail kernel (i + l) against the fused kernel
# ---------------------------------------------------------------------------


def _graph_us(fns, reps=25):
    """Median us of one pass over `fns` (each a launch sequence), replayed
    from one captured graph."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for f in fns:
            f()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in fns:
            f()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        a.record()
        graph.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / len(fns))
    return sorted(times)[reps // 2]


def benchmark(sizes):
    """Hot: one input set replayed 20 times (MoE2's rows in the caches, as
    right after MoE2). Cold: input sets rotated over ~1 GB."""
    print(
        "| M | i + l hot | fused hot | i + l cold | fused cold "
        "| fused cold TB/s | 17 TB/s floor |"
    )
    print("|---|---|---|---|---|---|---|")
    for m in sizes:
        per = (m * TOPK + EXPERTS * 16) * HIDDEN * 2 + 3 * m * HIDDEN * 2
        sets = [
            _inputs(m, seed=i, device="cuda", nvt=m)
            for i in range(max(2, min(64, int(1e9 // per))))
        ]

        def unfused(t):
            return lambda: _unfused(t, t.gate)

        def fused(t):
            moe = mt.MoEGatedSum(t.g, t.shared, t.gate)
            return lambda: mt.moe_gather_tail_rmsnorm(moe, t.residual, t.weight, EPS)

        n = max(20, len(sets))
        cold = [sets[i % len(sets)] for i in range(n)]
        hot_old = _graph_us([unfused(sets[0])] * 20)
        hot_new = _graph_us([fused(sets[0])] * 20)
        cold_old = _graph_us([unfused(t) for t in cold])
        cold_new = _graph_us([fused(t) for t in cold])
        moved = (m * TOPK + 4 * m) * HIDDEN * 2  # rows; shared, residual, 2 outs
        print(
            f"| {m} | {hot_old:.2f} | {hot_new:.2f} | {cold_old:.2f} | "
            f"{cold_new:.2f} | {moved / cold_new / 1e6:.1f} | "
            f"{moved / 17e6:.2f} |"
        )


def context_benchmark(sizes):
    """The layer's MoE half as the model runs it (gfx1250): top-k + routing +
    quant, MoE1, MoE2, the shared expert (gated gate_up GEMM, down GEMM), then
    i + l or the fused kernel -- MoE2's rows as fresh in the caches as in the
    model. `base` is the same without the tail."""
    import aiter
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.grouped_moe_gfx1250 import (
        defer_gather_reduce,
        defer_topk_softmax,
    )
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16_gated
    from aiter.tuned_gemm import tgemm

    del aiter
    inter, u8 = 512, torch.uint8
    gen = torch.Generator(device="cuda").manual_seed(1)

    def rand_u8(lo, hi, *shape):
        return torch.randint(lo, hi, shape, dtype=u8, device="cuda", generator=gen)

    def rn(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=gen) * scale).bfloat16()

    w1 = rand_u8(0, 256, EXPERTS, 2 * inter, HIDDEN // 2).view(dtypes.fp4x2)
    w2 = rand_u8(0, 256, EXPERTS, HIDDEN, inter // 2).view(dtypes.fp4x2)
    s1 = rand_u8(118, 126, EXPERTS, 2 * inter, HIDDEN // 32)
    s2 = rand_u8(118, 126, EXPERTS, HIDDEN, inter // 32)
    w_gate_up, w_down = rn(2 * inter, HIDDEN, scale=0.02), rn(HIDDEN, inter, scale=0.02)
    norm_w = rn(HIDDEN, scale=0.1)
    print(
        "| M | base | base + i + l | base + fused | i + l in place | fused in place |"
    )
    print("|---|---|---|---|---|---|")

    def one(m):
        logits = rn(m, EXPERTS + 1, scale=3.0)
        hidden, residual = rn(m, HIDDEN), rn(m, HIDDEN, scale=2.0)
        nvr = torch.tensor([m], dtype=torch.int32, device="cuda")
        w = torch.empty(m, TOPK, dtype=torch.float32, device="cuda")
        ids = torch.empty(m, TOPK, dtype=torch.int32, device="cuda")

        def experts():
            defer_topk_softmax(w, ids, logits[:, :EXPERTS], True, nvr)
            return fused_moe(
                hidden,
                w1,
                w2,
                w,
                ids,
                activation=ActivationType.Silu,
                quant_type=QuantType.per_1x32,
                w1_scale=s1,
                w2_scale=s2,
                num_local_tokens=nvr,
                gate_mode="interleave",
            )

        def shared_expert():
            act = gemm_a16w16_gated(hidden, w_gate_up, dtype=torch.bfloat16)
            return tgemm.mm(act, w_down, None, otype=torch.bfloat16)

        def base():
            experts()
            shared_expert()

        def unfused():
            routed, shared = experts(), shared_expert()
            return mt.moe_tail_rmsnorm(
                routed, shared, logits[:, EXPERTS:], residual, norm_w, EPS
            )

        def fused():
            with defer_gather_reduce() as req:
                experts()
            shared = shared_expert()
            g = mt.GatheredRouted(
                req.grouped, req.route_rows, req.weights, req.num_valid_tokens
            )
            moe = mt.MoEGatedSum(g, shared, logits[:, EXPERTS:])
            return mt.moe_gather_tail_rmsnorm(moe, residual, norm_w, EPS)

        a, b = unfused(), fused()
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), m
        t_base, t_old, t_new = (_graph_us([f] * 10) for f in (base, unfused, fused))
        print(
            f"| {m} | {t_base:.2f} | {t_old:.2f} | {t_new:.2f} | "
            f"{t_old - t_base:.2f} | {t_new - t_base:.2f} |"
        )

    for m in sizes:
        one(m)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    context = "--context" in args
    sizes = [int(a) for a in args if a != "--context"]
    sizes = sizes or [85, 160, 340, 384, 480, 8192]
    (context_benchmark if context else benchmark)(sizes)
