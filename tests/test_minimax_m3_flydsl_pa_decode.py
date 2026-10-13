# SPDX-License-Identifier: MIT
"""FlyDSL paged decode on MiniMax-M3's shapes, against gluon and an exact reference.

Run 38236783820 lost its prefill server to a FlyDSL launch that failed with
hipErrorInvalidValue; the last plan logged before it was the first 1-row
sparse one (`flydsl plan: batch=1 ql=1 context_bound=2048 kv_heads=1`).
FlyDSL launches through MLIR's ROCm runtime wrapper, which prints the failed
hipModuleLaunchKernel and returns; nothing raises until the next native torch
launch reads the sticky error -- there, a torch.gather after sampling. So
every call here primes its output with NaN, then reads the error state before
it reads the numbers.

Two value regimes. ``uniform`` (q = 0, power-of-two scales) gives every visible
key exactly the same weight, so the answer is the mean of the visible values up
to bf16 rounding, and one key too many or too few shows. ``random`` exercises
the running max and the split merge, at fp8 tolerance.

gluon runs on every case before FlyDSL. It is the production default, so a
miss there indicts the harness, not FlyDSL.

Needs a gfx942/gfx950 GPU and an aiter with #5809's FlyDSL decode API:

    AITER_LOG_LEVEL=WARNING AMD_SERIALIZE_KERNEL=3 python -m pytest tests/test_minimax_m3_flydsl_pa_decode.py -v -s
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("ROCm GPU required", allow_module_level=True)

pytest.importorskip("triton", reason="sparse_attn defines @triton.jit kernels")
pytest.importorskip("aiter", reason="both kernels under test are aiter's")
flydsl_pa = pytest.importorskip(
    "aiter.ops.flydsl.pa_decode", reason="needs aiter's FlyDSL paged decode"
)

import aiter

from atom.model_ops import base_attention as ba
from atom.model_ops.minimax_m3 import sparse_attn

if not ba._flydsl_arch_supported():
    pytest.skip("FlyDSL builds no paged decode for this GPU", allow_module_level=True)
if not ba._flydsl_explicit_plan_api():
    pytest.skip("aiter predates #5809's explicit work plan", allow_module_level=True)

FP8 = aiter.dtypes.fp8
X = 16  # SHUFFLE inner width: 16 bytes of fp8
HEAD_DIM = 128
GROUP = 16  # M3: 64 query heads over 4 kv heads, at any tp up to 4
TOPK_BLOCKS = 16  # the indexer keeps 16 blocks of 128 keys
PAGES = sparse_attn.PAGES_PER_SPARSE_BLOCK
SPARSE_WINDOW = TOPK_BLOCKS * sparse_attn.SPARSE_BLOCK_SIZE
SPARSE_POOL_BLOCKS = 64
DENSE_PAGE = 128  # M3's KV block; the sparse layers re-view it as 16-token pages
DENSE_BLOCKS = 64  # per sequence: an 8192-token bound
SM_SCALE = HEAD_DIM**-0.5
# Shared by every token and a power of two, so p * v_scale stays exact in fp8.
UNIFORM_SCALE = 2.0**-6

REGIMES = ("uniform", "random")
TOL = {
    # Only the bf16 partials and output round. One key too many or too few
    # moves an n-key mean by ~1/sqrt(n): 2% at the sparse window.
    "uniform": 1e-2,
    # fp8 rounding of scores, probabilities and per-token scales.
    "random": 8e-2,
}

SPARSE_LENGTHS = {
    # Within one 256-key partition, including the 241..255 band.
    "short": (1, 2, 15, 16, 17, 127, 128, 129, 241, 255, 256),
    "mixed": (257, 1, 2048, 256, 255, 1500, 16, 513, 2047, 128, 17, 1024),
}
DENSE_LENGTHS = {
    "mixed": (4097, 4, 8192, 255, 256, 257, 1000, 129, 2048, 17, 128, 6000),
}


@dataclass
class _Pages:
    k_cache: torch.Tensor  # SHUFFLE [pages, kv_heads, head_dim // X, page, X] fp8
    v_cache: torch.Tensor  # SHUFFLE [pages, kv_heads, page // X, head_dim, X] fp8
    k_scale: torch.Tensor  # [pages, kv_heads, page] fp32, one per token
    v_scale: torch.Tensor
    k_deq: torch.Tensor  # [pages, kv_heads, page, head_dim] fp32, dequantized
    v_deq: torch.Tensor


@dataclass
class _Case:
    q: torch.Tensor
    pages: _Pages
    bt: torch.Tensor
    ctx: torch.Tensor
    num_kv_heads: int = 1
    query_length: int = 1


def _fp8_pages(num_pages, num_kv_heads, page_size, regime):
    shape = (num_pages, num_kv_heads, page_size, HEAD_DIM)
    fp8_max = torch.finfo(FP8).max
    planes = []
    for _ in range(2):
        val = torch.randn(shape, device="cuda")
        if regime == "uniform":
            bound = fp8_max * UNIFORM_SCALE
            val = val.clamp(-bound, bound)
            scale = torch.full((*shape[:-1], 1), UNIFORM_SCALE, device="cuda")
        else:
            scale = val.abs().amax(-1, keepdim=True) / fp8_max
        quant = (val / scale).to(FP8)
        planes.append((quant, scale.squeeze(-1).contiguous(), quant.float() * scale))
    (kq, ks, kd), (vq, vs, vd) = planes
    n, h, p, d = shape
    return _Pages(
        k_cache=kq.view(n, h, p, d // X, X).permute(0, 1, 3, 2, 4).contiguous(),
        v_cache=vq.view(n, h, p // X, X, d).permute(0, 1, 2, 4, 3).contiguous(),
        k_scale=ks,
        v_scale=vs,
        k_deq=kd,
        v_deq=vd,
    )


def _query(rows, heads, regime):
    shape = (rows, heads, HEAD_DIM)
    if regime == "uniform":
        return torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    return torch.randn(shape, device="cuda").to(torch.bfloat16)


def _lengths(kind, rows, window, patterns):
    if kind == "full":
        return [window] * rows
    pattern = patterns[kind]
    return [pattern[i % len(pattern)] for i in range(rows)]


def _sparse_tables(rows, lengths, num_kv_heads):
    """sparse_bt / sparse_ctx as the fused top-k emit writes them.

    A row spans ceil(ctx / 128) distinct 128-blocks, 8 pages each, the partial
    one last. Page ids are kv-head encoded (phys16 * Hkv + kvh) for the
    collapsed cache view; unused slots hold page 0.
    """
    ctx = _lengths(lengths, rows, SPARSE_WINDOW, SPARSE_LENGTHS)
    bt = torch.zeros(rows, TOPK_BLOCKS * PAGES, dtype=torch.int32)
    page = torch.arange(PAGES)
    for r, n in enumerate(ctx):
        used = math.ceil(n / sparse_attn.SPARSE_BLOCK_SIZE)
        blocks = torch.randperm(SPARSE_POOL_BLOCKS)[:used]
        phys = (blocks[:, None] * PAGES + page).flatten()
        bt[r, : phys.numel()] = phys * num_kv_heads + r % num_kv_heads
    return bt.cuda(), torch.tensor(ctx, dtype=torch.int32, device="cuda")


def _sparse_case(tokens, lengths, regime, num_kv_heads=1):
    bt, ctx = _sparse_tables(tokens * num_kv_heads, lengths, num_kv_heads)
    return _Case(
        q=_query(tokens, GROUP * num_kv_heads, regime),
        pages=_fp8_pages(
            SPARSE_POOL_BLOCKS * PAGES,
            num_kv_heads,
            sparse_attn.ASM_PAGE_SIZE,
            regime,
        ),
        bt=bt,
        ctx=ctx,
        num_kv_heads=num_kv_heads,
    )


def _dense_case(num_seqs, query_length, lengths, regime):
    ctx = _lengths(lengths, num_seqs, DENSE_BLOCKS * DENSE_PAGE, DENSE_LENGTHS)
    pool = num_seqs * DENSE_BLOCKS
    bt = torch.randperm(pool, dtype=torch.int32).view(num_seqs, DENSE_BLOCKS)
    return _Case(
        q=_query(num_seqs * query_length, GROUP, regime),
        pages=_fp8_pages(pool, 1, DENSE_PAGE, regime),
        bt=bt.cuda(),
        ctx=torch.tensor(ctx, dtype=torch.int32, device="cuda"),
        query_length=query_length,
    )


def _sparse(case, out, owner=None):
    """The production sparse entry, shared by decode and prefill."""
    sparse_attn._sparse_pa_per_row(
        case.q,
        case.pages.k_cache,
        case.pages.v_cache,
        case.bt,
        case.ctx,
        case.num_kv_heads,
        SM_SCALE,
        out,
        case.pages.k_scale,
        case.pages.v_scale,
        plan_step_owner=owner,
    )


def _dense(case, out):
    """The dense decode call as attention_mha builds it for an M3 layer."""
    num_seqs = case.ctx.shape[0]
    splits = ba.dense_decode_splits(num_seqs, 1)
    partials = (num_seqs, 1, splits, case.query_length * GROUP)
    ba.run_pa_decode(
        output=out,
        q=case.q,
        k_cache=case.pages.k_cache,
        v_cache=case.pages.v_cache,
        context_lens=case.ctx,
        block_tables=case.bt,
        softmax_scale=SM_SCALE,
        max_seqlen_q=case.query_length,
        max_context_partition_num=splits,
        context_partition_size=256,
        compute_type=FP8,
        q_scale=None,
        k_scale=case.pages.k_scale.unsqueeze(-1),
        v_scale=case.pages.v_scale.unsqueeze(-1),
        exp_sums=torch.empty(partials, dtype=torch.float32, device="cuda"),
        max_logits=torch.empty(partials, dtype=torch.float32, device="cuda"),
        temporary_output=torch.empty(
            *partials, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        ),
        alibi_slopes=None,
        sinks=None,
        sliding_window=-1,
        ps=True,
    )


def _reference(case):
    """Exact softmax attention, one query row at a time.

    Query j of a sequence sees all but its last ``query_length - 1 - j`` keys:
    the draft tokens are the tail of the context.
    """
    page_size = case.pages.k_deq.shape[2]
    keys = case.pages.k_deq.reshape(-1, HEAD_DIM)
    values = case.pages.v_deq.reshape(-1, HEAD_DIM)
    q_rows = case.q.view(-1, GROUP, HEAD_DIM)
    out = torch.zeros(q_rows.shape, dtype=torch.float32, device="cuda")
    lens = case.ctx.tolist()
    for i in range(q_rows.shape[0]):
        seq, j = divmod(i, case.query_length)
        n = lens[seq] - (case.query_length - 1 - j)
        if n <= 0:
            continue
        tok = torch.arange(n, device="cuda")
        slot = case.bt[seq, tok // page_size].long() * page_size + tok % page_size
        scores = q_rows[i].float() @ keys[slot].T * SM_SCALE
        out[i] = scores.softmax(-1) @ values[slot]
    return out


def _raise_pending_launch_error():
    """Fail on a launch error that was printed but never raised.

    torch reads hipGetLastError after each of its own launches; this is that
    read, done here rather than layers later.
    """
    try:
        torch.ones(1, device="cuda").add_(1)
    except RuntimeError as exc:
        pytest.fail(f"a kernel launch failed silently: {exc}")


def _close(name, actual, expected, tol):
    """Per row: one wrong row must not hide in the mean of a thousand right ones."""
    actual = actual.float().flatten(1)
    expected = expected.flatten(1)
    unwritten = (~torch.isfinite(actual)).any(1).nonzero().flatten().tolist()
    assert not unwritten, f"{name}: rows {unwritten[:8]} are not finite"
    err = (actual - expected).norm(dim=1) / expected.norm(dim=1).clamp_min(1e-6)
    worst = int(err.argmax())
    print(
        f"{name}: worst row {worst} rel_rms={err[worst].item():.3g} "
        f"mean={err.mean().item():.3g}"
    )
    assert err[worst] < tol, f"{name}: row {worst} off by {err[worst].item():.3g}"


class _Kernels:
    """Runs a call on gluon or FlyDSL and proves which one it reached.

    ASM is switched off: above `_gluon_one_pass_max_rows` (64 on MI355X) the
    sparse rows would go there instead, and that floor has its own test.
    Plans start empty so a case does not inherit another's.
    """

    def __init__(self, monkeypatch):
        self._monkeypatch = monkeypatch
        self.flydsl_calls = []
        launch = flydsl_pa.pa_decode

        @functools.wraps(launch)
        def counted(output, *args, **kwargs):
            self.flydsl_calls.append(tuple(output.shape))
            return launch(output, *args, **kwargs)

        monkeypatch.setattr(flydsl_pa, "pa_decode", counted)
        monkeypatch.setattr(sparse_attn, "_sparse_pa_ran_on_asm", lambda *a: False)
        monkeypatch.setattr(ba, "_FLYDSL_RUNTIME_PLANS", {})
        monkeypatch.setattr(ba, "_FLYDSL_PLAN_SCRATCH", {})

    def select(self, flydsl):
        self._monkeypatch.setattr(ba.envs, "ATOM_PA_FLYDSL", flydsl)

    def run(self, op, out, *, flydsl):
        """``op(out)`` on one kernel, as [rows, GROUP, HEAD_DIM]."""
        self.select(flydsl)
        out.fill_(float("nan"))
        launched = len(self.flydsl_calls)
        op(out)
        _raise_pending_launch_error()
        torch.cuda.synchronize()
        ran = len(self.flydsl_calls) > launched
        if flydsl:
            assert ran, "FlyDSL refused this call; it went to gluon"
        else:
            assert not ran, "FlyDSL ran with ATOM_PA_FLYDSL off"
        return out.view(-1, GROUP, HEAD_DIM)

    def match_reference(self, case, op, tol):
        want = _reference(case)
        for flydsl, name in ((False, "gluon"), (True, "flydsl")):
            got = self.run(op, torch.empty_like(case.q), flydsl=flydsl)
            _close(name, got, want, tol)


@pytest.fixture
def kernels(monkeypatch):
    return _Kernels(monkeypatch)


def test_first_one_row_prefill_call(kernels):
    """The call run 38236783820 died on: a fresh 1-row plan over the full window."""
    torch.manual_seed(0)
    case = _sparse_case(1, "full", "random")
    got = kernels.run(
        lambda out: _sparse(case, out), torch.empty_like(case.q), flydsl=True
    )
    _close("flydsl", got, _reference(case), TOL["random"])


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("lengths", ("full", "short", "mixed"))
@pytest.mark.parametrize("rows", (1, 2, 3, 4, 5, 8, 16, 64, 65, 256, 1024))
def test_sparse_rows(kernels, rows, lengths, regime):
    """M3's per-rank sparse call -- 16 heads on one kv head, a 2048-key window --
    with no step owner, as prefill makes it."""
    torch.manual_seed(rows)
    case = _sparse_case(rows, lengths, regime)
    kernels.match_reference(case, lambda out: _sparse(case, out), TOL[regime])


@pytest.mark.parametrize("owned", (False, True), ids=("per_call", "per_step"))
@pytest.mark.parametrize("rows", (1, 4, 64))
def test_plan_follows_new_lengths(kernels, rows, owned):
    """Same rows, new lengths: a plan that missed its refresh answers the old ones.

    An owner stands for one decode step's metadata, so each call gets a new one.
    """
    torch.manual_seed(rows)
    for lengths in ("full", "short", "mixed", "full"):
        case = _sparse_case(rows, lengths, "uniform")
        owner = SimpleNamespace() if owned else None
        got = kernels.run(
            lambda out: _sparse(case, out, owner),
            torch.empty_like(case.q),
            flydsl=True,
        )
        _close(f"flydsl {lengths}", got, _reference(case), TOL["uniform"])


def test_row_counts_in_serving_order(kernels):
    """Decode (rows a multiple of 4, one owner per step) interleaved with odd
    prefill tails (no owner), the order a PD prefill server sees them in."""
    torch.manual_seed(0)
    for rows in (4, 8, 64, 1, 3, 4, 1):
        case = _sparse_case(rows, "full", "uniform")
        owner = SimpleNamespace() if rows % 4 == 0 else None
        got = kernels.run(
            lambda out: _sparse(case, out, owner),
            torch.empty_like(case.q),
            flydsl=True,
        )
        _close(f"flydsl {rows} rows", got, _reference(case), TOL["uniform"])


@pytest.mark.parametrize("tokens", (1, 3, 32))
def test_two_kv_heads_fold_into_rows(kernels, tokens):
    """M3 at tp2: two kv heads per rank fold into rows over kv-head-encoded pages."""
    torch.manual_seed(tokens)
    case = _sparse_case(tokens, "mixed", "random", num_kv_heads=2)
    kernels.match_reference(case, lambda out: _sparse(case, out), TOL["random"])


def test_empty_padding_rows(kernels):
    """A padded decode batch carries rows the top-k emit wrote ctx 0 for."""
    torch.manual_seed(0)
    case = _sparse_case(8, "full", "uniform")
    case.ctx[5:] = 0
    want = _reference(case)[:5]
    for flydsl, name in ((False, "gluon"), (True, "flydsl")):
        got = kernels.run(
            lambda out: _sparse(case, out), torch.empty_like(case.q), flydsl=flydsl
        )
        _close(name, got[:5], want, TOL["uniform"])
        print(f"{name}: empty rows finite={torch.isfinite(got[5:]).all().item()}")


@pytest.mark.parametrize("rows", (1, 4, 64))
def test_graph_replay_replans(kernels, rows):
    """Captured decode records the plan refresh, so a replay reads the lengths
    in the buffers at replay time, not the ones seen at capture."""
    torch.manual_seed(rows)
    case = _sparse_case(rows, "full", "uniform")
    out = torch.empty_like(case.q)
    kernels.select(flydsl=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Build, then refresh: both compile before the capture.
        for _ in range(2):
            _sparse(case, out, SimpleNamespace())
    warm = len(kernels.flydsl_calls)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _sparse(case, out, SimpleNamespace())
    assert len(kernels.flydsl_calls) > warm, "the capture went to gluon"

    bt, ctx = _sparse_tables(rows, "mixed", 1)
    case.bt.copy_(bt)
    case.ctx.copy_(ctx)
    out.fill_(float("nan"))
    graph.replay()
    _raise_pending_launch_error()
    torch.cuda.synchronize()
    _close(
        "flydsl replay",
        out.view(-1, GROUP, HEAD_DIM),
        _reference(case),
        TOL["uniform"],
    )


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("lengths", ("full", "mixed"))
@pytest.mark.parametrize("query_length", (1, 4), ids=("plain", "eagle3_k3"))
@pytest.mark.parametrize("num_seqs", (1, 3, 4, 64))
def test_dense_decode(kernels, num_seqs, query_length, lengths, regime):
    """M3's dense layers at tp4 on the block-128 cache: one token per sequence,
    or EAGLE3's three drafts plus the target (16 heads x 4, gluon's limit).

    gluon passing confirms the reference's causal convention for the drafts.
    """
    torch.manual_seed(num_seqs)
    case = _dense_case(num_seqs, query_length, lengths, regime)
    kernels.match_reference(case, lambda out: _dense(case, out), TOL[regime])
