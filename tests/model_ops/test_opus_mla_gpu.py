# SPDX-License-Identifier: MIT
"""GPU check of the opus flat-row MLA decode (``ATOM_OPUS_MLA``) as ATOM calls
it: ``opus_mla.decode`` vs ``mla_decode_fwd`` (the default path) vs an fp32
reference, on round-robin DCP8 shapes (96 gathered heads, one rank's KV
share) and on DCP1 shapes (no decode context parallelism: a rank's own 12
or 16 heads over its whole KV), eager and as a CUDA graph replayed with a later
step's plan.

Needs gfx950 and an aiter with ``OpusMlaFlatDecode``; skipped otherwise.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

aiter = pytest.importorskip("aiter")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

from atom.model_ops import opus_mla

if get_gfx() != "gfx950":
    pytest.skip("opus flat-row decode is gfx950-only", allow_module_level=True)
if opus_mla.OpusMlaFlatDecode is None:
    pytest.skip("aiter lacks OpusMlaFlatDecode", allow_module_level=True)

DEV = "cuda"
H, DIM, DV = opus_mla.HEADS, opus_mla.QK_DIM, opus_mla.V_DIM
W, RANK = 8, 3
SCALE = 192**-0.5
# aiter's own bounds for the kernel, which feeds P to the PV MFMA as fp8: O max
# error as a fraction of the reference's max, LSE max absolute error.
O_TOL, LSE_TOL = 5e-2, 1e-2
_META = (
    "work_meta_data",
    "work_indptr",
    "work_info_set",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)


def _indptr(lens) -> np.ndarray:
    out = np.zeros(len(lens) + 1, dtype=np.int32)
    out[1:] = np.cumsum(lens)
    return out


def _contexts(bs: int, seed: int) -> list[int]:
    """Global context lengths (new tokens included), lognormal around 16k."""
    rng = np.random.default_rng(seed)
    return [
        int(min(max(16000 * math.exp(0.6 * rng.standard_normal()), 200), 100000))
        for _ in range(bs)
    ]


def _local_tokens(ctxs: list[int], world: int = W, rank: int = RANK) -> int:
    return sum((c - rank + world - 1) // world for c in ctxs)


class _Step:
    """One rank's decode inputs, in fixed buffers sized for ``max_tokens``:
    round-robin DCP (``world`` > 1, ``heads`` gathered) or DCP1 (``world`` 1,
    rank 0, the rank's own ``heads``)."""

    def __init__(
        self,
        bs: int,
        q_len: int,
        max_tokens: int,
        heads: int = H,
        world: int = W,
        rank: int = RANK,
    ):
        self.bs, self.q_len = bs, q_len
        self.heads, self.world, self.rank = heads, world, rank
        i32 = {"dtype": torch.int32, "device": DEV}
        self.q = torch.empty((bs * q_len, heads, DIM), dtype=dtypes.fp8, device=DEV)
        self.kv = torch.empty((max_tokens, 1, 1, DIM), dtype=dtypes.fp8, device=DEV)
        self.qo_indptr = torch.arange(0, (bs + 1) * q_len, q_len, **i32)
        self.kv_indptr = torch.zeros(bs + 1, **i32)
        self.g_kv_indptr = torch.zeros(bs + 1, **i32)
        self.kv_indices = torch.zeros(max_tokens, **i32)
        self.last = torch.ones(bs, **i32)
        self.one = torch.ones(1, dtype=torch.float32, device=DEV)

    def fill(self, ctxs: list[int], seed: int) -> None:
        """Rewrite the buffers in place for a step with these contexts."""
        self.ctxs = ctxs
        self.lens = [(c - self.rank + self.world - 1) // self.world for c in ctxs]
        self.kv_indptr_np = _indptr(self.lens)
        total = int(self.kv_indptr_np[-1])
        assert total <= self.kv.shape[0]
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.q.copy_(torch.randn(self.q.shape, generator=g, device=DEV))
        self.kv.copy_(torch.randn(self.kv.shape, generator=g, device=DEV))
        self.kv_indptr.copy_(torch.from_numpy(self.kv_indptr_np))
        self.g_kv_indptr.copy_(torch.from_numpy(_indptr(ctxs)))
        perm = torch.randperm(self.kv.shape[0], generator=g, device=DEV)[:total]
        self.kv_indices[:total].copy_(perm.to(torch.int32))

    def call_kwargs(self, meta=None) -> dict:
        """The keyword arguments ATOM's ``_forward_decode`` passes."""
        kw = {
            "page_size": 1,
            "num_kv_splits": 16,
            "sm_scale": SCALE,
            "q_scale": self.one,
            "kv_scale": self.one,
            "return_lse": True,
            # ATOM passes no global indptr without CP
            "g_kv_indptr": self.g_kv_indptr if self.world > 1 else None,
            "cp_world_size": self.world,
            "cp_rank": self.rank,
            "causal": True,
        }
        if meta is not None:
            kw.update(meta)
        return kw

    def args(self, o):
        return (
            self.q,
            self.kv,
            o,
            self.qo_indptr,
            self.kv_indptr,
            self.kv_indices,
            self.last,
            self.q_len,
        )

    def default_metadata(self) -> dict:
        info = aiter.get_mla_metadata_info_v1(
            self.bs,
            self.q_len,
            self.heads,
            dtypes.fp8,
            dtypes.fp8,
            is_sparse=False,
            fast_mode=True,
        )
        meta = {
            n: torch.zeros(s, dtype=t, device=DEV) for n, (s, t) in zip(_META, info)
        }
        aiter.get_mla_metadata_v1(
            self.qo_indptr,
            self.kv_indptr,
            self.last,
            self.heads,
            1,
            True,
            meta["work_meta_data"],
            meta["work_info_set"],
            meta["work_indptr"],
            meta["reduce_indptr"],
            meta["reduce_final_map"],
            meta["reduce_partial_map"],
            page_size=1,
            kv_granularity=16,
            max_seqlen_qo=self.q_len,
            uni_seqlen_qo=self.q_len,
            fast_mode=True,
            max_split_per_batch=-1,
            dtype_q=dtypes.fp8,
            dtype_kv=dtypes.fp8,
            is_cp_round_robin=self.world > 1,
        )
        return meta

    def reference(self):
        """fp32 attention under the round-robin causal mask: local key j sits
        at global position j * world + rank (j without CP) and query token t
        at ctx - q_len + t."""
        H = self.heads
        q = self.q.float().view(self.bs, self.q_len, H, DIM)
        o = torch.empty((self.bs, self.q_len, H, DV), device=DEV)
        lse = torch.empty((self.bs, self.q_len, H), device=DEV)
        for b, (ctx, n) in enumerate(zip(self.ctxs, self.lens)):
            start = int(self.kv_indptr_np[b])
            kv = self.kv.view(-1, DIM)[self.kv_indices[start : start + n].long()]
            kv = kv.float()
            pos = torch.arange(n, device=DEV) * self.world + self.rank
            qpos = ctx - self.q_len + torch.arange(self.q_len, device=DEV)
            s = torch.einsum("thd,kd->thk", q[b], kv) * SCALE
            s.masked_fill_((pos[None, :] > qpos[:, None])[:, None, :], -math.inf)
            lse[b] = torch.logsumexp(s, -1)
            o[b] = torch.einsum("thk,kd->thd", torch.softmax(s, -1), kv[:, :DV])
        return o.view(-1, H, DV), lse.view(-1, H)


def _err(x, ref) -> float:
    return float((x.float() - ref).abs().max() / ref.abs().max())


def _rms_err(x, ref) -> float:
    return float((x.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())


def _new_decoder(monkeypatch, max_bs: int):
    monkeypatch.setenv("ATOM_OPUS_MLA", "1")
    monkeypatch.setenv("ATOM_OPUS_MLA_MAX_BS", str(max_bs))
    monkeypatch.setattr(opus_mla, "_decoder", None)


@pytest.mark.parametrize("q_len", [2, 4, 6])
@pytest.mark.parametrize("bs", [1, 4, 11, 32, 128])
def test_matches_default_decode_and_reference(monkeypatch, bs, q_len):
    _new_decoder(monkeypatch, 128)
    ctxs = _contexts(bs, seed=bs * 10 + q_len)
    step = _Step(bs, q_len, max_tokens=_local_tokens(ctxs))
    step.fill(ctxs, seed=bs + q_len)
    ref_o, ref_lse = step.reference()

    o_def = torch.empty((bs * q_len, H, DV), dtype=torch.bfloat16, device=DEV)
    _, lse_def = aiter.mla.mla_decode_fwd(
        *step.args(o_def), **step.call_kwargs(step.default_metadata())
    )

    marker = opus_mla.make_marker(bs, q_len)
    opus_mla.update_plan(step.kv_indptr_np, bs, q_len, W)
    o = torch.empty_like(o_def)
    kw = step.call_kwargs()
    assert opus_mla.wanted(
        SimpleNamespace(opus_plan=marker),
        q=step.q,
        kv_buffer=step.kv,
        o=o,
        q_scale=kw["q_scale"],
        kv_scale=kw["kv_scale"],
        causal=True,
        cp_world_size=W,
        max_q_len=q_len,
        page_size=1,
    )
    out, lse = opus_mla.decode(*step.args(o), **kw)
    torch.cuda.synchronize()

    assert out is o
    assert not o.isnan().any()
    assert _err(o_def, ref_o) < O_TOL
    assert _err(o, ref_o) < O_TOL
    torch.testing.assert_close(lse, ref_lse, atol=LSE_TOL, rtol=0)
    torch.testing.assert_close(lse, lse_def, atol=LSE_TOL, rtol=0)


def test_graph_replays_with_a_later_plan(monkeypatch):
    """ATOM's use: capture once, then every step rewrites the inputs in place
    and publishes a new plan before the replay."""
    bs, q_len = 32, 6
    _new_decoder(monkeypatch, 64)
    ctxs = {seed: _contexts(bs, seed=seed) for seed in (1, 2, 3)}
    step = _Step(bs, q_len, max_tokens=max(map(_local_tokens, ctxs.values())))
    o = torch.empty((bs * q_len, H, DV), dtype=torch.bfloat16, device=DEV)
    lse_out = torch.empty((bs * q_len, H), dtype=torch.float32, device=DEV)
    kw = step.call_kwargs()

    step.fill(ctxs[1], seed=1)
    opus_mla.capture_marker(bs, q_len)

    def fwd():
        lse_out.copy_(opus_mla.decode(*step.args(o), **kw)[1])

    fwd()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fwd()

    for seed in (2, 3):
        step.fill(ctxs[seed], seed=seed)
        opus_mla.update_plan(step.kv_indptr_np, bs, q_len, W)
        graph.replay()
        torch.cuda.synchronize()
        ref_o, ref_lse = step.reference()
        assert _err(o, ref_o) < O_TOL
        torch.testing.assert_close(lse_out, ref_lse, atol=LSE_TOL, rtol=0)


# ---- DCP1: one rank's own heads over its whole KV, no CP

# The largest single-element error is the tail of the fp8 P the kernel feeds
# to PV: over 36 seeded cases each it reaches 0.0496 at DCP8 (96 heads) and
# 0.041 at DCP1, against O_TOL 0.05. DCP1 checks the RMS error tightly
# (0.009 to 0.014 at DCP1, 0.017 to 0.022 at DCP8) and the max element loosely.
DCP1_RMS_TOL, DCP1_MAX_TOL = 2e-2, 1e-1
DCP1_SHAPES = [(16, 2), (16, 4), (16, 6), (16, 8), (12, 6)]


@pytest.mark.parametrize(
    "heads,q_len", DCP1_SHAPES, ids=[f"h{h}-q{q}" for h, q in DCP1_SHAPES]
)
@pytest.mark.parametrize("bs", [1, 4, 11, 32])
def test_dcp1_matches_default_decode_and_reference(monkeypatch, bs, heads, q_len):
    _new_decoder(monkeypatch, 128)
    ctxs = _contexts(bs, seed=bs * 10 + q_len + heads)
    step = _Step(bs, q_len, _local_tokens(ctxs, 1, 0), heads=heads, world=1, rank=0)
    step.fill(ctxs, seed=bs + q_len + heads)
    ref_o, ref_lse = step.reference()

    # the default path's LSE, where aiter's asm decode has a kernel for the
    # shape (not 16 heads at query length 2). Its O is not compared: at some
    # no-CP shapes (16 heads, query length 4) it is itself off the reference.
    o_def = torch.empty((bs * q_len, heads, DV), dtype=torch.bfloat16, device=DEV)
    try:
        _, lse_def = aiter.mla.mla_decode_fwd(
            *step.args(o_def), **step.call_kwargs(step.default_metadata())
        )
    except RuntimeError as exc:
        if "cannot get heuristic kernel" not in str(exc):
            raise
        lse_def = None

    marker = opus_mla.make_marker(bs, q_len, heads)
    opus_mla.update_plan(step.kv_indptr_np, bs, q_len, 1, heads)
    o = torch.empty((bs * q_len, heads, DV), dtype=torch.bfloat16, device=DEV)
    kw = step.call_kwargs()
    assert opus_mla.wanted(
        SimpleNamespace(opus_plan=marker),
        q=step.q,
        kv_buffer=step.kv,
        o=o,
        q_scale=kw["q_scale"],
        kv_scale=kw["kv_scale"],
        causal=True,
        cp_world_size=1,
        max_q_len=q_len,
        page_size=1,
    )
    out, lse = opus_mla.decode(*step.args(o), **kw)
    torch.cuda.synchronize()

    assert out is o
    assert not o.isnan().any()
    assert _rms_err(o, ref_o) < DCP1_RMS_TOL
    assert _err(o, ref_o) < DCP1_MAX_TOL
    torch.testing.assert_close(lse, ref_lse, atol=LSE_TOL, rtol=0)
    if lse_def is not None:
        torch.testing.assert_close(lse, lse_def, atol=LSE_TOL, rtol=0)


def test_dcp1_graph_replays_with_a_later_plan(monkeypatch):
    bs, q_len, heads = 16, 6, 16
    _new_decoder(monkeypatch, 64)
    ctxs = {seed: _contexts(bs, seed=seed) for seed in (1, 2, 3)}
    tokens = max(_local_tokens(c, 1, 0) for c in ctxs.values())
    step = _Step(bs, q_len, tokens, heads=heads, world=1, rank=0)
    o = torch.empty((bs * q_len, heads, DV), dtype=torch.bfloat16, device=DEV)
    lse_out = torch.empty((bs * q_len, heads), dtype=torch.float32, device=DEV)
    kw = step.call_kwargs()

    step.fill(ctxs[1], seed=1)
    opus_mla.capture_marker(bs, q_len, heads)

    def fwd():
        lse_out.copy_(opus_mla.decode(*step.args(o), **kw)[1])

    fwd()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fwd()

    for seed in (2, 3):
        step.fill(ctxs[seed], seed=seed)
        opus_mla.update_plan(step.kv_indptr_np, bs, q_len, 1, heads)
        graph.replay()
        torch.cuda.synchronize()
        ref_o, ref_lse = step.reference()
        assert _rms_err(o, ref_o) < DCP1_RMS_TOL
        assert _err(o, ref_o) < DCP1_MAX_TOL
        torch.testing.assert_close(lse_out, ref_lse, atol=LSE_TOL, rtol=0)
