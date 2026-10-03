# SPDX-License-Identifier: MIT
"""GPU check of the persistent HCA decode kernel through
``sparse_attn_v4_paged_decode``: persistent path vs the aiter decode ASM path
vs the torch reference (``mla_decode_fwd_v4_nm_ref``), on HCA-shaped inputs.

Needs gfx950 and an aiter with ``mla_decode_fwd_v4_nm_ps``; skipped otherwise.
"""

from __future__ import annotations

import random

import pytest
import torch

pytest.importorskip("aiter")
if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from aiter.jit.utils.chip_info import get_gfx

from atom.model_ops.v4_kernels import hca_persist, paged_decode
from atom.model_ops.v4_kernels.v4_quant import (
    mla_decode_fwd_v4_nm_ref,
    quantize_bf16_to_v4_2buff,
)

if get_gfx() != "gfx950":
    pytest.skip("persistent HCA kernel is gfx950-only", allow_module_level=True)
if hca_persist.mla_decode_fwd_v4_nm_ps is None:
    pytest.skip("aiter lacks mla_decode_fwd_v4_nm_ps", allow_module_level=True)

DEV = "cuda"
H = 128
MAX_ROWS_BUF = 1024


def _hca_k(pos: int) -> int:
    """HCA kv length at absolute position ``pos``: 128-token window + one
    compressed entry per 128 tokens."""
    return min(pos + 1, 128) + (pos + 1) // 128


def _q7_lens(rows, seed):
    """kv lens of `rows` q-len-7 verify rows of requests with random contexts."""
    rng = random.Random(seed)
    lens = []
    while len(lens) < rows:
        c = int(2 ** rng.uniform(13, 17))  # 8k .. 128k context
        lens += [_hca_k(c + i) for i in range(7)]
    return lens[:rows]


def _rnd(g, *shape):
    """bf16 randn with a per-64-group magnitude 2^U(-2, 2), so every tile
    scale differs."""
    x = torch.randn(*shape, 512, device=DEV, generator=g)
    mag = torch.exp2(torch.rand(*shape, 8, device=DEV, generator=g) * 4 - 2)
    return (x * mag.repeat_interleave(64, -1)).to(torch.bfloat16)


def _inputs(kv_lens, seed, pool_extra=257):
    g = torch.Generator(device=DEV).manual_seed(seed)
    n, tot = len(kv_lens), int(sum(kv_lens))
    pool = tot + pool_extra
    qp, qr = quantize_bf16_to_v4_2buff(_rnd(g, n, H))
    kp, kr = quantize_bf16_to_v4_2buff(_rnd(g, pool))
    indptr = torch.zeros(n + 1, dtype=torch.int32, device=DEV)
    indptr[1:] = torch.tensor(kv_lens, device=DEV).cumsum(0)
    idx = torch.randint(0, pool, (tot,), device=DEV, generator=g)
    return {
        "q_packed": qp,
        "q_rope": qr,
        "kv_packed": kp,
        "kv_rope": kr,
        "kv_indptr": indptr,
        "kv_page_indices": idx.to(torch.int32),
        "sink": (torch.randn(H, device=DEV, generator=g) * 2).float(),
    }


def _reference(inp, n):
    out = torch.empty((n, H, 512), dtype=torch.bfloat16, device=DEV)
    mla_decode_fwd_v4_nm_ref(
        inp["q_packed"][:n],
        inp["q_rope"][:n],
        inp["kv_packed"].view(-1, 1, 1, 512),
        inp["kv_rope"].view(-1, 1, 1, 64),
        out,
        torch.arange(n + 1, dtype=torch.int32, device=DEV),
        inp["kv_indptr"][: n + 1],
        inp["kv_page_indices"],
        torch.ones(n, dtype=torch.int32, device=DEV),
        1,
        sink=inp["sink"],
    )
    return out


def _close(a, b):
    """Per (row, head): cosine >= 0.9999 and relative L2 <= 1e-2."""
    a, b = a.double().reshape(-1, 512), b.double().reshape(-1, 512)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1).min().item()
    rel = ((a - b).norm(dim=1) / b.norm(dim=1).clamp_min(1e-30)).max().item()
    assert cos >= 0.9999 and rel <= 1e-2, (cos, rel)


@pytest.fixture
def mode(monkeypatch):
    def set_mode(persist, min_rows=15):
        monkeypatch.setenv("ATOM_V4_HCA_PERSIST", "1" if persist else "0")
        monkeypatch.setenv("ATOM_V4_HCA_PERSIST_MIN_ROWS", str(min_rows))

    return set_mode


def _run(inp, *, n=None, t_pad=None):
    """One ATOM decode call: q has `n` rows (default all), the CSR / qo_indptr
    have `t_pad` + 1 entries (default n), like the metadata builder stages them."""
    n = inp["q_packed"].shape[0] if n is None else n
    t_pad = n if t_pad is None else t_pad
    kv_indptr = inp["kv_indptr"][: t_pad + 1]
    kv_len = int((kv_indptr[1:] - kv_indptr[:-1]).max())
    plan = paged_decode.v4_decode_split_plan(
        t_pad, H, kv_len, torch.zeros(MAX_ROWS_BUF + 1, dtype=torch.int32, device=DEV)
    )
    return paged_decode.sparse_attn_v4_paged_decode(
        None,
        inp["kv_packed"],
        inp["kv_page_indices"],
        kv_indptr,
        inp["sink"],
        512**-0.5,
        unified_kv_rope=inp["kv_rope"],
        q_packed_in=inp["q_packed"][:n],
        q_rope_in=inp["q_rope"][:n],
        qo_indptr=torch.arange(t_pad + 1, dtype=torch.int32, device=DEV),
        split_plan=plan,
        compress_ratio=128,
    )


def _both(mode, inp, **kw):
    mode(False)
    asm = _run(inp, **kw)
    mode(True, min_rows=1)
    before = hca_persist.stats["persist"]
    per = _run(inp, **kw)
    assert hca_persist.stats["persist"] == before + 1, "persistent path not taken"
    torch.cuda.synchronize()
    return asm, per


@pytest.mark.parametrize("rows", [7, 15, 21, 112, 448, 896])
def test_rows_vs_asm_and_reference(mode, rows):
    inp = _inputs(_q7_lens(rows, rows), seed=rows)
    asm, per = _both(mode, inp)
    ref = _reference(inp, rows)
    _close(asm, ref)
    _close(per, ref)
    _close(per, asm)


@pytest.mark.parametrize("rows", [7, 14, 15, 112])
def test_default_min_rows_dispatch(mode, rows):
    inp = _inputs(_q7_lens(rows, 1), seed=1)
    mode(True)  # default MIN_ROWS 15
    before = hca_persist.stats["persist"]
    _run(inp)
    assert hca_persist.stats["persist"] - before == (1 if rows >= 15 else 0)


def test_graph_padded_rows(mode):
    """Captured-grid shape: T_pad rows, the tail has K = 0 (indptr repeats)."""
    real = _q7_lens(35, 3)
    inp = _inputs(real + [0] * 13, seed=3)
    asm, per = _both(mode, inp)
    r = len(real)
    _close(per[:r], _reference(inp, r))
    _close(per[:r], asm[:r])


def test_eager_fewer_rows_than_staged(mode):
    """Eager forward: q has N real rows, the CSR was staged for T_pad > N."""
    inp = _inputs(_q7_lens(21, 4) + [0] * 11, seed=4)
    asm, per = _both(mode, inp, n=21, t_pad=32)
    assert per.shape[0] == 21 and asm.shape[0] == 21
    _close(per, _reference(inp, 21))
    _close(per, asm)


def test_offset_kv_indptr(mode):
    inp = _inputs(_q7_lens(28, 5), seed=5)
    ref = _reference(inp, 28)
    off = 777
    shifted = dict(inp)
    shifted["kv_indptr"] = inp["kv_indptr"] + off
    junk = torch.randint(
        0, inp["kv_packed"].shape[0], (off,), dtype=torch.int32, device=DEV
    )
    shifted["kv_page_indices"] = torch.cat([junk, inp["kv_page_indices"]])
    asm, per = _both(mode, shifted)
    _close(per, ref)
    _close(asm, ref)


def test_cuda_graph_capture_replay_changing_csr(mode):
    """Capture once at T_pad rows, replay with new kv_indptr / indices / q."""
    t_pad, cap_k = 112, 1400
    mode(True)
    hca_persist.prepare(DEV)  # model load does this, before capture
    first = _inputs(_q7_lens(100, 6) + [0] * 12, seed=6, pool_extra=t_pad * cap_k)
    pool_rows = first["kv_packed"].shape[0]
    s = {
        "q_packed": first["q_packed"].clone(),
        "q_rope": first["q_rope"].clone(),
        "kv_packed": first["kv_packed"],
        "kv_rope": first["kv_rope"],
        "kv_indptr": torch.zeros(t_pad + 1, dtype=torch.int32, device=DEV),
        "kv_page_indices": torch.zeros(t_pad * cap_k, dtype=torch.int32, device=DEV),
        "sink": first["sink"],
    }

    def stage(lens, seed):
        g = torch.Generator(device=DEV).manual_seed(seed)
        ip = torch.zeros(t_pad + 1, dtype=torch.int32, device=DEV)
        ip[1:] = torch.tensor(lens, device=DEV).cumsum(0)
        idx = torch.randint(0, pool_rows, (int(ip[-1]),), generator=g, device=DEV)
        s["kv_indptr"].copy_(ip)
        s["kv_page_indices"][: idx.numel()].copy_(idx.to(torch.int32))
        other = _inputs([1] * t_pad, seed=seed)
        s["q_packed"].copy_(other["q_packed"])
        s["q_rope"].copy_(other["q_rope"])

    stage(_q7_lens(100, 6) + [0] * 12, 60)
    before = hca_persist.stats["persist"]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = paged_decode.sparse_attn_v4_paged_decode(
            None,
            s["kv_packed"],
            s["kv_page_indices"],
            s["kv_indptr"],
            s["sink"],
            512**-0.5,
            unified_kv_rope=s["kv_rope"],
            q_packed_in=s["q_packed"],
            q_rope_in=s["q_rope"],
            qo_indptr=torch.arange(t_pad + 1, dtype=torch.int32, device=DEV),
            compress_ratio=128,
        )
    assert hca_persist.stats["persist"] == before + 1
    for step, lens in enumerate(
        [
            _q7_lens(100, 6) + [0] * 12,
            _q7_lens(112, 7),
            _q7_lens(42, 8) + [0] * 70,
            [random.Random(9).randint(1, cap_k) for _ in range(t_pad)],
        ]
    ):
        stage(lens, 100 + step)
        graph.replay()
        torch.cuda.synchronize()
        r = sum(1 for k in lens if k > 0)
        _close(out[:r], _reference(s, r))
    # the merge counters are back to zero after every completed call
    ws = hca_persist._workspaces[torch.cuda.current_device()]
    assert int(ws.cnt.abs().sum()) == 0


def test_capture_without_workspace_stays_on_asm(mode, monkeypatch):
    """Nothing prepared the workspace before capture: the captured call keeps
    HCA decode on the ASM path instead of failing, and allocates nothing."""
    mode(True)
    monkeypatch.setattr(hca_persist, "_workspaces", {})
    t_pad = 112
    inp = _inputs(_q7_lens(100, 10) + [0] * 12, seed=10)
    plan = paged_decode.v4_decode_split_plan(
        t_pad,
        H,
        int((inp["kv_indptr"][1:] - inp["kv_indptr"][:-1]).max()),
        torch.zeros(MAX_ROWS_BUF + 1, dtype=torch.int32, device=DEV),
    )
    before = hca_persist.stats["persist"]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = paged_decode.sparse_attn_v4_paged_decode(
            None,
            inp["kv_packed"],
            inp["kv_page_indices"],
            inp["kv_indptr"],
            inp["sink"],
            512**-0.5,
            unified_kv_rope=inp["kv_rope"],
            q_packed_in=inp["q_packed"],
            q_rope_in=inp["q_rope"],
            qo_indptr=torch.arange(t_pad + 1, dtype=torch.int32, device=DEV),
            split_plan=plan,
            compress_ratio=128,
        )
    assert hca_persist.stats["persist"] == before
    assert hca_persist._workspaces == {}
    graph.replay()
    torch.cuda.synchronize()
    _close(out[:100], _reference(inp, 100))
