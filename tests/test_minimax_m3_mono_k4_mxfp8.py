# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mono K4's merge -> per-1x32 MXFP8 quantization -> o_proj over a native MXFP8 weight
(``attn_mxfp8``; gfx950; a single-rank layer kernel, K1 not fused, the MoE weights zero).

Every token attends over two keys whose scores differ by 22 nats (q = +-1, k1 = +1,
k2 = -1, all e4m3), so each head's attention output is exactly one of the two V rows
(the other one's softmax weight, 1.7e-10, is lost in the f32 sum): the merged output is
a known bf16 tensor, different per head, and the weights are {0, +-1, +-2} with
power-of-two E8M0 scales, so o_proj is exact in f32 whatever the accumulation order.
Bit for bit, the post-attention row the kernel stores (``h_mid``, zero residual; with
S >= 12 the reduce-scatter's ``a_ag`` row) is the float64 oracle of the native linear's
own activation quantization (``quantize_fp8`` of the bf16 attention output) times the
dequantized weight, rounded to bf16.
"""

import pytest
import torch

aiter = pytest.importorskip("aiter")
pytest.importorskip("flydsl")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="needs gfx950",
)

from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight

from atom.models.minimax_m3.mono.config import (
    HEAD_DIM,
    HIDDEN,
    INTER,
    LOCAL_Q_HEADS,
    N_ROUTED,
    O_K,
)
from atom.models.minimax_m3.mono.kernels.post_attn import K4_ABI, build_post_attn_kernel
from atom.models.minimax_m3.mono.layout import AG_RS_FROM, SCRATCH, sym_layout
from atom.mono.runtime.uncached import uncached_bytes


def _pow2(e):
    """2^e exactly (``torch.pow`` is not: 2^-2 * 3 came out as 2.9999999999999996)."""
    return torch.ldexp(torch.ones((), dtype=torch.float64), e.to(torch.int32))


PAGE16 = 16
NB = O_K // 32


def _case(tokens, seed=0, real=None):
    """``real``: (shuffled fp8 o_proj weight, E8M0 scale bytes, V scale) instead of the exact
    data (V rows then N(0, 1 / 2^blk) e4m3)."""
    g = torch.Generator().manual_seed(seed)
    dev = "cuda"
    # V rows: e4m3 values with short mantissas (2^j * {1, 1.5}), two per token
    mant = torch.tensor([1.0, 1.5, -1.0, -1.5])
    # (a power of two per 32 dims too, so the blocks' E8M0 codes differ)
    blk = torch.randint(-2, 4, (tokens, 2, HEAD_DIM // 32), generator=g)
    v = (
        mant[torch.randint(0, 4, (tokens, 2, HEAD_DIM), generator=g)]
        * 2.0 ** torch.randint(-1, 3, (tokens, 2, HEAD_DIM), generator=g)
        * 2.0 ** blk.repeat_interleave(32, dim=-1)
    ).to(dtypes.fp8)
    if real is not None:
        v = (torch.randn(tokens, 2, HEAD_DIM, generator=g) * 0.7).to(dtypes.fp8)
    pick = torch.randint(0, 2, (tokens, LOCAL_Q_HEADS), generator=g)  # head's key
    q = torch.where(pick == 0, 1.0, -1.0)[:, :, None].expand(-1, -1, HEAD_DIM)
    q = q.reshape(tokens, O_K).to(torch.bfloat16).to(dev)
    # one block a request, key 0 = +1 (K), key 1 = -1; request k's page16s k * 8 + j
    pages = tokens * 8
    kc = torch.zeros(pages, HEAD_DIM // 16, PAGE16, 16, dtype=torch.uint8)
    vc = torch.zeros(pages, HEAD_DIM, PAGE16, dtype=torch.uint8)
    one = torch.tensor([1.0, -1.0]).to(dtypes.fp8).view(torch.uint8)
    for k in range(tokens):
        for t in range(2):
            kc[k * 8, :, t, :] = one[t]
            vc[k * 8, :, t] = v[k, t].view(torch.uint8)
    ks = torch.ones(pages * PAGE16)
    vs = torch.ones(pages * PAGE16)
    block_table = torch.arange(tokens, dtype=torch.int32).view(tokens, 1)
    seq = torch.full((tokens,), 2, dtype=torch.int32)
    # the merged attention output the oracle expects, [tokens, O_K] bf16
    a = torch.stack(
        [
            torch.cat(
                [v[k, pick[k, h]].to(torch.bfloat16) for h in range(LOCAL_Q_HEADS)]
            )
            for k in range(tokens)
        ]
    )
    # o_proj weights: {0, +-1, +-2}, E8M0 codes 126..128 per (row, 32-block)
    wq = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0])[
        torch.randint(0, 5, (HIDDEN, O_K), generator=g)
    ]
    code = torch.randint(126, 129, (HIDDEN, NB), generator=g).to(torch.uint8)
    w_deq = (
        wq.double().view(HIDDEN, NB, 32) * _pow2(code.double() - 127)[..., None]
    ).view(HIDDEN, O_K)
    w = shuffle_weight(wq.to(dtypes.fp8).to(dev), layout=(16, 16)).contiguous()
    if real is not None:
        w, code, w_deq, wq = real[0].to(dev), real[1], None, None
    return {
        "q": q, "kc": kc.to(dev), "vc": vc.to(dev), "ks": ks.to(dev), "vs": vs.to(dev),
        "bt": block_table.to(dev), "seq": seq.to(dev), "a": a, "w": w, "code": code.to(dev), "w_deq": w_deq, "wq": wq, "code_cpu": code,
    }  # fmt: skip


def _run(tokens, real=None):
    c = _case(tokens, real=real)
    dev = torch.device("cuda")
    z = lambda *shape, dt=torch.uint8: torch.zeros(*shape, dtype=dt, device=dev)
    E = N_ROUTED
    w13, s13 = z(E * 2 * INTER * HIDDEN // 2), z(E * 2 * INTER * HIDDEN // 32)
    w2, s2 = z(E * HIDDEN * INTER // 2), z(E * HIDDEN * INTER // 32)
    w13s, s13s = z(2 * INTER * HIDDEN), z(2 * INTER * HIDDEN // 32)
    w2s, s2s = z(HIDDEN * INTER), z(HIDDEN * INTER // 32)
    gate = z(E, HIDDEN, dt=torch.bfloat16)
    bias = z(E, dt=torch.float32)
    g_post = z(HIDDEN, dt=torch.bfloat16)
    h_in = z(tokens, HIDDEN, dt=torch.bfloat16)
    h_mid = z(tokens, HIDDEN, dt=torch.bfloat16)
    ar_out = z(tokens, HIDDEN, dt=torch.bfloat16)
    scratch = uncached_bytes(SCRATCH["_bytes"], dev)
    sym = uncached_bytes(sym_layout(1)["_bytes"], dev)
    peers = torch.tensor([sym.data_ptr()], dtype=torch.int64, device=dev)
    k1 = torch.zeros(12, dtype=torch.int64, device=dev)
    ids = torch.arange(tokens, dtype=torch.int32, device=dev)
    launch = build_post_attn_kernel(
        1, 0.088, 1e-6, 1.0, 1.0, 7.0, 1, 1, tokens, shared_mxfp8=True,
        attn_mxfp8=True,
    )  # fmt: skip
    p = lambda t: t.data_ptr()
    args = K4_ABI.pack(
        {
            "h_in": p(h_in), "q": p(c["q"]), "block_table": p(c["bt"]), "seq_lens": p(c["seq"]),
            "k_cache": p(c["kc"]), "v_cache": p(c["vc"]), "k_scale": p(c["ks"]),
            "v_scale": p(c["vs"]), "w_o": p(c["w"]), "s_o": p(c["code"]), "g_post": p(g_post),
            "w_gate": p(gate), "bias": p(bias), "w13": p(w13), "s13": p(s13), "w2": p(w2), "s2": p(s2),
            "h_mid": p(h_mid), "ar_out": p(ar_out), "scratch": scratch.data_ptr(),
            "sym": sym.data_ptr(), "peers": p(peers), "rank": 0, "layer": 3, "bt_width": 1, "q_len": 1,
            "tl": 0, "k1_args": p(k1), "positions": p(ids), "slot_mapping": p(ids), "res": p(h_in),
            "batch_ids": p(ids), "w13s": p(w13s), "s13s": p(s13s), "w2s": p(w2s), "s2s": p(s2s),
        }
    )  # fmt: skip
    launch(*args)
    torch.cuda.synchronize()
    if tokens >= AG_RS_FROM:  # the owner's sums, gathered: pairs (bf16 x 2, tag)
        off = sym_layout(1)["a_ag"]
        words = sym[off : off + tokens * HIDDEN // 2 * 8].view(torch.int32)[0::2]
        got = words.contiguous().view(torch.bfloat16).view(tokens, HIDDEN).float()
    else:
        got = h_mid.float()
    return c, got.cpu()


@pytest.mark.parametrize("tokens", [1, 2, 4, 5, 8, 11, 12, 16])
def test_k4_mxfp8_merge_quant_and_o_proj_are_exact(tokens):
    from atom.model_ops.blockscale import quantize_fp8

    c, got = _run(tokens)
    xq, xs = quantize_fp8(c["a"].cuda())  # the native linear's activation quantization
    xdq = (
        xq.cpu().view(tokens, NB, 32).double()
        * _pow2(xs.view(torch.uint8).double().cpu() - 127).unsqueeze(-1)
    ).view(tokens, O_K)
    want = (xdq.cpu() @ c["w_deq"].T).to(torch.bfloat16).float()
    assert torch.isfinite(got).all() and got.abs().max() > 0
    assert torch.equal(got, want), (got - want).abs().max().item()


def test_the_oracle_tells_a_wrong_scale_mapping():
    """Negative control: the weight scales of the neighbouring block / row, or the
    activation codes of the neighbouring block (another head's), give another answer."""
    from atom.model_ops.blockscale import quantize_fp8

    tokens = 4
    c, got = _run(tokens)
    xq, xs = quantize_fp8(c["a"].cuda())
    xcode = xs.view(torch.uint8).double().cpu()

    def oracle(wcode, acode):
        xdq = (
            xq.cpu().view(tokens, NB, 32).double() * _pow2(acode - 127)[..., None]
        ).view(tokens, O_K)
        w = (
            c["wq"].double().view(HIDDEN, NB, 32)
            * _pow2(wcode.double() - 127)[..., None]
        ).view(HIDDEN, O_K)
        return (xdq @ w.T).to(torch.bfloat16).float()

    assert torch.equal(got, oracle(c["code_cpu"], xcode))
    for what, wcode, acode in (
        ("weight block", c["code_cpu"].roll(1, 1), xcode),
        ("weight row", c["code_cpu"].roll(1, 0), xcode),
        ("activation block", c["code_cpu"], xcode.roll(1, 1)),
        ("activation head", c["code_cpu"], xcode.roll(4, 1)),
    ):
        assert not torch.equal(got, oracle(wcode, acode)), what
