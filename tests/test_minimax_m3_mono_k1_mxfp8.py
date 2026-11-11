# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mono K1 over a native MXFP8 1x32 qkv weight (``attn_mxfp8``; gfx950, standalone build).

The data is chosen so that every number in the chain is exactly representable: the
residual row holds +-2^j (j = 0, 1, 2; 2048 / 3584 / 512 of them, mean square exactly
4) so the bf16-rounded gemma norm is +-{0.5, 1, 2}; the weights are {0, +-1, +-2}
with power-of-two E8M0 scales (2^-1 .. 2^1) so every partial sum of the f32
accumulation fits 24 bits, and the result does not depend on the accumulation order.
Then, bit for bit:

* the activation bytes and E8M0 codes the norm tasks hand over (S > 2) are what the
  native linear's own ``quantize_fp8`` makes of the same bf16 row;
* the GEMV rows K1 makes are a float64 oracle of the dequantized operands, rounded to
  bf16 (and, where AITER's MXFP8 GEMM runs, equal to its output).
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
    LOCAL_Q_HEADS,
    MAX_INDEX_BLOCKS,
    MAX_TOKENS,
    ROTARY_DIM,
    qkv_rows,
)
from atom.models.minimax_m3.mono.kernels.pre_attn import (
    K1_ABI,
    SCRATCH_BYTES,
    SCRATCH_QKV,
    SCRATCH_X8,
    SCRATCH_XSC,
    build_pre_attn_kernel,
)
from atom.mono.runtime.uncached import uncached_bytes


def _pow2(e):
    """2^e exactly (``torch.pow`` is not: 2^-2 * 3 came out as 2.9999999999999996)."""
    return torch.ldexp(torch.ones((), dtype=torch.float64), e.to(torch.int32))


SM_SCALE = HEAD_DIM**-0.5
ROWS = qkv_rows(1)
NB = HIDDEN // 32


def _residual_row(g):
    """+-2^j, mean square exactly 4, in a random order."""
    mag = torch.cat([torch.full((n,), 2.0**j) for j, n in enumerate((2048, 3584, 512))])
    mag = mag[torch.randperm(HIDDEN, generator=g)]
    return mag * (torch.randint(0, 2, (HIDDEN,), generator=g) * 2 - 1)


def _weights(g):
    """{0, +-1, +-2} e4m3, per-(row, 32-block) E8M0 codes 126..128; the (16, 16)-
    preshuffled fp8 weight, its row-major scales and the dequantized f64 weight."""
    q = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0])[
        torch.randint(0, 5, (ROWS, HIDDEN), generator=g)
    ]
    code = torch.randint(126, 129, (ROWS, NB), generator=g).to(torch.uint8)
    deq = q.double().view(ROWS, NB, 32) * _pow2(code.double() - 127).unsqueeze(-1)
    w = shuffle_weight(q.to(dtypes.fp8).cuda(), layout=(16, 16)).contiguous()
    return w, code.cuda().contiguous(), deq.view(ROWS, HIDDEN), q, code


def _run(tokens, seed=0, ctx=300, real=None):
    """``real``: (res [tokens, HIDDEN] bf16, g_in, shuffled fp8 weight, E8M0 scale bytes)
    instead of the exact-data case (the weights' oracle fields are then None)."""
    g = torch.Generator().manual_seed(seed)
    dev, bf = "cuda", torch.bfloat16
    if real is None:
        res = torch.stack([_residual_row(g) for _ in range(tokens)]).to(bf).to(dev)
        g_in = torch.zeros(HIDDEN, dtype=bf, device=dev)
        w, code, w_deq, q, code_cpu = _weights(g)
    else:
        res, g_in, w, code = (t.to(dev) for t in real)
        w_deq = q = code_cpu = None
    ar = torch.zeros(tokens, HIDDEN, dtype=bf, device=dev)
    gam = torch.zeros(HEAD_DIM, dtype=bf, device=dev)
    max_pos = ctx + 8
    cos_sin = torch.zeros(max_pos, ROTARY_DIM, dtype=bf, device=dev)
    positions = torch.full((tokens,), ctx - 1, dtype=torch.int64, device=dev)
    n_blk = (ctx + 127) // 128
    total = tokens * n_blk + 1
    block_table = (
        torch.arange(tokens * n_blk, dtype=torch.int32, device=dev)
        .view(tokens, n_blk)
        .contiguous()
    )
    seq = torch.full((tokens,), ctx, dtype=torch.int32, device=dev)
    slots = (block_table[:, (ctx - 1) // 128].long() * 128 + (ctx - 1) % 128).to(
        torch.int64
    )
    kc = torch.zeros(total * 128 * HEAD_DIM, dtype=torch.uint8, device=dev)
    vc, idx = torch.zeros_like(kc), torch.zeros_like(kc)
    ks = torch.zeros(total * 128, dtype=torch.float32, device=dev)
    vs = torch.zeros_like(ks)
    q_out = torch.zeros(MAX_TOKENS, LOCAL_Q_HEADS * HEAD_DIM, dtype=bf, device=dev)
    iq_out = torch.zeros(MAX_TOKENS, HEAD_DIM, dtype=bf, device=dev)
    h_out = torch.zeros(MAX_TOKENS, HIDDEN, dtype=bf, device=dev)
    iscore = torch.zeros(
        2 * MAX_INDEX_BLOCKS * MAX_TOKENS, dtype=torch.int32, device=dev
    )
    scratch = uncached_bytes(SCRATCH_BYTES, torch.device(dev))
    scratch.zero_()
    launch = build_pre_attn_kernel(1e-6, SM_SCALE, 0, 1, tokens, attn_mxfp8=True)
    args = K1_ABI.pack(
        {
            "ar": ar.data_ptr(), "res": res.data_ptr(), "h_out": h_out.data_ptr(),
            "g_in": g_in.data_ptr(), "w_qkv": w.data_ptr(), "s_qkv": code.data_ptr(),
            "g_q": gam.data_ptr(), "g_k": gam.data_ptr(), "g_iq": gam.data_ptr(),
            "g_ik": gam.data_ptr(), "cos_sin": cos_sin.data_ptr(),
            "positions": positions.data_ptr(), "slot_mapping": slots.data_ptr(),
            "k_cache": kc.data_ptr(), "v_cache": vc.data_ptr(), "k_scale": ks.data_ptr(),
            "v_scale": vs.data_ptr(), "index_cache": idx.data_ptr(), "q_out": q_out.data_ptr(),
            "iq_out": iq_out.data_ptr(), "block_table": block_table.data_ptr(),
            "seq_lens": seq.data_ptr(), "bt_width": n_blk, "q_len": 1, "iscore": iscore.data_ptr(),
            "scratch": scratch.data_ptr(), "layer": 3, "tl": 0,
        }
    )  # fmt: skip
    launch(*args)
    torch.cuda.synchronize()
    pairs = scratch[SCRATCH_QKV : SCRATCH_QKV + tokens * ROWS * 8].view(torch.int32)
    got = pairs.view(-1, 2)[:, 0].view(torch.float32).view(tokens, ROWS).clone()
    x8 = scratch[SCRATCH_X8 : SCRATCH_X8 + tokens * HIDDEN].view(tokens, HIDDEN).clone()
    xsc = (
        scratch[SCRATCH_XSC : SCRATCH_XSC + tokens * NB].view(tokens, NB).clone()
    )  # 4 codes a word
    return {
        "res": res,
        "w": w,
        "code": code,
        "w_deq": w_deq,
        "got": got,
        "x8": x8,
        "xsc": xsc,
        "q": q,
        "code_cpu": code_cpu,
    }


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 8, 16])
def test_k1_mxfp8_gemv_is_exact(tokens):
    from atom.model_ops.blockscale import quantize_fp8

    o = _run(tokens)
    # the norm's bf16 output is exactly x / 2 (see the module doc)
    normed = (o["res"].float() * 0.5).to(torch.bfloat16)
    xq, xs = quantize_fp8(normed)  # the native linear's activation quantization
    xdq = (
        xq.cpu().view(tokens, NB, 32).double()
        * _pow2(xs.view(torch.uint8).double().cpu() - 127).unsqueeze(-1)
    ).view(tokens, HIDDEN)
    want = (xdq.cpu() @ o["w_deq"].T).to(torch.bfloat16).float()
    got = o["got"].cpu()
    assert torch.isfinite(got).all() and got.abs().max() > 0
    assert torch.equal(got, want), (got - want).abs().max().item()
    if tokens > 2:  # the hand-off rows the norm tasks publish
        assert torch.equal(o["x8"].view(dtypes.fp8).float(), xq.float())
        assert torch.equal(o["xsc"].cpu(), xs.view(torch.uint8).cpu())


def test_the_oracle_tells_a_wrong_scale_mapping():
    """Negative control: the same operands with the weight scales of the neighbouring
    32-block (or neighbouring row) give another answer, so the exact equality above
    pins the block each lane's scale byte belongs to."""
    from atom.model_ops.blockscale import quantize_fp8

    o = _run(4)
    normed = (o["res"].float() * 0.5).to(torch.bfloat16)
    xq, xs = quantize_fp8(normed)
    xdq = (
        xq.cpu().view(4, NB, 32).double()
        * _pow2(xs.view(torch.uint8).double().cpu() - 127).unsqueeze(-1)
    ).view(4, HIDDEN)
    for what, code in (
        ("block", o["code_cpu"].roll(1, 1)),
        ("row", o["code_cpu"].roll(1, 0)),
    ):
        deq = (
            o["q"].double().view(ROWS, NB, 32)
            * _pow2(code.double() - 127).unsqueeze(-1)
        ).view(ROWS, HIDDEN)
        bad = (xdq.cpu() @ deq.T).to(torch.bfloat16).float()
        assert not torch.equal(o["got"].cpu(), bad), what
