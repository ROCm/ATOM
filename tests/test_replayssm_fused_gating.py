# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerics of the Qwen3.5 GDN decode fusions against the unfused kernels.

F1 (ATOM_GDN_REPLAYSSM_FUSED_GATING): `replayssm_gated_delta_rule` with the
gate computed in-kernel from the raw a/b columns must be bit-identical to
`fused_gdn_gating` + the same kernel reading g/beta -- output, record buffers,
checkpoint and cursor -- across the T1_TILED (T=1), serial (T=2, 4) and both
launch configs (BV=16 small-batch on gfx950, BV=64), padded entries, the
prefill sentinel, a flush step, and multi-step partial accepts.

GPU-only, run manually: `pytest tests/test_replayssm_fused_gating.py` on a
ROCm GPU. No CI job runs it -- the non-GPU unit runner has neither triton nor
aiter, and TRITON_INTERPRET=1 does not help without a GPU because importing
attention_gdn pulls in aiter's Triton utils, which probe the GPU arch at
import. The switch's dispatch and entry-point argument handling run without a
kernel in tests/test_qwen35_gdn_decode_fusion_switches.py.
"""

import pytest
import torch

if not (torch.cuda.is_available() and torch.version.hip is not None):
    pytest.skip("GPU-only (ROCm), run manually; not in CI", allow_module_level=True)

from atom.model_ops.attention_gdn import fused_gdn_gating
from atom.model_ops.fla_ops import replayssm as rs

CAP = 16
STATE = ("ckpt", "buf_k", "buf_u", "buf_g", "write_pos")
DEV = "cuda"


def _case(bs, T, H, HV, K, V, seed):
    gen = torch.Generator().manual_seed(seed)

    def rn(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(dtype).to(DEV)

    T_tot, nslots = bs * T, bs + 3
    # [q|k|v|z|b|a] row, so b and a are strided column views as in qwen3_5.py
    off = 2 * H * K + 2 * HV * V
    row = rn(T_tot, off + 2 * HV)
    slots = torch.randperm(nslots, generator=gen)[:bs].int()
    if bs > 1:
        slots[-1] = rs.PAD_SLOT_ID
    wp = torch.randint(0, CAP - 2 * T + 1, (nslots,), generator=gen).int()
    live = slots[slots >= 0].long()
    wp[live[0]] = -1  # prefill sentinel
    if len(live) > 1:
        wp[live[1]] = CAP - T  # flushes this step
    return {
        "q": rn(1, T_tot, H, K),
        "k": rn(1, T_tot, H, K),
        "v": rn(1, T_tot, HV, V),
        "b": row[:, off : off + HV],
        "a": row[:, off + HV :],
        "A_log": (torch.rand(HV, generator=gen) * 2 - 1).to(DEV),
        "dt_bias": rn(HV),
        "ckpt": rn(nslots, HV, K, V, scale=0.1),
        "buf_k": rn(nslots, HV, CAP, K, scale=0.1),
        "buf_u": rn(nslots, HV, CAP, V, scale=0.1),
        "buf_g": (-torch.rand(nslots, HV, CAP, generator=gen) * 0.5).to(DEV),
        "write_pos": wp.to(DEV),
        "slot_idx": slots.to(DEV),
        "cu": (torch.arange(bs + 1, dtype=torch.int32) * T).to(DEV),
    }


def _run(d, T, fused, route="auto"):
    kw = {
        "q": d["q"],
        "k": d["k"],
        "v": d["v"],
        "ckpt": d["ckpt"],
        "buf_k": d["buf_k"],
        "buf_u": d["buf_u"],
        "buf_g": d["buf_g"],
        "write_pos": d["write_pos"],
        "slot_idx": d["slot_idx"],
        "cu_seqlens": d["cu"],
        "max_query_len": T,
        "use_qk_l2norm_in_kernel": True,
        "route": route,
    }
    if fused:
        return rs.replayssm_gated_delta_rule(
            g=None,
            beta=None,
            a=d["a"],
            b=d["b"],
            A_log=d["A_log"],
            dt_bias=d["dt_bias"],
            **kw,
        )
    g, beta = fused_gdn_gating(d["A_log"], d["a"], d["b"], d["dt_bias"])
    return rs.replayssm_gated_delta_rule(g=g, beta=beta, **kw)


def _fork(d):
    return {k: (v.clone() if k in STATE else v) for k, v in d.items()}


BATCHES = [1, 2, 4, 6, 8, 16, 64]
SHAPES = [(4, 16), (8, 32)]  # (H, HV): TP4, TP2


@pytest.mark.parametrize("gfx", ["gfx950", "gfx942"])  # BV=16 vs BV=64 config
@pytest.mark.parametrize("H,HV", SHAPES)
@pytest.mark.parametrize("T", [1, 2, 4])
@pytest.mark.parametrize("bs", BATCHES)
def test_f1_in_kernel_gating_is_bit_identical(monkeypatch, bs, T, H, HV, gfx):
    monkeypatch.setattr(rs, "get_gfx", lambda: gfx)
    base = _case(bs, T, H, HV, 128, 128, seed=97 * bs + T + HV)
    ref, fus = _fork(base), _fork(base)
    live_tok = (base["slot_idx"] >= 0).repeat_interleave(T).cpu()
    gen = torch.Generator().manual_seed(bs + T)
    for step in range(3):
        o_ref = _run(ref, T, fused=False)[0, live_tok]
        o_fus = _run(fus, T, fused=True)[0, live_tok]
        assert torch.equal(o_ref, o_fus), f"o differs at step {step}"
        for key in STATE:
            assert torch.equal(ref[key], fus[key]), f"{key} differs at step {step}"
        accepted = torch.randint(1, T + 1, (bs,), generator=gen).int().to(DEV)
        for st in (ref, fus):
            rs.replayssm_commit(st["write_pos"], st["slot_idx"], accepted, T, CAP)


def test_f1_ut_route_falls_back_to_the_gating_kernel():
    T = 12
    base = _case(4, T, 4, 16, 128, 128, seed=5)
    ref, fus = _fork(base), _fork(base)
    live_tok = (base["slot_idx"] >= 0).repeat_interleave(T).cpu()
    o_ref = _run(ref, T, fused=False, route="ut")[0, live_tok]
    o_fus = _run(fus, T, fused=True, route="ut")[0, live_tok]
    assert torch.equal(o_ref, o_fus)
    for key in STATE:
        assert torch.equal(ref[key], fus[key]), key
