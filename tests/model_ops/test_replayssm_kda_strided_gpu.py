# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""KDA column views and copies agree across append and flush."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops.fla_ops.replayssm import (
    replayssm_flush_wait,
    replayssm_sigmoid_gating_delta_rule,
)

DEV = "cuda"
HV, D = 12, 128
FUSED_N = 4 * HV * D + HV + D
BETA_COL = 4 * HV * D
T_MAX, CAP = 8, 24


@pytest.mark.parametrize(
    "lens", [[8, 3, 8, 1], [8] * 4, [8] * 16], ids=["T20", "T32", "T128"]
)
@pytest.mark.parametrize("flush", [False, True])
@pytest.mark.parametrize("head_step", [1, 2])
@pytest.mark.parametrize("graph", [False, True])
def test_strided_beta_matches_contiguous(lens, flush, head_step, graph):
    torch.manual_seed(0)
    T = sum(lens)
    slots = max(8, 2 * len(lens))
    bf16 = {"device": DEV, "dtype": torch.bfloat16}
    q, k, v = (torch.randn(1, T, HV, D, **bf16) for _ in range(3))
    fused = torch.randn(T, FUSED_N, **bf16)
    gate = fused[:, BETA_COL - head_step * HV * D : BETA_COL]
    gate = gate.view(1, T, HV, head_step * D)[..., :D]
    beta = fused[:, BETA_COL : BETA_COL + head_step * HV : head_step].unsqueeze(0)
    assert not beta.is_contiguous() and not gate.is_contiguous()
    A_log = torch.randn(HV, device=DEV)
    dt_bias = torch.randn(HV * D, device=DEV)
    lo, hi = (T_MAX + 1, CAP - T_MAX) if flush else (1, CAP - 2 * T_MAX)
    state = {
        "ckpt": torch.randn(slots, HV, D, D, device=DEV) * 0.1,
        "buf_k": torch.randn(slots, HV, CAP, D, **bf16),
        "buf_u": torch.randn(slots, HV, CAP, D, **bf16) * 0.1,
        "buf_g": -torch.rand(slots, HV, CAP, D, device=DEV) * 0.1,
        "write_pos": torch.randint(lo, hi + 1, (slots,), device=DEV, dtype=torch.int32),
    }
    cu = torch.tensor([0] + lens, device=DEV).cumsum(0).to(torch.int32)
    slot_idx = torch.randperm(slots, device=DEV)[: len(lens)].to(torch.int32)

    wait = replayssm_flush_wait()

    def run(a, b):
        arrivals = torch.zeros(slots, HV, dtype=torch.int32, device=DEV)
        s = {name: tensor.clone() for name, tensor in state.items()}

        def launch():
            return replayssm_sigmoid_gating_delta_rule(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                slot_idx=slot_idx,
                cu_seqlens=cu,
                max_query_len=T_MAX,
                use_qk_l2norm_in_kernel=True,
                lower_bound=-5.0,
                flush_arrivals=arrivals,
                flush_wait=wait,
                **s,
            )

        if not graph:
            return launch(), s
        launch()  # Compile this layout before graph capture.
        torch.cuda.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            out = launch()
        torch.cuda.synchronize()
        for name, tensor in s.items():
            tensor.copy_(state[name])
        captured.replay()
        torch.cuda.synchronize()
        return out, s

    out_s, got = run(gate, beta)
    out_c, want = run(gate.contiguous(), beta.contiguous())
    assert torch.equal(out_s, out_c)
    for name in state:
        assert torch.equal(got[name], want[name]), name
    assert torch.equal(got["ckpt"], state["ckpt"]) != flush
