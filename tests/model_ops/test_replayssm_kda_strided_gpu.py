# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""ReplaySSM's KDA step reads beta (and the gate) as column views of a wider
buffer, as Kimi-K3 passes its in_proj output, exactly as it reads them
contiguous: output, checkpoints, record buffers and cursors equal bitwise."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops.fla_ops.replayssm import replayssm_sigmoid_gating_delta_rule

DEV = "cuda"
HV, D = 12, 128  # heads and head dim per rank, Kimi-K3 at TP8
FUSED_N = 4 * HV * D + HV + D  # [q | k | v | gate | beta | f_a]
BETA_COL = 4 * HV * D
T_MAX, CAP, SLOTS = 8, 24, 8


@pytest.mark.parametrize("lens", [[8, 3, 8, 1], [8, 8, 8, 8]], ids=["T20", "T32"])
@pytest.mark.parametrize("flush", [False, True])
# Heads a step of 2 apart are a layout the kernel cannot index; the op copies.
@pytest.mark.parametrize("head_step", [1, 2])
def test_strided_beta_matches_contiguous(lens, flush, head_step):
    torch.manual_seed(0)
    T = sum(lens)
    bf16 = {"device": DEV, "dtype": torch.bfloat16}
    q, k, v = (torch.randn(1, T, HV, D, **bf16) for _ in range(3))
    fused = torch.randn(T, FUSED_N, **bf16)
    # The gate is contiguous in Kimi-K3; a view here covers its offsets too.
    gate = fused[:, BETA_COL - head_step * HV * D : BETA_COL]
    gate = gate.view(1, T, HV, head_step * D)[..., :D]
    beta = fused[:, BETA_COL : BETA_COL + head_step * HV : head_step].unsqueeze(0)
    assert not beta.is_contiguous() and not gate.is_contiguous()
    A_log = torch.randn(HV, device=DEV)
    dt_bias = torch.randn(HV * D, device=DEV)
    # h + 2 * T_MAX > CAP folds the records into the checkpoint first.
    lo, hi = (T_MAX + 1, CAP - T_MAX) if flush else (1, CAP - 2 * T_MAX)
    state = {
        "ckpt": torch.randn(SLOTS, HV, D, D, device=DEV) * 0.1,
        "buf_k": torch.randn(SLOTS, HV, CAP, D, **bf16),
        "buf_u": torch.randn(SLOTS, HV, CAP, D, **bf16) * 0.1,
        "buf_g": -torch.rand(SLOTS, HV, CAP, D, device=DEV) * 0.1,
        "write_pos": torch.randint(lo, hi + 1, (SLOTS,), device=DEV, dtype=torch.int32),
    }
    cu = torch.tensor([0] + lens, device=DEV).cumsum(0).to(torch.int32)
    slot_idx = torch.tensor([5, 2, 7, 0], device=DEV, dtype=torch.int32)

    def run(a, b):
        s = {name: t.clone() for name, t in state.items()}
        out = replayssm_sigmoid_gating_delta_rule(
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
            **s,
        )
        return out, s

    out_s, got = run(gate, beta)
    out_c, want = run(gate.contiguous(), beta.contiguous())
    assert torch.equal(out_s, out_c)
    for name in state:
        assert torch.equal(got[name], want[name]), name
    assert torch.equal(got["ckpt"], state["ckpt"]) != flush
