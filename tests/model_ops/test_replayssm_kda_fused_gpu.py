# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3's KDA decode / verify rows through one kernel, against the three it
replaces -- `causal_conv1d_update`, f_b_proj and ReplaySSM's recurrence -- on
the same in_proj output.

The conv state, the cursors and the g records come out bitwise equal. The k
and u records and a flushed checkpoint pass through fp32 sums (k's norm, the
replayed state) that the fused kernel orders its own way, so they agree to
that rounding. The output is held to an fp32 reference as tightly as the
unfused path's is.
"""

import pytest
import torch
import torch.nn.functional as Fn

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops.fla_ops.replayssm import (
    replayssm_kda_fused_update,
    replayssm_sigmoid_gating_delta_rule,
)
from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_update

DEV = "cuda"
HV, D = 12, 128  # heads and head dim per rank, Kimi-K3 at TP8
LP = HV * D
FUSED_N = 4 * LP + HV + D  # [q | k | v | gate | beta | f_a]
BETA_COL, FA_COL = 4 * LP, 4 * LP + HV
T_MAX, CAP, LB = 8, 24, -5.0
SLOTS, CONV_SLOTS, STATE_LEN = 8, 10, 2 + T_MAX
SLOT_IDX = [5, 2, 7, 0, 3, 6, 1, 4]
CONV_IDX = [3, 6, 1, 4, 8, 0, 7, 2]
# A fork's first sequence reads these and writes its own slots.
FORK_SRC, CONV_FORK_SRC = 1, 9

# rows per sequence, accepted rows of the previous step (None: plain decode),
# write_pos per sequence (-1 parked; past CAP - 2 * T_MAX flushes), fork.
SCENES = {
    "T32_all_accepted": ([8, 8, 8, 8], [8, 8, 8, 8], [0, 5, 9, 16], False),
    "T20_fork_some_none": ([8, 3, 8, 1], [3, 1, 8, 2], [-1, 4, 12, 8], True),
    "T8_decode": ([1] * 8, None, [0, 1, 7, 8, 9, 16, -1, 3], False),
}


def make_inputs(lens, write_pos, seed=0):
    torch.manual_seed(seed)
    bf16 = {"device": DEV, "dtype": torch.bfloat16}
    T = sum(lens)
    fused = torch.randn(T, FUSED_N, **bf16)
    # f_a and f_b_proj's weight on a grid of k / 8 and k / 64: every partial
    # sum is exact in fp32, so the bf16 gate cannot depend on the GEMM's order.
    fused[:, FA_COL : FA_COL + D] = torch.randint(-8, 9, (T, D), **bf16) / 8
    pos = torch.randint(0, CAP - 2 * T_MAX + 1, (SLOTS,), device=DEV)
    pos[SLOT_IDX[: len(lens)]] = torch.tensor(write_pos, device=DEV)
    return {
        "fused": fused,
        "fb_w": torch.randint(-8, 9, (LP, D), **bf16) / 64,
        "conv_w": (torch.randn(3 * LP, 4, device=DEV) * 0.5).to(torch.bfloat16),
        "A_log": torch.randn(HV, device=DEV),
        "dt_bias": torch.randn(LP, device=DEV),
        "state": {
            # Physical [slot, state_len, dim]; the ops see it transposed.
            "conv": torch.randn(CONV_SLOTS, STATE_LEN, 3 * LP, **bf16),
            "ckpt": torch.randn(SLOTS, HV, D, D, device=DEV) * 0.1,
            "buf_k": Fn.normalize(torch.randn(SLOTS, HV, CAP, D, device=DEV), dim=-1)
            .to(torch.bfloat16)
            .contiguous(),
            "buf_u": torch.randn(SLOTS, HV, CAP, D, **bf16) * 0.1,
            "buf_g": -torch.rand(SLOTS, HV, CAP, D, device=DEV) * 0.1,
            "write_pos": pos.to(torch.int32),
        },
    }


def step_args(lens, accepted, fork):
    n = len(lens)
    conv_in = CONV_IDX[:n]
    fork_slots = None
    if fork:
        conv_in = [CONV_FORK_SRC] + conv_in[1:]
        fork_slots = torch.tensor([[SLOT_IDX[0]], [FORK_SRC]], device=DEV)

    def i32(x):
        return torch.tensor(x, device=DEV, dtype=torch.int32)

    return {
        "cu": i32([0] + lens).cumsum(0).to(torch.int32),
        "slot_idx": i32(SLOT_IDX[:n]),
        "conv_idx": i32(CONV_IDX[:n]),
        "conv_in": i32(conv_in),
        "accepted": None if accepted is None else i32(accepted),
        "fork_slots": fork_slots,
    }


def flushing(write_pos):
    """The sequences whose step flushes: its records restart at row 0."""
    return [i for i, h in enumerate(write_pos) if h + 2 * T_MAX > CAP]


def plant_step_records(inp, lens, write_pos, a):
    """Give a flushing slot's rows [0, T) the k / g records its step writes
    there, so the unfused kernel replays the same records whichever order its
    programs run in: `_replayssm_kda_fwd_kernel`'s `i_v == 0` program rewrites
    rows its sibling programs replay (wuhuikx/LLM-Booster#523)."""
    _, after = run_unfused(inp, a)
    for i in flushing(write_pos):
        slot, T = SLOT_IDX[i], lens[i]
        for name in ("buf_k", "buf_g"):
            inp["state"][name][slot, :, :T] = after[name][slot, :, :T]


def views(fused):
    return (
        fused[:, : 3 * LP],
        fused[:, FA_COL : FA_COL + D],
        fused[:, BETA_COL : BETA_COL + HV],
    )


def run_unfused(inp, a):
    """Kimi-K3's decode / verify branch as it stands: three ops and a copy."""
    s = {name: t.clone() for name, t in inp["state"].items()}
    conv = s.pop("conv")
    mixed, f_a, beta = views(inp["fused"])
    T = mixed.shape[0]
    if a["accepted"] is None:
        varlen = {}
    else:
        varlen = {
            "num_accepted_tokens": a["accepted"],
            "query_start_loc": a["cu"],
            "max_query_len": T_MAX,
        }
    q, k, v = causal_conv1d_update(
        mixed,
        conv.transpose(1, 2),
        inp["conv_w"],
        LP,
        LP,
        None,
        "silu",
        conv_state_indices=a["conv_idx"],
        conv_state_indices_in=a["conv_in"],
        **varlen,
    )
    gate = Fn.linear(f_a.contiguous(), inp["fb_w"])
    out = replayssm_sigmoid_gating_delta_rule(
        *(t.reshape(1, T, HV, D) for t in (q, k, v, gate)),
        beta.unsqueeze(0),
        inp["A_log"],
        inp["dt_bias"],
        slot_idx=a["slot_idx"],
        cu_seqlens=a["cu"],
        max_query_len=T_MAX,
        use_qk_l2norm_in_kernel=True,
        lower_bound=LB,
        fork_slots=a["fork_slots"],
        **s,
    )
    return out.view(T, HV, D), {"conv": conv, **s}


def run_fused(inp, a):
    s = {name: t.clone() for name, t in inp["state"].items()}
    conv = s.pop("conv")
    out = replayssm_kda_fused_update(
        *views(inp["fused"]),
        conv.transpose(1, 2),
        inp["conv_w"],
        inp["fb_w"],
        inp["A_log"],
        inp["dt_bias"],
        slot_idx=a["slot_idx"],
        cu_seqlens=a["cu"],
        max_query_len=T_MAX,
        conv_state_indices=a["conv_idx"],
        conv_state_indices_in=a["conv_in"],
        num_accepted_tokens=a["accepted"],
        lower_bound=LB,
        fork_slots=a["fork_slots"],
        **s,
    )
    return out, {"conv": conv, **s}


def reference(inp, lens, accepted, a):
    """The chain in fp32, nothing rounded between its stages, records replayed
    one at a time. Also returns each slot's state once its records are folded
    in: a flush's checkpoint."""
    s = inp["state"]
    ckpt = s["ckpt"].clone()
    if a["fork_slots"] is not None:
        ckpt[a["fork_slots"][0]] = ckpt[a["fork_slots"][1]]
    rows = inp["fused"].float()
    A = inp["A_log"].exp()[:, None]
    dt_bias = inp["dt_bias"].view(HV, D)
    out, replayed, bos = [], {}, 0
    for i, T in enumerate(lens):
        x = rows[bos : bos + T]
        bos += T
        n_acc = 1 if accepted is None else accepted[i]
        window = s["conv"][a["conv_in"][i], n_acc - 1 : n_acc + 2].float()
        seq = torch.cat([window, x[:, : 3 * LP]])
        conv = sum(inp["conv_w"][:, j] * seq[j : j + T] for j in range(4))
        q, k, v = Fn.silu(conv).view(T, 3, HV, D).unbind(1)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * D**-0.5
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        gate = (x[:, FA_COL : FA_COL + D] @ inp["fb_w"].float().T).view(T, HV, D)
        g = LB * torch.sigmoid(A * (gate + dt_bias))
        beta = torch.sigmoid(x[:, BETA_COL : BETA_COL + HV])[..., None]

        slot = SLOT_IDX[i]
        S = ckpt[slot]  # [HV, V, K]
        for j in range(max(int(s["write_pos"][slot]), 0)):
            u_j = s["buf_u"][slot, :, j, :, None].float()
            k_j = s["buf_k"][slot, :, j, None, :].float()
            S = S * s["buf_g"][slot, :, j, None, :].exp() + u_j * k_j
        replayed[slot] = S
        for t in range(T):
            S = S * g[t, :, None, :].exp()
            u = (v[t] - (S @ k[t, ..., None])[..., 0]) * beta[t]
            S = S + u[..., None] * k[t, :, None, :]
            out.append((S @ q[t, ..., None])[..., 0])
    return torch.stack(out), replayed


@pytest.mark.parametrize("scene", list(SCENES))
def test_fused_matches_conv_fb_proj_and_recurrence(scene):
    lens, accepted, write_pos, fork = SCENES[scene]
    inp = make_inputs(lens, write_pos)
    a = step_args(lens, accepted, fork)
    plant_step_records(inp, lens, write_pos, a)

    out_u, want = run_unfused(inp, a)
    out_f, got = run_fused(inp, a)

    for name in ("write_pos", "conv", "buf_g"):
        assert torch.equal(got[name], want[name]), name
    # A k or u record may round to the neighbouring bf16 value; a flushed
    # checkpoint sums the replayed records in another order.
    for name in ("buf_k", "buf_u"):
        torch.testing.assert_close(got[name], want[name], rtol=2**-7, atol=1e-6)
    torch.testing.assert_close(got["ckpt"], want["ckpt"], rtol=1e-5, atol=4e-6)
    # A flush rewrote checkpoints and no other slot's moved.
    flushed = [SLOT_IDX[i] for i in flushing(write_pos)]
    assert flushed
    moved = (got["ckpt"] != inp["state"]["ckpt"]).flatten(1).any(1)
    if fork:
        moved[SLOT_IDX[0]] = False
    assert moved.nonzero().flatten().tolist() == sorted(flushed)

    ref, _ = reference(inp, lens, accepted, a)
    err_u = (out_u.float() - ref).abs().max().item()
    err_f = (out_f.float() - ref).abs().max().item()
    assert err_u < 0.02 * ref.abs().max().item()
    assert err_f <= err_u * 1.05, (err_f, err_u)


def test_flush_folds_records_before_the_step_rewrites_them():
    """The step's k / g records land on rows the flush folds into the
    checkpoint; the checkpoint must hold the records that were there."""
    lens, accepted, write_pos, fork = SCENES["T32_all_accepted"]
    inp = make_inputs(lens, write_pos)
    a = step_args(lens, accepted, fork)

    _, got = run_fused(inp, a)

    _, replayed = reference(inp, lens, accepted, a)
    for i in flushing(write_pos):
        slot = SLOT_IDX[i]
        torch.testing.assert_close(
            got["ckpt"][slot], replayed[slot], rtol=1e-5, atol=4e-6
        )
