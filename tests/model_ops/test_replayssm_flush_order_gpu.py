# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A flushing ReplaySSM step folds records [0, h) into the checkpoint and then
writes its own k / g records over rows [0, T). Every V slice of a head replays
the old rows before they are overwritten, so the step repeated on one input
gives the same output and buffers every time, and they are the right ones."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops.fla_ops.replayssm import (
    replayssm_gated_delta_rule,
    replayssm_sigmoid_gating_delta_rule,
)

DEV = "cuda"
CAP, REPEATS = 16, 2000
# (rows per sequence, verify window, cursors). A cursor h flushes when
# h + 2 * window > CAP. 256 decode rows make the base kernel's race show.
SCENES = {
    "decode": ([1] * 256, 1, [CAP - 1] * 256),
    "verify_T20": ([8, 3, 8, 1], 8, [1, 8, 5, 0]),
}


def make_step(op, lens, window, cursors):
    torch.manual_seed(0)
    kda = op == "kda"
    H, HV, D = (12, 12, 128) if kda else (4, 16, 128)  # Kimi-K3, Qwen3.5 per rank
    T, n_slots = sum(lens), len(lens) + 3
    gate = (D,) if kda else ()
    bf16 = {"device": DEV, "dtype": torch.bfloat16}
    slots = torch.randperm(n_slots)[: len(lens)].tolist()
    write_pos = torch.zeros(n_slots, dtype=torch.int32, device=DEV)
    write_pos[slots] = torch.tensor(cursors, dtype=torch.int32, device=DEV)
    ckpt_dtype = torch.float32 if kda else torch.bfloat16
    state = {
        "ckpt": torch.randn(n_slots, HV, D, D, device=DEV).to(ckpt_dtype) * 0.1,
        "buf_k": torch.randn(n_slots, HV, CAP, D, **bf16),
        "buf_u": torch.randn(n_slots, HV, CAP, D, **bf16) * 0.1,
        "buf_g": -torch.rand(n_slots, HV, CAP, *gate, device=DEV) * 0.1,
    }
    args = {
        "q": torch.randn(1, T, H, D, **bf16),
        "k": torch.randn(1, T, H, D, **bf16),
        "v": torch.randn(1, T, HV, D, **bf16),
        "write_pos": write_pos,
        "slot_idx": torch.tensor(slots, dtype=torch.int32, device=DEV),
        "cu_seqlens": torch.tensor([0] + lens, device=DEV).cumsum(0).int(),
        "max_query_len": window,
        "use_qk_l2norm_in_kernel": True,
    }
    if kda:
        fn = replayssm_sigmoid_gating_delta_rule
        args["a"] = torch.randn(1, T, HV, D, **bf16)
        args["b"] = torch.randn(1, T, HV, **bf16)
        args["A_log"] = torch.randn(HV, device=DEV)
        args["dt_bias"] = torch.randn(HV * D, device=DEV)
        args["lower_bound"] = -5.0
    else:
        fn = replayssm_gated_delta_rule
        args["g"] = -torch.rand(1, T, HV, device=DEV) * 0.1
        args["beta"] = torch.rand(1, T, HV, **bf16)
    return fn, args, state, slots


def replayed(state, s, h, kda):
    """Slot s's checkpoint with records [0, h) folded in, fp32."""
    g = state["buf_g"][s, :, :h].float()
    g = g if kda else g[..., None]
    w = (g.sum(1, keepdim=True) - g.cumsum(1)).exp()
    k, u = state["buf_k"][s, :, :h].float(), state["buf_u"][s, :, :h].float()
    fold, decay = (k * w).transpose(1, 2) @ u, g.sum(1).exp()
    ckpt = state["ckpt"][s].float()
    if kda:  # [HV, V, K]
        return ckpt * decay[:, None, :] + fold.transpose(1, 2)
    return ckpt * decay[:, :, None] + fold


@pytest.mark.parametrize("scene", SCENES)
@pytest.mark.parametrize("op", ["kda", "gdn"])
def test_flush_replays_old_records_before_overwrite(op, scene):
    lens, window, cursors = SCENES[scene]
    fn, args, state, slots = make_step(op, lens, window, cursors)
    work = {name: t.clone() for name, t in state.items()}

    def step(**over):
        for name, t in work.items():
            t.copy_(state[name])
        return fn(**{**args, **over}, **work)

    first = {"out": step().clone(), **{n: t.clone() for n, t in work.items()}}
    differs = {n: torch.zeros((), dtype=torch.int32, device=DEV) for n in first}
    for _ in range(REPEATS - 1):
        got = {"out": step(), **work}
        for name, t in got.items():
            differs[name] += (t != first[name]).any()
    assert {n: int(c) for n, c in differs.items()} == dict.fromkeys(first, 0)

    # At h = 0 the step appends at row 0 without flushing: the k / g it stores.
    step(write_pos=torch.zeros_like(args["write_pos"]))
    for s, n, h in zip(slots, lens, cursors):
        flush = h + 2 * window > CAP
        base = 0 if flush else h
        for name in ("buf_k", "buf_g"):
            got = first[name][s, :, base : base + n]
            assert torch.equal(got, work[name][s, :, :n]), (name, s)
        if flush:
            want = replayed(state, s, h, op == "kda")
            torch.testing.assert_close(
                first["ckpt"][s].float(), want, rtol=2e-2, atol=2e-2
            )
