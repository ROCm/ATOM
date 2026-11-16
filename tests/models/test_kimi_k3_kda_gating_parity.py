# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter's gfx1250 KDA gating entry against ATOM's own kernel, on the shapes
Kimi-K3 decode and spec verify send at TP8 (12 heads, K = V = 128).

Runs on gfx1250 only. The dispatch rules are pinned on CPU in
test_kimi_k3_kda_gating_dispatch.py.
"""

import pytest
import torch

H, K = 12, 128
LOWER_BOUND = -5.0


def _gfx1250() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from aiter.jit.utils.chip_info import get_gfx
    except ImportError:
        return False
    return get_gfx() == "gfx1250"


pytestmark = pytest.mark.skipif(not _gfx1250(), reason="gfx1250 only")


def _kernels():
    atom = pytest.importorskip("atom.model_ops.fla_ops.fused_sigmoid_gating")
    aiter = pytest.importorskip("aiter.ops.triton.attention.kda_atom")
    return (
        atom.fused_sigmoid_gating_delta_rule_update,
        aiter.fused_sigmoid_gating_delta_rule_update,
    )


def _inputs(seqs, per_seq, slots):
    t = seqs * per_seq
    gen = {"dtype": torch.bfloat16, "device": "cuda"}
    return {
        "A_log": torch.randn(H, dtype=torch.float32, device="cuda") * 0.1,
        "dt_bias": torch.randn(H, K, dtype=torch.float32, device="cuda") * 0.1,
        "q": torch.randn(1, t, H, K, **gen),
        "k": torch.randn(1, t, H, K, **gen),
        "v": torch.randn(1, t, H, K, **gen),
        "a": torch.randn(1, t, H, K, **gen),
        "b": torch.randn(1, t, H, **gen),
        "state": torch.randn(slots, H, K, K, dtype=torch.float32, device="cuda"),
        "cu": torch.arange(0, t + 1, per_seq, dtype=torch.int32, device="cuda"),
        "out": torch.full((t, H, K), 7.0, dtype=torch.bfloat16, device="cuda"),
    }


def _run(kernel, inp, state, write, read, accepted):
    state = state.clone()
    out = inp["out"].clone()
    result = kernel(
        A_log=inp["A_log"],
        a=inp["a"],
        b=inp["b"],
        dt_bias=inp["dt_bias"],
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        o=out,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=inp["cu"],
        ssm_state_indices=write,
        ssm_state_indices_in=read,
        num_accepted_tokens=accepted,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        lower_bound=LOWER_BOUND,
    )
    # aiter's entry returns None for inputs that its kernel does not cover.
    assert result is not None, "Kimi-K3's decode shapes must reach aiter's kernel"
    return result[0], state


def _compare(inp, write, read, accepted):
    atom, aiter = _kernels()
    state = inp["state"]
    o_ref, s_ref = _run(atom, inp, state, write, read, accepted)
    o_new, s_new = _run(aiter, inp, state, write, read, accepted)
    torch.testing.assert_close(o_new, o_ref, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(s_new, s_ref, atol=1e-3, rtol=1e-3)


def test_decode_same_slots_matches_atom():
    seqs = 72
    inp = _inputs(seqs, per_seq=1, slots=seqs + 2)
    slots = torch.arange(seqs, dtype=torch.int32, device="cuda")
    _compare(inp, slots, slots, None)


def test_decode_fork_matches_atom():
    seqs = 72
    inp = _inputs(seqs, per_seq=1, slots=2 * seqs)
    write = torch.arange(seqs, dtype=torch.int32, device="cuda")
    read = write + seqs
    _compare(inp, write, read, None)


def test_spec_verify_matches_atom():
    seqs, width = 8, 3
    inp = _inputs(seqs, per_seq=width, slots=3 * seqs * width)
    accepted = torch.tensor([1, 3, 2, 1, 2, 3, 1, 2], dtype=torch.int32, device="cuda")
    write = torch.arange(seqs * width, dtype=torch.int32, device="cuda").view(
        seqs, width
    )
    read = write + seqs * width // 2
    _compare(inp, write, read, accepted)
