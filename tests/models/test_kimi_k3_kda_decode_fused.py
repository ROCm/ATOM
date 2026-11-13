# SPDX-License-Identifier: MIT
"""Kimi-K3 KDA under ReplaySSM: a decode or verify step that fits one wave runs
the fused kernel on views of the in_proj output, with no f_b_proj GEMM and no
conv kernel; the switch, or a step past one wave, keeps the two-kernel path."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytest.importorskip("aiter", reason="atom.models.kimi_k3.model imports aiter")

from atom.model_ops import linear
from atom.models.kimi_k3 import model as kimi

HIDDEN, HEADS, HEAD_DIM, SEQS, ROWS = 16, 2, 4, 3, 4
LP = HEADS * HEAD_DIM
FUSED_OUT, SPLIT_OUT = 2.0, 3.0


def _layer(monkeypatch, verify, cu_num=SEQS * HEADS):
    monkeypatch.setattr(
        linear, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(kimi, "get_cu_num", lambda: cu_num)
    torch.manual_seed(0)
    attn = kimi.KimiKDAAttention.__new__(kimi.KimiKDAAttention)
    nn.Module.__init__(attn)
    attn.local_proj_size, attn.num_local_heads, attn.head_dim = LP, HEADS, HEAD_DIM
    attn.layer_num, attn.fuse_f_a_quant, attn.conv_kernel_size = 0, False, 4
    attn.activation, attn.conv_weight = "silu", torch.randn(3 * LP, 4)
    attn.A_log, attn.dt_bias = torch.randn(HEADS), torch.randn(LP)
    attn._kda_gate_lower_bound = -5.0
    attn.in_proj = linear.MergedColumnParallelLinear(
        HIDDEN, [LP] * 4, bias=False, quant_config=None
    )
    attn.in_proj.weight = nn.Parameter(
        torch.randn(4 * LP + HEADS + HEAD_DIM, HIDDEN), requires_grad=False
    )
    attn.f_b_proj = linear.ColumnParallelLinear(
        HEAD_DIM, LP, bias=False, quant_config=None
    )
    attn.f_b_proj.weight = nn.Parameter(torch.randn(LP, HEAD_DIM), requires_grad=False)
    attn.o_norm = lambda out, gate, is_prefill: out
    attn.o_proj = lambda x: x

    rows = ROWS if verify else 1
    tokens = SEQS * rows
    cu = torch.arange(0, tokens + 1, rows, dtype=torch.int32)
    # Conv slots differ from the ReplaySSM slots so the test sees which is which.
    slots, conv_slots = torch.tensor([5, 1, 3]), torch.tensor([0, 2, 4])
    meta = SimpleNamespace(
        num_actual_tokens=tokens,
        num_prefills=0,
        num_decodes=0 if verify else SEQS,
        num_spec_decodes=SEQS if verify else 0,
        replayssm=True,
        slot_idx=slots,
        write_pos=torch.zeros(8, dtype=torch.int32),
        replayssm_max_query_len=rows,
        replayssm_fork_slots=None,
        non_spec_state_indices_tensor=None if verify else conv_slots,
        non_spec_state_indices_in_tensor=None,
        non_spec_query_start_loc=None if verify else cu,
        spec_state_indices_tensor=(
            torch.stack([conv_slots + i for i in range(ROWS)], 1) if verify else None
        ),
        spec_state_indices_in_tensor=None,
        spec_query_start_loc=cu if verify else None,
        num_accepted_tokens=torch.tensor([1, 4, 2]) if verify else None,
    )
    cache = SimpleNamespace(
        k_cache=torch.zeros(8, 3 * LP, 2 + rows),
        v_cache=torch.zeros(8, HEADS, HEAD_DIM, HEAD_DIM),
        replay_buf_k=torch.zeros(8, HEADS, 16, HEAD_DIM),
        replay_buf_u=torch.zeros(8, HEADS, 16, HEAD_DIM),
        replay_buf_g=torch.zeros(8, HEADS, 16, HEAD_DIM),
    )
    ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(kda_metadata=meta),
        kv_cache_data={"layer_0": cache},
    )
    monkeypatch.setattr(kimi, "get_forward_context", lambda: ctx)

    calls = {"gemm_weights": [], "gemm_outputs": [], "fused": [], "conv": []}

    def mm(self, inp, weights, bias=None, otype=None, **_):
        calls["gemm_weights"].append(weights)
        calls["gemm_outputs"].append(F.linear(inp, weights, bias))
        return calls["gemm_outputs"][-1]

    def fused(*args, o, **kwargs):
        calls["fused"].append((args, kwargs))
        return o.fill_(FUSED_OUT)

    def conv(x, *args, **kwargs):
        calls["conv"].append(kwargs)
        return x.split(LP, -1)

    def split_recurrence(q, k, v, gate, beta, *args, o, **kwargs):
        calls["split_gate"] = gate
        return o.fill_(SPLIT_OUT)

    monkeypatch.setattr(type(kimi.tgemm), "mm", mm)
    monkeypatch.setattr(kimi, "replayssm_kda_fused_update", fused)
    monkeypatch.setattr(kimi, "causal_conv1d_update", conv)
    monkeypatch.setattr(kimi, "replayssm_sigmoid_gating_delta_rule", split_recurrence)
    return attn, meta, calls


def _run(attn, meta):
    torch.manual_seed(1)
    return attn._forward_impl(torch.randn(meta.num_actual_tokens, HIDDEN))


def _same_storage(view, base):
    return view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()


@pytest.mark.parametrize("verify", [False, True], ids=["decode", "verify"])
def test_step_in_one_wave_runs_the_fused_kernel_on_in_proj_views(monkeypatch, verify):
    attn, meta, calls = _layer(monkeypatch, verify)
    out = _run(attn, meta)

    assert calls["conv"] == [] and "split_gate" not in calls
    assert [w.shape for w in calls["gemm_weights"]] == [attn.in_proj.weight.shape]
    in_proj_out = calls["gemm_outputs"][0]
    ((args, kwargs),) = calls["fused"]
    mixed_qkv, f_a, beta, _, conv_weight, f_b_weight = args[:6]
    for view, cols in (
        (mixed_qkv, slice(0, 3 * LP)),
        (beta.squeeze(0), slice(4 * LP, 4 * LP + HEADS)),
        (f_a, slice(4 * LP + HEADS, None)),
    ):
        assert _same_storage(view, in_proj_out)
        assert torch.equal(view, in_proj_out[:, cols])
    assert conv_weight is attn.conv_weight and f_b_weight is attn.f_b_proj.weight

    cu = meta.spec_query_start_loc if verify else meta.non_spec_query_start_loc
    assert torch.equal(kwargs["slot_idx"], meta.slot_idx)
    assert torch.equal(kwargs["conv_state_indices"], torch.tensor([0, 2, 4]))
    assert kwargs["num_accepted_tokens"] is meta.num_accepted_tokens
    assert torch.equal(kwargs["cu_seqlens"], cu)
    assert kwargs["max_query_len"] == meta.replayssm_max_query_len
    torch.testing.assert_close(out, torch.full_like(out, FUSED_OUT))


@pytest.mark.parametrize("verify", [False, True], ids=["decode", "verify"])
@pytest.mark.parametrize("reason", ["switch_off", "past_one_wave"])
def test_two_kernel_path_when_off_or_past_one_wave(monkeypatch, verify, reason):
    cu_num = SEQS * HEADS - 1 if reason == "past_one_wave" else SEQS * HEADS
    attn, meta, calls = _layer(monkeypatch, verify, cu_num)
    if reason == "switch_off":
        monkeypatch.setenv("ATOM_KIMI_K3_KDA_FUSED_DECODE", "0")
    out = _run(attn, meta)

    assert calls["fused"] == [] and len(calls["conv"]) == 1
    assert [w.shape for w in calls["gemm_weights"]] == [
        attn.in_proj.weight.shape,
        attn.f_b_proj.weight.shape,
    ]
    gate = calls["gemm_outputs"][1].view(1, meta.num_actual_tokens, HEADS, HEAD_DIM)
    torch.testing.assert_close(calls["split_gate"], gate)
    torch.testing.assert_close(out, torch.full_like(out, SPLIT_OUT))
