# SPDX-License-Identifier: MIT
"""KDA decode recurrence with the o_norm fused in (fused_sigmoid_gating onorm=)
against the recurrence's BF16 output fed to the gfx1250 E8M0 o_norm."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from aiter import QuantType, dtypes

from atom.model_ops.fla_ops.fused_sigmoid_gating import (
    ONormE8M0,
    fused_sigmoid_gating_delta_rule_update,
)
from atom.model_ops.kimi_k3 import rmsnorm_gated, rmsnorm_gated_e8m0_col_major

if rmsnorm_gated_e8m0_col_major(QuantType.per_1x128, dtypes.fp8, True) is None:
    pytest.skip("gfx1250 Gluon o_norm required", allow_module_level=True)

HEADS, DIM, FUSED_N = 12, 128, 6284  # TP8 KDA heads; fused in_proj row


def _inputs(requests, tokens_each, seed, heavy=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = requests * tokens_each

    def rn(*shape, dtype=torch.bfloat16, scale=1.0):
        return (scale * torch.randn(*shape, device="cuda", generator=g)).to(dtype)

    state = rn(t, HEADS, DIM, DIM, dtype=torch.float32, scale=0.05)
    if heavy:
        state *= torch.exp(1.5 * torch.randn(state.shape, device="cuda", generator=g))
    slots = torch.randperm(t, device="cuda", generator=g).to(torch.int32)
    return {
        "q": rn(1, t, HEADS, DIM),
        "k": rn(1, t, HEADS, DIM),
        "v": rn(1, t, HEADS, DIM),
        "a": rn(1, t, HEADS, DIM),
        "b": rn(1, t, HEADS),
        "A_log": rn(HEADS, dtype=torch.float32, scale=0.5),
        "dt_bias": rn(HEADS * DIM, dtype=torch.float32, scale=0.5),
        "state": state,
        "slots": slots if tokens_each == 1 else slots.view(requests, tokens_each),
        "accepted": (
            None
            if tokens_each == 1
            else torch.ones(requests, dtype=torch.int32, device="cuda")
        ),
        "cu": torch.arange(0, t + 1, tokens_each, dtype=torch.int32, device="cuda"),
        # the gate is a column slice of the fused in_proj output, as in the model
        "gate": rn(t, FUSED_N)[:, 4608:6144].view(t, HEADS, DIM),
        "weight": (1 + 0.1 * torch.randn(DIM, device="cuda", generator=g)).float(),
    }


def _first(s, requests):
    tokens_each = 1 if s["accepted"] is None else s["slots"].shape[1]
    t = requests * tokens_each
    r = dict(s, state=s["state"].clone())
    for name in ("q", "k", "v", "a", "b"):
        r[name] = s[name][:, :t]
    r["slots"] = s["slots"][: (t if tokens_each == 1 else requests)]
    r["accepted"] = None if s["accepted"] is None else s["accepted"][:requests]
    r["cu"], r["gate"] = s["cu"][: requests + 1], s["gate"][:t]
    return r


def _recurrence(s, o=None, onorm=None):
    return fused_sigmoid_gating_delta_rule_update(
        A_log=s["A_log"],
        a=s["a"],
        b=s["b"],
        dt_bias=s["dt_bias"],
        q=s["q"],
        k=s["k"],
        v=s["v"],
        o=o,
        initial_state=s["state"],
        inplace_final_state=True,
        cu_seqlens=s["cu"],
        ssm_state_indices=s["slots"],
        num_accepted_tokens=s["accepted"],
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        lower_bound=-5.0,
        onorm=onorm,
    )[0]


def _fused(s, col_major):
    return _recurrence(
        s, onorm=ONormE8M0(s["weight"], s["gate"], 1e-6, dtypes.fp8, col_major)
    )


def _unfused(s, col_major, monkeypatch):
    t = s["q"].shape[1]
    o = torch.empty(t, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    _recurrence(s, o=o)
    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", str(int(col_major)))
    return rmsnorm_gated(
        o,
        s["weight"],
        s["gate"],
        1e-6,
        quant_type=QuantType.per_1x128,
        quant_dtype=dtypes.fp8,
        e8m0_scale=True,
    )


def _assert_close_bytes(actual, expected):
    """E8M0 codes equal; FP8 values equal but for rare one-ulp steps (the
    variance is summed in another order)."""
    (q, s), (q_ref, s_ref) = actual, expected
    assert s.dtype == dtypes.fp8_e8m0 and s.shape == s_ref.shape
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))
    assert q.dtype == q_ref.dtype and q.shape == q_ref.shape
    a, b = q.view(torch.uint8).to(torch.int32), q_ref.view(torch.uint8).to(torch.int32)
    off = a != b
    assert not bool(((a ^ b)[off] & 0x80).any()), "sign flip"
    assert int(((a & 0x7F) - (b & 0x7F)).abs().max()) <= 1
    assert int(off.sum()) <= max(1, q.numel() // 10**5)


@pytest.mark.parametrize("col_major", [True, False])
@pytest.mark.parametrize(
    "requests, tokens_each", [(1, 1), (58, 1), (128, 1), (32, 5), (110, 5)]
)
def test_fused_o_norm_matches_the_recurrence_then_o_norm(
    requests, tokens_each, col_major, monkeypatch
):
    s = _inputs(requests, tokens_each, seed=requests * 10 + tokens_each)
    s_ref = dict(s, state=s["state"].clone())
    expected = _unfused(s_ref, col_major, monkeypatch)
    actual = _fused(s, col_major)
    # the recurrence itself keeps its bytes
    assert torch.equal(s["state"].view(torch.int32), s_ref["state"].view(torch.int32))
    _assert_close_bytes(actual, expected)


def test_fused_o_norm_on_heavy_tailed_rows(monkeypatch):
    s = _inputs(2000, 1, seed=7, heavy=True)
    expected = _unfused(dict(s, state=s["state"].clone()), True, monkeypatch)
    _assert_close_bytes(_fused(s, True), expected)


@pytest.mark.parametrize("col_major", [True, False])
@pytest.mark.parametrize("tokens_each", [1, 5])
def test_fused_o_norm_is_batch_invariant(tokens_each, col_major):
    requests = 96
    s = _inputs(requests, tokens_each, seed=3)
    q_full, s_full = _fused(_first(s, requests), col_major)
    t_full = requests * tokens_each
    codes_full = s_full.view(torch.uint8).reshape(-1)
    codes_full = (
        codes_full.view(HEADS, t_full).t()
        if col_major
        else codes_full.view(t_full, HEADS)
    )
    for n in (1, 3, 47, 95):
        t = n * tokens_each
        q, sc = _fused(_first(s, n), col_major)
        codes = sc.view(torch.uint8).reshape(-1)
        codes = codes.view(HEADS, t).t() if col_major else codes.view(t, HEADS)
        assert torch.equal(q.view(torch.uint8), q_full[:t].view(torch.uint8))
        assert torch.equal(codes, codes_full[:t])


def test_the_model_fuses_only_where_it_pays_and_o_norm_takes_the_gluon_kernel():
    from atom.models.kimi_k3.model import KimiKDAAttention

    norm = SimpleNamespace(
        weight=torch.ones(DIM, device="cuda"),
        variance_epsilon=1e-6,
        quant_type=QuantType.per_1x128,
        quant_dtype=dtypes.fp8,
        e8m0_scale=True,
    )
    layer = SimpleNamespace(o_norm=norm, head_dim=DIM, num_local_heads=HEADS)
    out_gate = torch.zeros(4, FUSED_N, dtype=torch.bfloat16, device="cuda")[
        :, 4608:6144
    ]
    fp32_state = torch.zeros(1, dtype=torch.float32, device="cuda")

    def onorm(state=fp32_state, num_seqs=4):
        return KimiKDAAttention._decode_onorm(layer, out_gate, state, num_seqs)

    fused = onorm()
    assert fused is not None and fused.gate.shape == (4, HEADS, DIM)
    assert fused.gate.stride() == (FUSED_N, DIM, 1)
    assert onorm(state=fp32_state.to(torch.bfloat16)) is None
    # one workgroup per (request, head); past a few per CU the unfused pair wins
    assert onorm(num_seqs=64) is not None  # 768 workgroups, 256 CUs on MI455X
    assert onorm(num_seqs=4096) is None
    norm.e8m0_scale = False
    assert onorm() is None
