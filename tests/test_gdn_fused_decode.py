"""AITER fused GDN decode (gfx1250) against ATOM's own GDN decode chain."""

import types

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("ROCm GPU required", allow_module_level=True)
if torch.cuda.get_device_properties().gcnArchName.split(":")[0] != "gfx1250":
    pytest.skip("AITER fused GDN decode is gfx1250-only", allow_module_level=True)
pytest.importorskip("aiter.ops.triton.gated_delta_net.fused_gdn_decode_qkvz")

from atom.model_ops.attention_gdn import fused_gdn_gating
from atom.model_ops.fla_ops import (
    fused_recurrent_gated_delta_rule,
    gdn_fused_decode,
)
from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_update

KH, VH, K, V = 4, 8, 128, 128  # Qwen3.5-35B-A3B at TP4
QKV, ZS = 2 * KH * K + VH * V, VH * V
SLOTS = 64
EPS = 1e-6


def inputs(tokens, a_log_dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    bf16 = {"dtype": torch.bfloat16, "device": "cuda"}
    # One projection output [q|k|v|z|b|a] with padded rows, as Qwen3.5 produces it.
    qkvzba = torch.randn(tokens, QKV + ZS + 2 * VH + 8, **bf16)
    mixed_qkv, z, b, a = qkvzba[:, : QKV + ZS + 2 * VH].split([QKV, ZS, VH, VH], -1)
    layer = types.SimpleNamespace(
        conv1d=types.SimpleNamespace(weight=torch.randn(QKV, 1, 4, **bf16) * 0.3),
        A_log=torch.randn(VH, device="cuda").to(a_log_dtype),
        dt_bias=torch.randn(VH, **bf16),
        head_k_dim=K,
        gated_norm=types.SimpleNamespace(weight=torch.randn(V, **bf16), eps=EPS),
    )
    # Pools as the cache builder binds them: the conv window stored [N, 3, C] and
    # passed transposed; the recurrent state stored V-major and exposed as KV.
    conv = torch.randn(SLOTS, 3, QKV, **bf16)
    state = torch.randn(SLOTS, VH, V, K, **bf16) * 0.1
    slots = torch.randperm(SLOTS, device="cuda")[:tokens].to(torch.int32)
    return layer, mixed_qkv, z.view(tokens, VH, V), b, a, conv, state, slots


def gated_rmsnorm_reference(core, z, norm):
    c, zf = core.float(), z.float()
    rms = torch.rsqrt(c.square().mean(-1, keepdim=True) + norm.eps)
    return c * rms * norm.weight.float() * zf * torch.sigmoid(zf)


def unfused_decode(layer, mixed_qkv, z, b, a, conv, state, slots):
    """causal_conv1d_update -> fused_gdn_gating -> fused_recurrent -> gated norm."""
    tokens = mixed_qkv.shape[0]
    q, k, v = causal_conv1d_update(
        mixed_qkv,
        conv.transpose(-1, -2),
        layer.conv1d.weight.view(QKV, 4),
        KH * K,
        VH * V,
        None,
        "silu",
        conv_state_indices=slots,
        conv_state_indices_in=slots,
        validate_data=True,
    )
    g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
    core, _ = fused_recurrent_gated_delta_rule(
        q=q.view(1, tokens, -1, K),
        k=k.view(1, tokens, -1, K),
        v=v.view(1, tokens, -1, V),
        g=g,
        beta=beta,
        initial_state=state.transpose(-1, -2),
        inplace_final_state=True,
        cu_seqlens=torch.arange(tokens + 1, dtype=torch.int32, device="cuda"),
        ssm_state_indices=slots,
        ssm_state_indices_in=slots,
        use_qk_l2norm_in_kernel=True,
    )
    return gated_rmsnorm_reference(core.view(tokens, VH, V), z, layer.gated_norm)


def relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@pytest.mark.parametrize("tokens", [1, 7, 64])
@pytest.mark.parametrize("a_log_dtype", [torch.float32, torch.bfloat16])
def test_decode_matches_unfused_chain(tokens, a_log_dtype):
    layer, mixed_qkv, z, b, a, conv, state, slots = inputs(tokens, a_log_dtype)
    conv_ref, state_ref = conv.clone(), state.clone()
    expected = unfused_decode(layer, mixed_qkv, z, b, a, conv_ref, state_ref, slots)
    out = torch.empty(tokens, VH, V, dtype=torch.bfloat16, device="cuda")
    assert gdn_fused_decode.decode(
        layer,
        mixed_qkv,
        b,
        a,
        z,
        conv.transpose(-1, -2),
        state.transpose(-1, -2),
        slots,
        out,
    )
    assert relative_error(out, expected) < 2e-2
    assert relative_error(state, state_ref) < 2e-2
    assert torch.equal(conv, conv_ref)


def test_decode_declines_unsupported_layouts():
    """Outside the kernel's contract nothing is launched and nothing changes."""
    layer, mixed_qkv, z, b, a, conv, state, slots = inputs(8)
    conv_before, state_before = conv.clone(), state.clone()
    out = torch.zeros(8, VH, V, dtype=torch.bfloat16, device="cuda")
    # z is not the column slice right after q|k|v.
    assert not gdn_fused_decode.decode(
        layer,
        mixed_qkv,
        b,
        a,
        z.clone(),
        conv.transpose(-1, -2),
        state.transpose(-1, -2),
        slots,
        out,
    )
    # A K-major pool has the same shape when K == V; only its strides differ.
    assert not gdn_fused_decode.decode(
        layer, mixed_qkv, b, a, z, conv.transpose(-1, -2), state, slots, out
    )
    assert torch.equal(conv, conv_before) and torch.equal(state, state_before)
    assert not out.any()


def test_gated_rmsnorm_in_place_with_strided_gate():
    """The norm the attention op applies to every batch the kernel does not run."""
    tokens = 300
    layer, _, z, *_ = inputs(tokens)
    core = torch.randn(tokens, VH, V, dtype=torch.bfloat16, device="cuda")
    expected = gated_rmsnorm_reference(core, z, layer.gated_norm)
    gdn_fused_decode.gated_rmsnorm_(core, z, layer.gated_norm)
    assert relative_error(core, expected) < 1e-2


def test_available_only_in_native_atom(monkeypatch):
    """The vLLM, SGLang and RTP-LLM plugins bring their own GDN attention, which
    does not take the output gate, so they keep their own path."""
    from atom.plugin import prepare

    monkeypatch.setenv("ATOM_GDN_FUSED_DECODE", "1")
    for framework, expected in [
        ("atom", True),
        ("vllm", False),
        ("sglang", False),
        ("rtpllm", False),
    ]:
        monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", framework)
        gdn_fused_decode.available.cache_clear()
        assert gdn_fused_decode.available() is expected, framework
    gdn_fused_decode.available.cache_clear()
