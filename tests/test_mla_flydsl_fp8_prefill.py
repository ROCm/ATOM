# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gate and dispatch logic for the flydsl fp8 prefill FMHA backend.

CPU-only: the flydsl callable, the aiter fallback and the aiter q/k/v quantizer
are all mocked, so what is under test is the eligibility predicate, the
softmax-scale fold, the descale ABI, and the AITER fallback -- not
kernel numerics.
"""

import math
import types
from unittest import mock

import pytest
import torch

from atom.model_ops import attention_mla as mla


@pytest.fixture(autouse=True)
def cpu_quantizer(monkeypatch):
    """Stand a torch quantizer in for the aiter one, for every test in the file.

    `quant_fp8_per_tensor` wraps an aiter HIP kernel. Handed a CPU tensor it
    does not raise -- it ABORTS the process, taking the whole pytest session
    down with it, so this cannot be left to an ImportError-style guard. The
    stand-in keeps the only part of the contract these tests lean on:
    ``(x_fp8, descale)`` with descale a shape-[1] tensor.
    """

    monkeypatch.setattr(mla, "_FLYDSL_FP8_MHA_AVAILABLE", True)
    monkeypatch.setattr(mla, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(
        mla,
        "flydsl_flash_attn_fp8_func",
        lambda *a, **k: pytest.fail("unexpected FlyDSL FP8 MHA call"),
        raising=False,
    )

    def quant(x):
        x8, descale = mla.dynamic_per_batched_tensor_quant(x)
        return x8, descale.reshape(1)

    def fused(q, k, v, *, q_scale_factor=1.0):
        q8, qs = quant(q)
        k8, ks = quant(k)
        v8, vs = quant(v)
        return (
            q8,
            k8,
            v8,
            qs * q_scale_factor,
            ks,
            vs,
            ks.clamp_min(1e-6) * 2,
            vs.clamp_min(1e-6) * 2,
        )

    with (
        mock.patch.object(mla, "quant_fp8_per_tensor", quant),
        mock.patch.object(mla, "fused_qkv_per_tensor_quant", fused),
    ):
        yield quant


def _layer(
    *,
    scale,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    rope_is_zero_pad=False,
    dtype=torch.bfloat16,
    enabled=True,
):
    """Build the attribute subset of MLAAttention that the gate reads.

    Constructing the real module needs a distributed group and a live config, so
    the tests drive the unbound methods against a stand-in. Every attribute here
    is one the production __init__ sets.
    """
    self = types.SimpleNamespace()
    self.scale = scale
    self.layer_num = 0
    self.qk_nope_head_dim = qk_nope_head_dim
    self.qk_rope_head_dim = qk_rope_head_dim
    self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    self.v_head_dim = v_head_dim
    self.rope_is_zero_pad = rope_is_zero_pad
    self.dtype = dtype
    # Mirror of the production __init__ block.
    self._fmha_d = qk_nope_head_dim if rope_is_zero_pad else self.qk_head_dim
    self._fmha_scale_fixup = scale * self._fmha_d**0.5
    self.use_flydsl_fp8_prefill_attn = bool(enabled)
    self._flydsl_fmha_callable = types.MethodType(
        mla.MLAAttention._flydsl_fmha_callable, self
    )
    self._quant_fmha_q = types.MethodType(mla.MLAAttention._quant_fmha_q, self)
    self._flash_attn_prefill = types.MethodType(
        mla.MLAAttention._flash_attn_prefill, self
    )
    return self


def _qkv(total_q=8, total_k=16, h=4, d=192, dv=128):
    return (
        torch.randn(total_q, h, d),
        torch.randn(total_k, h, d),
        torch.randn(total_k, h, dv),
    )


_K3_SCALE = 192**-0.5


def _call_kwargs(**over):
    kwargs = {
        "cu_seqlens_q": torch.tensor([0, 8], dtype=torch.int32),
        "cu_seqlens_k": torch.tensor([0, 16], dtype=torch.int32),
        "max_seqlen_q": 8,
        "max_seqlen_k": 16,
        "min_seqlen_q": 1,
        "dropout_p": 0.0,
        "causal": True,
    }
    kwargs.update(over)
    return kwargs


# --------------------------------------------------------------------------
# Static gate
# --------------------------------------------------------------------------


def test_gate_uses_post_drop_rope_pad_dim():
    """The head dim the kernel sees is neither `head_dim` (576) nor always
    `qk_head_dim`. A gate on the wrong one is a permanent silent no-op."""
    glm = _layer(
        scale=256**-0.5, qk_nope_head_dim=256, rope_is_zero_pad=True, v_head_dim=128
    )
    assert glm._fmha_d == 256
    assert glm._flydsl_fmha_callable(*_qkv(d=256), 0.0)
    assert glm._fmha_scale_fixup == pytest.approx(1.0)

    k3 = _layer(scale=_K3_SCALE)
    assert k3._fmha_d == 192
    assert k3._flydsl_fmha_callable(*_qkv(), 0.0)


def test_yarn_mscale_scale_is_folded_not_rejected():
    """DeepSeek-V3 carries scale = mscale**2 * D**-0.5. The kernel bakes
    rsqrt(D), so the ratio has to reappear in q_descale."""
    mscale_sq = 1.8738542070926265
    ds = _layer(scale=mscale_sq * _K3_SCALE)
    assert ds.use_flydsl_fp8_prefill_attn
    assert ds._fmha_scale_fixup == pytest.approx(mscale_sq)

    q = torch.randn(4, 2, 192)
    _, descale = ds._quant_fmha_q(q)
    _, plain = mla.dynamic_per_batched_tensor_quant(q)
    assert descale.item() == pytest.approx(plain.item() * mscale_sq, rel=1e-6)


def test_scale_fold_is_skipped_when_exactly_one():
    k3 = _layer(scale=_K3_SCALE)
    q = torch.randn(4, 2, 192)
    _, descale = k3._quant_fmha_q(q)
    _, plain = mla.dynamic_per_batched_tensor_quant(q)
    # Bit-equal, i.e. no extra multiply was issued.
    assert descale.item() == plain.item()


@pytest.mark.parametrize(
    "kw",
    [
        {"v_head_dim": 256},  # Dv > 192: builder has no LDS budget for it
        {"v_head_dim": 96 + 16},  # Dv % 32 != 0
        {"qk_nope_head_dim": 160},  # D = 224, not a multiple of 64
        {"dtype": torch.float16},  # the kernel always writes bf16
        {"gfx950": False},
    ],
)
def test_gate_rejects_unsupported_layers(kw, monkeypatch):
    kw = dict(kw)
    if not kw.pop("gfx950", True):
        monkeypatch.setattr(mla, "get_gfx", lambda: "gfx942")
    layer = _layer(scale=_K3_SCALE, **kw)
    q, k, v = _qkv(d=layer._fmha_d, dv=layer.v_head_dim)
    assert not layer._flydsl_fmha_callable(q, k, v, 0.0)
    ck = mock.Mock()
    with mock.patch.object(mla, "flash_attn_varlen_func", ck):
        layer._flash_attn_prefill(q, k, v, **_call_kwargs())
    ck.assert_called_once()


def test_gate_off_when_env_off():
    layer = _layer(scale=_K3_SCALE, enabled=False)
    assert not layer._flydsl_fmha_callable(*_qkv(), 0.0)


# --------------------------------------------------------------------------
# Per-call predicate + dispatch
# --------------------------------------------------------------------------


def test_dropout_falls_back_per_call_without_disabling():
    layer = _layer(scale=_K3_SCALE, v_head_dim=128)
    q, k, v = _qkv(d=192)
    ck = mock.Mock(return_value=torch.zeros(8, 4, 128))
    with mock.patch.object(mla, "flash_attn_varlen_func", ck):
        layer._flash_attn_prefill(q, k, v, **_call_kwargs(dropout_p=0.1))
    ck.assert_called_once()
    assert layer.use_flydsl_fp8_prefill_attn  # shape-driven, not permanent


def test_undropped_rope_pad_falls_back():
    """A caller that skipped `_drop_rope_pad` must not silently get the wrong
    baked rsqrt(D)."""
    layer = _layer(scale=256**-0.5, qk_nope_head_dim=256, rope_is_zero_pad=True)
    q, k, v = _qkv(d=320)  # still padded
    ck = mock.Mock(return_value=torch.zeros(8, 4, 128))
    with mock.patch.object(mla, "flash_attn_varlen_func", ck):
        layer._flash_attn_prefill(q, k, v, **_call_kwargs())
    ck.assert_called_once()


def test_softmax_scale_is_never_passed_to_flydsl():
    layer = _layer(scale=_K3_SCALE)
    q, k, v = _qkv()
    fly = mock.Mock(return_value=torch.zeros(8, 4, 128))
    with mock.patch.object(mla, "flydsl_flash_attn_fp8_func", fly):
        layer._flash_attn_prefill(q, k, v, **_call_kwargs())
    fly.assert_called_once()
    assert "softmax_scale" not in fly.call_args.kwargs
    assert fly.call_args.kwargs["stream"] is None
    assert fly.call_args.kwargs["cross_seqlen"] is True


def test_descales_are_one_dimensional():
    """flydsl wraps arguments as layout-dynamic memrefs, which need an axis of
    stride 1. The 0-dim scalar the quantizer returns is rejected at trace time,
    and the gate would then disable itself on the first real call."""
    layer = _layer(scale=_K3_SCALE)
    q, k, v = _qkv()
    fly = mock.Mock(return_value=torch.zeros(8, 4, 128))
    with mock.patch.object(mla, "flydsl_flash_attn_fp8_func", fly):
        layer._flash_attn_prefill(q, k, v, **_call_kwargs())
    for name in ("q_descale", "k_descale", "v_descale"):
        got = fly.call_args.kwargs[name]
        assert got.shape == (1,), f"{name} is {tuple(got.shape)}, must be (1,)"


def test_kernel_exception_propagates():
    """Kernel execution errors propagate without changing the selected backend."""
    layer = _layer(scale=_K3_SCALE)
    q, k, v = _qkv()
    fly = mock.Mock(side_effect=RuntimeError("JIT boom"))
    ck = mock.Mock(return_value=torch.zeros(8, 4, 128))
    with (
        mock.patch.object(mla, "flydsl_flash_attn_fp8_func", fly),
        mock.patch.object(mla, "flash_attn_varlen_func", ck),
        pytest.raises(RuntimeError, match="JIT boom"),
    ):
        layer._flash_attn_prefill(q, k, v, **_call_kwargs())
    assert ck.call_count == 0, "must not silently fall back to aiter"
    # Write-once at __init__: nothing may flip the backend mid-run any more.
    assert layer.use_flydsl_fp8_prefill_attn


@pytest.mark.parametrize("symbol_present", [False, True])
def test_missing_kernel_uses_aiter(monkeypatch, symbol_present):
    monkeypatch.setattr(mla, "_FLYDSL_FP8_MHA_AVAILABLE", False)
    if not symbol_present:
        monkeypatch.delattr(mla, "flydsl_flash_attn_fp8_func")
    layer = _layer(scale=_K3_SCALE)
    q, k, v = _qkv()
    ck = mock.Mock(return_value=torch.zeros(8, 4, 128))
    with mock.patch.object(mla, "flash_attn_varlen_func", ck):
        result = layer._flash_attn_prefill(q, k, v, **_call_kwargs())
    ck.assert_called_once()
    assert result is ck.return_value


def test_callable_at_the_k3_production_prefill_shape():
    """Production K3 dimensions must pass the per-call eligibility checks."""
    layer = _layer(scale=_K3_SCALE, v_head_dim=128)
    # `meta` because the predicate only reads shapes and numel -- materializing
    # these would cost ~450 MB in a suite that must run without a GPU.
    q = torch.empty(16384, 12, 192, device="meta")
    k = torch.empty(16384, 12, 192, device="meta")
    v = torch.empty(16384, 12, 128, device="meta")
    assert layer._flydsl_fmha_callable(q, k, v, 0.0)


# --------------------------------------------------------------------------
# Zero-KV entries (fixed in the kernel, not repaired here)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("return_lse", [False, True])
def test_kernel_result_is_passed_through_untouched(return_lse):
    """A seqlen_kv == 0 entry once came back all NaN where aiter writes 0.0, and
    merge_attn_states spread it across the layer output. That is fixed in the
    kernel (the unconditional floor_masked_max in _merge_tile_max), so this
    helper hands the result back verbatim -- no masking pass, no copy.

    The repair that used to live here keyed on lse == -inf, so it could only run
    when return_lse was set and left the causal sites unprotected. Both cases are
    asserted, so a reintroduced repair fails loudly instead of silently costing
    an elementwise pass over every prefill chunk.
    """
    layer = _layer(scale=_K3_SCALE)
    q, k, v = _qkv(total_q=6, h=2)
    out = torch.randn(6, 2, 128)
    lse = torch.randn(2, 6)
    lse[:, 2:4] = float("-inf")  # the rows a zero-KV entry marks

    fly = mock.Mock(return_value=(out, lse) if return_lse else out)
    with mock.patch.object(mla, "flydsl_flash_attn_fp8_func", fly):
        got = layer._flash_attn_prefill(
            q, k, v, **_call_kwargs(causal=not return_lse, return_lse=return_lse)
        )
    if return_lse:
        assert got[0] is out and got[1] is lse
    else:
        assert got is out


# --------------------------------------------------------------------------
# Q quantization reuse
# --------------------------------------------------------------------------


def test_supplied_q_fp8_skips_requantization(cpu_quantizer):
    layer = _layer(scale=_K3_SCALE)
    q, k, v = _qkv()
    q8 = (q.to(torch.float8_e4m3fn), torch.tensor(0.25))
    fly = mock.Mock(return_value=torch.zeros(8, 4, 128))
    quant = mock.Mock(wraps=cpu_quantizer)
    with (
        mock.patch.object(mla, "flydsl_flash_attn_fp8_func", fly),
        mock.patch.object(mla, "quant_fp8_per_tensor", quant),
    ):
        layer._flash_attn_prefill(q, k, v, **_call_kwargs(), q_fp8=q8)
    # K and V only -- Q came in pre-quantized.
    assert quant.call_count == 2
    assert fly.call_args.kwargs["q_descale"] is q8[1]
    assert fly.call_args.args[0] is q8[0]


def test_fixup_reproduces_the_layers_softmax_scale():
    """The kernel's logit scale is rsqrt(D) * q_descale * k_descale. Folding
    scale*sqrt(D) into q_descale must land back on the layer's own scale."""
    for scale in (_K3_SCALE, 1.8738542070926265 * _K3_SCALE, 0.5 * _K3_SCALE):
        layer = _layer(scale=scale)
        q = torch.randn(4, 2, 192)
        _, descale = layer._quant_fmha_q(q)
        _, plain = mla.dynamic_per_batched_tensor_quant(q)
        effective = math.sqrt(1.0 / layer._fmha_d) * (descale / plain).item()
        assert effective == pytest.approx(scale, rel=1e-6)


def test_supplied_kv_fp8_skips_kv_quantization(cpu_quantizer):
    layer = _layer(scale=_K3_SCALE)
    q, k, v = _qkv()
    k8, ks = cpu_quantizer(k)
    v8, vs = cpu_quantizer(v)
    fly = mock.Mock(return_value=torch.empty(8, 4, 128))
    quant = mock.Mock(wraps=cpu_quantizer)
    with (
        mock.patch.object(mla, "flydsl_flash_attn_fp8_func", fly),
        mock.patch.object(mla, "quant_fp8_per_tensor", quant),
    ):
        layer._flash_attn_prefill(q, k, v, kv_fp8=(k8, v8, ks, vs), **_call_kwargs())
    assert quant.call_count == 1  # Q only
    assert fly.call_args.args[1] is k8
    assert fly.call_args.args[2] is v8
    assert fly.call_args.kwargs["k_descale"] is ks
    assert fly.call_args.kwargs["v_descale"] is vs


def test_prequantized_kv_cannot_fall_back_to_aliased_bf16():
    layer = _layer(scale=_K3_SCALE, enabled=False)
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="prequantized K/V require"):
        layer._flash_attn_prefill(q, k, v, kv_fp8=(k, v, None, None), **_call_kwargs())


@pytest.mark.parametrize("enabled", [False, True])
def test_fused_qkv_dispatch_and_scale_fixup(enabled, cpu_quantizer, monkeypatch):
    monkeypatch.setenv("ATOM_USE_FUSED_MLA_QKV_QUANT", str(int(enabled)))
    layer = _layer(scale=1.7 * _K3_SCALE)
    q, k, v = _qkv()
    fused = mock.Mock(wraps=mla.fused_qkv_per_tensor_quant)
    quant = mock.Mock(wraps=cpu_quantizer)
    fly = mock.Mock()
    with (
        mock.patch.object(mla, "fused_qkv_per_tensor_quant", fused),
        mock.patch.object(mla, "quant_fp8_per_tensor", quant),
        mock.patch.object(mla, "flydsl_flash_attn_fp8_func", fly),
    ):
        layer._flash_attn_prefill(q, k, v, **_call_kwargs())
    assert fused.call_count == int(enabled)
    assert quant.call_count == (0 if enabled else 3)
    _, qs = cpu_quantizer(q)
    torch.testing.assert_close(fly.call_args.kwargs["q_descale"], qs * 1.7)


def test_prequantized_qkv_bypasses_all_preparation(cpu_quantizer):
    layer = _layer(scale=_K3_SCALE)
    q, k, v = _qkv()
    q8, qs = cpu_quantizer(q)
    k8, ks = cpu_quantizer(k)
    v8, vs = cpu_quantizer(v)
    fly = mock.Mock()
    with (
        mock.patch.object(
            mla, "fused_qkv_per_tensor_quant", side_effect=AssertionError
        ),
        mock.patch.object(mla, "quant_fp8_per_tensor", side_effect=AssertionError),
        mock.patch.object(mla, "flydsl_flash_attn_fp8_func", fly),
    ):
        layer._flash_attn_prefill(
            q, k, v, q_fp8=(q8, qs), kv_fp8=(k8, v8, ks, vs), **_call_kwargs()
        )
    assert fly.call_args.args == (q8, k8, v8)
