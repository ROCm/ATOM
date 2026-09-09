# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gate and dispatch logic for the flydsl fp8 prefill FMHA backend.

CPU-only: the flydsl callable and the aiter fallback are both mocked, so what is
under test is the eligibility predicate, the softmax-scale fold, the descale ABI,
and the fallback state machine -- not kernel numerics.
"""

import math
import types
from unittest import mock

import pytest
import torch

from atom.model_ops import attention_mla as mla


def _layer(
    *,
    scale,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    rope_is_zero_pad=False,
    dtype=torch.bfloat16,
    enabled=True,
    gfx950=True,
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
    with mock.patch.object(mla, "_is_gfx950", lambda: gfx950):
        self.use_flydsl_fp8_prefill_attn = bool(
            enabled
            and mla._is_gfx950()
            and dtype == torch.bfloat16
            and self._fmha_d % 64 == 0
            and 64 <= v_head_dim <= 192
            and v_head_dim % 32 == 0
        )
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
    assert glm.use_flydsl_fp8_prefill_attn
    assert glm._fmha_scale_fixup == pytest.approx(1.0)

    k3 = _layer(scale=_K3_SCALE)
    assert k3._fmha_d == 192
    assert k3.use_flydsl_fp8_prefill_attn


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
def test_gate_rejects_unsupported_layers(kw):
    scale = kw.pop("scale", None)
    layer = _layer(scale=scale or _K3_SCALE, **kw)
    assert not layer.use_flydsl_fp8_prefill_attn


def test_gate_off_when_env_off():
    assert not _layer(scale=_K3_SCALE, enabled=False).use_flydsl_fp8_prefill_attn


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
    with mock.patch.object(mla, "_load_flydsl_fp8_fmha", lambda: fly):
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
    with mock.patch.object(mla, "_load_flydsl_fp8_fmha", lambda: fly):
        layer._flash_attn_prefill(q, k, v, **_call_kwargs())
    for name in ("q_descale", "k_descale", "v_descale"):
        got = fly.call_args.kwargs[name]
        assert got.shape == (1,), f"{name} is {tuple(got.shape)}, must be (1,)"


def test_kernel_exception_propagates():
    """A kernel failure must reach the caller, not degrade to aiter.

    The old code caught it, latched the layer off and logged a warning, so a
    JIT or ABI break turned an A/B into a silent baseline re-run: the score
    still looked like flydsl's. Losing the run is the cheaper failure.
    """
    layer = _layer(scale=_K3_SCALE)
    q, k, v = _qkv()
    fly = mock.Mock(side_effect=RuntimeError("JIT boom"))
    ck = mock.Mock(return_value=torch.zeros(8, 4, 128))
    with (
        mock.patch.object(mla, "_load_flydsl_fp8_fmha", lambda: fly),
        mock.patch.object(mla, "flash_attn_varlen_func", ck),
        pytest.raises(RuntimeError, match="JIT boom"),
    ):
        layer._flash_attn_prefill(q, k, v, **_call_kwargs())
    assert ck.call_count == 0, "must not silently fall back to aiter"
    # Write-once at __init__: nothing may flip the backend mid-run any more.
    assert layer.use_flydsl_fp8_prefill_attn


def test_missing_kernel_raises_rather_than_falling_back():
    """`_load_flydsl_fp8_fmha` no longer returns None for an absent kernel."""
    layer = _layer(scale=_K3_SCALE)
    q, k, v = _qkv()
    ck = mock.Mock(return_value=torch.zeros(8, 4, 128))

    def _boom():
        raise ImportError("no flydsl_flash_attn_fp8_func in this build")

    with (
        mock.patch.object(mla, "_load_flydsl_fp8_fmha", _boom),
        mock.patch.object(mla, "flash_attn_varlen_func", ck),
        pytest.raises(ImportError),
    ):
        layer._flash_attn_prefill(q, k, v, **_call_kwargs())
    assert ck.call_count == 0


def test_callable_at_the_k3_production_prefill_shape():
    """Closes the one gap the armed-line count cannot see.

    The server log now proves only that every layer passed the *static* gate.
    An armed layer whose every call missed `_flydsl_fmha_callable` would still
    run on aiter silently, so pin the real K3 prefill shape here instead:
    12 heads/rank at TP8, D=192 (128 nope + 64 rope), Dv=128, one full
    attn_prefill_chunk_size tile.
    """
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
    with mock.patch.object(mla, "_load_flydsl_fp8_fmha", lambda: fly):
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


def test_supplied_q_fp8_skips_requantization():
    layer = _layer(scale=_K3_SCALE)
    q, k, v = _qkv()
    q8 = (q.to(torch.float8_e4m3fn), torch.tensor(0.25))
    fly = mock.Mock(return_value=torch.zeros(8, 4, 128))
    quant = mock.Mock(wraps=mla.dynamic_per_batched_tensor_quant)
    with (
        mock.patch.object(mla, "_load_flydsl_fp8_fmha", lambda: fly),
        mock.patch.object(mla, "dynamic_per_batched_tensor_quant", quant),
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
