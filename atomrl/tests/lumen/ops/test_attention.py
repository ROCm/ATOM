# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################
"""
Tests for atomrl.lumen.ops.attention: multi-head attention (BSHD layout).

Covers:
  - Forward vs PyTorch reference (BF16, compute_snr)
  - Forward + backward: dQ/dK/dV gradients vs reference
  - Causal attention forward + backward
  - GQA (num_head_q > num_head_kv) forward + backward
  - Cross-attention (seqlen_q != seqlen_kv)
  - Backend selection ("auto" dispatch)
  - aiter_csrc (CK) backend: forward, forward+backward, causal
  - FP8 quantized attention: blockwise/blockwise2d/dynamic/delayed/per_token
  - MXFP8 attention (gfx950+ only)
  - return_lse output verification
  - Edge cases: batch_size=1, head_dim_v != head_dim_qk

Reference: attention_ref from conftest (pure PyTorch BSHD).

SNR threshold rationale (empirical, measured on MI300X / MI355X):
  - BF16 forward:   >= 20 dB  (Triton/CK vs FP32 reference)
  - BF16 backward:  >= 15 dB  (gradients have higher variance)
  - BF16 causal bwd:>= 10 dB  (causal mask amplifies numerical differences)
  - FP8 forward:    >= 15 dB  (blockwise quantization adds ~5 dB noise)
  - FP8 backward:   >=  8 dB  (quantized backward is less precise)
  - MXFP8 forward:  >= 10 dB  (microscaling has coarser granularity)
  - MXFP8 backward: >=  5 dB  (lowest precision path)
"""

import atomrl.lumen.ops.attention as attn_ops
import pytest
import torch
from conftest import AttnConfig, attention_ref, compute_snr

# ---------------------------------------------------------------------------
# Hardware / backend detection
# ---------------------------------------------------------------------------


def _is_gfx950():
    try:
        from atomrl.lumen.kernels.attention.attention_impl import is_cdna4

        return is_cdna4()
    except (AssertionError, RuntimeError):
        return False


_gfx950 = _is_gfx950()

# ---------------------------------------------------------------------------
# Configurations (BSHD layout)
# ---------------------------------------------------------------------------

ATTN_CONFIGS = [
    AttnConfig(128, 128, 8, 8, 64, 64),  # standard MHA
    AttnConfig(256, 256, 16, 4, 64, 64),  # GQA
    AttnConfig(512, 512, 8, 8, 128, 128),  # larger head dim
    AttnConfig(256, 128, 8, 8, 64, 64),  # cross-attention: seqlen_q > seqlen_kv
    AttnConfig(128, 256, 8, 8, 64, 64),  # cross-attention: seqlen_q < seqlen_kv
]

ATTN_IDS = [repr(c) for c in ATTN_CONFIGS]

CSRC_CONFIGS = [
    AttnConfig(128, 128, 8, 8, 64, 64),  # standard MHA
    AttnConfig(256, 256, 16, 4, 64, 64),  # GQA
    AttnConfig(128, 256, 8, 8, 64, 64),  # cross-attention: seqlen_q < seqlen_kv
]
CSRC_IDS = [repr(c) for c in CSRC_CONFIGS]

FP8_CONFIGS = [
    AttnConfig(128, 128, 8, 8, 64, 64),  # standard MHA
    AttnConfig(256, 256, 16, 4, 64, 64),  # GQA
]
FP8_IDS = [repr(c) for c in FP8_CONFIGS]

FP8_QUANT_TYPES = ["blockwise", "blockwise2d", "dynamic", "delayed", "per_token"]

GQA_CONFIGS = [
    AttnConfig(256, 256, 16, 4, 64, 64),
    AttnConfig(128, 128, 8, 2, 64, 64),
]
GQA_IDS = [repr(c) for c in GQA_CONFIGS]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tensors(
    config: AttnConfig, batch_size: int, dtype, device, requires_grad=False
):
    """Create q, k, v in BSHD layout with deterministic CUDA seed."""
    torch.cuda.manual_seed(42)
    q = (
        torch.randn(
            batch_size,
            config.seqlen_q,
            config.num_head_q,
            config.head_dim_qk,
            device=device,
            dtype=dtype,
        )
        * 0.02
    ).requires_grad_(requires_grad)
    k = (
        torch.randn(
            batch_size,
            config.seqlen_kv,
            config.num_head_kv,
            config.head_dim_qk,
            device=device,
            dtype=dtype,
        )
        * 0.02
    ).requires_grad_(requires_grad)
    v = (
        torch.randn(
            batch_size,
            config.seqlen_kv,
            config.num_head_kv,
            config.head_dim_v,
            device=device,
            dtype=dtype,
        )
        * 0.02
    ).requires_grad_(requires_grad)
    return q, k, v


def _assert_fwd_snr(out_ref, out, min_snr, label):
    """Assert forward output SNR with a standardized message."""
    snr = compute_snr(out_ref, out)
    assert snr > min_snr, f"{label} forward SNR: {snr:.1f} dB (need > {min_snr})"


def _assert_grad_snr(q_ref, k_ref, v_ref, q, k, v, min_snr, label):
    """Assert dQ/dK/dV gradient SNR with standardized messages."""
    dq_snr = compute_snr(q_ref.grad, q.grad)
    dk_snr = compute_snr(k_ref.grad, k.grad)
    dv_snr = compute_snr(v_ref.grad, v.grad)
    assert dq_snr > min_snr, f"{label} dQ SNR: {dq_snr:.1f} dB (need > {min_snr})"
    assert dk_snr > min_snr, f"{label} dK SNR: {dk_snr:.1f} dB (need > {min_snr})"
    assert dv_snr > min_snr, f"{label} dV SNR: {dv_snr:.1f} dB (need > {min_snr})"


def _run_fwd_test(config, backend_type, causal, min_snr, label, batch_size=2):
    """Run a forward-only SNR test against attention_ref."""
    dtype = torch.bfloat16
    device = "cuda"
    sm_scale = config.head_dim_qk ** (-0.5)

    q, k, v = _make_tensors(config, batch_size, dtype, device)
    out_ref = attention_ref(q, k, v, sm_scale, causal=causal)
    out = attn_ops.attention(
        q, k, v, softmax_scale=sm_scale, causal=causal, backend_type=backend_type
    )
    _assert_fwd_snr(out_ref, out, min_snr, label)


def _run_fwd_bwd_test(config, backend_type, causal, min_snr, label, batch_size=2):
    """Run a forward+backward SNR test against attention_ref."""
    dtype = torch.bfloat16
    device = "cuda"
    sm_scale = config.head_dim_qk ** (-0.5)

    q, k, v = _make_tensors(config, batch_size, dtype, device, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out_ref = attention_ref(q_ref, k_ref, v_ref, sm_scale, causal=causal)
    out_ref.float().mean().backward()

    out = attn_ops.attention(
        q, k, v, softmax_scale=sm_scale, causal=causal, backend_type=backend_type
    )
    out.float().mean().backward()

    _assert_grad_snr(q_ref, k_ref, v_ref, q, k, v, min_snr, label)


def _run_fp8_fwd_test(config, quant_type, causal, min_snr, label, batch_size=2):
    """Run an FP8 quantized attention forward SNR test."""
    dtype = torch.bfloat16
    device = "cuda"
    sm_scale = config.head_dim_qk ** (-0.5)

    q, k, v = _make_tensors(config, batch_size, dtype, device)
    out_ref = attention_ref(q, k, v, sm_scale, causal=causal)
    out = attn_ops.attention_fp8_quant(
        q, k, v, softmax_scale=sm_scale, causal=causal, quant_type=quant_type
    )
    _assert_fwd_snr(out_ref, out, min_snr, label)


def _run_fp8_fwd_bwd_test(config, quant_type, causal, min_snr, label, batch_size=2):
    """Run an FP8 quantized attention forward+backward SNR test."""
    dtype = torch.bfloat16
    device = "cuda"
    sm_scale = config.head_dim_qk ** (-0.5)

    q, k, v = _make_tensors(config, batch_size, dtype, device, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out_ref = attention_ref(q_ref, k_ref, v_ref, sm_scale, causal=causal)
    out_ref.float().mean().backward()

    out = attn_ops.attention_fp8_quant(
        q, k, v, softmax_scale=sm_scale, causal=causal, quant_type=quant_type
    )
    out.float().mean().backward()

    _assert_grad_snr(q_ref, k_ref, v_ref, q, k, v, min_snr, label)


# ===================================================================
# aiter_triton: Forward vs reference
# ===================================================================


@pytest.mark.parametrize("config", ATTN_CONFIGS, ids=ATTN_IDS)
def test_attention_fwd(config):
    """BF16 forward vs PyTorch reference (attention_ref), using compute_snr."""
    _run_fwd_test(config, "aiter_triton", causal=False, min_snr=20, label="Triton")


# ===================================================================
# aiter_triton: Forward + backward
# ===================================================================


@pytest.mark.parametrize("config", ATTN_CONFIGS, ids=ATTN_IDS)
def test_attention_fwd_bwd(config):
    """BF16 forward+backward, compare dQ/dK/dV gradients vs reference."""
    _run_fwd_bwd_test(config, "aiter_triton", causal=False, min_snr=15, label="Triton")


# ===================================================================
# aiter_triton: Causal attention forward
# ===================================================================


@pytest.mark.parametrize("config", ATTN_CONFIGS, ids=ATTN_IDS)
def test_attention_causal(config):
    """Causal attention forward."""
    if config.seqlen_q > config.seqlen_kv:
        pytest.skip("Causal mask is ill-defined when seqlen_q > seqlen_kv")
    _run_fwd_test(
        config, "aiter_triton", causal=True, min_snr=20, label="Triton causal"
    )


# ===================================================================
# aiter_triton: Causal forward + backward
# ===================================================================


@pytest.mark.parametrize("config", ATTN_CONFIGS, ids=ATTN_IDS)
def test_attention_causal_fwd_bwd(config):
    """Causal BF16 forward+backward, compare dQ/dK/dV gradients vs reference."""
    if config.seqlen_q > config.seqlen_kv:
        pytest.skip("Causal mask is ill-defined when seqlen_q > seqlen_kv")
    # Causal backward has higher variance due to the triangular mask
    _run_fwd_bwd_test(
        config, "aiter_triton", causal=True, min_snr=10, label="Triton causal"
    )


# ===================================================================
# GQA (num_head_q > num_head_kv) forward
# ===================================================================


@pytest.mark.parametrize(
    "config",
    [
        AttnConfig(256, 256, 16, 4, 64, 64),  # GQA: 16 query heads, 4 KV heads
        AttnConfig(128, 128, 8, 2, 64, 64),  # GQA: 8 query heads, 2 KV heads
    ],
    ids=["sq256_sk256_hq16_hkv4_dqk64_dv64", "sq128_sk128_hq8_hkv2_dqk64_dv64"],
)
def test_attention_gqa(config):
    """GQA (num_head_q > num_head_kv) forward."""
    assert (
        config.num_head_q > config.num_head_kv
    ), "Config must be GQA (num_head_q > num_head_kv)"
    _run_fwd_test(config, "aiter_triton", causal=False, min_snr=20, label="GQA")


# ===================================================================
# GQA forward + backward
# ===================================================================


@pytest.mark.parametrize("config", GQA_CONFIGS, ids=GQA_IDS)
def test_attention_gqa_fwd_bwd(config):
    """GQA forward+backward, compare dQ/dK/dV gradients vs reference."""
    assert config.num_head_q > config.num_head_kv
    _run_fwd_bwd_test(config, "aiter_triton", causal=False, min_snr=15, label="GQA")


# ===================================================================
# Backend selection: "auto" dispatch
# ===================================================================


@pytest.mark.parametrize(
    "config",
    [AttnConfig(128, 128, 8, 8, 64, 64)],
    ids=["sq128_sk128_hq8_hkv8_dqk64_dv64"],
)
def test_attention_auto_backend(config):
    """backend_type='auto' agrees with the backend it resolves to."""
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 2
    sm_scale = config.head_dim_qk ** (-0.5)

    q, k, v = _make_tensors(config, batch_size, dtype, device)

    # The supported test image includes the AITER CK backend.
    explicit_backend = "aiter_csrc"
    out_explicit = attn_ops.attention(
        q, k, v, softmax_scale=sm_scale, causal=False, backend_type=explicit_backend
    )
    out_auto = attn_ops.attention(
        q, k, v, softmax_scale=sm_scale, causal=False, backend_type="auto"
    )

    _assert_fwd_snr(out_explicit, out_auto, 20, f"auto vs {explicit_backend}")


# ###################################################################
# aiter_csrc (CK) backend
# ###################################################################


@pytest.mark.parametrize("config", CSRC_CONFIGS, ids=CSRC_IDS)
def test_attention_csrc_fwd(config):
    """aiter_csrc (CK) BF16 forward vs PyTorch reference."""
    _run_fwd_test(config, "aiter_csrc", causal=False, min_snr=20, label="CK")


@pytest.mark.parametrize("config", CSRC_CONFIGS, ids=CSRC_IDS)
def test_attention_csrc_fwd_bwd(config):
    """aiter_csrc (CK) BF16 forward+backward, dQ/dK/dV vs reference."""
    _run_fwd_bwd_test(config, "aiter_csrc", causal=False, min_snr=15, label="CK")


@pytest.mark.parametrize("config", CSRC_CONFIGS, ids=CSRC_IDS)
def test_attention_csrc_causal(config):
    """aiter_csrc (CK) causal forward vs reference."""
    if config.seqlen_q > config.seqlen_kv:
        pytest.skip("Causal mask is ill-defined when seqlen_q > seqlen_kv")
    _run_fwd_test(config, "aiter_csrc", causal=True, min_snr=20, label="CK causal")


@pytest.mark.parametrize(
    "config",
    [AttnConfig(512, 512, 8, 2, 128, 128)],
    ids=["sq512_sk512_hq8_hkv2_d128"],
)
def test_attention_csrc_seq_major_out_matches_default(config):
    """``seq_major_out`` must change only where the values land, not what they are.

    The output buffer is handed to CK as a transposed view, so the kernel writes
    a strided destination and the backward then reads strided ``out`` / ``dout``.
    The forward is exact under that change. The gradients cannot be held to bit
    equality -- they accumulate through fp32 atomics, so the default path does
    not reproduce itself either -- but atomic reordering only costs a last bit,
    which leaves the SNR far above what any real layout error could survive.
    """
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 2
    sm_scale = config.head_dim_qk ** (-0.5)

    def run(seq_major):
        q, k, v = _make_tensors(config, batch_size, dtype, device)
        for t in (q, k, v):
            t.requires_grad_(True)
        out = attn_ops.attention(
            q,
            k,
            v,
            softmax_scale=sm_scale,
            causal=True,
            backend_type="aiter_csrc",
            seq_major_out=seq_major,
        )
        # randn_like would inherit the output's strides and so hand the two runs
        # different values; allocate the upstream gradient explicitly.
        torch.manual_seed(0)
        out.backward(torch.randn(out.shape, dtype=out.dtype, device=out.device))
        return out, q.grad, k.grad, v.grad

    ref_out, ref_dq, ref_dk, ref_dv = run(False)
    got_out, got_dq, got_dk, got_dv = run(True)

    assert not ref_out.permute(1, 0, 2, 3).is_contiguous()
    assert got_out.permute(1, 0, 2, 3).is_contiguous(), "seq-major layout not honoured"

    torch.testing.assert_close(got_out, ref_out, atol=0, rtol=0)
    for name, ref, got in (
        ("dQ", ref_dq, got_dq),
        ("dK", ref_dk, got_dk),
        ("dV", ref_dv, got_dv),
    ):
        _assert_fwd_snr(ref, got, 50, f"seq-major {name}")


# ###################################################################
# FP8 quantized attention (attention_fp8_quant)
# ###################################################################


@pytest.mark.parametrize("quant_type", FP8_QUANT_TYPES)
@pytest.mark.parametrize("config", FP8_CONFIGS, ids=FP8_IDS)
def test_attention_fp8_fwd(config, quant_type):
    """FP8 quantized attention forward vs BF16 reference."""
    _run_fp8_fwd_test(
        config, quant_type, causal=False, min_snr=15, label=f"FP8 {quant_type}"
    )


@pytest.mark.parametrize("quant_type", FP8_QUANT_TYPES)
@pytest.mark.parametrize("config", FP8_CONFIGS, ids=FP8_IDS)
def test_attention_fp8_fwd_bwd(config, quant_type):
    """FP8 quantized attention forward+backward, dQ/dK/dV vs reference."""
    _run_fp8_fwd_bwd_test(
        config, quant_type, causal=False, min_snr=8, label=f"FP8 {quant_type}"
    )


def test_attention_fp8_none():
    """quant_type='none' delegates to BF16 attention, output matches reference.

    Internally, attention_fp8_quant(quant_type="none") calls attention() with
    backend_type="auto" (aiter_csrc if AITER is available, else aiter_triton).
    """
    _run_fp8_fwd_test(
        AttnConfig(128, 128, 8, 8, 64, 64),
        "none",
        causal=False,
        min_snr=20,
        label="FP8 none (BF16 delegation)",
    )


@pytest.mark.parametrize("quant_type", FP8_QUANT_TYPES)
def test_attention_fp8_causal(quant_type):
    """FP8 quantized causal attention forward vs reference."""
    _run_fp8_fwd_test(
        AttnConfig(128, 128, 8, 8, 64, 64),
        quant_type,
        causal=True,
        min_snr=15,
        label=f"FP8 {quant_type} causal",
    )


# ###################################################################
# MXFP8 attention (gfx950+ only)
# ###################################################################


@pytest.mark.skipif(not _gfx950, reason="MXFP8 requires gfx950+")
@pytest.mark.parametrize("config", FP8_CONFIGS, ids=FP8_IDS)
def test_attention_mxfp8_fwd(config):
    """MXFP8 attention forward vs BF16 reference."""
    _run_fp8_fwd_test(config, "mxfp8", causal=False, min_snr=10, label="MXFP8")


@pytest.mark.skipif(not _gfx950, reason="MXFP8 requires gfx950+")
@pytest.mark.parametrize("config", FP8_CONFIGS, ids=FP8_IDS)
def test_attention_mxfp8_fwd_bwd(config):
    """MXFP8 attention forward+backward, dQ/dK/dV vs reference."""
    _run_fp8_fwd_bwd_test(config, "mxfp8", causal=False, min_snr=5, label="MXFP8")


# ###################################################################
# return_lse=True
# ###################################################################


def _compute_lse_ref(q, k, sm_scale, causal=False):
    """Compute reference log-sum-exp: log(sum(exp(Q K^T * scale))) per (batch, head, query)."""
    B, SQ, HQ, D = q.shape
    _, SK, HK, _ = k.shape
    gqa_ratio = HQ // HK

    q_f32 = q.float()
    k_f32 = k.float()

    if gqa_ratio > 1:
        k_f32 = k_f32.unsqueeze(3).expand(B, SK, HK, gqa_ratio, D).reshape(B, SK, HQ, D)

    # (B, HQ, SQ, SK)
    attn = (
        torch.matmul(q_f32.transpose(1, 2), k_f32.transpose(1, 2).transpose(-2, -1))
        * sm_scale
    )

    if causal:
        row_idx = torch.arange(SQ, device=q.device).unsqueeze(1)
        col_idx = torch.arange(SK, device=q.device).unsqueeze(0)
        col_offset = SQ - SK
        mask = row_idx >= (col_offset + col_idx)
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    return torch.logsumexp(attn, dim=-1)  # (B, HQ, SQ)


def _validate_lse(result, config, batch_size, q, k, sm_scale, label):
    """Validate return_lse=True result: shape, dtype, and values vs reference."""
    assert isinstance(result, tuple), "return_lse=True should return a tuple"
    assert len(result) == 2, f"Expected 2 elements, got {len(result)}"
    out, lse = result
    assert out.shape == (
        batch_size,
        config.seqlen_q,
        config.num_head_q,
        config.head_dim_v,
    )
    assert lse.dtype == torch.float32

    lse_ref = _compute_lse_ref(q, k, sm_scale, causal=False)  # (B, H, SQ)
    lse_snr = compute_snr(lse_ref, lse)
    assert lse_snr > 15, f"{label} LSE SNR: {lse_snr:.1f} dB (need > 15)"


def test_attention_return_lse():
    """return_lse=True returns (output, softmax_lse) with correct shape, dtype, and values."""
    config = AttnConfig(128, 128, 8, 8, 64, 64)
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 2
    sm_scale = config.head_dim_qk ** (-0.5)

    q, k, v = _make_tensors(config, batch_size, dtype, device)
    result = attn_ops.attention(
        q,
        k,
        v,
        softmax_scale=sm_scale,
        causal=False,
        backend_type="aiter_triton",
        return_lse=True,
    )
    _validate_lse(result, config, batch_size, q, k, sm_scale, "Triton")


def test_attention_csrc_return_lse():
    """aiter_csrc with return_lse=True returns (output, softmax_lse) with correct values."""
    config = AttnConfig(128, 128, 8, 8, 64, 64)
    dtype = torch.bfloat16
    device = "cuda"
    batch_size = 2
    sm_scale = config.head_dim_qk ** (-0.5)

    q, k, v = _make_tensors(config, batch_size, dtype, device)
    result = attn_ops.attention(
        q,
        k,
        v,
        softmax_scale=sm_scale,
        causal=False,
        backend_type="aiter_csrc",
        return_lse=True,
    )
    _validate_lse(result, config, batch_size, q, k, sm_scale, "CK")


# ###################################################################
# Edge cases
# ###################################################################


def test_attention_batch_size_1():
    """batch_size=1 forward works correctly."""
    _run_fwd_test(
        AttnConfig(128, 128, 8, 8, 64, 64),
        "aiter_triton",
        causal=False,
        min_snr=20,
        label="B=1",
        batch_size=1,
    )


def test_attention_head_dim_v_ne_qk():
    """head_dim_v != head_dim_qk forward works correctly."""
    config = AttnConfig(128, 128, 8, 8, 64, 128)  # d_qk=64, d_v=128
    _run_fwd_test(config, "aiter_triton", causal=False, min_snr=20, label="dv!=dqk")


def test_attention_head_dim_v_ne_qk_fwd_bwd():
    """head_dim_v != head_dim_qk forward+backward works correctly."""
    config = AttnConfig(128, 128, 8, 8, 64, 128)  # d_qk=64, d_v=128
    _run_fwd_bwd_test(config, "aiter_triton", causal=False, min_snr=15, label="dv!=dqk")


def test_attention_causal_cross_attn_sq_lt_sk():
    """Causal cross-attention where seqlen_q < seqlen_kv.

    When seqlen_q < seqlen_kv, attention_ref uses a shifted causal mask:
    query position i attends to key positions 0..(i + seqlen_kv - seqlen_q).
    This is the "prefix" causal pattern used by cross-attention in decoder layers.
    """
    config = AttnConfig(128, 256, 8, 8, 64, 64)
    _run_fwd_test(
        config, "aiter_triton", causal=True, min_snr=20, label="causal cross-attn sq<sk"
    )
