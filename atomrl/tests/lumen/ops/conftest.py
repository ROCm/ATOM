# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

import glob
import logging
import os
import time
from dataclasses import dataclass

import pytest
import torch

DEVICE = "cuda"

logger = logging.getLogger("atomrl.lumen.test")


# ---------------------------------------------------------------------------
# AITER JIT stale-lock cleanup (session-scoped)
# ---------------------------------------------------------------------------


def _get_aiter_build_dir():
    """Return the AITER JIT build directory, or None if not determinable."""
    try:
        from aiter.jit.core import get_user_jit_dir

        return os.path.join(get_user_jit_dir(), "build")
    except (AssertionError, ImportError, OSError, RuntimeError):
        return None


_LOCK_MAX_AGE_S = int(os.environ.get("LUMEN_AITER_LOCK_MAX_AGE_S", "3600"))


def _is_lock_stale(lock_path):
    """A lock file is stale if older than ``_LOCK_MAX_AGE_S`` (default 1 hour).

    The threshold is deliberately conservative: normal CK kernel compilation
    takes 2-15 minutes. A lock surviving beyond an hour almost certainly
    belongs to a process that crashed without calling ``FileBaton.release()``.
    Override with ``LUMEN_AITER_LOCK_MAX_AGE_S`` env var if needed.
    """
    try:
        mtime = os.path.getmtime(lock_path)
        return (time.time() - mtime) > _LOCK_MAX_AGE_S
    except OSError:
        return False


@pytest.fixture(scope="session", autouse=True)
def _cleanup_stale_aiter_jit_locks():
    """Remove stale AITER JIT build lock files that cause indefinite hangs.

    AITER uses two levels of ``FileBaton`` locks during JIT compilation:

    1. ``{bd_dir}/lock_{module}`` — held by ``build_module()`` in ``core.py``
    2. ``{bd_dir}/{module}/build/lock`` — held by ``_jit_compile()`` in
       ``cpp_extension.py`` (the inner ninja build lock)

    If either build process is killed (Ctrl-C, OOM, crash), the lock file
    is never released and subsequent runs block forever in
    ``FileBaton.wait()``.

    This fixture runs once per session and removes lock files older than
    ``LUMEN_AITER_LOCK_MAX_AGE_S`` seconds (default 3600 = 1 hour), well
    beyond any normal CK compilation time.  Set the env var to 0 to disable.
    """
    if _LOCK_MAX_AGE_S <= 0:
        return

    bd_dir = _get_aiter_build_dir()
    if bd_dir is None or not os.path.isdir(bd_dir):
        return

    stale_locks = []
    for pattern in [
        os.path.join(bd_dir, "lock_*"),
        os.path.join(bd_dir, "*", "build", "lock"),
    ]:
        stale_locks.extend(p for p in glob.glob(pattern) if _is_lock_stale(p))

    for lock_path in stale_locks:
        try:
            os.remove(lock_path)
            logger.warning(
                "Removed stale AITER JIT lock: %s "
                "(a previous build likely crashed — this would have caused "
                "tests to hang indefinitely)",
                lock_path,
            )
        except OSError as exc:
            logger.debug("Could not remove lock %s: %s", lock_path, exc)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def attention_ref(q, k, v, sm_scale, causal=False):
    """Pure PyTorch multi-head attention reference (BSHD layout)."""
    B, SQ, HQ, D = q.shape
    _, SK, HK, _ = k.shape
    DV = v.shape[-1]
    gqa_ratio = HQ // HK

    q_f32 = q.float()
    k_f32 = k.float()
    v_f32 = v.float()

    if gqa_ratio > 1:
        k_f32 = k_f32.unsqueeze(3).expand(B, SK, HK, gqa_ratio, D).reshape(B, SK, HQ, D)
        v_f32 = (
            v_f32.unsqueeze(3).expand(B, SK, HK, gqa_ratio, DV).reshape(B, SK, HQ, DV)
        )

    q_t = q_f32.transpose(1, 2)
    k_t = k_f32.transpose(1, 2)
    v_t = v_f32.transpose(1, 2)

    attn = torch.matmul(q_t, k_t.transpose(-2, -1)) * sm_scale

    if causal:
        row_idx = torch.arange(SQ, device=q.device).unsqueeze(1)
        col_idx = torch.arange(SK, device=q.device).unsqueeze(0)
        col_offset = SQ - SK
        mask = row_idx >= (col_offset + col_idx)
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v_t)
    out = out.transpose(1, 2)
    return out.to(q.dtype)


def rmsnorm_ref(x, weight, eps=1e-6):
    """Pure PyTorch RMSNorm reference."""
    x_f32 = x.float()
    rms = torch.sqrt(torch.mean(x_f32**2, dim=-1, keepdim=True) + eps)
    return (x_f32 / rms * weight.float()).to(x.dtype)


def layernorm_ref(x, weight, bias=None, eps=1e-5):
    """PyTorch LayerNorm reference."""
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def grouped_gemm_ref(lhs, rhs, group_sizes, bias=None):
    """Sequential per-expert GEMM reference for MoE."""
    outputs = []
    offset = 0
    for g, size in enumerate(group_sizes.tolist()):
        size = int(size)
        if size == 0:
            continue
        x_g = lhs[offset : offset + size]
        w_g = rhs[g]
        out_g = x_g @ w_g.T
        if bias is not None:
            out_g = out_g + bias[g]
        outputs.append(out_g)
        offset += size
    return (
        torch.cat(outputs, dim=0)
        if outputs
        else torch.empty(0, rhs.shape[1], device=lhs.device, dtype=lhs.dtype)
    )


def cross_entropy_ref(logits, target, label_smoothing=0.0, ignore_idx=-100):
    """PyTorch cross-entropy reference."""
    V = logits.shape[-1]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, V),
        target.reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
        ignore_index=ignore_idx,
    )


# ---------------------------------------------------------------------------
# FP8 quantization golden references (no TE dependency)
# ---------------------------------------------------------------------------


def fp8_dynamic_scale_ref(tensor, fp8_max, margin=0):
    """Golden: compute per-tensor dynamic FP8 (dequant) scale = amax / (fp8_max / 2^margin)."""
    amax = tensor.abs().amax().clamp(min=1e-12)
    effective_max = fp8_max / (2.0**margin)
    return amax / effective_max


def fp8_quant_dequant_ref(tensor, fp8_dtype=None):
    """Golden: per-tensor FP8 quant→dequant round-trip in pure PyTorch.

    Returns (reconstructed_bf16, scale) so callers can verify both.
    """
    if fp8_dtype is None:
        fp8_dtype = torch.float8_e4m3fnuz
    orig_dtype = tensor.dtype
    fp8_max = torch.finfo(fp8_dtype).max
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = fp8_max / amax
    clamped = (tensor.float() * scale).clamp(-fp8_max, fp8_max)
    quantized = clamped.to(fp8_dtype)
    dequantized = quantized.to(orig_dtype) / scale
    return dequantized, 1.0 / scale


def fp8_blockwise_quant_dequant_ref(tensor, block_size=128, fp8_dtype=None):
    """Golden: per-block FP8 quant→dequant for blockwise scaling.

    Processes each block_size chunk along the last dim independently.
    """
    if fp8_dtype is None:
        fp8_dtype = torch.float8_e4m3fnuz
    orig_dtype = tensor.dtype
    orig_shape = tensor.shape
    fp8_max = torch.finfo(fp8_dtype).max
    flat = tensor.reshape(-1, orig_shape[-1]).float()
    _M, N = flat.shape

    out = torch.zeros_like(flat)
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        block = flat[:, start:end]
        amax = block.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = fp8_max / amax
        q = (block * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
        out[:, start:end] = q.float() / scale

    return out.to(orig_dtype).view(orig_shape)


def delayed_scale_ref(amax_history, fp8_max, margin=0):
    """Golden: compute delayed scale from amax history.

    scale = fp8_max / (max(amax_history) * 2^margin)
    """
    amax = max(amax_history) if amax_history else 1.0
    if isinstance(amax, torch.Tensor):
        amax = amax.item()
    amax = max(amax, 1e-12)
    return fp8_max / (amax * (2.0**margin))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_snr(x, y):
    """Compute Signal-to-Noise Ratio in dB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    signal = torch.norm(x.float()).pow(2)
    if signal < 1e-12:
        return float("inf") if torch.allclose(x.float(), y.float(), atol=1e-7) else 0.0
    noise = torch.norm(x.float() - y.float()).pow(2)
    return 10.0 * torch.log10(signal / (noise + 1e-12)).item()


def check_close(a, b, atol=1e-2, rtol=1e-2, tol_err_ratio=0.05, msg=""):
    """Check that most elements are close, allowing a small fraction of outliers."""
    is_close = torch.isclose(a.float(), b.float(), rtol=rtol, atol=atol)
    err_ratio = 1.0 - is_close.float().mean().item()
    assert err_ratio <= tol_err_ratio, (
        f"{msg}: {err_ratio:.2%} elements exceed tolerance "
        f"(atol={atol}, rtol={rtol}, max_allowed={tol_err_ratio:.0%})"
    )


# ---------------------------------------------------------------------------
# Data-class configs
# ---------------------------------------------------------------------------


@dataclass
class AttnConfig:
    seqlen_q: int
    seqlen_kv: int
    num_head_q: int
    num_head_kv: int
    head_dim_qk: int
    head_dim_v: int

    def __repr__(self):
        return (
            f"sq{self.seqlen_q}_sk{self.seqlen_kv}_"
            f"hq{self.num_head_q}_hkv{self.num_head_kv}_"
            f"dqk{self.head_dim_qk}_dv{self.head_dim_v}"
        )


@dataclass
class NormConfig:
    M: int
    N: int

    def __repr__(self):
        return f"M{self.M}_N{self.N}"


@dataclass
class LinearConfig:
    M: int
    K: int
    N: int

    def __repr__(self):
        return f"M{self.M}_K{self.K}_N{self.N}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def seed_rng():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


@pytest.fixture
def device():
    return DEVICE
