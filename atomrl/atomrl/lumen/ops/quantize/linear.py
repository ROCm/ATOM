# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Quantized linear forward + backward with explicit autograd and
multi-backend ASM → Triton fallback.

All backends are AITER implementations — no torch.nn.functional fallbacks.

All AITER GEMM kernels follow TN layout convention:
    ``Y = X @ W^T``  where X is (M, K) and W is (N, K).

Supports all 8 scaling modes:
    - ``delayed``      — per-tensor FP8 (delayed scaling from amax history)
    - ``dynamic``      — per-tensor FP8 (current scaling from current amax)
    - ``per_token``    — per-row FP8 dynamic scaling
    - ``blockwise``    — per-block FP8 scaling (e.g. block=128)
    - ``blockwise2d``  — 2D block FP8 scaling (same kernel, 2D scale management)
    - ``mxfp8``        — microscaling FP8
    - ``mxfp4``        — microscaling FP4 (all GEMMs in FP4, SR on gradients only)
    - ``none``         — BF16 passthrough (no quantization)

GEMM backends are selected automatically:
    - **Triton**: ``gemm_a8w8`` (per-tensor), ``gemm_a8w8_blockscale``,
      ``gemm_a8w8_per_token_scale``, ``gemm_a16w16`` (BF16),
      ``gemm_mxfp8`` (MXFP8 with E8M0 scales)
    - **hipBLASLt**: ``hipb_mm`` (per-tensor)
"""

import functools
import logging as _logging
import os
import threading

import torch
from torch.autograd.function import once_differentiable

# Import the submodule by path. Importing this name from the package cycles
# with quantize/__init__.py while that package is still initializing.
from atomrl.lumen.ops.dispatch import Backend, try_backends
from atomrl.lumen.quantize.config import _get_float8_e4m3
from atomrl.lumen.quantize.descriptor import FP8Descriptor

_logger = _logging.getLogger(__name__)


def _to_2d(x: torch.Tensor) -> torch.Tensor:
    """Reshape to 2D, only calling .contiguous() when necessary."""
    x_2d = x.reshape(-1, x.shape[-1])
    return x_2d if x_2d.is_contiguous() else x_2d.contiguous()


_PREFER_HIPBLASLT = os.environ.get("LUMEN_PREFER_HIPBLASLT", "0") == "1"
_FAST_QUANT_DISPATCH = os.environ.get("LUMEN_FAST_QUANT_DISPATCH", "1") == "1"
_FP8_DGRAD_OUTPUT = os.environ.get("LUMEN_FP8_DGRAD_OUTPUT", "0") == "1"
# Route mixed-dtype (hybrid) dgrad/wgrad through torch._scaled_mm, which reaches
# hipBLASLt's F8B8 Tensile kernels (same path TE uses) instead of AITER Triton.
_MIXED_SCALED_MM = os.environ.get("LUMEN_MIXED_SCALED_MM", "0") == "1"

# ---------------------------------------------------------------------------
# MXFP4 Hadamard: deterministic sign vector (all +1 = pure Hadamard).
# arXiv:2605.09825 shows randomized signs cause Wgrad divergence;
# only deterministic Hadamard converges at 8B+ scale.
# ---------------------------------------------------------------------------
_MXFP4_RHT_SIGN: torch.Tensor | None = None
_MXFP4_RHT_G = 16


def _get_mxfp4_rht_sign(device: torch.device) -> torch.Tensor:
    """Return deterministic Hadamard sign vector (all +1)."""
    global _MXFP4_RHT_SIGN
    if _MXFP4_RHT_SIGN is None or _MXFP4_RHT_SIGN.device != device:
        _MXFP4_RHT_SIGN = torch.ones(_MXFP4_RHT_G, device=device, dtype=torch.float32)
    return _MXFP4_RHT_SIGN


# ---------------------------------------------------------------------------
# Tuned hipBLASLt GEMM solutions
# ---------------------------------------------------------------------------
_TUNED_GEMM_PATH = os.environ.get("LUMEN_TUNED_GEMM", "")
_tuned_gemm_solutions = {}  # (M, N, K) -> int
_hipblas_initialized = False


def _load_tuned_gemms():
    """Load tuned GEMM solutions from JSON file (called once at import)."""
    if not _TUNED_GEMM_PATH:
        return
    try:
        import json

        with open(_TUNED_GEMM_PATH) as f:
            data = json.load(f)
        candidates = {}
        for key, val in data.items():
            parts = tuple(int(x) for x in key.split(","))
            sol_idx = val if isinstance(val, int) else val.get("solution_index", -1)
            if sol_idx != -1:
                candidates[parts] = sol_idx
        if candidates and os.environ.get("LUMEN_TUNED_GEMM_VALIDATE", "0") == "1":
            _validate_tuned_solutions(candidates)
        _tuned_gemm_solutions.update(candidates)
        if _tuned_gemm_solutions:
            _logger.info(
                "Loaded %d tuned GEMM solutions from %s",
                len(_tuned_gemm_solutions),
                _TUNED_GEMM_PATH,
            )
    except (KeyError, OSError, TypeError, ValueError) as e:
        _logger.warning("Failed to load tuned GEMMs from %s: %s", _TUNED_GEMM_PATH, e)


def _validate_tuned_solutions(candidates):
    """Validate tuned solutions in a subprocess (SIGABRT-safe)."""
    import json as _json
    import subprocess
    import sys

    if not torch.cuda.is_available():
        return
    script = """
import os, sys, json, torch
os.environ.setdefault("HIP_VISIBLE_DEVICES", os.environ.get("LOCAL_RANK", "0"))
from aiter.ops.gradlib import hipb_create_extension, hipb_mm
hipb_create_extension()
candidates = json.loads(sys.argv[1])
valid = {}
props = torch.cuda.get_device_properties(torch.cuda.current_device())
fp8_dtype = (
    torch.float8_e4m3fn
    if (props.major, props.minor) >= (9, 5)
    else torch.float8_e4m3fnuz
)
for key, sol_idx in candidates.items():
    M, N, K = (int(x) for x in key.split(","))
    try:
        m = min(M, 64)
        a = torch.randn(m, K, device="cuda").to(fp8_dtype)
        w = torch.randn(N, K, device="cuda").to(fp8_dtype)
        s = torch.tensor([[1.0]], dtype=torch.float32, device="cuda")
        hipb_mm(a, w.t(), sol_idx, out_dtype=torch.bfloat16, scaleA=s, scaleB=s)
        torch.cuda.synchronize()
        valid[key] = sol_idx
    except Exception:
        pass
print(json.dumps(valid))
"""
    cand_json = _json.dumps(
        {f"{M},{N},{K}": sol for (M, N, K), sol in candidates.items()}
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, cand_json],
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            valid = _json.loads(result.stdout.strip())
            new_candidates = {}
            for key, sol in valid.items():
                parts = tuple(int(x) for x in key.split(","))
                new_candidates[parts] = sol
            removed = set(candidates.keys()) - set(new_candidates.keys())
            for k in removed:
                _logger.warning(
                    "Tuned GEMM solution %d invalid for shape %s, using default",
                    candidates[k],
                    k,
                )
            candidates.clear()
            candidates.update(new_candidates)
        else:
            _logger.warning(
                "GEMM validation subprocess failed (rc=%d), disabling tuned solutions",
                result.returncode,
            )
            candidates.clear()
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        _logger.warning("GEMM validation failed: %s, disabling tuned solutions", e)
        candidates.clear()


_load_tuned_gemms()


def _get_tuned_solution(M, N, K):
    """Return tuned hipBLASLt solution index for shape (M,N,K), or -1."""
    return _tuned_gemm_solutions.get((M, N, K), -1)


# ---------------------------------------------------------------------------
# Fast per-tensor quant dispatch (Opt C)
# ---------------------------------------------------------------------------
_PER_TENSOR_SCALING = frozenset({"delayed", "dynamic"})
_UNPROBED = object()
_fast_quant_cache = {}  # fp8_dtype -> callable or None


def _get_fast_quant_fn(fp8_dtype):
    """Return cached AITER per-tensor quant function for this dtype, or None."""
    fn = _fast_quant_cache.get(fp8_dtype, _UNPROBED)
    if fn is not _UNPROBED:
        return fn
    fn = None
    # Triton first. The HIP per-tensor kernel in this AITER build aborts the
    # process inside its JIT wrapper (SIGABRT), which cannot be caught, so it
    # is only a fallback for dtypes Triton does not implement.
    if not _is_e5m2(fp8_dtype):
        from aiter.ops.quant import per_tensor_quant_triton

        fn = per_tensor_quant_triton
    else:
        from aiter.ops.quant import per_tensor_quant_hip

        fn = per_tensor_quant_hip
    _fast_quant_cache[fp8_dtype] = fn
    return fn


# ---------------------------------------------------------------------------
# Thread-local FP8 gradient cache
#
# When GEMM epilogue produces FP8 output (dgrad), the FP8 tensor + scale
# are cached here keyed by data_ptr of the BF16 gradient. The next layer's
# backward can pop the entry and skip redundant BF16->FP8 quantization.
# Entries are consumed exactly once via pop(), so no cleanup is needed.
# ---------------------------------------------------------------------------
_fp8_grad_cache = threading.local()


def _fp8_cache_put(bf16_ptr: int, fp8_data: torch.Tensor, scale: torch.Tensor) -> None:
    """Store FP8 data + scale for a BF16 gradient tensor."""
    if not hasattr(_fp8_grad_cache, "store"):
        _fp8_grad_cache.store = {}
    _fp8_grad_cache.store[bf16_ptr] = (fp8_data, scale)


def _fp8_cache_pop(bf16_ptr: int, expected_shape: tuple | None = None):
    """Retrieve and remove FP8 data for a BF16 gradient. Returns (fp8, scale) or None.

    When ``expected_shape`` is provided, the cached entry is only returned
    if the FP8 tensor shape matches.  This guards against false hits from
    PyTorch's caching allocator reusing the same address for a different
    tensor.
    """
    store = getattr(_fp8_grad_cache, "store", None)
    if store is None:
        return None
    entry = store.pop(bf16_ptr, None)
    if (
        entry is not None
        and expected_shape is not None
        and entry[0].shape != expected_shape
    ):
        return None
    return entry


__all__ = ["QuantizedLinearFunction", "quantized_linear"]

# When per-step weight quantization is active (LUMEN_WEIGHT_QUANT_ONCE), the
# backward reuses the scaling manager's step-cached weight descriptor (whose
# transpose is materialized once per step) instead of rebuilding a fresh one
# and re-transposing every micro-batch. Invalidation is handled by the manager
# (optimizer post-step hook), not here.
_WEIGHT_QUANT_ONCE = os.environ.get("LUMEN_WEIGHT_QUANT_ONCE", "0") == "1"


def _mark_allow_in_graph(cls):
    try:
        from torch._dynamo import allow_in_graph

        allow_in_graph(cls)
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Quantization helpers (all via AITER)
# ---------------------------------------------------------------------------

_E5M2_DTYPES = frozenset({torch.float8_e5m2, torch.float8_e5m2fnuz})


def _is_e5m2(dtype):
    return dtype in _E5M2_DTYPES


def _quant_per_tensor_hip(x, dtype):
    from aiter.ops.quant import per_tensor_quant_hip

    return per_tensor_quant_hip(x, quant_dtype=dtype)


def _quant_per_tensor_finite(x, dtype):
    """Per-tensor FP8 quant that stays finite when ``x`` is all zeros.

    AITER's dynamic kernel stores ``scale = amax / fp8_max`` and quantizes
    with ``x * (1 / scale)``. An all-zero tensor has amax 0, so that product
    is ``0 * inf`` and the GEMM that consumes it returns NaN. Delayed-scaling
    MLP down-projections hit this once an upstream SwiGLU underflows to 0.
    ``compute_scale`` maps a non-positive amax to 1, which quantizes zeros
    to zeros.
    """
    from atomrl.lumen.kernels.compute_scale import compute_scale
    from atomrl.lumen.ops.quantize.quant_amax_fused import (
        fused_amax_abs,
        static_quant_with_amax,
    )

    x2d = _to_2d(x).contiguous()
    amax = fused_amax_abs(x2d)
    scale = compute_scale(amax, float(torch.finfo(dtype).max))
    y, _ = static_quant_with_amax(x2d, scale, dtype)
    if y.shape != x.shape:
        y = y.view(x.shape)
    return y, scale


def _quant_per_tensor_triton(x, dtype):
    if not _is_e5m2(dtype):
        return _quant_per_tensor_finite(x, dtype)
    from aiter.ops.quant import per_tensor_quant_triton

    return per_tensor_quant_triton(x, quant_dtype=dtype)


def _quant_per_token_hip(x, dtype):
    from aiter.ops.quant import per_token_quant_hip

    return per_token_quant_hip(x, quant_dtype=dtype)


def _quant_per_token_triton(x, dtype):
    from aiter.ops.quant import pertoken_quant

    return pertoken_quant(x, quant_dtype=dtype)


def _quant_blockwise(x, dtype, block_size=128):
    from atomrl.lumen.ops.quantize.ops import quant_fp8_blockwise_impl

    orig_shape = x.shape
    flat = _to_2d(x)
    x_fp8, x_scales = quant_fp8_blockwise_impl(
        flat, dtype=dtype, axis=1, block_size=block_size
    )
    return x_fp8.view(orig_shape), x_scales


def _quant_blockwise2d_activation(x, dtype, block_size=128):
    """1×block FP8 activation quant via AITER's HIP kernel.

    Delegates to ``aiter.ops.quant.get_hip_quant(QuantType.per_1x128)``
    (compiled HIP ``dynamic_per_token_scaled_quant``), producing a
    ``(M, N/block_size)`` fp32 scale matching ``gemm_a8w8_blockscale``.
    """
    from aiter.ops.enum import QuantType
    from aiter.ops.quant import get_hip_quant

    assert (
        block_size == 128
    ), f"blockwise2d activation quant only supports block_size=128, got {block_size}"
    orig_shape = x.shape
    flat = _to_2d(x)
    quant_fn = get_hip_quant(QuantType.per_1x128)
    x_fp8, x_scale = quant_fn(flat, quant_dtype=dtype)
    return x_fp8.view(orig_shape), x_scale


def _quant_blockwise2d_weight(w, dtype, block_size=128):
    """2D (block × block) FP8 weight quant via AITER's HIP per-token kernel.

    DeepSeek-V3 / Jet-RL scheme: weight ``(N, K)`` split into
    ``block_size × block_size`` tiles, one scale per tile.  Reshapes
    ``(M, N)`` into tile-major ``(sm·sn, block·block)`` so each tile is one
    contiguous row, then runs HIP ``per_token_quant_hip`` (one scale per
    row); reshapes back.  Returns ``(w_fp8 (N, K), w_scales (N/block, K/block))``.
    """
    from aiter.ops.quant import per_token_quant_hip

    assert (
        block_size == 128
    ), f"blockwise2d weight quant only supports block_size=128, got {block_size}"
    orig_shape = w.shape
    flat = _to_2d(w)
    # per_token_quant_hip kernel does not support fp32; cast to bf16 first.
    if flat.dtype == torch.float32:
        flat = flat.bfloat16()
    M, N = flat.shape
    assert (
        M % block_size == 0 and N % block_size == 0
    ), f"shape ({M}, {N}) not divisible by block_size={block_size}"

    sm, sn = M // block_size, N // block_size
    # (M,N) → (sm,blk,sn,blk) → permute → (sm,sn,blk,blk) → flatten tile rows.
    tiles = (
        flat.view(sm, block_size, sn, block_size)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(sm * sn, block_size * block_size)
    )
    y, tile_scale = per_token_quant_hip(tiles, quant_dtype=dtype)
    # Invert reshape to recover (M, N).
    w_fp8 = (
        y.view(sm, sn, block_size, block_size)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(M, N)
    )
    return w_fp8.view(orig_shape), tile_scale.view(sm, sn)


_IN_GRAPH_CAPTURE = False


def set_graph_capture_mode(active: bool):
    """Toggle graph-capture flag for FP8 linear and dispatch safety."""
    global _IN_GRAPH_CAPTURE
    _IN_GRAPH_CAPTURE = active
    from atomrl.lumen.ops.dispatch import set_graph_capture_mode as _set_dispatch

    _set_dispatch(active)


def set_warmup_mode(active: bool):
    """No-op.  Kept for backward compatibility with megatron.py."""


def _safe_fp8_desc(x_fp8, x_scale, fp8_dtype):
    """Return an FP8Descriptor.

    Zero-scale handling (which previously required a CPU-GPU sync via
    ``.item()``) is now done inside the Triton quantization kernels
    themselves via ``tl.where(scale > 0, scale, 1.0)`` — no Python-side
    fixup needed.
    """
    return FP8Descriptor(data=x_fp8, scale=x_scale, fp8_dtype=fp8_dtype)


def _mxfp8_k_alignment(x: torch.Tensor) -> int:
    if not x.is_cuda:
        return 32
    arch = torch.cuda.get_device_properties(x.device).gcnArchName
    return 128 if arch.startswith("gfx1250") else 32


def quantize_input(
    x_2d,
    scaling_type,
    fp8_dtype,
    block_size=128,
    manager=None,
    tensor_id=None,
    backward=False,
    is_weight=False,
    use_sr=False,
    swizzle_scale=False,
    shuffle_data=False,
) -> FP8Descriptor | None:
    """Quantize input tensor according to scaling_type (all via AITER).

    Returns an :class:`~atomrl.lumen.quantize.descriptor.FP8Descriptor` bundling ``data`` and
    ``scale``, or ``None`` when ``scaling_type == "none"`` (BF16 passthrough).

    Args:
        use_sr: For MXFP4 only — use stochastic rounding (True for gradients,
            False/RTN for weights and activations per NVFP4 paper §4.4).
        swizzle_scale: Deprecated. MXFP4 layout conversion is owned by AITER.
        shuffle_data: Deprecated. MXFP4 layout conversion is owned by AITER.

    Scale tensor shapes by mode:
    - per-tensor: ``(1,)``
    - per-token: ``(M, 1)``
    - blockwise: ``(M, ceil(N/block_size))``
    - blockwise2d:
        * activation / gradient (``is_weight=False``): ``(M, ceil(N/block_size))``
          — 1×block per-group quantization along the K axis
        * weight (``is_weight=True``): ``(ceil(M/block_size), ceil(N/block_size))``
          — block×block 2D tile quantization (DeepSeek-V3 / Jet-RL scheme)
    - mxfp4:
        * activation / gradient (``is_weight=False``): 1×32 block scales
        * weight (``is_weight=True``): 32×32 2D block scales (chain-rule consistent)
    - mxfp8: ``(scales_shape,)``
    """
    if scaling_type == "none":
        return None

    # Fast path: per-tensor quant without manager — skip try_backends overhead.
    # E5M2 stays on the AITER fallback: hipBLASLt's per-tensor GEMM rejects it.
    # E4M3 must not use AITER's dynamic quant: a zero tensor becomes NaN.
    if (
        _FAST_QUANT_DISPATCH
        and manager is None
        and scaling_type in _PER_TENSOR_SCALING
        and not _is_e5m2(fp8_dtype)
    ):
        x_fp8, x_scale = _quant_per_tensor_finite(x_2d, fp8_dtype)
        return _safe_fp8_desc(x_fp8, x_scale, fp8_dtype)

    if scaling_type == "delayed":
        if manager is not None:
            result = manager.quantize(tensor_id or "input", x_2d, backward=backward)
            if isinstance(result, tuple):
                return FP8Descriptor(
                    data=result[0], scale=result[1], fp8_dtype=fp8_dtype
                )
            return result
        backends = [(Backend.TRITON, lambda: _quant_per_tensor_triton(x_2d, fp8_dtype))]
        x_fp8, x_scale = try_backends(backends, op_name="quant_delayed_per_tensor")
        return _safe_fp8_desc(x_fp8, x_scale, fp8_dtype)

    if scaling_type == "dynamic":
        backends = [(Backend.TRITON, lambda: _quant_per_tensor_triton(x_2d, fp8_dtype))]
        x_fp8, x_scale = try_backends(backends, op_name="quant_per_tensor")
        return _safe_fp8_desc(x_fp8, x_scale, fp8_dtype)

    if scaling_type == "per_token":
        backends = [(Backend.TRITON, lambda: _quant_per_token_triton(x_2d, fp8_dtype))]
        x_fp8, x_scale = try_backends(backends, op_name="quant_per_token")
        return FP8Descriptor(data=x_fp8, scale=x_scale, fp8_dtype=fp8_dtype)

    if scaling_type == "blockwise":
        x_fp8, x_scale = _quant_blockwise(x_2d, fp8_dtype, block_size)
        return FP8Descriptor(data=x_fp8, scale=x_scale, fp8_dtype=fp8_dtype)

    if scaling_type == "blockwise2d":
        # Jet-RL scheme: weight uses 2D (block × block) tiles; activation /
        # gradient uses 1×block per-group along the K axis.  Both feed into
        # AITER's gemm_a8w8_blockscale, which routes the 2D weight scale
        # natively via its GROUP_N / GROUP_K parameters.  Both quantizers are
        # pure-torch (activation via AITER's get_torch_quant reference, weight
        # via in-place tile reshape) — no AITER kernel patch required.
        if is_weight:
            x_fp8, x_scale = _quant_blockwise2d_weight(x_2d, fp8_dtype, block_size)
        else:
            x_fp8, x_scale = _quant_blockwise2d_activation(x_2d, fp8_dtype, block_size)
        return FP8Descriptor(data=x_fp8, scale=x_scale, fp8_dtype=fp8_dtype)

    if scaling_type == "mxfp8":
        from atomrl.lumen.ops.quantize.ops import convert_to_mxfp8
        from atomrl.lumen.ops.quantize.padding import pad_to_block

        unpadded_shape = x_2d.shape
        mxfp8_block = 32
        k_alignment = _mxfp8_k_alignment(x_2d)
        x_2d, _orig_m = pad_to_block(x_2d, mxfp8_block, dim=0)
        x_2d, _orig_n = pad_to_block(x_2d, k_alignment, dim=-1)
        x_fp8, x_scale = convert_to_mxfp8(
            x_2d, block_size=mxfp8_block, axis=-1, float8_dtype_pt=fp8_dtype
        )
        return FP8Descriptor(
            data=x_fp8,
            scale=x_scale,
            fp8_dtype=fp8_dtype,
            unpadded_shape=unpadded_shape,
        )

    if scaling_type == "mxfp4":
        from atomrl.lumen.ops.quantize.ops import convert_to_mxfp4, convert_to_mxfp4_2d
        from atomrl.lumen.ops.quantize.padding import pad_to_block

        mxfp4_block = 32
        # Only the 2D weight tiles and the swizzled scale layout are organised
        # along rows; the row-wise activation path takes any M. Padding it too
        # would hand the GEMM more rows than the caller asked for, and nothing
        # downstream knows to slice them off again.
        if is_weight and x_2d.shape[0] % mxfp4_block != 0:
            # Padding the rows here is what the 2D tile grid needs, but nothing
            # carries the original count forward, so the caller's reshape of the
            # GEMM output meets a wider tensor and dies on the row count -- an
            # unreadable RuntimeError several frames away. N is hidden size,
            # vocab or a TP shard, so it is fixed for the run: say so here, and
            # let the patch-time check leave such layers in BF16.
            raise ValueError(
                f"MXFP4 needs a weight whose output width is a multiple of "
                f"{mxfp4_block}; got {tuple(x_2d.shape)}. The reduction dim is "
                "free -- only the row count matters. Leave this layer in BF16."
            )
        if is_weight:
            x_2d, _orig_m = pad_to_block(x_2d, mxfp4_block, dim=0)
        x_2d, _orig_n = pad_to_block(x_2d, mxfp4_block, dim=-1)
        if is_weight:
            x_fp4, x_scale = convert_to_mxfp4_2d(
                x_2d,
                block_size=mxfp4_block,
                use_sr=use_sr,
                shuffle_data=shuffle_data,
            )
        else:
            x_fp4, x_scale = convert_to_mxfp4(
                x_2d,
                block_size=mxfp4_block,
                axis=-1,
                use_sr=use_sr,
                swizzle_scale=swizzle_scale,
            )
        return FP8Descriptor(data=x_fp4, scale=x_scale, fp8_dtype=None)

    raise ValueError(f"Unknown scaling_type={scaling_type!r}")


def _fp8_store_activation(input_2d, fp8_dtype):
    """Quantize activation for memory-efficient storage in save_for_backward."""
    desc = quantize_input(input_2d, "dynamic", fp8_dtype)
    return desc.data, desc.scale


def _fp8_restore_activation(input_fp8, scale, target_dtype):
    """Dequantize stored FP8 activation. Convention: dequant = fp8 * scale."""
    if target_dtype == torch.bfloat16 and input_fp8.is_cuda and input_fp8.dim() >= 2:
        try:
            from atomrl.lumen.ops.quantize.quant_amax_fused import dequant_fp8_to_bf16

            return dequant_fp8_to_bf16(input_fp8, scale)
        except RuntimeError:
            pass
    return (input_fp8.to(torch.float32) * scale).to(target_dtype)


# ---------------------------------------------------------------------------
# GEMM dispatch with fallback (all via AITER)
#
# Convention: all AITER kernels compute Y = X @ W^T (TN layout).
#   - x: (M, K)    — activation / LHS
#   - w: (N, K)    — weight / RHS (internally transposed by the kernel)
#   - Y: (M, N)    — output
# ---------------------------------------------------------------------------


def ensure_hipblaslt_ready():
    """Pre-initialize hipBLASLt workspace (256 MiB via raw HIP).

    Must be called **after** ``torch.cuda.set_device(local_rank)`` so the
    allocation lands on the correct GPU.  Safe to call multiple times — only
    the first invocation per process allocates.

    By reserving the workspace early (before model loading), PyTorch's caching
    allocator sees less free VRAM and adjusts peak usage downward, preventing
    the OOM that occurs when hipBLASLt is lazily initialized during backward.
    """
    global _hipblas_initialized
    if _hipblas_initialized:
        return

    from aiter.ops.gradlib import hipb_create_extension

    hipb_create_extension()
    _hipblas_initialized = True

    _logger.info("hipBLASLt workspace pre-allocated (256 MiB)")


def _scale_to_f32_1x1(scale, device):
    """Convert scale to float32 (1,1) shape, reusing if already correct."""
    if isinstance(scale, torch.Tensor):
        if scale.dtype == torch.float32 and scale.shape == (1, 1):
            return scale
        if scale.dtype == torch.float32 and scale.numel() == 1:
            return scale.view(1, 1)
        return scale.float().reshape(1, 1)
    return torch.tensor([[scale]], dtype=torch.float32, device=device)


def _hipb_mm_accepts(*tensors):
    """Whether AITER's ``hipb_mm`` takes these operand dtypes.

    AITER marshals ``hipb_mm`` arguments through ``aiter_tensor_t``, whose dtype
    table holds a single FP8 type (E4M3), so an E5M2 operand raises before it
    reaches hipBLASLt.
    """
    from aiter.utility.dtypes import _torch_to_aiter_dtype

    return all(t.dtype in _torch_to_aiter_dtype for t in tensors)


def _fp8_mm(mat1, mat2, sol, scale1, scale2):
    """``mat1 @ mat2`` in BF16 on hipBLASLt, via ``hipb_mm`` or ``torch._scaled_mm``.

    ``torch._scaled_mm`` covers the dtypes ``hipb_mm`` rejects (E5M2 grads) but
    needs a row-major LHS and a column-major RHS, so a mismatched operand is
    transposed once.
    """
    if _hipb_mm_accepts(mat1, mat2):
        from aiter.ops.gradlib import hipb_mm

        return hipb_mm(
            mat1, mat2, sol, out_dtype=torch.bfloat16, scaleA=scale1, scaleB=scale2
        )

    from atomrl.lumen.ops.quantize.fast_transpose import fast_transpose_fp8

    if mat1.stride(1) != 1:
        mat1 = (
            fast_transpose_fp8(mat1.t()) if mat1.stride(0) == 1 else mat1.contiguous()
        )
    if mat2.stride(0) != 1:
        mat2 = fast_transpose_fp8(mat2.contiguous()).t()
    return torch._scaled_mm(
        mat1,
        mat2,
        scale_a=scale1.reshape(1),
        scale_b=scale2.reshape(1),
        out_dtype=torch.bfloat16,
    )


def _gemm_per_tensor_hipblas(
    a_fp8, w_fp8, scale_a, scale_w, w_transposed=None, bias=None
):
    """hipBLASLt per-tensor GEMM via AITER hipb_mm.

    hipb_mm computes mat1 @ mat2 (NN layout).  Lumen's dispatch convention
    passes w as (N, K), so we need (K, N).  hipBLASLt's C++ kernel
    (``hipbsolgemm.cu``) detects non-contiguous strides and applies
    ``HIPBLAS_OP_T`` internally, so a metadata-only ``.t()`` view suffices
    — no expensive ``.t().contiguous()`` copy needed.

    If ``w_transposed`` is already a contiguous (K, N) tensor it is used
    directly.

    When *bias* is provided, it is fused into the GEMM epilogue (avoiding a
    separate ``aten::add`` kernel launch).
    """
    from aiter.ops.gradlib import hipb_mm

    ensure_hipblaslt_ready()
    sa = _scale_to_f32_1x1(scale_a, a_fp8.device)
    sw = _scale_to_f32_1x1(scale_w, w_fp8.device)
    w_t = w_transposed if w_transposed is not None else w_fp8.t()
    M, K = a_fp8.shape
    N = w_fp8.shape[0]
    sol = _get_tuned_solution(M, N, K)
    return hipb_mm(
        a_fp8, w_t, sol, bias=bias, out_dtype=torch.bfloat16, scaleA=sa, scaleB=sw
    )


def _expand_per_tensor_scale(scale, size):
    """Broadcast a per-tensor scale (1,) to a contiguous per-row vector.

    AITER's gemm_a8w8 Triton kernels index into scales with per-row offsets.
    A (1,)-shaped tensor causes out-of-bounds reads.
    """
    if scale.dtype == torch.float32 and scale.ndim == 1 and scale.numel() == 1:
        return scale.expand(size).contiguous()
    return scale.float().reshape(1).expand(size).contiguous()


def _gemm_per_tensor_triton(
    a_fp8, w_fp8, scale_a, scale_w, w_transposed=None, bias=None
):
    # w_transposed is hipBLASLt-only (Triton computes Y = A @ W^T directly); ignored here.
    from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

    M, N = a_fp8.shape[0], w_fp8.shape[0]
    sa = _expand_per_tensor_scale(scale_a, M)
    sw = _expand_per_tensor_scale(scale_w, N)
    out = gemm_a8w8(a_fp8, w_fp8, sa, sw)
    if bias is not None:
        out = out + bias
    return out


def gemm_per_tensor(a_fp8, w_fp8, scale_a, scale_w, w_transposed=None, bias=None):
    """Per-tensor FP8 GEMM: Y = X @ W^T. Triton first, hipBLASLt as fallback.

    ``w_transposed`` is an optional pre-computed ``(K, N)`` contiguous
    transpose of ``w_fp8``, used by hipBLASLt to skip ``.t().contiguous()``.

    When *bias* is provided and hipBLASLt is the selected backend, the bias
    is fused into the GEMM epilogue, saving a separate kernel launch.
    """
    if _FAST_GEMM_DISPATCH:
        fn = _get_fast_gemm_fn("gemm_per_tensor")
        if fn is not None:
            return fn(a_fp8, w_fp8, scale_a, scale_w, w_transposed, bias)
    backends = [
        (
            Backend.TRITON,
            lambda: _gemm_per_tensor_triton(a_fp8, w_fp8, scale_a, scale_w),
        ),
        (
            Backend.HIPBLAS,
            lambda: _gemm_per_tensor_hipblas(
                a_fp8, w_fp8, scale_a, scale_w, w_transposed, bias
            ),
        ),
    ]
    return try_backends(backends, op_name="gemm_per_tensor")


def _gemm_per_tensor_hipblas_mixed(a_fp8, w_fp8, scale_a, scale_w):
    """hipBLASLt mixed-dtype FP8 GEMM for dgrad: ``grad(M,N) @ weight(N,K)``.

    For dgrad: ``a_fp8`` is the gradient ``(M, N_out)`` and ``w_fp8`` is the
    weight ``(N_out, K_in)``.  hipb_mm computes ``mat1 @ mat2`` so we pass
    them directly — no transpose needed (saving ~224 MiB contiguous copy).
    """
    ensure_hipblaslt_ready()
    sa = _scale_to_f32_1x1(scale_a, a_fp8.device)
    sw = _scale_to_f32_1x1(scale_w, w_fp8.device)
    M = a_fp8.shape[0]
    N, K = w_fp8.shape
    sol = _get_tuned_solution(M, K, N)
    return _fp8_mm(a_fp8, w_fp8, sol, sa, sw)


def gemm_per_tensor_mixed(a_fp8, w_fp8, scale_a, scale_w, w_transpose=None):
    """Mixed-dtype FP8 dgrad GEMM: ``Y = grad(M,N) @ weight(N,K)``.

    For hybrid backward: ``a_fp8`` is E5M2 gradient ``(M, N_out)`` and
    ``w_fp8`` is E4M3 weight ``(N_out, K_in)``.  Uses hipBLASLt which
    supports mixed FP8 dtypes and NN layout natively.

    ``w_transpose`` (optional) is the pre-computed ``(K_in, N_out)`` FP8
    transpose of the weight (``weight_desc.transpose_cached``).  When given,
    the hipBLASLt path reuses it as the column-major RHS with zero copies.

    The hipBLASLt workspace must be pre-allocated (via
    :func:`ensure_hipblaslt_ready`) before model loading to avoid OOM.
    """
    # Mixed FP8 dtypes (E5M2 grad x E4M3 weight): AITER's hipb_mm wrapper
    # rejects them (single-dtype layout API). Two FP8 paths, both no-BF16:
    #  1. torch._scaled_mm -> hipBLASLt F8B8 (same kernels as TE), ~20% faster
    #     GEMM, but needs a column-major RHS. Free when the weight transpose is
    #     already cached (dgrad): transpose_cached is (K_in,N_out) row-major, so
    #     .t() is exactly the (N_out,K_in) column-major RHS scaled_mm wants.
    #  2. AITER Triton gemm_a8w8 (FP32 tl.dot, any FP8 mix) — consumes any strides, no
    #     transpose copy needed. Fallback when no cached transpose is available.
    if a_fp8.dtype != w_fp8.dtype:
        if w_transpose is not None and w_transpose.dtype == w_fp8.dtype:
            sa = _scale_to_f32_1x1(scale_a, a_fp8.device).reshape(1)
            sw = _scale_to_f32_1x1(scale_w, w_fp8.device).reshape(1)
            return torch._scaled_mm(
                a_fp8,
                w_transpose.t(),
                scale_a=sa,
                scale_b=sw,
                out_dtype=torch.bfloat16,
            )
        from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8

        M = a_fp8.shape[0]
        K_in = w_fp8.shape[1]
        sa = _expand_per_tensor_scale(scale_a, M)
        sw = _expand_per_tensor_scale(scale_w, K_in)
        # gemm_a8w8 computes X @ W^T and requires W shaped (N, K) before its
        # own transpose. ``w_fp8`` is (N_out, K_in), which is already the
        # (K, N) product layout (K = N_out). Transpose it once so the kernel's
        # transpose restores grad @ weight. There is no FP8 dtype check, so
        # E5M2 x E4M3 is accepted.
        return gemm_a8w8(a_fp8, w_fp8.t(), sa, sw, dtype=torch.bfloat16)
    if _FAST_GEMM_DISPATCH:
        fn = _get_fast_gemm_fn("gemm_per_tensor_mixed")
        if fn is not None:
            return fn(a_fp8, w_fp8, scale_a, scale_w)
    backends = [
        (
            Backend.HIPBLAS,
            lambda: _gemm_per_tensor_hipblas_mixed(a_fp8, w_fp8, scale_a, scale_w),
        )
    ]
    return try_backends(backends, op_name="gemm_per_tensor_mixed")


def gemm_per_tensor_mixed_fp8out(
    a_fp8, w_fp8, scale_a, scale_w, fp8_out_dtype, scale_out
):
    """Mixed-dtype FP8 dgrad GEMM with FP8 output + AMAX_D epilogue.

    NOTE: not used. An FP8-output + AMAX_D dgrad epilogue was prototyped to match
    TE's delayed-scaling grad, but TE on ROCm also outputs BF16 dgrad
    (rocm_gemm.cu forbids FP8 GEMM output), and gfx942 hipBLASLt has no tuned
    kernel for the mixed-dtype FP8-output combo (~38x slower fallback). Kept as a
    reference stub; dgrad stays BF16.
    """
    raise RuntimeError(
        "gemm_per_tensor_mixed_fp8out: FP8-output dgrad epilogue unsupported on "
        "gfx942 hipBLASLt (no tuned mixed-dtype FP8-output kernel)"
    )


def _gemm_wgrad_hipblas(grad_fp8, input_fp8, scale_grad, scale_input):
    """FP8 wgrad via hipBLASLt: ``dW = grad^T @ input``.

    Computes ``hipb_mm(grad^T, input)`` where grad is ``(M, N_out)`` and
    input is ``(M, K_in)``, producing ``(N_out, K_in)`` = weight shape.

    hipBLASLt's C++ kernel detects non-contiguous strides from ``.t()``
    and applies ``HIPBLAS_OP_T`` internally — no ``.contiguous()`` needed.

    Supports mixed-dtype: grad can be E5M2, input can be E4M3 (matching TE).
    """
    ensure_hipblaslt_ready()
    g_t = grad_fp8.t()
    sg = (
        scale_grad.float().reshape(1, 1)
        if isinstance(scale_grad, torch.Tensor)
        else torch.tensor([[scale_grad]], dtype=torch.float32, device=grad_fp8.device)
    )
    si = (
        scale_input.float().reshape(1, 1)
        if isinstance(scale_input, torch.Tensor)
        else torch.tensor([[scale_input]], dtype=torch.float32, device=input_fp8.device)
    )
    M, N_out = grad_fp8.shape
    K_in = input_fp8.shape[1]
    sol = _get_tuned_solution(N_out, K_in, M)
    return _fp8_mm(g_t, input_fp8, sol, sg, si)


def gemm_wgrad_fp8(grad_fp8, input_fp8, scale_grad, scale_input):
    """FP8 weight gradient GEMM: ``dW = grad^T @ input``.

    Uses hipBLASLt for mixed or same-dtype FP8 GEMM, matching TE's
    ``fp8_wgrad=True`` behavior.
    """
    if _FAST_GEMM_DISPATCH:
        fn = _get_fast_gemm_fn("gemm_wgrad_fp8")
        if fn is not None:
            return fn(grad_fp8, input_fp8, scale_grad, scale_input)
    backends = [
        (
            Backend.HIPBLAS,
            lambda: _gemm_wgrad_hipblas(grad_fp8, input_fp8, scale_grad, scale_input),
        )
    ]
    return try_backends(backends, op_name="gemm_wgrad_fp8")


def gemm_wgrad_mixed(grad_fp8, input_fp8, scale_grad, scale_input, grad_t=None):
    """Mixed-dtype FP8 wgrad GEMM: ``dW = grad^T @ input``.

    For hybrid backward ``grad_fp8`` is E5M2 ``(M, N_out)`` and ``input_fp8``
    is E4M3 ``(M, K_in)``.

    Preferred path (matches TE on ROCm): materialize the grad transpose with
    AITER's cheap dedicated FP8 transpose kernel (~0.1-0.25 ms), then run a
    plain ``hipb_mm(grad_t, input)`` on hipBLASLt. Even including the transpose,
    this beats the copy-free ``grad.t()`` stride path — that path forces a
    slower kernel variant, so avoiding the transpose is a false economy for the
    large-K wgrad shapes (e.g. down/gate_up: ~6-14% faster with the transpose).
    Falls back to AITER Triton ``gemm_a8w8`` when hipBLASLt is absent.

    ``grad_t`` (optional) is a pre-computed ``(N_out, M)`` contiguous transpose
    of ``grad_fp8``, produced upstream by the fused cast+transpose grad
    quantization (``LUMEN_FUSED_QUANT_TRANSPOSE_CPP``). When supplied it lets
    wgrad skip the dedicated transpose kernel entirely — the fused quant already
    paid for it — so grad is transposed once instead of twice.
    """
    ensure_hipblaslt_ready()
    if grad_t is None:
        from atomrl.lumen.ops.quantize.fast_transpose import fast_transpose_fp8

        grad_t = fast_transpose_fp8(grad_fp8)  # (N_out, M) contiguous
    sg = _scale_to_f32_1x1(scale_grad, grad_fp8.device)
    si = _scale_to_f32_1x1(scale_input, input_fp8.device)
    N_out = grad_fp8.shape[1]
    K_in = input_fp8.shape[1]
    M = grad_fp8.shape[0]
    sol = _get_tuned_solution(N_out, K_in, M)
    return _fp8_mm(grad_t, input_fp8, sol, sg, si)


def _gemm_per_token_triton(a_fp8, w_fp8, scale_a, scale_w):
    from aiter.ops.triton.gemm.basic.gemm_a8w8_per_token_scale import (
        gemm_a8w8_per_token_scale,
    )

    return gemm_a8w8_per_token_scale(a_fp8, w_fp8, scale_a, scale_w)


def gemm_per_token(a_fp8, w_fp8, scale_a, scale_w):
    """Per-token FP8 GEMM: Y = X @ W^T via AITER Triton."""
    backends = [
        (
            Backend.TRITON,
            lambda: _gemm_per_token_triton(a_fp8, w_fp8, scale_a, scale_w),
        )
    ]
    return try_backends(backends, op_name="gemm_per_token")


def _column_major(tensor):
    """Return ``tensor`` so that ``tensor.T`` is contiguous.

    AITER's blockscale GEMM transposes the weight with ``w.T`` and then loads
    that view. On gfx950 the kernel drops elements along a non-contiguous N
    axis (columns ``..., 60, 124, 188, 252``), so two equal weights at
    different addresses disagree by O(1). A column-major input makes the
    transposed view contiguous and the kernel deterministic.
    """
    if tensor.ndim == 2 and not tensor.T.is_contiguous():
        return tensor.t().contiguous().t()
    return tensor


def _gemm_blockscale_triton(a_fp8, w_fp8, scale_a, scale_w):
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale

    return gemm_a8w8_blockscale(
        a_fp8,
        _column_major(w_fp8),
        scale_a,
        _column_major(scale_w),
    )


@functools.lru_cache(maxsize=1)
def _skip_frozen_wgrad_enabled():
    return os.environ.get("LUMEN_SKIP_FROZEN_WGRAD", "0") == "1"


def gemm_blockscale(a_fp8, w_fp8, scale_a, scale_w):
    """Blockwise FP8 GEMM: Y = X @ W^T via Triton."""
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale_preshuffle as _triton_preshuffle,
    )

    # Preshuffle fast path: frozen weight pre-cached as (N//16, K*16) layout
    w_sh = getattr(w_fp8, "_lumen_wsh", None)
    if w_sh is not None:
        try:
            N_orig = w_fp8.shape[0]
            y = torch.empty(
                a_fp8.shape[0], N_orig, device=a_fp8.device, dtype=torch.bfloat16
            )
            # x_scale is (M, K//128) — not transposed; tell the kernel explicitly.
            return _triton_preshuffle(
                a_fp8,
                w_sh,
                scale_a,
                scale_w,
                dtype=torch.bfloat16,
                y=y,
                is_x_scale_tranposed=False,
            )
        except RuntimeError as e:
            _logger.warning(
                "preshuffle GEMM failed (%s); falling back to Triton blockscale", e
            )

    backends = [
        (
            Backend.TRITON,
            lambda: _gemm_blockscale_triton(a_fp8, w_fp8, scale_a, scale_w),
        )
    ]
    return try_backends(backends, op_name="gemm_blockscale")


def _gemm_blockscale_fused_bias(a_fp8, w_fp8, scale_a, scale_w, bias):
    """Blockscale FP8 GEMM with bias fused into epilogue: Y = 1.0 * (X @ W^T) + bias."""
    from aiter.ops.triton.gemm.fused.fused_gemm_a8w8_blockscale_mul_add import (
        fused_gemm_a8w8_blockscale_mul_add,
    )

    return fused_gemm_a8w8_blockscale_mul_add(
        a_fp8,
        _column_major(w_fp8),
        scale_a,
        _column_major(scale_w),
        a=1.0,
        b=bias,
        dtype=torch.bfloat16,
        fuse_type=0,
    )


def gemm_blockscale_with_bias(a_fp8, w_fp8, scale_a, scale_w, bias):
    """Blockwise FP8 GEMM + bias: tries Triton fused epilogue, falls back to Triton + separate add."""
    backends = [
        (
            Backend.TRITON,
            lambda: _gemm_blockscale_fused_bias(a_fp8, w_fp8, scale_a, scale_w, bias),
        ),
        (
            Backend.TRITON,
            lambda: _gemm_blockscale_triton(a_fp8, w_fp8, scale_a, scale_w) + bias,
        ),
    ]
    return try_backends(backends, op_name="gemm_blockscale_fused_bias")


def _gemm_mxfp8_aiter(a_fp8, w_fp8, scale_a, scale_w):
    """Call AITER's native MXFP8 GEMM with its required preshuffled layouts."""
    arch = torch.cuda.get_device_properties(a_fp8.device).gcnArchName
    if not arch.startswith(("gfx950", "gfx1250")):
        raise NotImplementedError(
            "AITER native MXFP8 GEMM supports gfx950/gfx1250, "
            f"not {arch.split(':', 1)[0]}"
        )
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_mxfp8_bpreshuffle
    from aiter.ops.shuffle import (
        shuffle_mxfp8_bpreshuffle_scale,
        shuffle_weight,
    )

    w_preshuffled = shuffle_weight(w_fp8, layout=(16, 16))
    w_scale_preshuffled = shuffle_mxfp8_bpreshuffle_scale(scale_w)
    return gemm_a8w8_mxfp8_bpreshuffle(
        a_fp8,
        w_preshuffled,
        scale_a,
        w_scale_preshuffled,
        dtype=torch.bfloat16,
    )


def gemm_mxfp8(a_fp8, w_fp8, scale_a, scale_w):
    """MXFP8 GEMM through AITER's public native MXFP8 operator."""
    return _gemm_mxfp8_aiter(a_fp8, w_fp8, scale_a, scale_w)


_mxfp4_bwd_fallback_warned = set()


def _warn_mxfp4_backward_fallback(reason, shape):
    """Report once per shape that a layer's backward left FP4.

    Every other fallback in the MXFP4 backward logs; the alignment one did not,
    so a layer computing its gradients in BF16 was indistinguishable from one
    that was not. Once per shape rather than per step, since a shape that
    misses alignment misses it on every micro-batch.
    """
    key = (reason, shape)
    if key in _mxfp4_bwd_fallback_warned:
        return
    _mxfp4_bwd_fallback_warned.add(key)
    _logger.warning(
        "mxfp4 backward: %s for shape (M=%d, N=%d, K=%d); this layer's gradients are BF16",
        reason,
        *shape,
    )


def _expand_2d_scale_to_1d(scale, data_shape, block_size=32):
    """Expand 2D block scales (M//b, K//b) → 1D per-row block scales (M, K//b).

    AITER's gemm_afp4wfp4 expects 1D scales (one per block along K for each row).
    2D scales replicate each tile scale across the block_size rows it covers
    (NVFP4 paper §4.3: "2D block scales are replicated for each of the 1×16 blocks").
    """
    if scale.dim() == 1 or (scale.dim() == 2 and scale.shape[0] == data_shape[0]):
        return scale
    sm, sn = scale.shape
    M = data_shape[0]
    if sm == M // block_size:
        return scale.unsqueeze(1).expand(sm, block_size, sn).reshape(M, sn)
    return scale


def _gemm_mxfp4_aiter(a_fp4, w_fp4, scale_a, scale_w):
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    # Expand 2D weight scales to 1D if needed
    sa = _expand_2d_scale_to_1d(scale_a, (a_fp4.shape[0], a_fp4.shape[1] * 2))
    sw = _expand_2d_scale_to_1d(scale_w, (w_fp4.shape[0], w_fp4.shape[1] * 2))
    return gemm_afp4wfp4(a_fp4, w_fp4, sa, sw, dtype=torch.bfloat16)


def _gemm_mxfp4_fallback(a_fp4, w_fp4, scale_a, scale_w):
    """Dequant both operands to BF16, do BF16 GEMM (TN layout)."""
    from atomrl.lumen.ops.quantize.ops import convert_from_mxfp4, convert_from_mxfp4_2d

    K_packed_a = a_fp4.shape[1]
    block_size = (K_packed_a * 2) // scale_a.shape[-1]

    # Use 2D dequant for 2D scales, 1D dequant for 1D scales
    if scale_a.dim() == 2 and scale_a.shape[0] < a_fp4.shape[0]:
        a_bf16 = convert_from_mxfp4_2d(
            a_fp4, scale_a, output_dtype=torch.bfloat16, block_size=block_size
        )
    else:
        a_bf16 = convert_from_mxfp4(
            a_fp4, scale_a, output_dtype=torch.bfloat16, block_size=block_size
        )

    K_packed_w = w_fp4.shape[1]
    block_size_w = (K_packed_w * 2) // scale_w.shape[-1]
    if scale_w.dim() == 2 and scale_w.shape[0] < w_fp4.shape[0]:
        w_bf16 = convert_from_mxfp4_2d(
            w_fp4, scale_w, output_dtype=torch.bfloat16, block_size=block_size_w
        )
    else:
        w_bf16 = convert_from_mxfp4(
            w_fp4, scale_w, output_dtype=torch.bfloat16, block_size=block_size_w
        )

    return gemm_bf16(a_bf16, w_bf16)


def gemm_mxfp4_dispatch(a_fp4, w_fp4, scale_a, scale_w):
    """MXFP4 GEMM through AITER's row-major public path, with BF16 fallback."""
    m, n, k = a_fp4.shape[0], w_fp4.shape[0], a_fp4.shape[1] * 2
    return try_backends(
        [
            (
                Backend.TRITON,
                lambda: _gemm_mxfp4_aiter(a_fp4, w_fp4, scale_a, scale_w),
                "aiter_plain",
            ),
            (
                Backend.TRITON,
                lambda: _gemm_mxfp4_fallback(a_fp4, w_fp4, scale_a, scale_w),
                "dequant_bf16",
            ),
        ],
        op_name=f"gemm_mxfp4:{m}x{n}x{k}:rowmajor",
        slow_labels=("dequant_bf16",),
    )


def _gemm_bf16_tuned(a, w, bias):
    from aiter.tuned_gemm import gemm_a16w16

    return gemm_a16w16(a, w, bias=bias)


# ---------------------------------------------------------------------------
# Fast GEMM dispatch — bypass try_backends list/lambda overhead after warmup
# ---------------------------------------------------------------------------
_FAST_GEMM_DISPATCH = _FAST_QUANT_DISPATCH

_fast_gemm_cache = {}  # op_name -> callable or None


def _get_fast_gemm_fn(op_name):
    """Return cached GEMM function for this op, or None."""
    fn = _fast_gemm_cache.get(op_name, _UNPROBED)
    if fn is not _UNPROBED:
        return fn
    fn = None
    if op_name == "gemm_per_tensor":
        # Forward GEMM (Y = X @ W^T). hipBLASLt's tuned F8 kernel is ~1.1-1.2x
        # faster than the AITER Triton one and reuses the weight's cached FP8
        # transpose (w_transposed) with no copy, so prefer it when requested.
        if _PREFER_HIPBLASLT:
            fn = _gemm_per_tensor_hipblas
        else:
            fn = _gemm_per_tensor_hipblas
    elif op_name == "gemm_per_tensor_mixed":
        fn = _gemm_per_tensor_hipblas_mixed
    elif op_name == "gemm_wgrad_fp8":
        fn = _gemm_wgrad_hipblas
    elif op_name == "gemm_bf16":
        fn = _gemm_bf16_tuned
    _fast_gemm_cache[op_name] = fn
    return fn


def gemm_bf16(a, w, bias=None):
    """BF16 GEMM ``Y = X @ W.T`` via AIter tuned_gemm. No autograd."""
    if a.device.type != "cuda":
        return torch.nn.functional.linear(a, w, bias)
    if _FAST_GEMM_DISPATCH:
        fn = _get_fast_gemm_fn("gemm_bf16")
        if fn is not None:
            return fn(a, w, bias)
    backends = [(Backend.ASM, lambda: _gemm_bf16_tuned(a, w, bias))]
    return try_backends(backends, op_name="gemm_bf16")


def dispatch_gemm(a, w, scale_a=None, scale_w=None, scaling_type="none", bias=None):
    """Route GEMM to the appropriate AITER backend based on scaling_type.

    All kernels compute ``Y = A @ W^T`` (TN layout).  ``w`` must have
    shape ``(N, K)`` — same as PyTorch Linear weight convention.

    For backward dgrad, pass ``weight.t().contiguous()`` as ``w`` so that
    ``A @ W^T = grad @ weight``.

    ``a`` and ``w`` may each be a plain :class:`torch.Tensor` or an
    :class:`~atomrl.lumen.quantize.descriptor.FP8Descriptor`.  When an argument
    is a descriptor, its ``.data`` and ``.scale`` fields are unpacked
    automatically, overriding the corresponding ``scale_*`` parameter.

    Args:
        a: Activation / LHS tensor ``(M, K)``, or :class:`FP8Descriptor`.
        w: Weight / RHS tensor ``(N, K)``, or :class:`FP8Descriptor`.
        scale_a: Activation scale (ignored for that side if ``a`` is a descriptor).
        scale_w: Weight scale (ignored for that side if ``w`` is a descriptor).
        scaling_type: One of the 6 supported modes.
        bias: Optional bias.

    Returns:
        Output tensor ``(M, N)``.
    """
    w_transposed = None
    a_unpadded_shape = None
    w_unpadded_shape = None
    # scale_f32_1x1 collapses the scale to (1,1) for hipBLASLt's per-tensor
    # epilogue; that is only valid for per-tensor scalings. block/token/mx
    # scalings carry multi-element scales that must be passed through as-is.
    _per_tensor = scaling_type in ("delayed", "dynamic")
    if isinstance(a, FP8Descriptor):
        a_unpadded_shape = a.unpadded_shape
        scale_a = a.scale_f32_1x1 if (_PREFER_HIPBLASLT and _per_tensor) else a.scale
        a = a.data
    if isinstance(w, FP8Descriptor):
        w_unpadded_shape = w.unpadded_shape
        scale_w = w.scale_f32_1x1 if (_PREFER_HIPBLASLT and _per_tensor) else w.scale
        if w._transpose is not None:
            w_transposed = w._transpose
        w = w.data

    if scaling_type == "none":
        return gemm_bf16(a, w, bias)

    _fuse_bias = (
        bias is not None
        and scaling_type in ("delayed", "dynamic")
        and _PREFER_HIPBLASLT
    )

    _fuse_blockscale_bias = bias is not None and scaling_type in (
        "blockwise",
        "blockwise2d",
    )

    if scaling_type in ("delayed", "dynamic"):
        out = gemm_per_tensor(
            a, w, scale_a, scale_w, w_transposed, bias=bias if _fuse_bias else None
        )
    elif scaling_type == "per_token":
        out = gemm_per_token(a, w, scale_a, scale_w)
    elif scaling_type in ("blockwise", "blockwise2d"):
        if _fuse_blockscale_bias:
            out = gemm_blockscale_with_bias(a, w, scale_a, scale_w, bias)
        else:
            out = gemm_blockscale(a, w, scale_a, scale_w)
    elif scaling_type == "mxfp8":
        out = gemm_mxfp8(a, w, scale_a, scale_w)
    elif scaling_type == "mxfp4":
        out = gemm_mxfp4_dispatch(a, w, scale_a, scale_w)
    else:
        raise ValueError(f"Unknown scaling_type={scaling_type!r}")

    if scaling_type == "mxfp8":
        rows = a_unpadded_shape[0] if a_unpadded_shape is not None else out.shape[0]
        columns = w_unpadded_shape[0] if w_unpadded_shape is not None else out.shape[1]
        out = out[:rows, :columns]

    if bias is not None and not _fuse_bias and not _fuse_blockscale_bias:
        out = out + bias
    return out


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------


_MXFP4_MAX_OPERAND_ELEMS = 2**31
_mxfp4_int32_warned: set[str] = set()


def _mxfp4_operands_fit_int32(input: torch.Tensor, weight: torch.Tensor) -> bool:
    """Can every operand this layer's MXFP4 path builds be indexed in 32 bits?

    A Qwen3-8B run at micro-batch 8 dies with an illegal memory access in the
    vocab projection's dgrad -- M=16384, N=151936, so the grad_output operand
    holds 2.49e9 elements. 2**31 is 2.15e9. Every other GEMM in that same step
    stays under the line (the widest is 16384x12288 = 2.01e9) and every one of
    them runs, and halving the micro-batch puts the vocab layer at 1.24e9 and
    it runs too. That is a 32-bit index wrapping, in a kernel Lumen does not
    own.

    Catching it after the fact is not an option: an illegal access poisons the
    HIP context, so the BF16 fallback in backward never gets to run, and the
    error surfaces from whatever happens to synchronize next rather than from
    the kernel that caused it. The shape simply must not be dispatched.

    Checks all three products because the operands differ per pass: forward
    reads M*K, dgrad reads M*N, wgrad reads both, and the weight is N*K.
    """
    k = input.shape[-1]
    m = input.numel() // k if k else 0
    n = weight.shape[0]
    return max(m * k, m * n, n * k) < _MXFP4_MAX_OPERAND_ELEMS


class QuantizedLinearFunction(torch.autograd.Function):
    """FP8 quantized linear: quant -> GEMM -> dequant, for both fwd and bwd.

    Supports all 7 scaling modes via ``scaling_type`` parameter.
    Backend selection uses ASM → Triton fallback automatically.
    All backends are AITER implementations.

    When ``delay_wgrad=True``, the backward pass computes only dgrad
    (input gradient) and defers the wgrad computation to a later
    ``deferred_wgrad.execute()`` call.  This enables overlapping the
    deferred wgrad GEMM with the next layer's communication.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        scaling_manager,
        scaling_type: str,
        fp8_dtype: torch.dtype,
        block_size: int,
        tensor_id: str = "weight",
        quantize_activation: bool = True,
        fp8_wgrad: bool = True,
        gradient_accumulation_fusion: bool = False,
        delay_wgrad: bool = False,
        deferred_wgrad=None,
        fp8_activation_store: bool = False,
        activation_tensor_id: str | None = None,
        pre_quantized_input: tuple | None = None,
        fp8_weight_cache: torch.Tensor | None = None,
        fp8_weight_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # weight is [N, K] — standard PyTorch Linear convention
        if scaling_type == "mxfp4" and not _mxfp4_operands_fit_int32(input, weight):
            if tensor_id not in _mxfp4_int32_warned:
                _mxfp4_int32_warned.add(tensor_id)
                _logger.warning(
                    "mxfp4 %s: operand exceeds %.2fe9 elements at input %s x weight "
                    "%s; AITER's kernels index in 32 bits, so this layer stays BF16. "
                    "Shrink the micro-batch to quantize it.",
                    tensor_id,
                    _MXFP4_MAX_OPERAND_ELEMS / 1e9,
                    tuple(input.shape),
                    tuple(weight.shape),
                )
            scaling_type = "none"

        if scaling_type == "none":
            output = gemm_bf16(input, weight, bias)
            if fp8_activation_store:
                input_2d = _to_2d(input)
                input_store, input_scale = _fp8_store_activation(input_2d, fp8_dtype)
                ctx.save_for_backward(input_store, input_scale, weight)
                ctx._input_shape = input.shape
            else:
                ctx.save_for_backward(input, weight)
            ctx.fp8_activation_store = fp8_activation_store
            ctx.has_bias = bias is not None
            ctx.scaling_type = "none"
            ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
            ctx.delay_wgrad = delay_wgrad
            ctx.deferred_wgrad = deferred_wgrad
            ctx.weight_ref = weight
            return output

        if not quantize_activation:
            if fp8_weight_cache is not None and fp8_weight_scale is not None:
                from atomrl.lumen.quantize.descriptor import FP8Descriptor as _FP8D

                _cache = (
                    fp8_weight_cache
                    if fp8_weight_cache.is_contiguous()
                    else fp8_weight_cache.contiguous()
                )
                weight_desc = _FP8D(
                    data=_cache,
                    scale=(
                        fp8_weight_scale.to(fp8_weight_cache.device)
                        if fp8_weight_scale.device != fp8_weight_cache.device
                        else fp8_weight_scale
                    ),
                    fp8_dtype=fp8_dtype,
                )
            else:
                _w = weight if weight.is_contiguous() else weight.contiguous()
                weight_desc = quantize_input(
                    _w,
                    scaling_type,
                    fp8_dtype,
                    block_size,
                    scaling_manager,
                    tensor_id,
                )
            weight_dequant = (weight_desc.data.to(input.dtype) * weight_desc.scale).to(
                input.dtype
            )
            output = gemm_bf16(input, weight_dequant, bias)
            if fp8_activation_store:
                input_2d = _to_2d(input)
                input_store, input_scale = _fp8_store_activation(input_2d, fp8_dtype)
                ctx.save_for_backward(
                    input_store, input_scale, weight_desc.data, weight_desc.scale
                )
                ctx._input_shape = input.shape
            else:
                ctx.save_for_backward(input, weight_desc.data, weight_desc.scale)
            ctx.fp8_activation_store = fp8_activation_store
            ctx.scaling_manager = scaling_manager
            ctx.has_bias = bias is not None
            ctx.quantize_activation = False
            ctx.scaling_type = scaling_type
            ctx.fp8_wgrad = True
            ctx.tensor_id = tensor_id
            ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
            ctx.delay_wgrad = delay_wgrad
            ctx.deferred_wgrad = deferred_wgrad
            ctx.weight_ref = weight
            return output

        input_2d = _to_2d(input)
        input_wgrad_operand = None

        if pre_quantized_input is not None:
            _pqi_fp8, _pqi_scale = pre_quantized_input
            # blockwise2d expects a 2D (M, K/block) scale; fused SwiGLU quant
            # produces a per-tensor 1D scale — discard and re-quantize correctly.
            if (
                scaling_type in ("blockwise", "blockwise2d")
                and _pqi_scale.dim() < 2
                or scaling_type in ("mxfp4", "mxfp8")
            ):
                pre_quantized_input = None
        if pre_quantized_input is not None:
            from atomrl.lumen.quantize.descriptor import FP8Descriptor

            input_fp8, input_scale = pre_quantized_input
            input_desc = FP8Descriptor(
                data=input_fp8,
                scale=input_scale,
                fp8_dtype=fp8_dtype,
            )
            if scaling_manager is not None and activation_tensor_id:
                try:
                    from atomrl.lumen.modules.parallel_linear import _pop_swiglu_amax

                    _sw_amax = _pop_swiglu_amax()
                    if _sw_amax is not None:
                        scaling_manager.update_amax_value(
                            activation_tensor_id, _sw_amax
                        )
                except ImportError:
                    pass
        else:
            _act_mgr = scaling_manager if activation_tensor_id else None
            _act_tid = activation_tensor_id or "activation"
            _fuse_in_swizzle = False
            input_wgrad_operand = None
            if input_wgrad_operand is not None:
                input_desc, _wg_fp4, _wg_scale, _wg_shuffled = input_wgrad_operand
            else:
                input_desc = quantize_input(
                    input_2d,
                    scaling_type,
                    fp8_dtype,
                    block_size,
                    _act_mgr,
                    _act_tid,
                    swizzle_scale=_fuse_in_swizzle,
                )
        if fp8_weight_cache is not None and fp8_weight_scale is not None:
            from atomrl.lumen.quantize.descriptor import FP8Descriptor as _FP8D

            # blockwise2d expects a 2D (N/block, K/block) scale — fail fast if
            # the cache was produced by per-tensor store_weights_fp8 (which
            # yields a scalar scale) and the user picked blockwise2d.
            if scaling_type == "blockwise2d":
                N_w, K_w = fp8_weight_cache.shape[-2], fp8_weight_cache.shape[-1]
                expected = (N_w // block_size, K_w // block_size)
                assert (
                    fp8_weight_scale.dim() == 2
                    and tuple(fp8_weight_scale.shape) == expected
                ), (
                    f"blockwise2d + fp8_weight_cache expects scale shape "
                    f"{expected}, got {tuple(fp8_weight_scale.shape)} — the "
                    f"cache must be 128×128 2D-block quantized (not per-tensor)."
                )
            weight_desc = _FP8D(
                data=fp8_weight_cache.contiguous(),
                scale=(
                    fp8_weight_scale.to(fp8_weight_cache.device)
                    if fp8_weight_scale.device != fp8_weight_cache.device
                    else fp8_weight_scale
                ),
                fp8_dtype=fp8_dtype,
            )
        else:
            weight_desc = quantize_input(
                weight.contiguous(),
                scaling_type,
                fp8_dtype,
                block_size,
                scaling_manager,
                tensor_id,
                is_weight=True,
            )

        # Forward: Y = input @ weight^T  (TN layout, weight is [N, K])
        _fuse_bias = (
            bias is not None
            and scaling_type in ("delayed", "dynamic")
            and _PREFER_HIPBLASLT
        )
        output = dispatch_gemm(
            input_desc,
            weight_desc,
            scaling_type=scaling_type,
            bias=bias if _fuse_bias else None,
        )
        output = output.view(*input.shape[:-1], weight.shape[0])

        if bias is not None and not _fuse_bias:
            output = output + bias

        # blockwise / blockwise2d bwd (Jet-RL §4.2): activation is stored in FP8
        # (1×128 quantized) for memory efficiency.  WGrad "Requantizes" it to
        # 128×1 by dequantizing back to BF16 and re-quantizing along axis=0.
        # Both route through the same full-FP8 backward block; blockwise(1D)
        # differs only in DGrad's weight source (columnwise re-quant copy vs the
        # 2D-tile transpose used by blockwise2d).
        if scaling_type in ("blockwise", "blockwise2d"):
            # Frozen-weight cache may carry the precomputed transposed weight
            # (data_t, scale_t) for DGrad and a skip-wgrad marker (frozen base →
            # its grad is discarded) — thread both onto ctx (see
            # atomrl.lumen.quantize._maybe_cache_frozen_weight).
            ctx._weight_t = getattr(weight_desc.data, "_lumen_wt", None)
            # Skip the frozen base weight's WGrad (its grad is discarded). Two ways:
            #   - cached path: marker on the cache tensor (_lumen_skip_wgrad);
            #   - uncached (e.g. 70B FSDP): LUMEN_SKIP_FROZEN_WGRAD=1 + the patch-time
            #     frozen marker threaded onto the weight (weight._lumen_frozen).
            ctx._skip_wgrad = getattr(weight_desc.data, "_lumen_skip_wgrad", False) or (
                getattr(weight, "_lumen_frozen", False) and _skip_frozen_wgrad_enabled()
            )
            ctx.save_for_backward(
                input_desc.data,
                input_desc.scale,
                weight_desc.data,
                weight_desc.scale,
            )
        elif scaling_type == "mxfp4":
            # Reuse pre-transposed weight from module cache if available.
            _wt_cached = getattr(weight_desc.data, "_mxfp4_wt_cached", None)
            if _wt_cached is not None:
                w_fp4_t, w_scale_t = _wt_cached
            else:
                from atomrl.lumen.ops.quantize.ops import transpose_packed_fp4

                w_fp4_t = transpose_packed_fp4(weight_desc.data)
                w_scale_t = weight_desc.scale.t().contiguous()
            ctx.mxfp4_input_scale_swizzled = False
            ctx.mxfp4_weight_t_shuffled = False
            ctx.mxfp4_weight_t_scale_swizzled = False
            ctx.mxfp4_wgrad_activation = False
            ctx.save_for_backward(
                input_desc.data,
                input_desc.scale,
                w_fp4_t,
                w_scale_t,
            )
        else:
            ctx.save_for_backward(
                input_desc.data,
                input_desc.scale,
                weight_desc.data,
                weight_desc.scale,
            )
        ctx.fp8_activation_store = False
        ctx.scaling_manager = scaling_manager
        ctx.scaling_type = scaling_type
        ctx.fp8_dtype = fp8_dtype
        ctx.block_size = block_size
        ctx.has_bias = bias is not None
        ctx.tensor_id = tensor_id
        ctx.quantize_activation = True
        ctx.fp8_wgrad = fp8_wgrad
        ctx.input_shape = input.shape
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.delay_wgrad = delay_wgrad
        ctx.deferred_wgrad = deferred_wgrad
        ctx.weight_ref = weight
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        scaling_type = ctx.scaling_type

        if scaling_type == "none":
            if ctx.fp8_activation_store:
                input_store, input_scale, weight = ctx.saved_tensors
                input_tensor = _fp8_restore_activation(
                    input_store, input_scale, grad_output.dtype
                )
                input_tensor = input_tensor.view(ctx._input_shape)
            else:
                input_tensor, weight = ctx.saved_tensors
            grad_input = dispatch_gemm(
                grad_output,
                weight.t().contiguous(),
                None,
                None,
                "none",
            )

            if ctx.delay_wgrad and ctx.deferred_wgrad is not None:
                grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
                input_flat = input_tensor.reshape(
                    -1, input_tensor.shape[-1]
                ).contiguous()
                w_ref = ctx.weight_ref
                gaf = ctx.gradient_accumulation_fusion

                def _wgrad_fn():
                    gw = dispatch_gemm(
                        grad_flat.t().contiguous(),
                        input_flat.t().contiguous(),
                        None,
                        None,
                        "none",
                    )
                    if gaf and hasattr(w_ref, "main_grad"):
                        w_ref.main_grad.add_(gw)
                    elif w_ref.grad is not None:
                        w_ref.grad.add_(gw)
                    else:
                        w_ref.grad = gw

                ctx.deferred_wgrad.defer(_wgrad_fn)
                grad_weight = None
            else:
                grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
                input_flat = input_tensor.reshape(-1, input_tensor.shape[-1])
                grad_weight = dispatch_gemm(
                    grad_flat.t().contiguous(),
                    input_flat.t().contiguous(),
                    None,
                    None,
                    "none",
                )
                if ctx.gradient_accumulation_fusion and hasattr(
                    ctx.weight_ref, "main_grad"
                ):
                    ctx.weight_ref.main_grad.add_(grad_weight)
                    grad_weight = None

            grad_bias = (
                grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
                if ctx.has_bias
                else None
            )
            return (
                grad_input,
                grad_weight,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        if not ctx.quantize_activation:
            if ctx.fp8_activation_store:
                input_store, input_scale, weight_fp8, weight_scale = ctx.saved_tensors
                input_tensor = _fp8_restore_activation(
                    input_store, input_scale, grad_output.dtype
                )
                input_tensor = input_tensor.view(ctx._input_shape)
            else:
                input_tensor, weight_fp8, weight_scale = ctx.saved_tensors

            weight_dequant = (weight_fp8.to(grad_output.dtype) * weight_scale).to(
                grad_output.dtype
            )
            grad_input = dispatch_gemm(
                grad_output,
                weight_dequant.t().contiguous(),
                None,
                None,
                "none",
            )

            if ctx.delay_wgrad and ctx.deferred_wgrad is not None:
                grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
                input_flat = input_tensor.reshape(
                    -1, input_tensor.shape[-1]
                ).contiguous()
                mgr = ctx.scaling_manager
                w_ref = ctx.weight_ref
                gaf = ctx.gradient_accumulation_fusion

                def _wgrad_fn():
                    gw = dispatch_gemm(
                        grad_flat.t().contiguous(),
                        input_flat.t().contiguous(),
                        None,
                        None,
                        "none",
                    )
                    if mgr is not None:
                        gw = mgr.quantize_grad(gw)
                    if gaf and hasattr(w_ref, "main_grad"):
                        w_ref.main_grad.add_(gw)
                    elif w_ref.grad is not None:
                        w_ref.grad.add_(gw)
                    else:
                        w_ref.grad = gw

                ctx.deferred_wgrad.defer(_wgrad_fn)
                grad_weight = None
            else:
                grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
                input_flat = input_tensor.reshape(-1, input_tensor.shape[-1])
                grad_weight = dispatch_gemm(
                    grad_flat.t().contiguous(),
                    input_flat.t().contiguous(),
                    None,
                    None,
                    "none",
                )
                if ctx.scaling_manager is not None:
                    grad_weight = ctx.scaling_manager.quantize_grad(grad_weight)
                if ctx.gradient_accumulation_fusion and hasattr(
                    ctx.weight_ref, "main_grad"
                ):
                    ctx.weight_ref.main_grad.add_(grad_weight)
                    grad_weight = None

            grad_bias = (
                grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
                if ctx.has_bias
                else None
            )
            return (
                grad_input,
                grad_weight,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        if scaling_type == "mxfp4":
            wgrad_act = None
            input_data, input_scale, weight_data, weight_scale = ctx.saved_tensors
        else:
            input_data, input_scale, weight_data, weight_scale = ctx.saved_tensors
        fp8_dtype = ctx.fp8_dtype
        block_size = ctx.block_size
        # Per-step weight-quant mode: reuse the manager's cached weight
        # descriptor (transpose materialized once per step) so dgrad does not
        # re-transpose the identical FP8 weight every micro-batch. The cached
        # desc's data equals the saved weight_data (forward used the same cache).
        weight_desc = None
        if _WEIGHT_QUANT_ONCE and ctx.scaling_manager is not None:
            _cache = getattr(ctx.scaling_manager, "_fp8_param_cache", None)
            if _cache is not None:
                _cd = _cache.get(getattr(ctx, "tensor_id", None))
                if _cd is not None and _cd.data.shape == weight_data.shape:
                    weight_desc = _cd
        if weight_desc is None:
            weight_desc = FP8Descriptor.from_tensors(
                weight_data, weight_scale, fp8_dtype
            )

        mgr = ctx.scaling_manager
        bwd_dtype = getattr(mgr, "fp8_dtype_bwd", fp8_dtype) if mgr else fp8_dtype

        # ----- blockwise / blockwise2d: full FP8 DGrad + WGrad (Jet-RL §4.2) -----
        # Activation is stored in FP8 (1×128) from fwd for memory efficiency.
        # DGrad: blockwise2d transposes W data + its 2D scale (square-tile
        # symmetry — O(1)); blockwise(1D) instead builds a columnwise re-quant
        # copy of W (1×128 along N), since its 1×128-along-K scale can't transpose.
        # WGrad runs a (1×128)×(1×128) FP8 GEMM via Triton; activation is
        # "Requantized" — dequantize FP8(1×128) back to BF16, then quantize
        # along the token axis (128×1).  Grad is quantized once into both
        # axis layouts via the fused dual-axis kernel.  WGrad is identical for
        # both modes (no weight operand).
        #
        # weight_data may come from fp8_weight_cache (VeRL offline FP8 quant)
        # or fwd inline quant; blockwise2d → 128×128, blockwise → 1×128.
        if scaling_type in ("blockwise", "blockwise2d"):
            from aiter.ops.enum import QuantType
            from aiter.ops.quant import get_hip_quant

            from atomrl.lumen.ops.quantize.gemm_primitives import _dequant_fp8_weight
            from atomrl.lumen.ops.quantize.ops import (
                quant_fp8_blockwise_dual_axis_impl,
                quant_fp8_blockwise_impl,
            )

            grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
            M, N_out = grad_flat.shape
            K_in = input_data.shape[-1]

            # DGrad = ∇Y(1×128) × W(128×128) — needs N_out, K_in aligned to
            # block; M is unconstrained.
            _aligned_dgrad = N_out % block_size == 0 and K_in % block_size == 0
            # WGrad = ∇Y^T(1×128) × X(1×128) sharing the token axis M —
            # needs M aligned for the axis=0 grad / X quantization grids.
            _aligned_wgrad = M % block_size == 0

            def _bf16_dgrad():
                w_bf16 = _dequant_fp8_weight(
                    weight_data, weight_scale, block_size
                ).bfloat16()
                return dispatch_gemm(
                    grad_flat.bfloat16(), w_bf16.t().contiguous(), None, None, "none"
                )

            def _bf16_wgrad():
                # Recover BF16 X from saved FP8 (1×128) via dequant, then
                # do BF16 GEMM ∇W = ∇Y^T @ X.  dispatch_gemm("none") is
                # Y = A @ B^T, so feed (∇Y^T, X^T).
                x_bf16 = _dequant_fp8_weight(
                    input_data, input_scale, block_size
                ).bfloat16()
                return dispatch_gemm(
                    grad_flat.t().contiguous().bfloat16(),
                    x_bf16.t().contiguous(),
                    None,
                    None,
                    "none",
                )

            # Fused dual-axis grad quant: one BF16 read, both layouts
            # (Jet-RL §4.2 "we fuse these quantization processes").  Only
            # usable when M, N_out are both block-aligned (kernel grid).
            g_row = g_row_s = g_col = g_col_s = None
            if _aligned_dgrad and _aligned_wgrad:
                try:
                    g_row, g_row_s, g_col, g_col_s = quant_fp8_blockwise_dual_axis_impl(
                        grad_flat,
                        dtype=fp8_dtype,
                        block_size=block_size,
                    )
                except (AssertionError, RuntimeError) as e:
                    _logger.warning(
                        "blockwise2d fused grad quant rejected (%s); falling back to single-axis",
                        e,
                    )
            if g_row is None and _aligned_dgrad:
                # axis=1 (per_1x128 along K) — use HIP kernel.
                try:
                    g_row, g_row_s = get_hip_quant(QuantType.per_1x128)(
                        grad_flat,
                        quant_dtype=fp8_dtype,
                    )
                except (AssertionError, RuntimeError) as e:
                    _logger.warning("blockwise2d dgrad grad quant rejected (%s)", e)
            if g_col is None and _aligned_wgrad:
                try:
                    g_col, g_col_s = quant_fp8_blockwise_impl(
                        grad_flat,
                        dtype=fp8_dtype,
                        axis=0,
                        block_size=block_size,
                    )
                except (AssertionError, RuntimeError) as e:
                    _logger.warning("blockwise2d wgrad grad quant rejected (%s)", e)

            # DGrad (FP8)
            if g_row is not None:
                try:
                    if scaling_type == "blockwise":
                        # blockwise(1D): the fwd weight is 1×128 along K, whose
                        # scale does not transpose into a 1×128-along-N layout.
                        # Build a columnwise pre-quantized copy of W^T (K,N) so
                        # DGrad runs in FP8 instead of dequant→BF16.  The copy
                        # uses 128×128 2D tiles: gemm_a8w8_blockscale needs a
                        # (N/128, K/128) weight scale, and a (K, N/128) 1×128
                        # scale is read out of bounds (garbage dX, seen as
                        # -inf SNR and run-to-run variation).  Prefer the original high-precision weight
                        # (no double quantization); fall back to dequantizing the
                        # saved FP8 weight when no BF16 master is available
                        # (e.g. frozen FP8 cache).  FP8 dtypes are 1 byte, so
                        # element_size() >= 2 means a real high-precision weight.
                        w_ref = ctx.weight_ref
                        if (
                            w_ref is not None
                            and w_ref.is_floating_point()
                            and w_ref.element_size() >= 2
                        ):
                            w_src = w_ref
                        else:
                            w_src = _dequant_fp8_weight(
                                weight_data, weight_scale, block_size
                            ).bfloat16()
                        w_t, w_s_t = _quant_blockwise2d_weight(
                            w_src.t().contiguous(), fp8_dtype, block_size
                        )
                    else:
                        # blockwise2d: reuse the frozen weight's cached transpose
                        # if available, else materialize it (per-backward copy
                        # hotspot).  2D square tiles transpose directly.
                        _wt = getattr(ctx, "_weight_t", None)
                        if _wt is not None:
                            w_t, w_s_t = _wt
                        else:
                            w_t, w_s_t = (
                                weight_data.t().contiguous(),
                                weight_scale.t().contiguous(),
                            )
                    grad_input = gemm_blockscale(g_row, w_t, g_row_s, w_s_t)
                except (AssertionError, RuntimeError) as e:
                    _logger.warning(
                        "%s dgrad: kernel rejected (%s); BF16 fallback", scaling_type, e
                    )
                    grad_input = _bf16_dgrad()
            else:
                grad_input = _bf16_dgrad()
            grad_input = grad_input.view(*grad_output.shape[:-1], K_in)

            # WGrad (FP8 (1×128)×(1×128) via Triton blockscale).
            # Activation was saved as FP8 (1×128, row-wise).  Re-quantize to
            # FP8 (128×1, col-wise) using a direct FP8→FP8 Triton kernel —
            # avoids the BF16 intermediate copy (dequant→BF16→requant(axis=0)).
            def _compute_grad_weight():
                if g_col is None:
                    return _bf16_wgrad()
                try:
                    from atomrl.lumen.ops.quantize.ops import requant_fp8_row_to_col

                    # FP8(1×128 row) → FP8(128×1 col) without BF16 roundtrip.
                    # input_scale shape: (M, K_in//block_size) — row dequant multipliers.
                    x_col, x_col_s = requant_fp8_row_to_col(
                        input_data,
                        input_scale,
                        fp8_dtype,
                        block_size,
                    )
                    return _gemm_blockscale_triton(
                        g_col.t().contiguous(),  # (N_out, M) ∇Y^T
                        x_col.t().contiguous(),  # (K_in,  M) X^T
                        g_col_s.t().contiguous(),  # (N_out, M/128)
                        x_col_s.t().contiguous(),  # (K_in,  M/128)
                    )
                except (AssertionError, RuntimeError) as e:
                    _logger.warning(
                        "blockwise wgrad fp8-requant: kernel rejected (%s); BF16 fallback",
                        e,
                    )
                    return _bf16_wgrad()

            if getattr(ctx, "_skip_wgrad", False):
                # Frozen base weight (LoRA): its grad is discarded, so skip the
                # whole WGrad (dequant→requant + transpose copies + GEMM). Under
                # FSDP a frozen view can report requires_grad=True, so this is
                # gated by the patch-time frozen flag (cache_frozen_weight path).
                grad_weight = None
            elif ctx.delay_wgrad and ctx.deferred_wgrad is not None:
                _w_ref = ctx.weight_ref
                _gaf = ctx.gradient_accumulation_fusion
                _mgr = mgr

                def _wgrad_fn():
                    gw = _compute_grad_weight()
                    if _mgr is not None:
                        gw = _mgr.quantize_grad(gw)
                    if _gaf and hasattr(_w_ref, "main_grad"):
                        _w_ref.main_grad.add_(gw)
                    elif _w_ref.grad is not None:
                        _w_ref.grad.add_(gw)
                    else:
                        _w_ref.grad = gw

                ctx.deferred_wgrad.defer(_wgrad_fn)
                grad_weight = None
            else:
                grad_weight = _compute_grad_weight()
                if mgr is not None:
                    grad_weight = mgr.quantize_grad(grad_weight)
                if ctx.gradient_accumulation_fusion and hasattr(
                    ctx.weight_ref, "main_grad"
                ):
                    ctx.weight_ref.main_grad.add_(grad_weight)
                    grad_weight = None

            grad_bias = (
                grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
                if ctx.has_bias
                else None
            )
            return (
                grad_input,
                grad_weight,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        # ----- end blockwise2d -----

        # ----- MXFP4: FP4 DGrad + WGrad -----
        # Against the NVFP4 paper §4 baseline: DGrad reuses the FP4 weight cached
        # in forward (2D block scales are transpose-invariant); the gradient is
        # quantized once into both layouts DGrad and WGrad need; and the
        # activation's WGrad operand goes from stored FP4 straight to rotated,
        # transposed FP4, never writing the BF16 form in between.
        # Per layer: 3 quant, 3 FP4 GEMM, no dequant/Hadamard/transpose.
        if scaling_type == "mxfp4":
            from atomrl.lumen.ops.quantize.ops import (
                convert_from_mxfp4,
                convert_to_mxfp4,
                dequant_hadamard_quant_mxfp4,
                dequant_transpose_mxfp4,
                dual_layout_quant_mxfp4,
            )

            grad_flat = (
                grad_output.reshape(-1, grad_output.shape[-1])
                .to(torch.bfloat16)
                .contiguous()
            )
            M, N_out = grad_flat.shape
            K_in = input_data.shape[-1] * 2  # input_data is packed (M, K//2)
            mxfp4_block = 32

            # Of the three axes only M is not a property of the model: it is
            # seq x mbs, which a last batch, a variable-length run or a MoE
            # expert's token count makes ragged. It is also the one that can be
            # repaired rather than refused. M is WGrad's reduction axis, where a
            # zero row contributes nothing to dY^T @ X, and DGrad's free axis,
            # where the rows padding adds slice back off -- so pad it and keep
            # both gradients in FP4.
            #
            # Folding M into a single _aligned flag sent the whole layer to
            # BF16 instead, and that path took DGrad from the BF16 master while
            # WGrad went through the dequantized activation: two gradient terms
            # against two different versions of the forward, reachable by
            # accident on a last batch.
            m_unpadded = M
            if M % mxfp4_block != 0:
                from atomrl.lumen.ops.quantize.padding import pad_to_block

                grad_flat, _ = pad_to_block(grad_flat, mxfp4_block, dim=0)
                input_data, _ = pad_to_block(
                    input_data.reshape(-1, input_data.shape[-1]),
                    mxfp4_block,
                    dim=0,
                )
                input_scale, _ = pad_to_block(
                    input_scale.reshape(-1, input_scale.shape[-1]),
                    mxfp4_block,
                    dim=0,
                )
                M = grad_flat.shape[0]

            # What is left are the layer's own dims, so reaching the fallback is
            # a property of the model rather than of a batch.
            _aligned = N_out % mxfp4_block == 0 and K_in % mxfp4_block == 0
            if not _aligned:
                _warn_mxfp4_backward_fallback(
                    f"N={N_out} and K={K_in} must both be multiples of {mxfp4_block}",
                    (m_unpadded, N_out, K_in),
                )

            if _aligned:
                try:
                    # --- DGrad: dX = dY @ W_cached^T ---
                    # Reuse FP4 weight + pre-transposed weight from forward.
                    # The FP4 tensors are independent allocations (not views of
                    # the BF16 param), so they survive FSDP2 resharding.
                    rht_g = _MXFP4_RHT_G
                    _rht_ok = M % rht_g == 0

                    if _rht_ok:
                        # The gradient feeds both GEMMs: row-major for DGrad and
                        # rotated + transposed for WGrad. Quantizing it once for
                        # both keeps the read dense; taking dY^T as a view instead
                        # costs 1.70x on the same shape (report §5.10).
                        sign_m = _get_mxfp4_rht_sign(grad_flat.device)
                        g_fp4, g_scale, grad_t_fp4, grad_t_scale = (
                            dual_layout_quant_mxfp4(
                                grad_flat,
                                sign_m,
                                block_size=mxfp4_block,
                                g=rht_g,
                                use_sr_row=True,
                                use_sr_transposed=True,
                                swizzle_scale=False,
                            )
                        )
                    else:
                        from atomrl.lumen.ops.quantize.padding import pad_to_block

                        g_padded, _ = pad_to_block(grad_flat, mxfp4_block, dim=0)
                        g_padded, _ = pad_to_block(g_padded, mxfp4_block, dim=-1)
                        g_fp4, g_scale = convert_to_mxfp4(
                            g_padded,
                            block_size=mxfp4_block,
                            axis=-1,
                            use_sr=True,
                        )
                        # convert_to_mxfp4 can route to AITER's quant, which wants
                        # a dense operand.
                        grad_t_fp4, grad_t_scale = convert_to_mxfp4(
                            grad_flat.t().contiguous(),
                            block_size=mxfp4_block,
                            axis=-1,
                            use_sr=True,
                        )

                    # weight_data / weight_scale are already the pre-transposed
                    # forms saved in forward (W^T packed, scales^T).
                    grad_input = gemm_mxfp4_dispatch(
                        g_fp4, weight_data, g_scale, weight_scale
                    )

                    # --- WGrad: dW = fused_HQ(dY^T) @ fused_HQ(X^T)^T ---
                    # NVFP4 §4.4 / E.3: stochastic rounding belongs on the
                    # gradient only. On activations it buys little and can
                    # diverge, so the activation stays round-to-nearest.
                    if wgrad_act is not None:
                        # Forward already emitted this operand off its own read of
                        # the activation (_mxfp4_wgrad_activation_operand).
                        input_t_fp4, input_t_scale = wgrad_act
                    elif _rht_ok:
                        # Decode, transpose, rotate and requantize in one pass, so
                        # the BF16 (K, M) form of the activation — four times the
                        # bytes of either FP4 end — never reaches memory.
                        input_t_fp4, input_t_scale = dequant_hadamard_quant_mxfp4(
                            input_data.reshape(-1, input_data.shape[-1]),
                            input_scale.reshape(-1, input_scale.shape[-1]),
                            sign_m,
                            block_size=mxfp4_block,
                            g=rht_g,
                            use_sr=False,
                            swizzle_scale=False,
                            shuffle_data=False,
                            in_scale_swizzled=False,
                        )
                    else:
                        # Without the rotation there is nothing to fuse into; the
                        # transposing dequant still lands dense, which the
                        # quantizer below needs.
                        input_t = dequant_transpose_mxfp4(
                            input_data,
                            input_scale,
                            block_size=mxfp4_block,
                        )
                        input_t_fp4, input_t_scale = convert_to_mxfp4(
                            input_t,
                            block_size=mxfp4_block,
                            axis=-1,
                            use_sr=False,
                        )

                    def _compute_wgrad():
                        return gemm_mxfp4_dispatch(
                            grad_t_fp4,
                            input_t_fp4,
                            grad_t_scale,
                            input_t_scale,
                        )

                except (AssertionError, RuntimeError) as e:
                    _logger.warning(
                        "mxfp4 backward: kernel rejected (%s); BF16 fallback", e
                    )
                    _aligned = False

            if not _aligned:
                input_bf16 = convert_from_mxfp4(
                    input_data,
                    input_scale,
                    output_dtype=torch.bfloat16,
                    block_size=mxfp4_block,
                )
                # Take DGrad against the weight the forward actually used.
                # Reaching for ctx.weight_ref here was the precision split:
                # WGrad reads the activation as the forward quantized it, so a
                # DGrad against the unquantized master makes the two terms the
                # gradient of two different forwards. The blockwise2d path in
                # this same function already dequantizes its saved weight for
                # exactly this reason.
                weight_rowmajor = weight_data
                # weight_rowmajor is W^T, the (K_in, N_out) operand
                # dispatch_gemm wants. Its scales are the 2D tile grid, so they
                # need the same expansion as the row-major FP4 kernel.
                w_dgrad = convert_from_mxfp4(
                    weight_rowmajor,
                    _expand_2d_scale_to_1d(
                        weight_scale,
                        (weight_rowmajor.shape[0], weight_rowmajor.shape[1] * 2),
                        mxfp4_block,
                    ),
                    output_dtype=torch.bfloat16,
                    block_size=mxfp4_block,
                )
                grad_input = dispatch_gemm(
                    grad_flat,
                    w_dgrad,
                    None,
                    None,
                    "none",
                )

                def _compute_wgrad():
                    return dispatch_gemm(
                        grad_flat.t().contiguous(),
                        input_bf16.t().contiguous(),
                        None,
                        None,
                        "none",
                    )

            if m_unpadded != M:
                # Drop the rows padding added. They exist only so WGrad's
                # reduction axis tiles; DGrad computed them and nobody wants them.
                grad_input = grad_input[:m_unpadded].contiguous()
            grad_input = grad_input.view(*grad_output.shape[:-1], K_in)

            mgr = ctx.scaling_manager
            if ctx.delay_wgrad and ctx.deferred_wgrad is not None:
                w_ref = ctx.weight_ref
                gaf = ctx.gradient_accumulation_fusion
                _mgr = mgr

                def _wgrad_fn():
                    gw = _compute_wgrad()
                    if _mgr is not None:
                        gw = _mgr.quantize_grad(gw)
                    if gaf and hasattr(w_ref, "main_grad"):
                        w_ref.main_grad.add_(gw)
                    elif w_ref.grad is not None:
                        w_ref.grad.add_(gw)
                    else:
                        w_ref.grad = gw

                ctx.deferred_wgrad.defer(_wgrad_fn)
                grad_weight = None
            else:
                grad_weight = _compute_wgrad()
                if mgr is not None:
                    grad_weight = mgr.quantize_grad(grad_weight)
                if ctx.gradient_accumulation_fusion and hasattr(
                    ctx.weight_ref, "main_grad"
                ):
                    ctx.weight_ref.main_grad.add_(grad_weight)
                    grad_weight = None

            grad_bias = (
                grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
                if ctx.has_bias
                else None
            )
            return (
                grad_input,
                grad_weight,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        # ----- end mxfp4 -----

        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])

        if scaling_type == "mxfp8":
            from atomrl.lumen.ops.quantize.ops import convert_from_mxfp8

            input_rows = 1
            for size in ctx.input_shape[:-1]:
                input_rows *= size
            input_width = ctx.input_shape[-1]
            weight_shape = ctx.weight_ref.shape
            input_bf16 = convert_from_mxfp8(
                input_data,
                input_scale,
                output_dtype=torch.bfloat16,
                block_size=32,
            )[:input_rows, :input_width]
            weight_bf16 = convert_from_mxfp8(
                weight_data,
                weight_scale,
                output_dtype=torch.bfloat16,
                block_size=32,
            )[: weight_shape[0], : weight_shape[1]]

            grad_bf16 = grad_flat.bfloat16().contiguous()
            grad_input = dispatch_gemm(
                grad_bf16,
                weight_bf16.t().contiguous(),
                None,
                None,
                "none",
            ).view(ctx.input_shape)

            def _compute_mxfp8_wgrad():
                return dispatch_gemm(
                    grad_bf16.t().contiguous(),
                    input_bf16.t().contiguous(),
                    None,
                    None,
                    "none",
                )

            if ctx.delay_wgrad and ctx.deferred_wgrad is not None:
                w_ref = ctx.weight_ref
                gaf = ctx.gradient_accumulation_fusion
                manager = ctx.scaling_manager

                def _wgrad_fn():
                    grad = _compute_mxfp8_wgrad()
                    if manager is not None:
                        grad = manager.quantize_grad(grad)
                    if gaf and hasattr(w_ref, "main_grad"):
                        w_ref.main_grad.add_(grad)
                    elif w_ref.grad is not None:
                        w_ref.grad.add_(grad)
                    else:
                        w_ref.grad = grad

                ctx.deferred_wgrad.defer(_wgrad_fn)
                grad_weight = None
            else:
                grad_weight = _compute_mxfp8_wgrad()
                if ctx.scaling_manager is not None:
                    grad_weight = ctx.scaling_manager.quantize_grad(grad_weight)
                if ctx.gradient_accumulation_fusion and hasattr(
                    ctx.weight_ref, "main_grad"
                ):
                    ctx.weight_ref.main_grad.add_(grad_weight)
                    grad_weight = None

            grad_bias = (
                grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
                if ctx.has_bias
                else None
            )
            return (
                grad_input,
                grad_weight,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        # Check FP8 grad cache: if a previous layer's dgrad produced FP8
        # output, reuse it directly instead of re-quantizing BF16->FP8.
        _cached = _fp8_cache_pop(
            grad_flat.data_ptr(), expected_shape=tuple(grad_flat.shape)
        )

        # Only per_token reaches here for the dequant→BF16 fallback; blockwise
        # and blockwise2d are handled by the full-FP8 block above (which returns).
        bwd_scaling = "dynamic" if scaling_type == "per_token" else scaling_type
        # ``grad_t`` is the fused cast+transpose grad^T, populated only when the
        # fused quant kernel ran (LUMEN_FUSED_QUANT_TRANSPOSE_CPP). When present,
        # wgrad reuses it instead of transposing grad a second time.
        grad_t = None
        if _cached is not None:
            grad_fp8, grad_scale = _cached
        else:
            # Pass manager + backward=True so delayed scaling uses the
            # single-kernel static_quant_with_amax path instead of falling
            # through to the 2-kernel dynamic_per_tensor_quant_fp8_i8.
            grad_desc = quantize_input(
                grad_flat,
                bwd_scaling,
                bwd_dtype,
                block_size,
                manager=mgr,
                tensor_id=(ctx.tensor_id or "linear") + "_bwd",
                backward=True,
            )
            grad_fp8, grad_scale = grad_desc.data, grad_desc.scale
            grad_t = grad_desc._transpose

        _needs_dequant = scaling_type == "per_token"
        if _needs_dequant:
            from atomrl.lumen.ops.quantize.gemm_primitives import _dequant_fp8_weight

            grad_bf16 = grad_fp8.bfloat16() * grad_scale.bfloat16()
            weight_bf16 = _dequant_fp8_weight(
                weight_data, weight_scale, block_size
            ).bfloat16()
            grad_input = dispatch_gemm(
                grad_bf16, weight_bf16.t().contiguous(), None, None, "none"
            )
        else:
            # dgrad -> BF16 (matches TE on ROCm: rocm_gemm.cu forbids FP8 GEMM
            # output, so gradients propagate in BF16 and each layer re-quantizes
            # its own grad). An FP8-output+amaxD epilogue path was prototyped
            # (LUMEN_FP8_DGRAD_OUTPUT) but hipBLASLt on gfx942 has no tuned kernel
            # for the mixed-dtype FP8-output combo (~38x slower fallback), so it
            # is not used — see gemm_per_tensor_mixed_fp8out.
            grad_input = gemm_per_tensor_mixed(
                grad_fp8,
                weight_data,
                grad_scale,
                weight_desc.scale,
                w_transpose=weight_desc.transpose_cached,
            )
        grad_input = grad_input.view(*grad_output.shape[:-1], weight_data.shape[-1])

        # wgrad: dW = grad^T @ input
        #
        # Triton dispatch_gemm crashes (SIGABRT) on transposed wgrad
        # tensors in delayed/dynamic mode.  hipBLASLt handles NN layout
        # natively via gemm_wgrad_fp8, so use it when available.  This
        # also matches TE's fp8_wgrad=True behavior (FP8 GEMM for wgrad).
        _MIN_FP8_K = 64
        wgrad_k = grad_fp8.shape[0]
        _hipblas_ok = True
        _use_fp8_wgrad = (
            ctx.fp8_wgrad
            and wgrad_k >= _MIN_FP8_K
            and (_hipblas_ok or bwd_scaling not in ("delayed", "dynamic"))
        )

        if ctx.delay_wgrad and ctx.deferred_wgrad is not None:
            _use_fp8 = _use_fp8_wgrad
            _use_hipblas = _hipblas_ok
            _grad_fp8 = grad_fp8
            _input_data = input_data
            _grad_scale = grad_scale
            _input_scale = input_scale
            _bwd_scaling = bwd_scaling
            _mgr = mgr
            _w_ref = ctx.weight_ref
            _gaf = ctx.gradient_accumulation_fusion
            _grad_t = grad_t  # fused grad^T (or None); reused by mixed wgrad

            def _wgrad_fn():
                if _use_fp8:
                    if (
                        _use_hipblas
                        and _bwd_scaling in ("delayed", "dynamic")
                        and _grad_fp8.dtype == _input_data.dtype
                    ):
                        gw = gemm_wgrad_fp8(
                            _grad_fp8,
                            _input_data,
                            _grad_scale,
                            _input_scale,
                        )
                    elif _grad_fp8.dtype != _input_data.dtype and _bwd_scaling in (
                        "delayed",
                        "dynamic",
                    ):
                        # Hybrid: keep wgrad fully FP8 via Triton mixed GEMM.
                        gw = gemm_wgrad_mixed(
                            _grad_fp8,
                            _input_data,
                            _grad_scale,
                            _input_scale,
                            grad_t=_grad_t,
                        )
                    else:
                        gw = dispatch_gemm(
                            _grad_fp8.t().contiguous(),
                            _input_data.t().contiguous(),
                            _grad_scale,
                            _input_scale,
                            _bwd_scaling,
                        )
                else:
                    g_bf16 = (
                        _grad_fp8.bfloat16() * _grad_scale.bfloat16()
                    ).contiguous()
                    i_bf16 = (
                        _input_data.bfloat16() * _input_scale.bfloat16()
                    ).contiguous()
                    gw = dispatch_gemm(
                        g_bf16.t().contiguous(),
                        i_bf16.t().contiguous(),
                        None,
                        None,
                        "none",
                    )
                if _mgr is not None:
                    gw = _mgr.quantize_grad(gw)
                if _gaf and hasattr(_w_ref, "main_grad"):
                    _w_ref.main_grad.add_(gw)
                elif _w_ref.grad is not None:
                    _w_ref.grad.add_(gw)
                else:
                    _w_ref.grad = gw

            ctx.deferred_wgrad.defer(_wgrad_fn)
            grad_weight = None
        else:
            if _use_fp8_wgrad:
                if (
                    _hipblas_ok
                    and bwd_scaling in ("delayed", "dynamic")
                    and grad_fp8.dtype == input_data.dtype
                ):
                    grad_weight = gemm_wgrad_fp8(
                        grad_fp8,
                        input_data,
                        grad_scale,
                        input_scale,
                    )
                elif grad_fp8.dtype != input_data.dtype and bwd_scaling in (
                    "delayed",
                    "dynamic",
                ):
                    # Hybrid: E5M2 grad x E4M3 input. Keep wgrad fully FP8 via
                    # Triton mixed GEMM instead of requanting grad to E4M3.
                    grad_weight = gemm_wgrad_mixed(
                        grad_fp8,
                        input_data,
                        grad_scale,
                        input_scale,
                        grad_t=grad_t,
                    )
                else:
                    grad_weight = dispatch_gemm(
                        grad_fp8.t().contiguous(),
                        input_data.t().contiguous(),
                        grad_scale,
                        input_scale,
                        bwd_scaling,
                    )
            else:
                grad_bf16 = (grad_fp8.bfloat16() * grad_scale.bfloat16()).contiguous()
                input_bf16 = (
                    input_data.bfloat16() * input_scale.bfloat16()
                ).contiguous()
                grad_weight = dispatch_gemm(
                    grad_bf16.t().contiguous(),
                    input_bf16.t().contiguous(),
                    None,
                    None,
                    "none",
                )

            if mgr is not None:
                grad_weight = mgr.quantize_grad(grad_weight)

            if ctx.gradient_accumulation_fusion and hasattr(
                ctx.weight_ref, "main_grad"
            ):
                ctx.weight_ref.main_grad.add_(grad_weight)
                grad_weight = None

        grad_bias = (
            grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            if ctx.has_bias
            else None
        )

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


_mark_allow_in_graph(QuantizedLinearFunction)


class FP8StoredLinearFunction(torch.autograd.Function):
    """Linear for weights already stored in FP8 (FP8 param storage).

    Unlike :class:`QuantizedLinearFunction`, the weight is never passed as a
    BF16 tensor — only the compact FP8 data + scale enter the autograd graph.
    This prevents PyTorch from pinning a full BF16 copy per layer for the
    entire forward pass, which is critical for fitting 70B models in 192 GB.

    Forward:  quantize input → FP8 GEMM with pre-quantized weight
    Backward: re-dequantize weight from saved FP8 for dgrad; no wgrad
              (frozen base weights).
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        scaling_manager,
        scaling_type: str,
        fp8_dtype: torch.dtype,
        block_size: int,
        tensor_id: str,
        gradient_accumulation_fusion: bool,
        delay_wgrad: bool,
        deferred_wgrad,
        activation_tensor_id: str | None = None,
        pre_quantized_input: tuple | None = None,
    ) -> torch.Tensor:
        input_2d = _to_2d(input)
        if scaling_type == "mxfp8":
            raise NotImplementedError(
                "FP8StoredLinear does not carry MXFP8 unpadded weight metadata"
            )

        if pre_quantized_input is not None:
            input_fp8, input_scale = pre_quantized_input
            if scaling_manager is not None and activation_tensor_id:
                try:
                    from atomrl.lumen.modules.parallel_linear import _pop_swiglu_amax

                    _sw_amax = _pop_swiglu_amax()
                    if _sw_amax is not None:
                        scaling_manager.update_amax_value(
                            activation_tensor_id, _sw_amax
                        )
                except ImportError:
                    pass
        else:
            _act_mgr = scaling_manager if activation_tensor_id else None
            _act_tid = activation_tensor_id or "activation"
            input_desc = quantize_input(
                input_2d,
                scaling_type,
                fp8_dtype,
                block_size,
                _act_mgr,
                _act_tid,
            )
            input_fp8, input_scale = input_desc.data, input_desc.scale

        N = weight_fp8.shape[0]
        _fuse_bias = (
            bias is not None
            and scaling_type in ("delayed", "dynamic")
            and _PREFER_HIPBLASLT
        )
        output = dispatch_gemm(
            input_fp8,
            weight_fp8,
            input_scale,
            weight_scale,
            scaling_type,
            bias=bias if _fuse_bias else None,
        )
        output = output.view(*input.shape[:-1], N)

        if bias is not None and not _fuse_bias:
            output = output + bias

        ctx.save_for_backward(weight_fp8, weight_scale)
        ctx.scaling_manager = scaling_manager
        ctx.scaling_type = scaling_type
        ctx.fp8_dtype = fp8_dtype
        ctx.block_size = block_size
        ctx.has_bias = bias is not None
        ctx.tensor_id = tensor_id
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        weight_fp8, weight_scale = ctx.saved_tensors
        fp8_dtype = ctx.fp8_dtype
        block_size = ctx.block_size
        scaling_type = ctx.scaling_type

        mgr = ctx.scaling_manager
        bwd_dtype = getattr(mgr, "fp8_dtype_bwd", fp8_dtype) if mgr else fp8_dtype

        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])

        bwd_scaling = (
            "dynamic"
            if scaling_type in ("per_token", "blockwise", "blockwise2d")
            else scaling_type
        )

        # Pop the FP8 dgrad cache eagerly to maintain cache coherence (the entry must
        # be consumed regardless of which DGrad path fires). Defer quantize_input until
        # actually needed: blockwise2d aligned uses g_row (1×128) and never needs
        # grad_fp8, saving one per-tensor quant kernel + up to ~470 MB FP8 allocation
        # per FP8StoredLinear backward call.
        _cache_entry = _fp8_cache_pop(
            grad_flat.data_ptr(), expected_shape=tuple(grad_flat.shape)
        )
        _grad_fp8_lazy = [None, None]  # [grad_fp8, grad_scale] — filled on first access

        def _ensure_grad_fp8():
            if _grad_fp8_lazy[0] is None:
                if _cache_entry is not None:
                    _grad_fp8_lazy[0], _grad_fp8_lazy[1] = _cache_entry
                else:
                    # Pass manager + backward=True so delayed scaling uses the
                    # single-kernel fused path instead of 2-kernel dynamic quant.
                    grad_desc = quantize_input(
                        grad_flat,
                        bwd_scaling,
                        bwd_dtype,
                        block_size,
                        manager=mgr,
                        tensor_id=(ctx.tensor_id or "linear") + "_bwd",
                        backward=True,
                    )
                    _grad_fp8_lazy[0], _grad_fp8_lazy[1] = (
                        grad_desc.data,
                        grad_desc.scale,
                    )
            return _grad_fp8_lazy[0], _grad_fp8_lazy[1]

        weight_desc = FP8Descriptor.from_tensors(weight_fp8, weight_scale, fp8_dtype)

        def _bf16_dgrad():
            # Dequant weight -> BF16, transpose+contiguous, BF16 GEMM. Used for
            # per_token / blockwise(1D) (1D scales are not transpose-symmetric)
            # and as the non-128-aligned fallback for blockwise2d.
            from atomrl.lumen.ops.quantize.gemm_primitives import _dequant_fp8_weight

            grad_fp8, grad_scale = _ensure_grad_fp8()
            grad_bf16 = grad_fp8.bfloat16() * grad_scale.bfloat16()
            weight_bf16 = _dequant_fp8_weight(
                weight_fp8, weight_scale, block_size
            ).bfloat16()
            return dispatch_gemm(
                grad_bf16,
                weight_bf16.t().contiguous(),
                None,
                None,
                "none",
            )

        if scaling_type == "blockwise2d":
            # blockwise2d weight is 128×128 quantized with a 2D scale that is
            # transpose-symmetric, so DGrad runs as an FP8 blockscale GEMM
            # (transpose the 1-byte FP8 weight + 2D scale, quantize grad 1×128)
            # instead of dequantizing the full weight to BF16 and doing a
            # weight_bf16.t().contiguous() (~205 GB/step of copies) + BF16 mm.
            # Mirrors QuantizedLinearFunction's blockwise2d DGrad.
            N_out, K_in = grad_flat.shape[1], weight_fp8.shape[1]
            if N_out % block_size == 0 and K_in % block_size == 0:
                from aiter.ops.enum import QuantType
                from aiter.ops.quant import get_hip_quant

                try:
                    # Frozen weight may carry a precomputed FP8 transpose.
                    _wt = getattr(weight_fp8, "_lumen_wt", None)
                    if _wt is not None:
                        w_t, w_s_t = _wt
                    else:
                        w_t = weight_fp8.t().contiguous()
                        w_s_t = weight_scale.t().contiguous()
                    g_row, g_row_s = get_hip_quant(QuantType.per_1x128)(
                        grad_flat,
                        quant_dtype=fp8_dtype,
                    )
                    grad_input = gemm_blockscale(g_row, w_t, g_row_s, w_s_t)
                except (AssertionError, RuntimeError) as e:
                    _logger.warning(
                        "FP8StoredLinear blockwise2d dgrad: kernel rejected (%s); "
                        "BF16 fallback",
                        e,
                    )
                    grad_input = _bf16_dgrad()
            else:
                grad_input = _bf16_dgrad()
        elif scaling_type in ("per_token", "blockwise"):
            grad_input = _bf16_dgrad()
        else:
            # dgrad -> BF16 (matches TE on ROCm; FP8-output epilogue not viable on
            # gfx942 hipBLASLt — see gemm_per_tensor_mixed_fp8out).
            grad_fp8, grad_scale = _ensure_grad_fp8()
            grad_input = gemm_per_tensor_mixed(
                grad_fp8,
                weight_fp8,
                grad_scale,
                weight_desc.scale,
                w_transpose=weight_desc.transpose_cached,
            )
        grad_input = grad_input.view(*grad_output.shape[:-1], weight_fp8.shape[-1])

        grad_bias = (
            grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            if ctx.has_bias
            else None
        )

        return (
            grad_input,
            None,  # weight_fp8 (no grad — frozen)
            None,  # weight_scale
            grad_bias,
            None,  # scaling_manager
            None,  # scaling_type
            None,  # fp8_dtype
            None,  # block_size
            None,  # tensor_id
            None,  # gradient_accumulation_fusion
            None,  # delay_wgrad
            None,  # deferred_wgrad
            None,  # activation_tensor_id
            None,  # pre_quantized_input
        )


_mark_allow_in_graph(FP8StoredLinearFunction)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def quantized_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    scaling_manager=None,
    backend: str = "auto",
    scaling_type: str = "delayed",
    fp8_dtype: torch.dtype | None = None,
    block_size: int = 128,
    tensor_id: str = "weight",
    quantize_activation: bool = True,
    fp8_wgrad: bool = True,
    gradient_accumulation_fusion: bool = False,
    delay_wgrad: bool = False,
    deferred_wgrad=None,
    fp8_activation_store: bool = False,
    fp8_weight_cache: torch.Tensor | None = None,
    fp8_weight_scale: torch.Tensor | None = None,
    pre_quantized_weight: tuple | None = None,
    activation_tensor_id: str | None = None,
    pre_quantized_input: tuple | None = None,
) -> torch.Tensor:
    """Functional quantized linear with multi-backend fallback (all AITER).

    Args:
        input: Input tensor ``[*, in_features]``.
        weight: Weight matrix ``[out_features, in_features]``.
        bias: Optional bias ``[out_features]``.
        scaling_manager: A :class:`~atomrl.lumen.quantize.ScalingManager`.
        backend: Legacy parameter (ignored, auto-fallback is always used).
        scaling_type: One of ``"delayed"``, ``"dynamic"``, ``"per_token"``,
            ``"blockwise"``, ``"blockwise2d"``, ``"mxfp8"``, ``"mxfp4"``, ``"none"``.
        fp8_dtype: Target FP8 dtype.  ``None`` auto-detects based on GPU
            architecture (``float8_e4m3fnuz`` on gfx942, ``float8_e4m3fn``
            on gfx950+).
        block_size: Block size for blockwise/MXFP8 quantization.
        tensor_id: Unique identifier for this layer's weight.
        quantize_activation: If ``True``, quantize both input and weight.
        fp8_wgrad: If ``True``, compute weight gradient in FP8.
        delay_wgrad: If ``True``, defer weight gradient computation.
        deferred_wgrad: A :class:`~atomrl.lumen.modules.parallel_linear._DeferredWgrad`
            instance that collects deferred wgrad closures.
        fp8_weight_cache: Pre-quantized FP8 weight tensor (from
            :func:`~atomrl.lumen.quantize.store_weights_fp8`).  When provided,
            ``quantize_input`` is skipped for the weight.
        fp8_weight_scale: Scale for *fp8_weight_cache*.
        pre_quantized_weight: Optional ``(fp8_tensor, scale)`` tuple when the
            weight is already stored in FP8.  Bypasses weight quantization
            and avoids materializing a full BF16 weight tensor in the
            autograd graph.

    Returns:
        Output tensor ``[*, out_features]``.
    """
    if fp8_dtype is None:
        fp8_dtype = _get_float8_e4m3()
        _logger.info("quantized_linear: auto-detected fp8_dtype=%s", fp8_dtype)

    if scaling_manager is None:
        from atomrl.lumen.quantize import ScalingManager

        scaling_manager = ScalingManager(fp8_dtype=fp8_dtype)

    if pre_quantized_weight is not None:
        return FP8StoredLinearFunction.apply(
            input,
            pre_quantized_weight[0],
            pre_quantized_weight[1],
            bias,
            scaling_manager,
            scaling_type,
            fp8_dtype,
            block_size,
            tensor_id,
            gradient_accumulation_fusion,
            delay_wgrad,
            deferred_wgrad,
            activation_tensor_id,
            pre_quantized_input,
        )

    return QuantizedLinearFunction.apply(
        input,
        weight,
        bias,
        scaling_manager,
        scaling_type,
        fp8_dtype,
        block_size,
        tensor_id,
        quantize_activation,
        fp8_wgrad,
        gradient_accumulation_fusion,
        delay_wgrad,
        deferred_wgrad,
        fp8_activation_store,
        activation_tensor_id,
        pre_quantized_input,
        fp8_weight_cache,
        fp8_weight_scale,
    )
