# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused Q/K/V dynamic FP8 quantization and MLA descale preparation."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _load_tensor(X, offsets, N, SHAPE: tl.constexpr, STRIDE: tl.constexpr):
    heads, dim = SHAPE
    physical = (
        offsets // (heads * dim) * STRIDE[0]
        + offsets // dim % heads * STRIDE[1]
        + offsets % dim * STRIDE[2]
    )
    return tl.load(X + physical, offsets < N, 0).to(tl.float32)


@triton.jit
def _partial_amax(
    X,
    Partial,
    N,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for step in range(tl.cdiv(N, PARTS * BLOCK)):
        x = _load_tensor(X, offsets + step * PARTS * BLOCK, N, SHAPE, STRIDE)
        acc = tl.maximum(acc, tl.abs(x))
    tl.store(Partial + tl.program_id(0), tl.max(acc, 0))


@triton.jit
def _fused_qkv_amax(
    Q,
    K,
    V,
    Partial,
    NQ,
    NK,
    NV,
    SHAPES: tl.constexpr,
    STRIDES: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    kind = tl.program_id(1)
    if kind == 0:
        _partial_amax(Q, Partial, NQ, SHAPES[0], STRIDES[0], PARTS, BLOCK)
    elif kind == 1:
        _partial_amax(K, Partial + PARTS, NK, SHAPES[1], STRIDES[1], PARTS, BLOCK)
    else:
        _partial_amax(V, Partial + 2 * PARTS, NV, SHAPES[2], STRIDES[2], PARTS, BLOCK)


@triton.jit
def _quant_tensor(
    X,
    Y,
    Partial,
    Scale,
    GatherScale,
    N,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    SCALE_FACTOR: tl.constexpr,
    GATHER: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = _load_tensor(X, offsets, N, SHAPE, STRIDE)
    if SINGLE_PASS:
        amax = tl.max(tl.abs(x), 0)
    else:
        amax = tl.max(tl.load(Partial + tl.arange(0, PARTS)), 0)
    # Match AITER for nonzero inputs. A zero descale gives FMHA NaNs when it
    # scales masked -inf logits by zero. Use a positive scale for zero tensors;
    # 1e-6 also preserves the cached-gather clamp/headroom policy exactly.
    raw_descale = amax * (1.0 / 448.0)
    descale = tl.where(raw_descale > 0, raw_descale, 1e-6)
    inv = tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1;",
        constraints="=v,v",
        args=[descale],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    y = tl.minimum(tl.maximum(x * inv, -448.0), 448.0)
    tl.store(Y + offsets, y, offsets < N)
    if tl.program_id(0) == 0:
        tl.store(Scale, descale * SCALE_FACTOR)
        if GATHER:
            tl.store(GatherScale, tl.maximum(descale, 1e-6) * 2.0)


@triton.jit
def _fused_qkv_quant(
    Q,
    K,
    V,
    Q8,
    K8,
    V8,
    Partial,
    Scales,
    NQ,
    NK,
    NV,
    SHAPES: tl.constexpr,
    STRIDES: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    Q_SCALE_FACTOR: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    kind = tl.program_id(1)
    if kind == 0:
        _quant_tensor(
            Q,
            Q8,
            Partial,
            Scales,
            Scales,
            NQ,
            SHAPES[0],
            STRIDES[0],
            PARTS,
            BLOCK,
            Q_SCALE_FACTOR,
            False,
            SINGLE_PASS,
        )
    elif kind == 1:
        _quant_tensor(
            K,
            K8,
            Partial + PARTS,
            Scales + 1,
            Scales + 3,
            NK,
            SHAPES[1],
            STRIDES[1],
            PARTS,
            BLOCK,
            1.0,
            True,
            SINGLE_PASS,
        )
    else:
        _quant_tensor(
            V,
            V8,
            Partial + 2 * PARTS,
            Scales + 2,
            Scales + 4,
            NV,
            SHAPES[2],
            STRIDES[2],
            PARTS,
            BLOCK,
            1.0,
            True,
            SINGLE_PASS,
        )


def fused_qkv_per_tensor_quant(q, k, v, *, q_scale_factor=1.0):
    """Quantize 3-D Q/K/V to E4M3 with independent per-tensor FP32 descales.

    Returns ``(q8, k8, v8, qs, ks, vs, gather_ks, gather_vs)``. Outputs are
    contiguous and scales have shape [1]. ``qs`` includes ``q_scale_factor``;
    this changes the attention logit scale, not Q's stored quantized values.
    Gather descales are ``max(ks/vs, 1e-6) * 2`` for cached-KV range headroom.

    Reads strided tensors directly, including V sliced from a K/V projection.
    Uses one launch for small tensors, otherwise two: partial amax reduction,
    then quantization with final reduction and scale preparation. No atomics,
    initialized workspace, contiguous copies, or host synchronization.
    All-zero/empty tensors use descale 1e-6 (before Q's scale adjustment), so
    FMHA can safely scale masked logits. Their FP8 values are zero and the
    cached-gather descales retain the existing 2e-6 floor.
    """
    tensors = (q, k, v)
    if any(x.ndim != 3 or x.shape[1] == 0 or x.shape[2] == 0 for x in tensors):
        raise ValueError("Q/K/V must have shape [tokens, nonzero heads, nonzero dim]")
    if not q.is_cuda or any(
        x.device != q.device or x.dtype != q.dtype for x in tensors
    ):
        raise ValueError("Q/K/V must have the same GPU device and dtype")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Q/K/V must be BF16 or FP16")
    if not math.isfinite(q_scale_factor) or q_scale_factor <= 0:
        raise ValueError("q_scale_factor must be positive and finite")
    shapes = tuple(tuple(x.shape[1:]) for x in tensors)
    strides = tuple(x.stride() for x in tensors)
    outputs = tuple(
        torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
        for x in tensors
    )
    scales = torch.empty(5, device=q.device, dtype=torch.float32)
    sizes = tuple(x.numel() for x in tensors)
    n = max(sizes)
    single_pass = n <= 8192
    block = (
        max(256, triton.next_power_of_2(n))
        if single_pass
        else (8192 if n >= 4 * 1024 * 1024 else 4096)
    )
    parts = min(128, triton.next_power_of_2(triton.cdiv(n, block))) if n else 1
    partial = torch.empty((3, parts), device=q.device, dtype=torch.float32)
    if not single_pass:
        _fused_qkv_amax[(parts, 3)](
            *tensors, partial, *sizes, shapes, strides, parts, block, num_warps=4
        )
    _fused_qkv_quant[(max(1, triton.cdiv(n, block)), 3)](
        *tensors,
        *outputs,
        partial,
        scales,
        *sizes,
        shapes,
        strides,
        parts,
        block,
        q_scale_factor,
        single_pass,
        num_warps=4,
    )
    return (*outputs, *(scales[i : i + 1] for i in range(5)))
