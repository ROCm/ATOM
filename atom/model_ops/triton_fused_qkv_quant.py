# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused QKV and KV dynamic FP8 quantization and MLA descale preparation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _load_tensor(
    X,
    offsets,
    N,
    SHAPE: tl.constexpr,
    STRIDE: tl.constexpr,
    Rope=None,
    ROPE_DIM: tl.constexpr = 0,
    ROPE_STRIDE: tl.constexpr = (0, 0, 0),
):
    heads, dim = SHAPE
    if ROPE_DIM:
        # K is logically cat(k_nope, k_rope), without a BF16 intermediate.
        token = offsets // (heads * dim)
        head = offsets // dim % heads
        col = offsets % dim
        nope_dim = dim - ROPE_DIM
        nope = tl.load(
            X + token * STRIDE[0] + head * STRIDE[1] + col * STRIDE[2],
            (offsets < N) & (col < nope_dim),
            0,
        ).to(tl.float32)
        rope = tl.load(
            Rope
            + token * ROPE_STRIDE[0]
            + head * ROPE_STRIDE[1]
            + (col - nope_dim) * ROPE_STRIDE[2],
            (offsets < N) & (col >= nope_dim),
            0,
        ).to(tl.float32)
        return tl.where(col < nope_dim, nope, rope)
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
    Rope=None,
    ROPE_DIM: tl.constexpr = 0,
    ROPE_STRIDE: tl.constexpr = (0, 0, 0),
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for step in range(tl.cdiv(N, PARTS * BLOCK)):
        x = _load_tensor(
            X,
            offsets + step * PARTS * BLOCK,
            N,
            SHAPE,
            STRIDE,
            Rope,
            ROPE_DIM,
            ROPE_STRIDE,
        )
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
    K_ROPE=None,
    ROPE_DIM: tl.constexpr = 0,
    ROPE_STRIDE: tl.constexpr = (0, 0, 0),
):
    kind = tl.program_id(1)
    if kind == 0:
        _partial_amax(Q, Partial, NQ, SHAPES[0], STRIDES[0], PARTS, BLOCK)
    elif kind == 1:
        _partial_amax(
            K,
            Partial + PARTS,
            NK,
            SHAPES[1],
            STRIDES[1],
            PARTS,
            BLOCK,
            K_ROPE,
            ROPE_DIM,
            ROPE_STRIDE,
        )
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
    GATHER: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
    Rope=None,
    ROPE_DIM: tl.constexpr = 0,
    ROPE_STRIDE: tl.constexpr = (0, 0, 0),
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = _load_tensor(X, offsets, N, SHAPE, STRIDE, Rope, ROPE_DIM, ROPE_STRIDE)
    if SINGLE_PASS:
        amax = tl.max(tl.abs(x), 0)
    else:
        amax = tl.max(tl.load(Partial + tl.arange(0, PARTS)), 0)
    # Match AITER for nonzero inputs; positive zero-input descales avoid FMHA NaNs.
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
        tl.store(Scale, descale)
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
    SINGLE_PASS: tl.constexpr,
    K_ROPE=None,
    ROPE_DIM: tl.constexpr = 0,
    ROPE_STRIDE: tl.constexpr = (0, 0, 0),
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
            True,
            SINGLE_PASS,
            K_ROPE,
            ROPE_DIM,
            ROPE_STRIDE,
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
            True,
            SINGLE_PASS,
        )


def fused_qkv_per_tensor_quant(q, k, v, *, k_rope=None):
    """Quantize 3-D Q/K/V to E4M3 with independent per-tensor FP32 descales.

    Returns ``(q8, k8, v8, qs, ks, vs, gather_ks, gather_vs)`` with contiguous
    outputs and shape-[1] scales. Gather descales are ``max(ks/vs, 1e-6) * 2``.
    Reads strided inputs in one launch for small tensors, otherwise two.
    Zero/empty inputs use descale 1e-6 and gather descale 2e-6.

    With ``k_rope``, ``k`` is the NoPE part and K8 is the quantized logical
    concatenation along the last dimension. K's amax includes both parts.
    ``k_rope`` may have one head (broadcast) or the same head count as K.
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
    rope_dim = 0
    rope_stride = (0, 0, 0)
    k_shape = tuple(k.shape)
    if k_rope is not None:
        if (
            k_rope.ndim != 3
            or k_rope.shape[0] != k.shape[0]
            or k_rope.shape[1] not in (1, k.shape[1])
            or k_rope.shape[2] == 0
        ):
            raise ValueError("K RoPE must match K tokens and have one or K heads")
        if k_rope.device != k.device or k_rope.dtype != k.dtype:
            raise ValueError("K RoPE must have the same GPU device and dtype as K")
        rope_dim = k_rope.shape[2]
        rope_stride = (
            k_rope.stride(0),
            0 if k_rope.shape[1] == 1 else k_rope.stride(1),
            k_rope.stride(2),
        )
        k_shape = (*k.shape[:2], k.shape[2] + rope_dim)
    output_shapes = (tuple(q.shape), k_shape, tuple(v.shape))
    shapes = tuple(shape[1:] for shape in output_shapes)
    strides = tuple(x.stride() for x in tensors)
    outputs = tuple(
        torch.empty(shape, device=q.device, dtype=torch.float8_e4m3fn)
        for shape in output_shapes
    )
    scales = torch.empty(5, device=q.device, dtype=torch.float32)
    sizes = tuple(x.numel() for x in outputs)
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
            *tensors,
            partial,
            *sizes,
            shapes,
            strides,
            parts,
            block,
            k_rope,
            rope_dim,
            rope_stride,
            num_warps=4,
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
        single_pass,
        k_rope,
        rope_dim,
        rope_stride,
        num_warps=4,
    )
    return (*outputs, *(scales[i : i + 1] for i in range(5)))


@triton.jit(do_not_specialize=["TOKENS"])
def _kv_amax(
    K,
    V,
    Partial,
    TOKENS: tl.int64,
    K_ROW: tl.constexpr,
    V_ROW: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    part = tl.program_id(0)
    kind = tl.program_id(1)
    X = K if kind == 0 else V
    # Fixed row widths preserve vector alignment for every token count.
    row = K_ROW if kind == 0 else V_ROW
    n = TOKENS * row
    acc = tl.full((BLOCK,), 0, tl.float32)
    # The small-input configuration benefits from overlapping two loads;
    # larger reductions favor the occupancy of the compact loop.
    for start in tl.range(
        part * BLOCK, n, PARTS * BLOCK, loop_unroll_factor=2 if PARTS == 256 else 1
    ):
        index = tl.multiple_of(start, BLOCK) + tl.arange(0, BLOCK)
        x = tl.load(X + index, index < n, 0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(x))
    tl.store(Partial + kind * PARTS + part, tl.max(acc, 0))


@triton.jit(do_not_specialize=["TOKENS"])
def _kv_quant(
    K,
    V,
    K8,
    V8,
    Partial,
    Scales,
    TOKENS: tl.int64,
    K_ROW: tl.constexpr,
    V_ROW: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    worker = tl.program_id(0)
    kind = tl.program_id(1)
    X = K if kind == 0 else V
    Y = K8 if kind == 0 else V8
    # Fixed row widths preserve vector alignment for every token count.
    row = K_ROW if kind == 0 else V_ROW
    n = TOKENS * row
    offsets = worker * BLOCK + tl.arange(0, BLOCK)
    if SINGLE_PASS:
        x = tl.load(X + offsets, offsets < n, 0).to(tl.float32)
        amax = tl.max(tl.abs(x), 0)
    else:
        amax = tl.max(tl.load(Partial + kind * PARTS + tl.arange(0, PARTS)), 0)
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
    if worker == 0:
        tl.store(Scales + kind, descale)
    if SINGLE_PASS:
        value = tl.minimum(tl.maximum(x * inv, -448.0), 448.0)
        tl.store(Y + offsets, value, offsets < n)
    else:
        for start in range(worker * BLOCK, n, tl.num_programs(0) * BLOCK):
            index = tl.multiple_of(start, BLOCK) + tl.arange(0, BLOCK)
            x = tl.load(X + index, index < n, 0).to(tl.float32)
            value = tl.minimum(tl.maximum(x * inv, -448.0), 448.0)
            tl.store(Y + index, value, index < n)


def _kv_config(n):
    if n <= 8192:
        return 1, max(256, triton.next_power_of_2(n)), 1, 4
    # gfx950: preserve the small-input launch cost, then increase occupancy
    # and vector width as the two-pass input traffic exceeds cache capacity.
    if n <= 2**24:
        return 256, 4096, min(1024, triton.cdiv(n, 4096)), 4
    if n <= 2**26:
        return 512, 4096, min(2048, triton.cdiv(n, 4096)), 4
    return 512, 8192, min(4096, triton.cdiv(n, 8192)), 8


def fused_kv_per_tensor_quant(k: torch.Tensor, v: torch.Tensor):
    """Return ``(k8, v8, k_descale, v_descale)`` for expanded cached K/V.

    Inputs are contiguous BF16/FP16 ``[tokens, heads, dim]`` tensors on the
    same GPU, with matching token/head counts. Each output uses its own amax
    over the complete input tensor, including K's RoPE columns. Outputs are
    contiguous E4M3; descales are FP32 shape-[1] device tensors, compatible
    with FlyDSL FMHA. Zero/empty inputs use a positive descale of 1e-6.

    A fused partial-amax pass is followed by a fused reduce-and-quantize pass.
    The kernel boundary provides the global reduction barrier. Every partial
    and scale is overwritten, so no initialization or atomics are required.
    Inputs fitting in one workgroup per tensor use a single launch instead.
    All work runs on the current stream and supports CUDA/HIP graph capture.
    """
    if k.ndim != 3 or v.ndim != 3:
        raise ValueError("K/V must have shape [tokens, heads, dim]")
    if k.shape[:2] != v.shape[:2] or min(*k.shape[1:], *v.shape[1:]) <= 0:
        raise ValueError("K/V must have matching tokens/heads and nonzero heads/dim")
    if not k.is_cuda or v.device != k.device or v.dtype != k.dtype:
        raise ValueError("K/V must have the same GPU device and dtype")
    if k.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("K/V must be BF16 or FP16")
    if not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("K/V must be contiguous")

    tokens = k.shape[0]
    k_row, v_row = k.shape[1] * k.shape[2], v.shape[1] * v.shape[2]
    n = tokens * max(k_row, v_row)
    parts, block, workers, warps = _kv_config(n)
    k8 = torch.empty(k.shape, dtype=torch.float8_e4m3fn, device=k.device)
    v8 = torch.empty(v.shape, dtype=torch.float8_e4m3fn, device=v.device)
    scales = torch.empty(2, dtype=torch.float32, device=k.device)
    partial = None
    single = n <= 8192
    if not single:
        partial = torch.empty((2, parts), dtype=torch.float32, device=k.device)
        _kv_amax[(parts, 2)](
            k, v, partial, tokens, k_row, v_row, parts, block, num_warps=warps
        )
    _kv_quant[(workers, 2)](
        k,
        v,
        k8,
        v8,
        partial,
        scales,
        tokens,
        k_row,
        v_row,
        parts,
        block,
        single,
        num_warps=warps,
    )
    return k8, v8, scales[:1], scales[1:]
