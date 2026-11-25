# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The DSpark draft's context K/V for all its layers in one MXFP8 GEMM.

Per target forward every draft layer projects the same context through its
fused QKV linear and keeps only K and V: per layer one MXFP8 quantization of
the same input and one GEMM over all of Q, K and V (Q is 8x the K/V bytes).
Here the K/V rows of every layer's preshuffled weight (and its 32-row-grouped
E8M0 scale) are concatenated once into one weight, the input is quantized
once, and one GEMM writes every layer's K/V.

Numerics: bit-identical per output column to the per-layer path. Each column of
a GEMM is the same K reduction whichever other columns share the launch, as long
as the kernel (tile, split-K) is the same; the launch is pinned to the kernel
the tuned table picks for the FULL fused-QKV shape, which is what the per-layer
path runs. Rows are taken in whole 32-row groups, so the 16x16 weight preshuffle
and the 32x4 scale shuffle stay intact.
"""

import functools
import itertools
import re

import torch
import triton
import triton.language as tl

# Decode widths: the K/V-only launch is checked bit-exact up to here; a prefill
# keeps the per-layer path (its kernels tile N in multiples a 5-layer K/V width
# need not be).
MAX_ROWS = 64


@functools.cache
def _kernel_for(m: int, n: int, k: int, n_kv: int):
    """The tuned kernel name the per-layer fused-QKV GEMM runs at (m, n, k), if
    its N tiling also covers ``n_kv`` columns exactly; else None."""
    from aiter.ops.gemm_op_a8w8 import get_mxscale_bpreshuffle_config

    cfg = get_mxscale_bpreshuffle_config(m, n, k, "1x32")
    if cfg is None:
        return None
    name = str(cfg["kernelName"])
    tile = re.search(r"_t\d+x(\d+)x\d+_", name)
    cn = re.search(r"_cn(\d+)", name)
    if tile is None or n_kv % (int(tile.group(1)) * (int(cn.group(1)) if cn else 1)):
        return None
    return name


class ContextKV:
    """K/V rows ``[q, q + 2 * kv)`` of each layer's fused-QKV linear, as one GEMM."""

    def __init__(self, linears, q_size: int, kv_size: int):
        w = [lin.weight.data for lin in linears]
        s = [lin.weight_scale.data for lin in linears]
        lo, hi = q_size, q_size + 2 * kv_size
        if lo % 32 or hi % 32:
            raise ValueError("K/V rows must be whole 32-row groups")
        self.n_full, self.k = w[0].shape
        self.weight = torch.cat([x[lo:hi] for x in w]).contiguous()
        self.scale = torch.cat([x[lo:hi] for x in s]).contiguous()
        self.width = hi - lo

    def __call__(self, x: torch.Tensor) -> "torch.Tensor | None":
        """``[M, layers * 2 * kv]`` bf16, or None when the full shape has no
        tuned kernel to pin (the caller keeps the per-layer path)."""
        from aiter.ops.flydsl.batched_gemm_a8w8 import run_bmm_a8w8_mxfp8

        from atom.model_ops.blockscale import quantize_fp8

        m = x.shape[0]
        if m > MAX_ROWS:
            return None
        name = _kernel_for(m, self.n_full, self.k, self.weight.shape[0])
        if name is None:
            return None
        xq, xs = quantize_fp8(x)
        n = self.weight.shape[0]
        y = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
        e8m0 = torch.float8_e8m0fnu
        run_bmm_a8w8_mxfp8(
            xq.view(m, 1, self.k),
            self.weight.view(1, n, self.k),
            xs.view(e8m0).view(m, 1, xs.shape[-1]),
            self.scale.view(e8m0).view(1, *self.scale.shape),
            y.view(m, 1, n),
            kernel_name=name,
            x_scale_transposed=False,
        )
        return y


# ---- every layer's context K/V into its cache, one launch --------------------
# `_fused_qkv_norm_rope_cache_kernel`'s K/V half (triton_fused_qkv_norm_rope_cache.py:
# Gemma RMSNorm + neox partial RoPE on K, per-token FP8 for K and V, the SHUFFLE
# cache writes), with a second grid axis over the layers: layer l reads its K/V
# columns at `l * LAYER_KV` of the fused row, its K norm at `l * D` of the stacked
# weights, and writes its caches at a fixed element delta from layer 0's (each
# layer's views of one pool). Same arithmetic in the same order as that kernel.


@triton.jit
def _ctx_kv_write_kernel(
    kv_ptr,  # [T, layers * 2 * kv_size] bf16, the ContextKV output
    kv_stride_t,
    kw_ptr,  # [layers, D] stacked K norm weights
    cos_cache_ptr,
    sin_cache_ptr,
    cos_sin_stride_pos,
    pos_ptr,
    k_cache_ptr,  # layer 0's views; layer l at + l * delta
    v_cache_ptr,
    kc_stride_block,
    kc_stride_head,
    kc_stride_dx,
    kc_stride_slot,
    kc_stride_x,
    vc_stride_block,
    vc_stride_head,
    vc_stride_sc,
    vc_stride_d,
    vc_stride_x,
    k_scale_ptr,
    v_scale_ptr,
    ks_stride_block,
    ks_stride_head,
    vs_stride_block,
    vs_stride_head,
    slot_mapping_ptr,
    kc_layer_delta,
    vc_layer_delta,
    ks_layer_delta,
    vs_layer_delta,
    num_kv_heads: tl.constexpr,
    kv_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    X_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    ROTARY_DIM_HALF: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_AMAX: tl.constexpr,
):
    pid = tl.program_id(0)
    layer = tl.program_id(1)
    token_id = pid // num_kv_heads
    kv_h = pid % num_kv_heads
    d_offs = tl.arange(0, BLOCK_D)
    k_ptr = kv_ptr + layer * (2 * kv_size)
    v_ptr = k_ptr + kv_size
    kw_l = kw_ptr + layer * BLOCK_D
    k_cache_l = k_cache_ptr + layer.to(tl.int64) * kc_layer_delta
    v_cache_l = v_cache_ptr + layer.to(tl.int64) * vc_layer_delta

    k_in_offset = token_id * kv_stride_t + kv_h * BLOCK_D
    k = tl.load(k_ptr + k_in_offset + d_offs).to(tl.float32)
    k_variance = tl.sum(k * k, axis=0) / BLOCK_D
    k_normed = k * tl.math.rsqrt(k_variance + eps)
    kw = tl.load(kw_l + d_offs).to(tl.float32)
    k_normed = k_normed * (1.0 + kw)

    rot_mask = d_offs < ROTARY_DIM
    first_half_mask = d_offs < ROTARY_DIM_HALF
    d_cos_idx = tl.where(
        first_half_mask,
        d_offs,
        tl.where(d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, tl.zeros_like(d_offs)),
    )
    pos = tl.load(pos_ptr + token_id)
    cos_base = pos * cos_sin_stride_pos
    cos_vals = tl.load(
        cos_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=1.0
    ).to(tl.float32)
    sin_vals = tl.load(
        sin_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=0.0
    ).to(tl.float32)
    gather_idx = tl.where(
        first_half_mask,
        d_offs + ROTARY_DIM_HALF,
        tl.where(d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, d_offs),
    )
    k_gathered_raw = tl.load(k_ptr + k_in_offset + gather_idx).to(tl.float32)
    kw_gathered = tl.load(kw_l + gather_idx).to(tl.float32)
    k_gathered_normed = (
        k_gathered_raw * tl.math.rsqrt(k_variance + eps) * (1.0 + kw_gathered)
    )
    k_rot = tl.where(first_half_mask, -k_gathered_normed, k_gathered_normed)
    k_rot = tl.where(rot_mask, k_rot, 0.0)
    k_roped = k_normed * cos_vals + k_rot * sin_vals

    v_in_offset = token_id * kv_stride_t + kv_h * BLOCK_D
    v = tl.load(v_ptr + v_in_offset + d_offs)

    slot = tl.load(slot_mapping_ptr + token_id).to(tl.int64)
    if slot >= 0:
        block_idx = slot // BLOCK_SIZE
        slot_in_block = slot % BLOCK_SIZE
        if IS_FP8:
            k_abs_max = tl.max(tl.abs(k_roped), axis=0)
            k_scale = k_abs_max / FP8_AMAX
            k_scale = tl.where(k_scale == 0.0, 1.0, k_scale)
            k_quant = (k_roped / k_scale).to(k_cache_ptr.dtype.element_ty)
            tl.store(
                k_scale_ptr
                + layer.to(tl.int64) * ks_layer_delta
                + block_idx * ks_stride_block
                + kv_h * ks_stride_head
                + slot_in_block,
                k_scale,
            )
        else:
            k_quant = k_roped.to(k_cache_ptr.dtype.element_ty)
        k_quant_2d = tl.reshape(k_quant, (BLOCK_D // X_SIZE, X_SIZE))
        dx_offs = tl.arange(0, BLOCK_D // X_SIZE).to(tl.int64)
        x_offs = tl.arange(0, X_SIZE).to(tl.int64)
        k_cache_ptrs = (
            k_cache_l
            + block_idx * kc_stride_block
            + kv_h * kc_stride_head
            + dx_offs[:, None] * kc_stride_dx
            + slot_in_block * kc_stride_slot
            + x_offs[None, :] * kc_stride_x
        )
        tl.store(k_cache_ptrs, k_quant_2d)
        if IS_FP8:
            v_f32 = v.to(tl.float32)
            v_abs_max = tl.max(tl.abs(v_f32), axis=0)
            v_scale = v_abs_max / FP8_AMAX
            v_scale = tl.where(v_scale == 0.0, 1.0, v_scale)
            v_quant = (v_f32 / v_scale).to(v_cache_ptr.dtype.element_ty)
            tl.store(
                v_scale_ptr
                + layer.to(tl.int64) * vs_layer_delta
                + block_idx * vs_stride_block
                + kv_h * vs_stride_head
                + slot_in_block,
                v_scale,
            )
        else:
            v_quant = v.to(v_cache_ptr.dtype.element_ty)
        slot_chunk = slot_in_block // X_SIZE
        x_off = slot_in_block % X_SIZE
        v_cache_ptrs = (
            v_cache_l
            + block_idx * vc_stride_block
            + kv_h * vc_stride_head
            + slot_chunk * vc_stride_sc
            + d_offs.to(tl.int64) * vc_stride_d
            + x_off * vc_stride_x
        )
        tl.store(v_cache_ptrs, v_quant)


def _layer_deltas(tensors) -> "int | None":
    """Element delta between consecutive tensors when it is one constant and the
    tensors are alike (dtype, shape, strides); else None."""
    t0 = tensors[0]
    if t0 is None:
        return 0 if all(t is None for t in tensors) else None
    es = t0.element_size()
    deltas = set()
    for a, b in itertools.pairwise(tensors):
        if b is None or (b.dtype, b.shape, b.stride()) != (
            t0.dtype,
            t0.shape,
            t0.stride(),
        ):
            return None
        d = b.data_ptr() - a.data_ptr()
        if d % es:
            return None
        deltas.add(d // es)
    if len(deltas) > 1:
        return None
    return deltas.pop() if deltas else 0


class ContextKVWrite:
    """Every layer's context K/V write (`write_context_kv_rows`) as one launch,
    when the layers' caches are one pool's equally spaced, alike views and the
    layers share the rotary tables and the norm epsilon; ``build`` returns None
    otherwise (the caller keeps one launch a layer)."""

    def __init__(self, attns, caches):
        a0 = attns[0]
        self.n = len(attns)
        self.kv_size = a0.kv_size
        self.num_kv_heads = a0.num_kv_heads
        self.head_dim = a0.head_dim
        self.kv_cache_dtype = a0.kv_cache_dtype
        # The per-layer write takes its epsilon from the q norm.
        self.eps = a0.q_norm.variance_epsilon
        rot = a0.rotary_emb
        self.rotary_dim = rot.rotary_dim
        self.cos = rot.cos_cache.squeeze(-2).squeeze(-2)
        self.sin = rot.sin_cache.squeeze(-2).squeeze(-2)
        self.kw = torch.stack([a.k_norm.weight.data for a in attns]).contiguous()
        self.k_cache, self.v_cache, self.k_scale, self.v_scale = caches[0]
        self.deltas = [_layer_deltas([c[i] for c in caches]) for i in range(4)]

    @classmethod
    def build(cls, attns, caches):
        if not caches or any(c is None for c in caches):
            return None
        a0 = attns[0]
        rot0 = a0.rotary_emb
        for a in attns[1:]:
            r = a.rotary_emb
            if (
                a.kv_size != a0.kv_size
                or a.num_kv_heads != a0.num_kv_heads
                or a.head_dim != a0.head_dim
                or a.kv_cache_dtype != a0.kv_cache_dtype
                or a.q_norm.variance_epsilon != a0.q_norm.variance_epsilon
                or r.rotary_dim != rot0.rotary_dim
                or not torch.equal(r.cos_cache, rot0.cos_cache)
                or not torch.equal(r.sin_cache, rot0.sin_cache)
            ):
                return None
        w = cls(attns, caches)
        return w if all(d is not None for d in w.deltas) else None

    def __call__(self, kv: torch.Tensor, positions, slot_mapping) -> None:
        T = kv.shape[0]
        if T == 0:
            return
        kc, vc, ks, vs = self.k_cache, self.v_cache, self.k_scale, self.v_scale
        is_fp8 = self.kv_cache_dtype == "fp8"
        _ctx_kv_write_kernel[(T * self.num_kv_heads, self.n)](
            kv,
            kv.stride(0),
            self.kw,
            self.cos,
            self.sin,
            self.cos.stride(0),
            positions,
            kc,
            vc,
            *kc.stride(),
            *vc.stride(),
            ks if ks is not None else kv,
            vs if vs is not None else kv,
            ks.stride(0) if ks is not None and ks.dim() >= 1 else 0,
            ks.stride(1) if ks is not None and ks.dim() >= 2 else 0,
            vs.stride(0) if vs is not None and vs.dim() >= 1 else 0,
            vs.stride(1) if vs is not None and vs.dim() >= 2 else 0,
            slot_mapping,
            *self.deltas,
            num_kv_heads=self.num_kv_heads,
            kv_size=self.kv_size,
            eps=self.eps,
            BLOCK_SIZE=kc.shape[3],
            X_SIZE=kc.shape[4],
            BLOCK_D=self.head_dim,
            ROTARY_DIM=self.rotary_dim,
            ROTARY_DIM_HALF=self.rotary_dim // 2,
            IS_FP8=is_fp8,
            FP8_AMAX=float(torch.finfo(kc.dtype).max),
        )
