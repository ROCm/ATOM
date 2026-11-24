# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""MiniMax-M3.1 DSpark: every draft layer's context K/V in one GEMM and one cache write,
bit-identical to the per-layer `M3DSparkAttention.write_context_kv`."""

import torch
import triton
import triton.language as tl

# A multiple of every N tile (16 to 256) the qkv GEMM's tuned rows run.
CTX_KV_ROW_PAD = 768


@triton.jit
def _ctx_kv_norm_rope_cache_kernel(
    kv_ptr,  # [T, >= layers * 2 * NKV * D]: per layer, K heads then V heads
    kv_stride_t,
    kw_ptr,  # [layers, D] k_norm weights
    cos_cache_ptr,
    sin_cache_ptr,
    cos_sin_stride_pos,
    pos_ptr,
    k_cache_ptr,  # layer 0's page-16 SHUFFLE caches; layer l's at the offsets' [l]
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    k_off_ptr,
    v_off_ptr,
    ks_off_ptr,
    vs_off_ptr,
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
    ks_stride_block,
    ks_stride_head,
    vs_stride_block,
    vs_stride_head,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    X_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    ROTARY_DIM_HALF: tl.constexpr,
    FP8_AMAX: tl.constexpr,
):
    # Grid (tokens, layers). Per head: `_fused_qkv_norm_rope_cache_kernel`'s KV
    # branch, unchanged.
    token_id = tl.program_id(0)
    layer = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + token_id).to(tl.int64)
    if slot >= 0:
        d_offs = tl.arange(0, BLOCK_D)
        rot_mask = d_offs < ROTARY_DIM
        first_half_mask = d_offs < ROTARY_DIM_HALF
        d_cos_idx = tl.where(
            first_half_mask,
            d_offs,
            tl.where(
                d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, tl.zeros_like(d_offs)
            ),
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
        block_idx = slot // BLOCK_SIZE
        slot_in_block = slot % BLOCK_SIZE
        slot_chunk = slot_in_block // X_SIZE
        x_off = slot_in_block % X_SIZE
        dx_offs = tl.arange(0, BLOCK_D // X_SIZE).to(tl.int64)
        x_offs = tl.arange(0, X_SIZE).to(tl.int64)
        k_off = tl.load(k_off_ptr + layer)
        v_off = tl.load(v_off_ptr + layer)
        ks_off = tl.load(ks_off_ptr + layer)
        vs_off = tl.load(vs_off_ptr + layer)
        row = token_id * kv_stride_t + layer * (2 * num_kv_heads * BLOCK_D)
        for kv_h in tl.static_range(num_kv_heads):
            k_in_offset = row + kv_h * BLOCK_D
            k = tl.load(kv_ptr + k_in_offset + d_offs).to(tl.float32)
            k_variance = tl.sum(k * k, axis=0) / BLOCK_D
            k_normed = k * tl.math.rsqrt(k_variance + eps)
            kw = tl.load(kw_ptr + layer * BLOCK_D + d_offs).to(tl.float32)
            k_normed = k_normed * (1.0 + kw)
            k_gathered_raw = tl.load(kv_ptr + k_in_offset + gather_idx).to(tl.float32)
            kw_gathered = tl.load(kw_ptr + layer * BLOCK_D + gather_idx).to(tl.float32)
            k_gathered_normed = (
                k_gathered_raw * tl.math.rsqrt(k_variance + eps) * (1.0 + kw_gathered)
            )
            k_rot = tl.where(first_half_mask, -k_gathered_normed, k_gathered_normed)
            k_rot = tl.where(rot_mask, k_rot, 0.0)
            k_roped = k_normed * cos_vals + k_rot * sin_vals

            v = tl.load(kv_ptr + row + (num_kv_heads + kv_h) * BLOCK_D + d_offs)

            k_abs_max = tl.max(tl.abs(k_roped), axis=0)
            k_scale = k_abs_max / FP8_AMAX
            k_scale = tl.where(k_scale == 0.0, 1.0, k_scale)
            k_quant = (k_roped / k_scale).to(k_cache_ptr.dtype.element_ty)
            tl.store(
                k_scale_ptr
                + ks_off
                + block_idx * ks_stride_block
                + kv_h * ks_stride_head
                + slot_in_block,
                k_scale,
            )
            k_quant_2d = tl.reshape(k_quant, (BLOCK_D // X_SIZE, X_SIZE))
            tl.store(
                k_cache_ptr
                + k_off
                + block_idx * kc_stride_block
                + kv_h * kc_stride_head
                + dx_offs[:, None] * kc_stride_dx
                + slot_in_block * kc_stride_slot
                + x_offs[None, :] * kc_stride_x,
                k_quant_2d,
            )

            v_f32 = v.to(tl.float32)
            v_abs_max = tl.max(tl.abs(v_f32), axis=0)
            v_scale = v_abs_max / FP8_AMAX
            v_scale = tl.where(v_scale == 0.0, 1.0, v_scale)
            v_quant = (v_f32 / v_scale).to(v_cache_ptr.dtype.element_ty)
            tl.store(
                v_scale_ptr
                + vs_off
                + block_idx * vs_stride_block
                + kv_h * vs_stride_head
                + slot_in_block,
                v_scale,
            )
            tl.store(
                v_cache_ptr
                + v_off
                + block_idx * vc_stride_block
                + kv_h * vc_stride_head
                + slot_chunk * vc_stride_sc
                + d_offs.to(tl.int64) * vc_stride_d
                + x_off * vc_stride_x,
                v_quant,
            )


def _element_offsets(tensors, base) -> torch.Tensor:
    """Each tensor's element offset from ``base``, whose allocation must hold them at or
    above it: Triton may address a pointer into <= 2 GiB with 32-bit offsets."""
    size = base.element_size()
    storage = base.untyped_storage().data_ptr()
    offs = []
    for t in tensors:
        delta = t.data_ptr() - base.data_ptr()
        if t.untyped_storage().data_ptr() != storage or delta < 0 or delta % size:
            raise ValueError(
                "context layers' caches are not views of layer 0's allocation "
                "at or above it"
            )
        offs.append(delta // size)
    return torch.tensor(offs, dtype=torch.int64, device=base.device)


class CtxKvWriter:
    """k_norm, RoPE and the fp8 SHUFFLE cache write of every layer's context K/V, in one
    launch; ``caches`` are each layer's page-16 ``(k, v, k_scale, v_scale)`` views."""

    def __init__(self, caches, k_norm_weights, rotary_emb, eps):
        layouts = {
            tuple((t.stride(), t.dtype) for t in layer_caches)
            for layer_caches in caches
        }
        if len(layouts) != 1:
            raise ValueError("context layers' caches differ in layout")
        self.offsets = [
            _element_offsets([c[i] for c in caches], caches[0][i]) for i in range(4)
        ]
        self.kw = torch.stack([w.detach() for w in k_norm_weights])
        k0 = caches[0][0]
        self.num_layers = len(caches)
        self.num_kv_heads = k0.shape[1]
        self.head_dim = k0.shape[2] * k0.shape[4]
        self.cos = rotary_emb.cos_cache.squeeze(-2).squeeze(-2)
        self.sin = rotary_emb.sin_cache.squeeze(-2).squeeze(-2)
        self.rotary_dim = rotary_emb.rotary_dim
        self.eps = eps
        self.fp8_amax = float(torch.finfo(k0.dtype).max)

    def __call__(self, kv, positions, slot_mapping, cache) -> None:
        """``kv`` holds each layer's K then V heads in its leading columns; ``cache`` is
        layer 0's live views, the others are reached through the offsets."""
        k0, v0, ks0, vs0 = cache
        _ctx_kv_norm_rope_cache_kernel[(kv.shape[0], self.num_layers)](
            kv,
            kv.stride(0),
            self.kw,
            self.cos,
            self.sin,
            self.cos.stride(0),
            positions,
            k0,
            v0,
            ks0,
            vs0,
            *self.offsets,
            *k0.stride(),
            *v0.stride(),
            ks0.stride(0),
            ks0.stride(1),
            vs0.stride(0),
            vs0.stride(1),
            slot_mapping,
            num_kv_heads=self.num_kv_heads,
            eps=self.eps,
            BLOCK_SIZE=k0.shape[3],
            X_SIZE=k0.shape[4],
            BLOCK_D=self.head_dim,
            ROTARY_DIM=self.rotary_dim,
            ROTARY_DIM_HALF=self.rotary_dim // 2,
            FP8_AMAX=self.fp8_amax,
        )


def stack_kv_rows(weights, scales, rows: slice):
    """Every layer's ``rows`` of its preshuffled qkv weight and 1x32 scale, stacked and
    padded to a multiple of ``CTX_KV_ROW_PAD`` rows (zero weights, unit scales)."""
    w0, s0 = weights[0], scales[0]
    pad = -(len(weights) * (rows.stop - rows.start)) % CTX_KV_ROW_PAD
    weight = torch.cat([w[rows] for w in weights] + [w0.new_zeros(pad, w0.shape[1])])
    # e8m0 127 is 1.0; the padded rows' outputs are never read.
    one = torch.full((pad, s0.shape[1]), 127, dtype=torch.uint8, device=s0.device)
    scale = torch.cat([s[rows] for s in scales] + [one.view(s0.dtype)])
    return weight, scale


def ctx_kv_gemm(x, weight, weight_scale, qkv_rows: int):
    """``x @ weight.T`` on the kernel the ``[qkv_rows, K]`` qkv GEMM's tuned row runs at
    this M, so each column is that GEMM's bit for bit; None where it has no such row."""
    from aiter.ops.flydsl.batched_gemm_a8w8 import run_bmm_a8w8_mxfp8
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
        run_gemm_a8w8_mxfp8_gfx950,
    )
    from aiter.ops.gemm_op_a8w8 import get_mxscale_bpreshuffle_config

    from atom.model_ops.blockscale import quantize_fp8

    m, k = x.shape
    row = get_mxscale_bpreshuffle_config(m, qkv_rows, k, "1x32")
    if (
        row is None
        or row["libtype"] != "flydsl"
        or row["kernelId"] not in ("bmm", "mxfp8")
    ):
        return None
    n = weight.shape[0]
    xq, xs = quantize_fp8(x)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    name = str(row["kernelName"])
    if row["kernelId"] == "mxfp8":
        run_gemm_a8w8_mxfp8_gfx950(xq, weight, xs, weight_scale, out, name, "1x32")
    else:
        run_bmm_a8w8_mxfp8(
            xq.view(m, 1, k),
            weight.view(1, n, k),
            xs.view(m, 1, xs.shape[-1]),
            weight_scale.view(1, *weight_scale.shape),
            out.view(m, 1, n),
            kernel_name=name,
        )
    return out
