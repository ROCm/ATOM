# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MiniMax-M3.1 decode QKV front chain (``ATOM_M3_QKV_CHAIN=1``, default off).

In pure-decode steps of ``MIN_ROWS``..``MAX_ROWS`` rows (TP4, NVFP4 K/V and index caches) a sparse attention layer's
``qkv_proj`` runs only the MXFP8 activation quantizer (codes in the decode GEMM's (16, 16) tile layout, AITER-shuffled
E8M0 scales), and the attention op's ``rope_cache`` runs aiter's one-launch ``qkv_chain_cooperative_insert`` (QKV MXFP8
GEMM + q / k / index RMSNorm + partial RoPE + NVFP4 K / V / index-K insert) on the parked codes. Both decisions are taken
at run time inside the two custom ops (concrete row count and forward context) by the same predicate ``take``, so the
compiled graph is the same for every row count; the codes travel from the linear to the attention op through
a per-thread hand-off slot (same forward, same stream, consumed by the next attention op of the same layer).

Techniques and the kernel's numerics (bitwise the unfused split-3 chain) are amd-kernels'
``mxfp8_qkv_norm_rope_nvfp4_insert``, ported into aiter (aiter.ops.flydsl.kernels.qkv_chain).
"""

import functools
import importlib.util
import threading

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.utils import envs

N, K = 2560, 6144
MIN_ROWS, MAX_ROWS = 17, 64

# (rows, codes, scales, weight, chain_scale) of the last qkv_front that took the chain, per thread: TBO decode runs
# its two micro-batches in two threads, each doing linear -> attention for its own layer.
_STATE = threading.local()


def _get():
    return getattr(_STATE, "pending", None)


def _set(value):
    _STATE.pending = value


def enabled() -> bool:
    return bool(envs.ATOM_M3_QKV_CHAIN) and available()


@functools.cache
def available() -> bool:
    """aiter has the chain kernel and its activation quantizer."""
    try:
        return all(
            importlib.util.find_spec(m) is not None
            for m in (
                "aiter.ops.flydsl.kernels.qkv_chain",
                "aiter.ops.flydsl.kernels.mxfp8_quant_shuffled",
            )
        )
    except ImportError:
        return False


def layer_ok(
    num_heads,
    num_kv_heads,
    num_idx_heads,
    rotary_dim,
    kv_dtype,
    index_dtype,
    skip_index_topk,
    per_token_fp8_out,
):
    """Static (init-time) eligibility of a sparse attention layer: the kernel's TP4 head layout and NVFP4 caches."""
    return (
        enabled()
        and (num_heads, num_kv_heads, num_idx_heads, rotary_dim) == (16, 1, 1, 64)
        and kv_dtype == "nvfp4"
        and index_dtype == "nvfp4"
        and not skip_index_topk
        and not per_token_fp8_out
    )


def linear_ok(linear) -> bool:
    """Static eligibility of the qkv linear's weight path: a native group32 (1x32) MXFP8 weight with its own FP8 GEMM
    (``_qkv_chain_scale`` exists only on that path); online-requantized, per-token, 32x32, MXFP4 or BF16-fallback
    weights keep their existing linear."""
    return (
        getattr(linear, "native_a8_group_rows", None) == 1
        and getattr(linear, "params_dtype", None) == torch.float8_e4m3fn
        and not getattr(linear, "runs_bf16", False)
        and getattr(linear, "bias", None) is None
    )


def take(m: int) -> bool:
    """Run time: this forward's qkv linear + insert take the chain.

    Never inside a token graph (TR_HYBRID decode, the level-3 default): replay runs the captured quantizer without
    the Python hand-off and the attention op (an eager region) would read the uncomputed QKV.
    """
    if not (MIN_ROWS <= m <= MAX_ROWS):
        return False
    from atom.utils.forward_context import get_forward_context

    c = get_forward_context().context
    if c is None or c.is_prefill or c.is_dummy_run:
        return False
    mode = getattr(c, "forward_mode", None)
    return not (mode is not None and getattr(mode, "token_graph", False))


def chain_weight_scale(weight_scale: torch.Tensor) -> torch.Tensor:
    """Row-major E8M0 weight scale [2560, 192] -> AITER's shuffled layout the chain's GEMM reads."""
    from aiter import dtypes
    from aiter.ops.shuffle import shuffle_scale_a16w4

    return shuffle_scale_a16w4(
        weight_scale.view(dtypes.fp8_e8m0).contiguous(), 1, False
    )


def _front_fake(x, weight, weight_scale, chain_scale, weight_preshuffled):
    return torch.empty(
        (*x.shape[:-1], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )


@torch_compile_guard(gen_fake=_front_fake, mutates_args=[])
def qkv_front(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    chain_scale: torch.Tensor,
    weight_preshuffled: bool,
) -> torch.Tensor:
    """The qkv linear of a chain-eligible layer. Chain rows: MXFP8 quant only, codes parked in the hand-off slot, the returned
    qkv is NOT computed (the attention op's chain never reads it). Other rows: today's linear.
    """
    x2 = x.reshape(-1, K)
    m = x2.shape[0]
    if weight_preshuffled and take(m):
        from aiter.ops.flydsl.kernels.mxfp8_quant_shuffled import (
            mxfp8_quantize_shuffled,
        )

        codes, scales = mxfp8_quantize_shuffled(x2, 16 * ((m + 15) // 16))
        _set((m, codes, scales, weight, chain_scale))
        return torch.empty((*x.shape[:-1], N), dtype=torch.bfloat16, device=x.device)
    _set(None)
    from atom.model_ops.blockscale import native_quant_linear

    return native_quant_linear(
        x,
        weight,
        weight_scale,
        x_scale=None,
        weight_group_rows=1,
        weight_preshuffled=weight_preshuffled,
        dtype=torch.bfloat16,
    )


def pop(m: int):
    """The attention op's side: the parked codes of this layer's linear (or None = today's path)."""
    p = _get()
    _set(None)
    if p is None:
        return None
    assert p[0] == m, f"qkv_chain: parked codes for {p[0]} rows, attention has {m}"
    return p


def run_chain(
    pend,
    q_out,
    index_q_out,
    q_norm_w,
    k_norm_w,
    iq_norm_w,
    ik_norm_w,
    cos_sin_cache,
    positions,
    slot_mapping,
    kv_pages,
    index_pages,
    k_scale,
    v_scale,
    num_heads,
    num_kv_heads,
    num_index_heads,
    eps,
):
    from aiter.ops.flydsl.kernels.qkv_chain import qkv_chain_cooperative_insert

    m, codes, scales, weight, chain_scale = pend
    qkv_chain_cooperative_insert(
        codes.view(torch.float8_e4m3fn),
        scales,
        weight,
        chain_scale,
        m,
        q_norm_w,
        k_norm_w,
        iq_norm_w,
        ik_norm_w,
        cos_sin_cache,
        positions,
        slot_mapping,
        slot_mapping,
        kv_pages,
        index_pages,
        k_scale,
        v_scale,
        q_out,
        index_q_out,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_index_heads=num_index_heads,
        eps=eps,
    )


def warm(weight: torch.Tensor, chain_scale: torch.Tensor) -> None:
    """Eager calls at load (QKV linear prepare): compile the chain's row tiers and create its counter pool outside graph
    capture (a first call inside capture raises). Every slot -1: no page is written."""
    from aiter.ops.flydsl.kernels.mxfp8_quant_shuffled import mxfp8_quantize_shuffled
    from aiter.ops.flydsl.kernels.qkv_chain import qkv_chain_cooperative_insert

    dev = weight.device
    nw = torch.zeros(128, dtype=torch.bfloat16, device=dev)
    cs = torch.zeros((128, 64), dtype=torch.bfloat16, device=dev)
    kv = torch.zeros((1, 2, 128, 72), dtype=torch.uint8, device=dev)
    idx = torch.zeros((1, 128, 72), dtype=torch.uint8, device=dev)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    for m in (MIN_ROWS, 33, 49):
        codes, scales = mxfp8_quantize_shuffled(
            torch.zeros((m, K), dtype=torch.bfloat16, device=dev), 16 * ((m + 15) // 16)
        )
        pos = torch.zeros(m, dtype=torch.int64, device=dev)
        skip = torch.full((m,), -1, dtype=torch.int64, device=dev)
        q = torch.empty((m, 2048), dtype=torch.bfloat16, device=dev)
        iq = torch.empty((m, 128), dtype=torch.float8_e4m3fn, device=dev)
        qkv_chain_cooperative_insert(
            codes.view(torch.float8_e4m3fn),
            scales,
            weight,
            chain_scale,
            m,
            nw,
            nw,
            nw,
            nw,
            cs,
            pos,
            skip,
            skip,
            kv,
            idx,
            one,
            one,
            q,
            iq,
            num_heads=16,
            num_kv_heads=1,
            num_index_heads=1,
            eps=1e-6,
        )
    torch.cuda.synchronize(dev)
