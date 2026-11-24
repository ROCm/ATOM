# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache + launch wrapper of the fused norm / RoPE / NVFP4 insert."""

import functools
import struct

import torch

from . import kernel as _kernel
from . import kernel_q16_kv1_iq1_streaming as _kernel_streaming
from . import kernel_prefill_q64_kv4_iq4 as _kernel_prefill
from . import splitk_reduce_kernel_q64_kv4_iq4 as _reduce_kernel


def _run_compiled(exe, *args):
    """The first call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


# (q heads, KV heads, index heads, rotary dim) of the layer at TP4, the one head set the entries take
HEADS = (16, 1, 1, 64)


def insert_blocker(
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    rotary_dim: int,
    *,
    head_dim: int = _kernel.D,
    index_head_dim: int = _kernel.D,
    neox: bool = True,
    index_q_dtype: torch.dtype = torch.float8_e4m3fn,
    cos_sin_cache: torch.Tensor | None = None,
) -> str | None:
    """Why a sparse attention layer (this rank's heads) does not fit the insert and its one-call QKV forms (None: it
    fits): 16 query, 1 KV and 1 index head of 128 dims (TP4), NeoX RoPE on dims 0-63, an e4m3 index-q, a contiguous
    bf16 cos / sin cache [rows, 64]. The entries assert the same."""
    if (num_heads, num_kv_heads, num_index_heads) != HEADS[:3]:
        return (
            f"{num_heads} query / {num_kv_heads} KV / {num_index_heads} index heads per rank "
            f"(q{HEADS[0]} kv{HEADS[1]} iq{HEADS[2]}, TP4)"
        )
    if (head_dim, index_head_dim) != (_kernel.D, _kernel.D):
        return f"head dim {head_dim}, index head dim {index_head_dim} ({_kernel.D})"
    if rotary_dim != HEADS[3] or not neox:
        return f"RoPE on {rotary_dim} dims{'' if neox else ', not NeoX'} (NeoX on {HEADS[3]})"
    if index_q_dtype != torch.float8_e4m3fn:
        return f"index-q {index_q_dtype} (e4m3)"
    c = cos_sin_cache
    if c is not None and (c.dtype != torch.bfloat16 or c.dim() != 2 or c.shape[1] != HEADS[3]
                          or not c.is_contiguous()):
        return f"cos / sin cache {tuple(c.shape)} {c.dtype} (contiguous bf16 [rows, {HEADS[3]}])"
    return None


def _check_heads(nq: int, nkv: int, niq: int, rotary_dim: int) -> None:
    why = insert_blocker(nq, nkv, niq, rotary_dim)
    assert why is None, f"norm_rope_nvfp4_insert: {why}"


@functools.cache
def _get_kernel(nq: int, nkv: int, niq: int, rotary_dim: int, tp: int):
    return _kernel.compile_norm_rope_insert(nq, nkv, niq, rotary_dim, tp)


@functools.cache
def _get_prefill_kernel(nq: int, nkv: int, niq: int, rotary_dim: int, tp: int):
    return _kernel_prefill.compile_norm_rope_insert(nq, nkv, niq, rotary_dim, tp)


@functools.cache
def _get_streaming_q16_kernel():
    return _kernel_streaming.compile_streaming_q16_kv1_iq1()


def _token_pairs_per_wave(n: int) -> int:
    return 1 if n <= 512 else 4


# the prefill variant from here (four pairs, nt q stores): QKV chain 2048 / 3072 / 4096 tokens -2.7 / -3.3 / -1.5% vs
# the main kernel
PREFILL_TOKENS = 1537


def _use_prefill(n: int, prefill: bool, tp: int) -> bool:
    return (prefill or n >= PREFILL_TOKENS) and tp % 2 == 0


# eight pairs (nt q stores): QKV chain ~-1% in this measured window; four pairs elsewhere
STREAMING_PREFILL_RANGE = (24576, 32768)


def _main_token_pairs(n):
    # four pairs from 489 tokens: QKV chain -0.5..-1.0% at 489-511, no gain at 385-488
    if 489 <= n <= 512:
        return 4
    return _token_pairs_per_wave(n)


def _prefill_token_pairs(n):
    if STREAMING_PREFILL_RANGE[0] <= n <= STREAMING_PREFILL_RANGE[1]:
        return 8
    return _main_token_pairs(n)


def norm_rope_nvfp4_insert(
    qkv: torch.Tensor,  # [n, (nq + 2 nkv + niq + 1) * 128] bf16
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim] bf16
    positions: torch.Tensor,  # [n] int64
    slot_mapping: torch.Tensor,  # [n] int64, -1 skips the token
    index_slot_mapping: torch.Tensor,  # [n] int64, -1 skips the token
    kv_cache: torch.Tensor,  # [blocks, 2 nkv, 128, 72] uint8
    index_cache: torch.Tensor,  # [blocks, 128, 72] uint8
    k_scale: torch.Tensor,  # fp32 scalar dequantization (global) scales
    v_scale: torch.Tensor,
    q_out: torch.Tensor,  # [n, nq * 128] bf16
    index_q_out: torch.Tensor,  # [n, niq * 128] e4m3, row stride free
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    eps: float,
    prefill: bool = False,
) -> None:
    """Norm + RoPE of the q / k / index-q / index-k heads into ``q_out``, ``index_q_out`` and the NVFP4 K / V /
    index-K pages, in one launch (contract: ``reference.py``); ``qkv`` is not written. ``prefill`` takes the prefill
    variant below its token threshold too; ``warmup_prefill_kernels`` compiles that variant at init."""
    n = qkv.shape[0]
    if n == 0:
        return
    nq, nkv, niq = num_heads, num_kv_heads, num_index_heads
    rotary_dim = cos_sin_cache.shape[1]
    _check_heads(nq, nkv, niq, rotary_dim)
    assert eps >= 2.0**-126  # the kernel's rsqrt skips ocml's denormal scaling
    assert qkv.dtype == torch.bfloat16 and qkv.is_contiguous()
    assert qkv.shape[1] == (nq + 2 * nkv + niq + 1) * _kernel.D
    assert cos_sin_cache.dtype == torch.bfloat16 and cos_sin_cache.is_contiguous()
    for w in (q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight):
        assert w.dtype == torch.bfloat16 and w.is_contiguous()
    assert q_out.dtype == torch.bfloat16 and q_out.is_contiguous()
    assert index_q_out.dtype == torch.float8_e4m3fn and index_q_out.stride(1) == 1
    assert index_q_out.stride(0) % 4 == 0
    assert kv_cache.shape[1:] == (2 * nkv, _kernel.BLK, 72)
    assert kv_cache.stride(1) == _kernel.BLK * 72 and kv_cache.stride(3) == 1
    assert index_cache.shape[1:] == (_kernel.BLK, 72) and index_cache.stride(2) == 1
    assert index_cache.stride(1) == 72
    tp = _main_token_pairs(n)
    if _use_prefill(n, prefill, tp):
        exe = _get_prefill_kernel(nq, nkv, niq, rotary_dim, _prefill_token_pairs(n))
    elif 289 <= n <= 384:
        exe = _get_streaming_q16_kernel()
    else:
        exe = _get_kernel(nq, nkv, niq, rotary_dim, tp)
    _run_compiled(
        exe,
        qkv.data_ptr(),
        q_out.data_ptr(),
        index_q_out.data_ptr(),
        q_norm_weight.data_ptr(),
        k_norm_weight.data_ptr(),
        index_q_norm_weight.data_ptr(),
        index_k_norm_weight.data_ptr(),
        cos_sin_cache.data_ptr(),
        positions.data_ptr(),
        slot_mapping.data_ptr(),
        index_slot_mapping.data_ptr(),
        kv_cache.data_ptr(),
        index_cache.data_ptr(),
        k_scale.data_ptr(),
        v_scale.data_ptr(),
        n,
        index_q_out.stride(0),
        kv_cache.stride(0),
        index_cache.stride(0),
        cos_sin_cache.shape[0],
        struct.unpack("<i", struct.pack("<f", eps))[0],
        torch.cuda.current_stream(),
    )


@functools.cache
def _get_reduce_kernel(nq: int, nkv: int, niq: int, rotary_dim: int, tp: int, sk: int):
    return _reduce_kernel.compile_reduce_insert(nq, nkv, niq, rotary_dim, tp, sk)


def _reduce_token_pairs(n, split_k):
    # one pair on the split-2 rows at 177-288 tokens: QKV chain -0.4..-1.7% at 193-288
    if split_k == 2 and 176 < n <= 288:
        return 1
    return 1 if n <= 32 else (2 if n <= 512 else 4)


def splitk_reduce_norm_rope_nvfp4_insert(
    slab: torch.Tensor,  # [sk, n, (nq + 2 nkv + niq + 1) * 128] fp32 QKV partials
    split_k: int,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim] bf16
    positions: torch.Tensor,  # [n] int64
    slot_mapping: torch.Tensor,  # [n] int64, -1 skips the token
    index_slot_mapping: torch.Tensor,  # [n] int64, -1 skips the token
    kv_cache: torch.Tensor,  # [blocks, 2 nkv, 128, 72] uint8
    index_cache: torch.Tensor,  # [blocks, 128, 72] uint8
    k_scale: torch.Tensor,  # fp32 scalar dequantization (global) scales
    v_scale: torch.Tensor,
    q_out: torch.Tensor,  # [n, nq * 128] bf16
    index_q_out: torch.Tensor,  # [n, niq * 128] e4m3, row stride free
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    eps: float,
) -> None:
    """The QKV GEMM's split-K reduce + ``norm_rope_nvfp4_insert`` in one launch, bit for bit: the slabs are summed
    in the reduce's order and rounded to bf16 as its store does; the bf16 QKV is never written."""
    n = slab.shape[1]
    if n == 0:
        return
    nq, nkv, niq = num_heads, num_kv_heads, num_index_heads
    rotary_dim = cos_sin_cache.shape[1]
    _check_heads(nq, nkv, niq, rotary_dim)
    assert eps >= 2.0**-126  # the kernel's rsqrt skips ocml's denormal scaling
    assert slab.dtype == torch.float32 and slab.is_contiguous()
    assert slab.shape[0] == split_k
    assert slab.shape[2] == (nq + 2 * nkv + niq + 1) * _kernel.D
    assert cos_sin_cache.dtype == torch.bfloat16 and cos_sin_cache.is_contiguous()
    for w in (q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight):
        assert w.dtype == torch.bfloat16 and w.is_contiguous()
    assert q_out.dtype == torch.bfloat16 and q_out.is_contiguous()
    assert index_q_out.dtype == torch.float8_e4m3fn and index_q_out.stride(1) == 1
    assert index_q_out.stride(0) % 4 == 0
    assert kv_cache.shape[1:] == (2 * nkv, _kernel.BLK, 72)
    assert kv_cache.stride(1) == _kernel.BLK * 72 and kv_cache.stride(3) == 1
    assert index_cache.shape[1:] == (_kernel.BLK, 72) and index_cache.stride(2) == 1
    assert index_cache.stride(1) == 72
    token_pairs = _reduce_token_pairs(n, split_k)
    _run_compiled(
        _get_reduce_kernel(nq, nkv, niq, rotary_dim, token_pairs, split_k),
        slab.data_ptr(),
        q_out.data_ptr(),
        index_q_out.data_ptr(),
        q_norm_weight.data_ptr(),
        k_norm_weight.data_ptr(),
        index_q_norm_weight.data_ptr(),
        index_k_norm_weight.data_ptr(),
        cos_sin_cache.data_ptr(),
        positions.data_ptr(),
        slot_mapping.data_ptr(),
        index_slot_mapping.data_ptr(),
        kv_cache.data_ptr(),
        index_cache.data_ptr(),
        k_scale.data_ptr(),
        v_scale.data_ptr(),
        n,
        index_q_out.stride(0),
        kv_cache.stride(0),
        index_cache.stride(0),
        cos_sin_cache.shape[0],
        struct.unpack("<i", struct.pack("<f", eps))[0],
        torch.cuda.current_stream(),
    )


def warmup_prefill_kernels(
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    rotary_dim: int = 64,
    device=None,
) -> str | None:
    """Compiles the prefill variants at init (root README rule 4: graph capture reaches only the main kernel) by
    launching each tile on a zero batch with slots -1, which writes no page. A layer ``insert_blocker`` rejects
    compiles nothing: returns that reason (the caller keeps its stock path), else None."""
    device = device or torch.device("cuda")
    nq, nkv, niq = num_heads, num_kv_heads, num_index_heads
    why = insert_blocker(nq, nkv, niq, rotary_dim)
    if why is not None:
        return why
    for n in (PREFILL_TOKENS, STREAMING_PREFILL_RANGE[0]):
        row = (nq + 2 * nkv + niq + 1) * _kernel.D
        qkv = torch.zeros((n, row), dtype=torch.bfloat16, device=device)
        w = torch.zeros(_kernel.D, dtype=torch.bfloat16, device=device)
        cs = torch.zeros((n, rotary_dim), dtype=torch.bfloat16, device=device)
        pos = torch.arange(n, device=device)
        slots = torch.full((n,), -1, dtype=torch.int64, device=device)
        kv = torch.zeros((1, 2 * nkv, _kernel.BLK, 72), dtype=torch.uint8, device=device)
        idx = torch.zeros((1, _kernel.BLK, 72), dtype=torch.uint8, device=device)
        one = torch.ones((), dtype=torch.float32, device=device)
        iq = torch.empty((n, niq * _kernel.D), dtype=torch.float8_e4m3fn, device=device)
        q = torch.empty((n, nq * _kernel.D), dtype=torch.bfloat16, device=device)
        norm_rope_nvfp4_insert(qkv, w, w, w, w, cs, pos, slots, slots, kv, idx, one, one, q, iq,
                               num_heads=nq, num_kv_heads=nkv, num_index_heads=niq, eps=1e-6)
    return None
