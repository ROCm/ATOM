# SPDX-License-Identifier: MIT
"""p1/glue-front: the layer-front QKV chain of MiniMax-M3.1 at TP4 through amd-kernels' one-launch
``mxfp8_qkv_norm_rope_nvfp4_insert`` (QKV MXFP8 GEMM + q/k/index norm + partial RoPE + NVFP4 K/V/index-K insert).

``ATOM_P1_FRONT_QKV_CHAIN=1`` (default off; needs ``ATOM_AK_MXFP8_GEMM=1`` and the amd-kernels checkout on PYTHONPATH):
in pure-decode steps of MIN_ROWS..MAX_ROWS rows a sparse attention layer's ``qkv_proj`` runs only
  quant_shuffled(x)   [FlyDSL MXFP8 quant: codes in the decode GEMM's (16, 16) tile + AITER-shuffled E8M0 scales]
and its attention op's ``rope_cache`` (today: amd-kernels norm_rope_nvfp4_insert on the bf16 qkv) runs
  mxfp8_qkv_norm_rope_nvfp4_insert   [one launch at 1-64 / 129-192 rows, two at 65-128 / 193-288]
Both decisions are taken at run time, inside the two custom ops (concrete row count and forward context), by the
same predicate ``take``; the compiled graph is the same for every row count. The codes travel from the linear to
the attention op through ``_PENDING`` (same forward, same stream, consumed by the next attention op).
Numerics (bench_front.py, round 2): quant bytes == ATOM quantize_fp8; q / index-q / pages bit-identical on N(0,1),
last-bf16-bit q differences on wide-range inputs from the GEMM's split-K order (fp32_order).
"""

import functools
import os

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard

QKV_CHAIN = os.environ.get("ATOM_P1_FRONT_QKV_CHAIN", "0") == "1"
MIN_ROWS = 8    # M4 (k3 G1): X measured +0.28 us -> today's path
MAX_ROWS = 288  # M320 / M384: tie -> today's path
N, K = 2560, 6144

_PENDING = None  # (rows, codes, scales, weight, ak_weight_scale, x, weight_scale) of the last qkv_front that took the chain

# p1/ar-epilogue (chain Y, default off): the pre-attention all-reduce + add + Gemma norm of a chain layer emits the
# entry's tiled MXFP8 codes + AITER-shuffled scales itself (amd-kernels Bf16ArNormComm, quant of the bf16-rounded
# norm = this module's quant rule) and parks them here; the layer's qkv_front then skips its quant. The AR op hands
# the linear an UNCOMPUTED bf16 x on those rows (nothing else reads the pre-attention norm output of a chain layer).
AR_CODES = QKV_CHAIN and os.environ.get("ATOM_P1_AR_QKV_CODES", "0") == "1"
# p1/L4-comm: chain Y on the staged pull exchange (overlay l4c_y.pullq) for registered graph rows in this range
L4_PULLQ = AR_CODES and os.environ.get("ATOM_P1_L4_PULLQ", "0") == "1"
L4_PULLQ_ROWS = (16, 88)
_CODES = None  # (rows, codes, scales) parked by the AR epilogue for the next qkv_front


def park_codes(m: int, codes: torch.Tensor, scales: torch.Tensor) -> None:
    global _CODES
    _CODES = (m, codes, scales)


@functools.cache
def _g():
    from amd_kernels.gemm.mxfp8_common import gemm

    return gemm


@functools.cache
def _entry():
    from amd_kernels.attention.mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128 import (
        mxfp8_qkv_norm_rope_nvfp4_insert,
    )

    return mxfp8_qkv_norm_rope_nvfp4_insert


def tile_rows(m: int) -> int:
    return _g().prequant_code_tile_rows(m, N, K, torch.bfloat16)


def layer_ok(num_heads, num_kv_heads, num_idx_heads, rotary_dim, kv_dtype, index_dtype, skip_index_topk,
             per_token_fp8_out) -> bool:
    """Static (init-time) eligibility of a sparse attention layer: the entry's TP4 head layout and NVFP4 caches."""
    return (QKV_CHAIN and (num_heads, num_kv_heads, num_idx_heads, rotary_dim) == (16, 1, 1, 64)
            and kv_dtype == "nvfp4" and index_dtype == "nvfp4" and not skip_index_topk and not per_token_fp8_out)


def take(m: int) -> bool:
    """Run-time: this forward's qkv linear + insert take the chain."""
    if not (QKV_CHAIN and MIN_ROWS <= m <= MAX_ROWS):
        return False
    from atom.utils.forward_context import get_forward_context

    c = get_forward_context().context
    if c is None or c.is_prefill or c.is_dummy_run:
        return False
    return tile_rows(m) > 0


def _front_fake(x, weight, weight_scale, ak_weight_scale):
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=torch.bfloat16, device=x.device)


@torch_compile_guard(gen_fake=_front_fake, mutates_args=[])
def qkv_front(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
              ak_weight_scale: torch.Tensor) -> torch.Tensor:
    """The qkv linear of a chain-eligible layer. Chain rows: MXFP8 quant only, codes parked in _PENDING, the
    returned qkv is NOT computed (the attention op's insert never reads it on this path). Else: today's linear."""
    global _PENDING, _CODES
    x2 = x.reshape(-1, K)
    m = x2.shape[0]
    pc, _CODES = _CODES, None
    if take(m):
        if pc is not None:  # chain Y: the AR epilogue already wrote this layer's codes (x is not computed)
            assert pc[0] == m, f"p1_front: AR parked codes for {pc[0]} rows, qkv has {m}"
            c, s = pc[1], pc[2]
        else:
            c, s = _g().mxfp8_e4m3_quantize_shuffled(x2, tile_rows(m))
        # chain Y: x2 was never computed, so park None (a future reader of pend[5] fails loudly, not on garbage)
        _PENDING = (m, c.view(torch.float8_e4m3fn), s.view(torch.uint8), weight, ak_weight_scale,
                    None if pc is not None else x2, weight_scale)
        return torch.empty((*x.shape[:-1], N), dtype=torch.bfloat16, device=x.device)
    if pc is not None:
        raise RuntimeError("p1_front: AR parked qkv codes but this qkv_front does not take the chain")
    _PENDING = None
    from atom.model_ops.ak_gemm import ak_mxfp8_linear

    return ak_mxfp8_linear(x, weight, weight_scale, ak_weight_scale)


def pop(m: int):
    """The attention op's side: the parked codes of this layer's linear (or None = today's path)."""
    global _PENDING
    p, _PENDING = _PENDING, None
    if p is None:
        return None
    assert p[0] == m, f"p1_front: parked codes for {p[0]} rows, attention has {m}"
    return p


def run_chain(codes, scales, weight, ak_weight_scale, q_out, index_q_out, q_norm_w, k_norm_w, iq_norm_w, ik_norm_w,
              cos_sin_cache, positions, slot_mapping, kv_pages, index_pages, k_scale, v_scale, num_heads,
              num_kv_heads, num_index_heads, eps):
    """The one entry; writes q_out [m, 2048] bf16, index_q_out [m, 128] e4m3 and the pages."""
    m = q_out.shape[0]
    _entry()(codes, scales, weight,
             ak_weight_scale, m, q_norm_w, k_norm_w, iq_norm_w, ik_norm_w, cos_sin_cache, positions, slot_mapping,
             slot_mapping, kv_pages, index_pages, k_scale, v_scale, q_out, index_q_out,
             num_heads=num_heads, num_kv_heads=num_kv_heads, num_index_heads=num_index_heads, eps=eps)


# Per-regime chain table (decode rows M = 8 G for k = 7, 4 G for k = 3), measured round 2 (bench_front.py, cold).
PLAN = {
    (1, 7): "atom",      # M4: X +0.28 us
    (8, 64): "X",        # one launch; -1.09 .. -3.78 us
    (65, 128): "X",      # split-K GEMM + reduce-insert; -2.43 .. -3.67 us
    (129, 192): "X",     # one launch; -2.48 .. -4.63 us
    (193, 288): "X",     # 2 launches; -1.89 .. -2.10 us
    (289, 384): "atom",  # tie
}


def warm(weight: torch.Tensor, ak_weight_scale: torch.Tensor) -> None:
    """Eager first call at load (QKV linear prepare): runs the entry's own warm-up (compiles its rows) and creates
    its counter pool outside graph capture (a first call inside capture raises). Every slot -1: no page is written."""
    dev, m = weight.device, 8
    codes = torch.zeros((16, K), dtype=torch.float8_e4m3fn, device=dev)
    scales = torch.full((32, K // 32), 127, dtype=torch.uint8, device=dev)
    nw = torch.zeros(128, dtype=torch.bfloat16, device=dev)
    cs = torch.zeros((128, 64), dtype=torch.bfloat16, device=dev)
    pos = torch.zeros(m, dtype=torch.int64, device=dev)
    skip = torch.full((m,), -1, dtype=torch.int64, device=dev)
    kv = torch.zeros((1, 2, 128, 72), dtype=torch.uint8, device=dev)
    idx = torch.zeros((1, 128, 72), dtype=torch.uint8, device=dev)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    q = torch.empty((m, 2048), dtype=torch.bfloat16, device=dev)
    iq = torch.empty((m, 128), dtype=torch.float8_e4m3fn, device=dev)
    _entry()(codes, scales, weight, ak_weight_scale, m, nw, nw, nw, nw, cs, pos, skip, skip, kv, idx, one, one, q, iq,
             num_heads=16, num_kv_heads=1, num_index_heads=1, eps=1e-6)
    torch.cuda.synchronize(dev)
