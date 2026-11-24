# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The torch contract of ``mxfp8_qkv_norm_rope_nvfp4_insert`` (root README, rule 3): the QKV GEMM's contract
(``amd_kernels/gemm/mxfp8_common/reference.py``) followed by the norm / RoPE / NVFP4 insert's
(``amd_kernels/attention/norm_rope_nvfp4_insert_d128_rot64_p128/reference.py``), both imported unchanged.

1. ``qkv = A @ W^T``: the exact values of the MXFP8 activation (row-major E4M3 codes + E8M0 exponents, per 1x32
   block along K) and weight, accumulated in fp32, rounded once to bf16 (``mxfp8_gemm_reference``).
2. ``norm_rope_nvfp4_insert(qkv, ...)``: per token and head the Gemma RMS norm, partial NeoX RoPE (rotary 64), bf16
   q, e4m3 index-q, NVFP4 K / V / index-K into 128-token pages (that reference's docstring: layouts, rounding).

Bytes (the entry's README): bit for bit the deferred chain (``gemm.mxfp8_gemm_shuffled(defer_reduce=True)`` +
``splitk_reduce_norm_rope_nvfp4_insert``) at every row count; against this reference the GEMM's K split and
summation order move the bf16 QKV in its last bit now and then, so the tests hold the insert reference's
tolerances (``COS_MIN``, ``REL_L2``, ``PAGE_REL_L2``, ``CODE_MISMATCH``, re-exported here), except index-q: its e4m3
(3 mantissa bits) turns a moved bf16 input into a flip of 1/8 now and then, so it holds ``INDEX_Q_REL_L2``
(measured on the tests' inputs: relative L2 <= 4.7e-4 at 1-33 rows, 1.9e-3 at 129, the deferred chain's bytes).
"""

import torch

from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import reference as _insert
from amd_kernels.gemm.mxfp8_common.reference import mxfp8_gemm_reference

COS_MIN = _insert.COS_MIN
REL_L2 = _insert.REL_L2
PAGE_REL_L2 = _insert.PAGE_REL_L2
CODE_MISMATCH = _insert.CODE_MISMATCH
INDEX_Q_REL_L2 = 5e-3


def mxfp8_qkv_norm_rope_nvfp4_insert(
    codes: torch.Tensor,  # [m, 6144] e4m3, row-major
    exponents: torch.Tensor,  # [m, 192] e8m0, row-major
    weight_codes: torch.Tensor,  # [2560, 6144] e4m3, row-major (not preshuffled)
    weight_exponents: torch.Tensor,  # [2560, 192] e8m0
    q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight, cos_sin_cache, positions,
    slot_mapping, index_slot_mapping, kv_cache, index_cache, k_scale, v_scale, q_out, index_q_out,
    *, num_heads: int, num_kv_heads: int, num_index_heads: int, eps: float,
) -> None:
    """The op's contract: writes ``q_out``, ``index_q_out`` and the pages in place."""
    if codes.shape[0] == 0:
        return
    qkv = mxfp8_gemm_reference(codes, exponents, weight_codes, weight_exponents, torch.bfloat16)
    _insert.norm_rope_nvfp4_insert(
        qkv, q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight, cos_sin_cache, positions,
        slot_mapping, index_slot_mapping, kv_cache, index_cache, k_scale, v_scale, q_out, index_q_out,
        num_heads=num_heads, num_kv_heads=num_kv_heads, num_index_heads=num_index_heads, eps=eps)
