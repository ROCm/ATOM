# SPDX-License-Identifier: MIT
"""A8W4 attention over NVFP4 CSA and MXFP8 SWA source planes."""

import struct

import torch
from aiter.jit.utils.chip_info import get_gfx_runtime

from atom.utils import envs
from atom.utils.decorators import mark_trace

try:
    from aiter.ops.mla_sparse_prefill import mla_sparse_prefill_v41
except ImportError:
    mla_sparse_prefill_v41 = None

try:
    from aiter.ops.opus.mla import opus_mla_v41_prefill_a8w4_gfx950_fwd
except ImportError:
    opus_mla_v41_prefill_a8w4_gfx950_fwd = None

try:
    from aiter.jit.utils.asm_guard import is_gfx1250_asm_supported
    from aiter.ops.mla_sparse_prefill_v41_asm import asm_mla_sparse_prefill_v41_fwd
except ImportError:
    asm_mla_sparse_prefill_v41_fwd = None
    is_gfx1250_asm_supported = None


def select_a8w4_backend(heads, head_dim, softmax_scale):
    """Choose before Q preparation; older AITER builds keep the BF16 adapter."""
    mode = envs.ATOM_DSV41_A8W4_ATTN
    if mode not in ("auto", "off"):
        raise ValueError("ATOM_DSV41_A8W4_ATTN must be auto or off")
    if mode == "off" or mla_sparse_prefill_v41 is None or head_dim != 512:
        return None
    gfx = get_gfx_runtime()
    if gfx == "gfx950" and opus_mla_v41_prefill_a8w4_gfx950_fwd is not None:
        return "opus"
    if (
        gfx == "gfx1250"
        and heads == 64
        and asm_mla_sparse_prefill_v41_fwd is not None
        and is_gfx1250_asm_supported()
        and struct.pack("<f", softmax_scale) == struct.pack("<f", 512**-0.5)
    ):
        return "asm"
    return None


@mark_trace
def sparse_attn_v41_a8w4_paged_2src(
    query,
    pool,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
    *,
    backend,
    out,
    num_kv_splits=None,
):
    """Both backends consume the same planes and per-source int32 row IDs.

    Prefill's SWA source holds batch-local history followed by current KV;
    decode reads the ring directly. No cache row is dequantized here. AITER
    owns the backend's split heuristic and partial-workspace layout, as it
    does for the A8W8 wrapper.
    """
    values, scales = query
    data, scale = pool
    window, window_scale = kv
    return mla_sparse_prefill_v41(
        values,
        scales,
        data,
        scale.view(torch.float8_e4m3fn),
        kv_indices_prefix,
        kv_indptr_prefix,
        window.view(torch.float8_e4m3fn),
        window_scale.view(torch.float8_e8m0fnu),
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out=out,
        num_kv_splits=num_kv_splits,
        backend=backend,
    )
