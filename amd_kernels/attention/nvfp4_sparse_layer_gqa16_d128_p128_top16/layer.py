# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One NVFP4 sparse attention layer's kernels at their served compute dtypes: ``init`` builds the decode indexer and
compiles every insert, decode and prefill variant; the decode / prefill calls bind the dtypes; ``quant_out_buffers``
sizes the decode's MXFP8 output (o_proj's input) or declines it."""

import functools

import torch

from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import warmup_prefill_kernels
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import host as decode_host
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import index_decode
from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import host as prefill_host
from amd_kernels.gemm.mxfp8_common import gemm
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import prefill_top16
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.decode_top16 import IndexDecodeTop16

DECODE_COMPUTE = torch.float16  # exact K / V, the fastest decode at TP4 (decode README, Performance)
PREFILL_COMPUTE = "fp16"  # exact K / V
PREFILL_LAZY = True  # fp16 P re-based past 2^8: the fastest prefill at every T (prefill README, Performance)
QUANT_OUT_MIN_ROWS = decode_host.MXFP8_OUT_MIN_ROWS

sparse_decode = functools.partial(decode_host.nvfp4_sparse_decode, compute_type=DECODE_COMPUTE)
index_sparse_decode = functools.partial(index_decode.nvfp4_index_sparse_decode, compute_type=DECODE_COMPUTE)
sparse_prefill = functools.partial(prefill_host.nvfp4_sparse_prefill, compute_dtype=PREFILL_COMPUTE,
                                   lazy_rescale=PREFILL_LAZY)


def quant_out_buffers(rows: int, n: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """``quant_out`` of a ``rows``-row decode whose output feeds an MXFP8 GEMM with an ``[n, k]`` weight, in that
    GEMM's pre-quantized layout (``gemm.prequant_buffers``); None below ``QUANT_OUT_MIN_ROWS`` (bf16 output)."""
    if rows < QUANT_OUT_MIN_ROWS:
        return None
    return gemm.prequant_buffers(rows, n, k, device)


def init(
    device,
    *,
    num_heads: int,
    num_kv_heads: int,
    num_index_heads: int,
    query_len: int,
    max_reqs: int,
    max_seq_len: int,
    insert: bool = True,
    attention: bool = True,
    index: bool = True,
    quant_out: bool = True,
    rank: int | None = None,
    group=None,
) -> IndexDecodeTop16 | None:
    """Before serving: the decode indexer (``index``; with ``group`` the four-GPU route, a collective on every rank in
    order) and every variant the layer launches, compiled: the insert's prefill tiles (``insert``), the decode of
    1-``max_reqs`` requests x ``query_len`` rows with the bf16 output, the MXFP8 output from ``QUANT_OUT_MIN_ROWS``
    rows (``quant_out``) and the in-kernel top-16 (``index``), the prefill (``attention``) and the prefill indexer
    (``index``: ``prefill_top16.index_prefill_top16``). Returns the decode indexer."""
    device = torch.device(device)
    indexer = None
    if index:
        indexer = IndexDecodeTop16(device, query_len=query_len, max_reqs=max_reqs, max_seq_len=max_seq_len,
                                   num_index_heads=num_index_heads, rank=rank, group=group)
        prefill_top16.warm_up(device, num_index_heads)
    with torch.inference_mode():
        if insert:
            warmup_prefill_kernels(num_heads, num_kv_heads, num_index_heads, device=device)
        if attention:
            decode_host.warm_up_decode(device, num_heads=num_heads, num_kv_heads=num_kv_heads, query_len=query_len,
                                       max_reqs=max_reqs, quant_min_rows=QUANT_OUT_MIN_ROWS if quant_out else None,
                                       index_select=index, compute_type=DECODE_COMPUTE)
            prefill_host.warm_up_prefill(device, num_heads=num_heads, num_kv_heads=num_kv_heads,
                                         compute_dtype=PREFILL_COMPUTE, lazy_rescale=PREFILL_LAZY)
    torch.accelerator.synchronize(device)
    torch.accelerator.empty_cache()
    return indexer
