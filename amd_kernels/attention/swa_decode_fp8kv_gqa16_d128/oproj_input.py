# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The attention of one layer as its o_proj's input: the MXFP8 epilogue where the o_proj's table row reads its codes
(from ``MXFP8_MIN_ROWS`` rows, query blocks of ``MXFP8_QUERY_ROWS`` rows), else the bf16 output. The buffers, the
choice and the warm-up of every request count live here; the caller passes its metadata tensors."""

import torch

from amd_kernels.warm_up import entry

from . import host as _host

# The MXFP8 epilogue as o_proj's input pays from 8 rows: attention + o_proj per layer, TP4 o_proj 6144 x 2048, graph
# replay, cold K/V and weights (benchmarks/attention/bench_attn_oproj_mxfp8.py): 7-row query
# blocks 14-28 rows -2.2 / -6.7%, 49-448 rows -2.6..-6.1%, 7 rows +1.3%.
MXFP8_MIN_ROWS = 8
# The query blocks whose MXFP8 epilogue the tests cover (tests/.../test_oproj_input.py, 1-64 requests): the 7-row and
# the 3-row block (the 3-row kernels need the native fptrunc bf16 pack).
MXFP8_QUERY_ROWS = (3, 7)
# The request counts the warm-up walks (the TP4 draft serves up to 256 requests per rank).
MAX_REQS = 256


def mxfp8_out(rows: int, max_query_len: int) -> bool:
    """Whether ``swa_decode_fp8kv_oproj_input`` writes MXFP8 codes for this call (given an o_proj)."""
    return rows >= MXFP8_MIN_ROWS and max_query_len in MXFP8_QUERY_ROWS


def oproj_codes(rows: int, n: int, k: int, device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Uninitialized (codes, scales, tile) for ``rows`` rows of an o_proj of N n, K k: codes in the layout of
    ``gemm.mxfp8_common``'s table row for the row count (the (16, 16) tiles of a decode row, ``tile`` its rows; tile 0:
    row-major), e8m0 scales in AITER's layout. The attention writes the real rows; the GEMM masks the padding."""
    from amd_kernels.gemm.mxfp8_common import gemm as G

    tile = G.prequant_code_tile_rows(rows, n, k, torch.bfloat16)
    pad = max(tile, 16)
    padded = -(-rows // pad) * pad
    codes = torch.empty((padded, k), dtype=torch.uint8, device=device)
    scales = torch.empty((-(-padded // 32) * 32, k // 32), dtype=torch.uint8, device=device)
    return codes, scales, tile


def _oproj_input(q, kv_cache, block_table, seq_lens, query_start_loc, max_query_len, sm_scale, window, causal,
                 k_scale, v_scale, *, oproj_n=None, output=None):
    """The attention of ``q`` ([rows, H, 128] or [rows, H * 128] bf16; ``swa_decode_fp8kv``'s other arguments) as an
    o_proj input: with ``oproj_n`` (the o_proj's N; its K is H * 128) and ``mxfp8_out(rows, max_query_len)``, the
    tuple (codes uint8, scales uint8, tile) of ``oproj_codes`` written by the MXFP8 epilogue (the bytes of the o_proj
    quantizer on the bf16 output); else the bf16 output, shaped like ``q`` (into ``output`` when given). Rows of no
    request are not written. The first call of each mode compiles every variant of 1..256 requests."""
    rows = q.shape[0]
    heads = q.numel() // (rows * 128) if rows else 0
    q3 = q.view(rows, heads, 128)
    if oproj_n is not None and mxfp8_out(rows, max_query_len):
        quant = oproj_codes(rows, oproj_n, heads * 128, q.device)
        _host.swa_decode_fp8kv(q3, kv_cache, block_table, seq_lens, query_start_loc, max_query_len, sm_scale, window,
                               causal, k_scale, v_scale, None, quant_out=quant)
        return quant
    out = torch.empty_like(q) if output is None else output
    _host.swa_decode_fp8kv(q3, kv_cache, block_table, seq_lens, query_start_loc, max_query_len, sm_scale, window,
                           causal, k_scale, v_scale, out.view(rows, heads, 128))
    return out


def _mode(q, kv_cache, block_table, seq_lens, query_start_loc, max_query_len, sm_scale, window, causal, k_scale,
          v_scale, *, oproj_n=None, output=None):
    """What a call fixes about its kernels besides the request count (``amd_kernels.warm_up``)."""
    heads = q.shape[1] if q.dim() == 3 else q.shape[1] // 128
    return (int(max_query_len), int(window), bool(causal), heads, kv_cache.shape[1], kv_cache.shape[2],
            kv_cache.dtype, q.dim(), oproj_n)


def _inputs(mode, n, q, kv_cache, block_table, seq_lens, query_start_loc, max_query_len, sm_scale, window, causal,
            k_scale, v_scale, *, oproj_n=None, output=None):
    """``n`` requests of ``max_query_len`` rows over two zero pages (no request reads a key)."""
    ql, _, _, heads, nkv, page, dtype, qdim, _ = mode
    dev = q.device
    shape = (n * ql, heads, 128) if qdim == 3 else (n * ql, heads * 128)
    qn = torch.zeros(shape, dtype=torch.bfloat16, device=dev)
    pages = torch.zeros((2, nkv, page, 256), dtype=dtype, device=dev)
    bt = torch.zeros((n, 1), dtype=torch.int32, device=dev)
    sl = torch.zeros(n, dtype=torch.int32, device=dev)
    qsl = torch.arange(n + 1, dtype=torch.int32, device=dev) * ql
    return (qn, pages, bt, sl, qsl, ql, sm_scale, window, causal, k_scale, v_scale), {"oproj_n": oproj_n}


swa_decode_fp8kv_oproj_input = entry("attention/swa_decode_fp8kv_gqa16_d128:oproj_input", mode=_mode,
                                     sizes=lambda mode: range(1, MAX_REQS + 1), inputs=_inputs)(_oproj_input)
