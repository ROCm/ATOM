# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of ``swa_decode_fp8kv`` (root README, rule 3): torch only.

Inputs. ``q [rows, H, 128]`` bf16 with ``H = 16 * Hkv`` (query head h reads KV head
h // 16); ``kv_cache [pages, Hkv, page_size, 256]`` e4m3 bytes, per (token, KV head)
128 K bytes then 128 V bytes; ``block_table [reqs, max_blocks]``, ``seq_lens [reqs]``,
``query_start_loc [reqs + 1]`` int32; per-tensor fp32 ``k_scale`` / ``v_scale``.
Request r's rows ``query_start_loc[r] .. query_start_loc[r + 1]`` (at most 8) are the
last positions of its sequence: row t sits at position ``p = seq_len - rows_r + t``
and attends the keys ``p - window < k < seq_len`` (causal: ``k <= p``).

Where the kernel rounds (its docstring): K and V widened from e4m3 to bf16 (exact);
bf16 Q; ``S = Q K^T`` in fp32 MFMAs, ``k_scale * sm_scale * log2(e)`` folded into the
fp32 score scale; online softmax in fp32 (exp2); P rounded to bf16 for ``P V``; fp32
accumulation; ``v_scale / sum`` applied once at the end; the output rounded to bf16
(RNE). With splits > 1 the partial (O, max, sum) rows are combined in split order.

``decode_reference`` computes the same attention in fp64 from the exact K / V values
(no bf16 rounding of P). Tolerances (``check_close``), per (request, KV head) over its
rows: cosine >= ``COS_MIN``, relative L2 <= ``REL_L2_MAX``, and elementwise
``atol ATOL, rtol RTOL``. Measured on the tests' cases
(MI355X): cosine >= 0.9999977, relative L2 <= 2.14e-3 (the bf16 output
alone is 2^-9 = 2.0e-3 per value; the bf16 P adds little). The kernel is deterministic: the same inputs give
the same bits (split count, row grouping and graph replay included).

MXFP8 output (``quant_out``): byte for byte what the Triton MXFP8 quantizer
(``_mxfp8_quantize_kernel``) makes of the bf16 output, ``mxfp8_quant_reference``:
per 32 consecutive values of a row, ``amax = max |x|`` floored at fp32 ``tiny``;
``e = clamp(ceil(log2(amax / 448)) + 127, 0, 254)`` (the division in fp32); codes
``e4m3_rne(x * 2^(127 - e))``; scale byte ``e``. Codes are
row-major (tile 0) or in the decode GEMM's (16, 16) tiles (tile > 0,
``code_offsets``); scales in AITER's shuffled layout (``scale_offsets``). Only the
rows of requests are written. The test compares bytes exactly.
"""

import torch

D = 128
HG = 16  # query heads per KV head
TOK_B = 2 * D
FP8_MAX = 448.0
TINY_F32 = torch.finfo(torch.float32).tiny

COS_MIN = 0.99998
REL_L2_MAX = 4e-3
ATOL, RTOL = 8e-3, 1e-2  # assert_close


def decode_reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    sm_scale: float,
    window: int,
    causal: bool,
    k_scale: float,
    v_scale: float,
) -> torch.Tensor:
    """fp64 ``[rows, H, 128]``; rows of no request (or of a request without keys)
    stay NaN."""
    rows, num_heads, head_dim = q.shape
    assert head_dim == D
    num_kv_heads, page = kv_cache.shape[1], kv_cache.shape[2]
    assert num_heads == HG * num_kv_heads
    cache = kv_cache.view(torch.float8_e4m3fn)
    out = torch.full((rows, num_heads, D), float("nan"), dtype=torch.float64, device=q.device)
    seqs, starts = seq_lens.tolist(), query_start_loc.tolist()
    for r, seq in enumerate(seqs):
        lo, hi = starts[r], starts[r + 1]
        ql = hi - lo
        if ql <= 0 or seq < ql:
            continue
        ctx = seq - ql
        keys = torch.arange(max(0, ctx - window + 1), seq, device=q.device)
        kv = cache[block_table[r][keys // page].long(), :, keys % page]  # [keys, Hkv, 256]
        k = kv[..., :D].double() * k_scale
        v = kv[..., D:].double() * v_scale
        qr = q[lo:hi].double().view(ql, num_kv_heads, HG, D)
        s = torch.einsum("tgjd,kgd->tgjk", qr, k) * sm_scale
        pos = ctx + torch.arange(ql, device=q.device)
        mask = keys[None, :] > pos[:, None] - window
        if causal:
            mask &= keys[None, :] <= pos[:, None]
        s = s.masked_fill(~mask[:, None, None, :], float("-inf"))
        o = torch.einsum("tgjk,kgd->tgjd", torch.softmax(s, dim=-1), v)
        out[lo:hi] = o.reshape(ql, num_heads, D)
    return out


def check_close(got: torch.Tensor, ref: torch.Tensor, name: str) -> tuple[float, float]:
    """``got`` / ``ref`` ``[rows, 16, 128]`` of one (request, KV head): the
    tolerances above. Returns (cosine, relative L2)."""
    g, r = got.double().flatten(), ref.flatten()
    cos = torch.nn.functional.cosine_similarity(g, r, dim=0).item()
    rel = ((g - r).norm() / r.norm()).item()
    assert cos >= COS_MIN and rel <= REL_L2_MAX, (name, cos, rel)
    torch.testing.assert_close(got.double(), ref, atol=ATOL, rtol=RTOL, msg=name)
    return cos, rel


def mxfp8_quant_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 ``[rows, K]`` -> (e4m3 codes ``[rows, K]`` uint8, e8m0 scales
    ``[rows, K // 32]`` uint8), both row-major."""
    rows, k = x.shape
    v = x.float().view(rows, k // 32, 32)
    amax = v.abs().amax(-1).clamp_min(TINY_F32)
    e = torch.ceil(torch.log2((amax / FP8_MAX).double())) + 127.0
    e = e.clamp(0.0, 254.0)
    rescale = torch.exp2(127.0 - e).float()  # a power of two: x * rescale is exact
    codes = (v * rescale[..., None]).to(torch.float8_e4m3fn)
    return codes.view(torch.uint8).view(rows, k), e.to(torch.uint8)


def code_offsets(rows: int, k: int, tiled: bool) -> torch.Tensor:
    """Byte offset of code (m, c) for m < rows: row-major, or the (16, 16) tiles
    ``(m//16)*16K + (c//64)*1024 + ((c%64)//16)*256 + (m%16)*16 + c%16``."""
    m = torch.arange(rows)[:, None]
    c = torch.arange(k)[None, :]
    if not tiled:
        return m * k + c
    return (m // 16) * (16 * k) + (c // 64) * 1024 + ((c % 64) // 16) * 256 + (m % 16) * 16 + c % 16


def scale_offsets(rows: int, k: int) -> torch.Tensor:
    """Byte offset of scale (m, gi), gi = c // 32: AITER's shuffled layout
    ``[M//32, K//256, klane 4, mlane 16, kpack 2, mpack 2]``."""
    m = torch.arange(rows)[:, None]
    gi = torch.arange(k // 32)[None, :]
    return (
        ((((m // 32) * (k // 256) + gi // 8) * 4 + gi % 4) * 16 + m % 16) * 2 + (gi % 8) // 4
    ) * 2 + (m % 32) // 16


def quant_buffers_reference(
    out_bf16: torch.Tensor, codes_buf: torch.Tensor, scales_buf: torch.Tensor, tile: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copies of the kernel's (codes, scales) buffers as they were before the launch,
    with the reference bytes of ``out_bf16 [rows, K]`` written where the kernel
    writes them (only rows < rows)."""
    rows, k = out_bf16.shape
    codes, scales = mxfp8_quant_reference(out_bf16)
    c_ref, s_ref = codes_buf.clone().flatten(), scales_buf.clone().flatten()
    c_ref[code_offsets(rows, k, tile > 0).flatten().to(c_ref.device)] = codes.flatten()
    s_ref[scale_offsets(rows, k).flatten().to(s_ref.device)] = scales.flatten()
    return c_ref.view_as(codes_buf), s_ref.view_as(scales_buf)
