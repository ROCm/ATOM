# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""amd-kernels (inferact/amd-kernels, FlyDSL) entries for the NVFP4 sparse layer of MiniMax-M3.1 at TP4.

Switches (default off = today's aiter path); they need the amd-kernels checkout on PYTHONPATH:

``ATOM_M3_AK_INSERT=1``: the q / k / index norm + RoPE + NVFP4 K / V / index-K insert of every step (decode and
prefill) through ``norm_rope_nvfp4_insert``: the same page bytes, q and index-q as
``aiter.fused_qknorm_idxrqknorm`` (NVFP4 mode).

``ATOM_M3_AK_DECODE=1``: the decode index top-16 + sparse attention through ``nvfp4_index_sparse_decode`` (one GPU:
scorer, then the attention selecting the top-16 in its prologue up to 256 rows, else scorer + selector + attention)
and, on a top-k reusing layer, ``nvfp4_sparse_decode`` on the cached ids. Same ids as ATOM's selector; the
attention computes in fp16 (exact K / V) instead of aiter's kernel: a different rounding of P, not bit-exact.

Both keep ATOM's page layouts: a KV page ``[K codes 8192 | K scales 1024 | V codes 8192 | V scales 1024]`` is the
amd-kernels ``[2, 128, 72]`` view of the same bytes, an index page ``[128, 72]``.
"""

import functools
import os

import torch

INSERT = os.environ.get("ATOM_M3_AK_INSERT", "0") == "1"
DECODE = os.environ.get("ATOM_M3_AK_DECODE", "0") == "1"
_COMPUTE = {"fp16": torch.float16, "bf16": torch.bfloat16}[os.environ.get("ATOM_M3_AK_DECODE_COMPUTE", "fp16")]
# p1/attn-decode: the ATOM_M3_AK_DECODE route writes o_proj's MXFP8 input itself (``quant_out``: the bytes of the
# quantizer amd-kernels' mxfp8_linear runs on the bf16 output, in the prequant layout of its GEMM) and o_proj runs
# ``mxfp8_gemm_prequant`` inside the attention op (no separate o_proj input quant). Default off.
QOUT = os.environ.get("ATOM_M3_AK_DECODE_QOUT", "0") == "1"
QOUT_MIN_ROWS = 16
# p1/indexer-cp: the ATOM_M3_AK_DECODE route's index top-16 on four GPUs from SPLIT4_MIN_REQS requests
# (8 at 8 or 4 rows a request): vLLM's ``IndexDecodeTop16`` on the replicated index cache -- rank r scores the
# blocks b % 4 == r of every rank's head (each rank's own e4m3 index-q rows, the bytes its TP path scores with, are
# copied to the peers) and stores fp32 scores into the head owner's buffer, which selects (amd-kernels' tie rule,
# the same as the one-GPU route's prologue). Same ids / outputs as the one-GPU route when the ranks' index pages are
# equal (the replicated cache). Default off.
SPLIT4 = os.environ.get("ATOM_P1_INDEXER_SPLIT4", "0") == "1"
# requests from which the four-GPU route is taken, by rows per request (amd-kernels: 5 at 8 rows). p1/indexer-cp
# kl1 (4 GPUs, cold, us per layer one-GPU -> split4): 8 rows x 5: 175K 25.93 -> 25.90, 100K 21.28 -> 25.70, 32K
# 17.42 -> 20.72; x 8: 175K 32.38 -> 26.63, 100K 25.81 -> 26.20, 32K 18.78 -> 21.30 (AgentX contexts ~175K mean)
SPLIT4_MIN_REQS = {8: 8, 4: 8}
# p1/decode-gap: ATOM_P1_INDEXER_SPLIT4_MIN8 (integration-only, default 8 = unchanged) moves the 8-rows threshold
# down. Route by request count only (rank-invariant, fixed per graph); bit-identical ids / outputs (indexer-cp kl1).
# Live AgentX decode contexts at 5-7 requests average 172-201K (milestone2 C runs), where split4 wins when
# requests x context >= ~0.9-1M tokens (kl1: N5 175K tie, N8 175K -5.75 us/layer, N8 100K +0.4).
SPLIT4_MIN_REQS[8] = int(os.environ.get("ATOM_P1_INDEXER_SPLIT4_MIN8", "8"))

_HEAD_SIDE = 9216  # one KV head's K (or V) of one 128-token page
BLK = 128


def kv_pages(kv_cache: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """ATOM's NVFP4 KV pool [N, 2 * Hkv * 9216] as amd-kernels' [N, 2 * Hkv, 128, 72] (same bytes)."""
    return torch.as_strided(
        kv_cache, (kv_cache.shape[0], 2 * num_kv_heads, BLK, 72), (kv_cache.stride(0), _HEAD_SIDE, 72, 1)
    )


def index_pages(index_cache: torch.Tensor) -> torch.Tensor:
    return index_cache.view(-1, BLK, 72)


@functools.cache
def _insert_fn():
    from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128.host import (
        insert_blocker,
        norm_rope_nvfp4_insert,
        warmup_prefill_kernels,
    )

    return insert_blocker, norm_rope_nvfp4_insert, warmup_prefill_kernels


@functools.cache
def insert_blocker(num_heads: int, num_kv_heads: int, num_index_heads: int, rotary_dim: int) -> str | None:
    """None when the layer fits the amd-kernels insert (TP4: q16 kv1 iq1, RoPE 64); compiles its prefill variants
    once (graph capture reaches only the main kernel)."""
    blocker, _, warm = _insert_fn()
    why = blocker(num_heads, num_kv_heads, num_index_heads, rotary_dim)
    if why is None:
        why = warm(num_heads, num_kv_heads, num_index_heads, rotary_dim)
    return why


def insert(qkv, q_norm_w, k_norm_w, iq_norm_w, ik_norm_w, cos_sin_cache, positions, slot_mapping, kv_cache,
           index_cache, k_scale, v_scale, q_out, index_q_out, num_heads, num_kv_heads, num_index_heads, eps,
           prefill: bool = False):
    """``aiter.fused_qknorm_idxrqknorm`` (NVFP4 K / V / index-K insert mode) through ``norm_rope_nvfp4_insert``."""
    _, fn, _ = _insert_fn()
    fn(qkv, q_norm_w, k_norm_w, iq_norm_w, ik_norm_w, cos_sin_cache, positions, slot_mapping, slot_mapping,
       kv_pages(kv_cache, num_kv_heads), index_pages(index_cache), k_scale, v_scale, q_out,
       index_q_out.view(index_q_out.shape[0], -1), num_heads, num_kv_heads, num_index_heads, eps, prefill=prefill)


# ---- ATOM_P1_L2_D2 (p1/L2-attn-decode "dist2" tail; integration branches only, default off) ----
# The amd-kernels decode host's kernel choice is unchanged (same grid, splits, arguments); for the shapes below the
# compiled kernel is the overlay variant ``*_p1d2`` (overlay package ``p1l2`` on PYTHONPATH), whose split combine and
# MXFP8 quant run on all four waves instead of wave 0 alone: bit-identical outputs, codes, scales and ids
# (verify-L2-attn-decode). d2 covers the row_xcd select kernel (splits 4, fp16; k7 G1-4) and kernel.py's padded XCD
# grid with split partials (fp16, xcd8, splits > 1, no tail grid; k7 G6-16). Every other variant (tail grid, one
# split = AK_WIDE 129-192 rows, fp8, prefetch, fixed, generic select at G5) keeps the snapshot kernel. Rank-invariant
# (the choice follows the host's shape-only schedule).
L2_D2 = DECODE and os.environ.get("ATOM_P1_L2_D2", "0") == "1"


def _install_l2_d2() -> None:
    from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import host as H

    if getattr(H, "_p1_l2_d2", False):
        return
    orig_sel, orig_gen = H._get_kernel_select, H._get_kernel

    @functools.cache
    def _sel(splits: int, quant: bool, exact_fp16: bool):
        if exact_fp16 and splits == 4:
            from p1l2.sel_d2 import compile_nvfp4_f16mfma_gqa16_d128_p128_top16_q8_rowxcd_select as comp

            return comp(splits, quant)
        return orig_sel(splits, quant, exact_fp16)

    @functools.cache
    def _gen(splits, topk, xcds, quant, exact_fp16, tail=False, fp8=False, row_xcd=False, prefetch=False,
             fixed=False):
        if (exact_fp16 and splits > 1 and topk == 16 and xcds == 8
                and not (tail or fp8 or row_xcd or prefetch or fixed)):
            from p1l2.gen_d2 import compile_nvfp4_sparse_decode as comp

            return comp(splits, topk, xcds, quant, exact_fp16, tail)
        return orig_gen(splits, topk, xcds, quant, exact_fp16, tail, fp8, row_xcd, prefetch, fixed)

    H._get_kernel_select, H._get_kernel = _sel, _gen
    H._p1_l2_d2 = True


@functools.cache
def _decode_fns():
    if L2_D2:
        _install_l2_d2()
    from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.host import nvfp4_sparse_decode
    from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.index_decode import (
        nvfp4_index_sparse_decode,
    )

    return nvfp4_index_sparse_decode, nvfp4_sparse_decode


# the one-GPU scorer of amd-kernels is faster than aiter's up to 16 requests x 8 rows and slower from 32 (32K-175K
# contexts, akSurvey/res/dec1.jsonl); the attention alone is on par at every batch
DECODE_MAX_REQS = int(os.environ.get("ATOM_M3_AK_DECODE_MAX_REQS", "16"))


def decode_supported(num_heads: int, num_kv_heads: int, num_index_heads: int, topk: int, max_query_len: int,
                     num_reqs: int) -> bool:
    if not ((num_heads, num_kv_heads, num_index_heads, topk) == (16, 1, 1, 16) and 1 <= max_query_len <= 8):
        return False
    return num_reqs <= min(DECODE_MAX_REQS, 256) or ak_wide_takes(num_reqs, max_query_len, num_index_heads)


# ---- ATOM_P1_AK_WIDE: the AK decode block above DECODE_MAX_REQS while split4 scores the step (p1/cliff17) ----
# DECODE_MAX_REQS = 16 was set when the AK block's scorer was one-GPU (it loses to aiter's at 175K above 16
# requests). With split4 the AK block's index top-16 runs on four GPUs, so above 16 requests the block is split4's
# fused score + top-16, then amd-kernels' attention (with ATOM_M3_AK_DECODE_QOUT writing o_proj's MXFP8 input),
# the same block as 8-16 requests, instead of split4 scores -> restore -> ATOM's q8 decode + o_proj's own quant.
# Taken only where split4 takes the step (rank-invariant: request count only), up to AK_WIDE_MAX_REQS by rows per
# request. Numerics: the <= 16-request route's kernels (amd-kernels tie rule, fp16 P), not the q8 route's bits.
AK_WIDE = SPLIT4 and DECODE and os.environ.get("ATOM_P1_AK_WIDE", "0") == "1"
AK_WIDE_MAX_REQS = {8: 32}
AK_WIDE_ONE_SPLIT = (128, 192)  # rows (exclusive, inclusive) where the attention runs one split


def ak_wide_takes(num_reqs: int, query_len: int, num_index_heads: int) -> bool:
    if not AK_WIDE or _Split4.index is None or num_reqs > AK_WIDE_MAX_REQS.get(query_len, 0):
        return False
    return _Split4.index.route(num_reqs, query_len, num_index_heads) == "split4"


class _Split4:
    index = None  # amd-kernels IndexDecodeTop16 with the TP group (the four-GPU route), made once per process
    tried = False


# ---- ATOM_P1_K2_SCORER=<arm> (p1/k2-scorer; integration branches only, default off) ----
# The split4 fused scorer + top-16 launch (score_topk, <= 128 rows) runs the overlay variant p1k2.s4k2 named by
# <arm> (overlay package ``p1k2`` on PYTHONPATH): half runs up to 512 runs/rank (h1024), the longest request's runs
# first (lf), the selector's row poll batched (wr2). Same grid, arguments and buffers; scores, ids and the receive
# buffer's MARKER state bit-identical (/shared_nfs/ygan/p1/k2-scorer/NOTES.md). Rank-invariant (every rank runs the
# same kernel with the same device-side schedule).
K2_SCORER = os.environ.get("ATOM_P1_K2_SCORER", "")


def init_split4(device, query_len: int, max_reqs: int, max_seq_len: int) -> None:
    """``ATOM_P1_INDEXER_SPLIT4``: the four-GPU decode indexer's IPC buffers and compiled variants (a collective on the
    TP CPU group, outside graph capture, the same call on every rank); no-op when off, done, or not four ranks."""
    if not (SPLIT4 and DECODE) or _Split4.tried:
        return
    _Split4.tried = True
    import gc
    import logging

    import torch.distributed as dist
    from aiter.dist.parallel_state import get_tp_group
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.decode_top16 import IndexDecodeTop16

    log = logging.getLogger("atom")
    tp = get_tp_group()
    if dist.get_world_size(tp.cpu_group) != 4:
        log.warning("ATOM_P1_INDEXER_SPLIT4: needs four TP ranks; keeping the one-GPU indexer")
        return
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import decode_top16

    decode_top16.SPLIT_MIN_REQS.update(SPLIT4_MIN_REQS)
    if K2_SCORER:
        from p1k2.atom_patch import install as _install_k2

        _install_k2(K2_SCORER)
        log.info("ATOM_P1_K2_SCORER: split4 fused scorer + top-16 variant %s", K2_SCORER)
    gc.collect()  # the warm-up launches every variant; no GC-unloaded Triton module in between
    index = IndexDecodeTop16(device, query_len=query_len, max_reqs=max_reqs, max_seq_len=max_seq_len,
                             rank=tp.rank_in_group, group=tp.cpu_group, sharded=False, warm_up=True)
    torch.cuda.synchronize(device)
    if index.split is None:
        log.warning("ATOM_P1_INDEXER_SPLIT4: off (%s)", index.split_off)
        return
    _Split4.index = index
    log.info("ATOM_P1_INDEXER_SPLIT4: four-GPU decode index top-16, %d rows a request, up to %d rows",
             query_len, index.split.max_rows)
    if SPLIT4_WIDE:
        warm_up_split4_wide(index.split, device)
        log.info("ATOM_P1_INDEXER_SPLIT4_WIDE: four-GPU index scores into the fused NVFP4 decode from %s requests",
                 SPLIT4_WIDE_MIN_REQS.get(query_len))


# ---- ATOM_P1_INDEXER_SPLIT4_WIDE: the four-GPU index SCORES above the AK route (p1/split4-beyond16) ----
# Above ATOM_M3_AK_DECODE_MAX_REQS requests the decode runs aiter's NVFP4 scorer, then ATOM's fused q8 decode, which
# selects the top-16 itself (aiter's lane-chunk tie rule) and attends. This route keeps the q8 decode and its tie rule
# and replaces only the scorer: split4's ``score_push`` (rank r scores blocks b % 4 == r of every head, fp32 raw
# max(q . k) into the head owner's buffer, rank-owned order), then ``_split4_restore`` puts this rank's head back in
# natural block order times fp32(sm_scale * LOG2E) -- the multiply aiter's scorer does -- into the scorer's own
# workspace. Same scores bit for bit on every live block, so the same ids and attention bytes (4-GPU check:
# /shared_nfs/ygan/p1/split4-beyond16/NOTES.md). Needs ATOM_P1_INDEXER_SPLIT4 (its IPC buffers). Default off.
SPLIT4_WIDE = SPLIT4 and os.environ.get("ATOM_P1_INDEXER_SPLIT4_WIDE", "0") == "1"
# from this many requests, by rows per request (only reached above DECODE_MAX_REQS): see NOTES.md for the crossover
SPLIT4_WIDE_MIN_REQS = {8: 17, 4: 17}
_RESTORE_TILE = 512


@functools.cache
def _split4_restore_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _split4_restore(Src, Dst, Lengths, SRC_STRIDE, DST_STRIDE, SCALE, QUERY_LEN: tl.constexpr,
                        TILE: tl.constexpr):
        """Row ``pid``'s live blocks (``(position + 128) // 128`` of them, position = length - QUERY_LEN + row %
        QUERY_LEN) from split4's rank-owned order (block 64 c + 4 j + r at 64 c + 16 r + j) into natural order, times
        ``SCALE`` (an fp32 multiply, as aiter's scorer applies its folded scale after the max). Columns past the
        row's live blocks are not written (no reader looks there)."""
        row = tl.program_id(0)
        length = tl.load(Lengths + row // QUERY_LEN)
        live = (length - QUERY_LEN + row % QUERY_LEN + 1 + 127) // 128
        for t in range(0, tl.cdiv(live, TILE)):
            b = t * TILE + tl.arange(0, TILE)
            src = (b // 64) * 64 + (b % 4) * 16 + (b % 64) // 4
            ok = b < live
            v = tl.load(Src + row * SRC_STRIDE + src, mask=ok)
            tl.store(Dst + row * DST_STRIDE + b, v * SCALE, mask=ok)

    return _split4_restore


def split4_wide_takes(num_reqs: int, query_len: int, num_index_heads: int) -> bool:
    """Whether a fused-NVFP4 decode of ``num_reqs`` x ``query_len`` rows scores on four GPUs (rank-invariant)."""
    if not SPLIT4_WIDE or _Split4.index is None or _Split4.index.split is None:
        return False
    cp = _Split4.index.split
    return (num_index_heads == 1 and query_len == cp.query_len and num_reqs * query_len <= cp.max_rows
            and num_reqs >= SPLIT4_WIDE_MIN_REQS.get(query_len, 1 << 30))


def split4_wide_scores(index_q, index_cache, block_table, seq_lens, init_blocks, local_blocks, sm_scale, out,
                       query_len, split=None):
    """aiter's NVFP4 decode scores of this rank's index head, on four GPUs: ``index_q`` [rows, 1, 128] e4m3,
    ``index_cache`` ATOM's NVFP4 index pool (replicated), ``out`` [1, >= rows, width] fp32 (the scorer workspace view
    the fused decode reads; only live blocks are written). A collective: every rank, same call, same order."""
    import triton
    from aiter.ops.flydsl.kernels.minimax_m3_index_score import LOG2E

    cp = split if split is not None else _Split4.index.split
    rows = index_q.shape[0]
    score = cp.score_push(index_q[:, 0], index_pages(index_cache), block_table, seq_lens, init_blocks, local_blocks,
                          cp.max_width, sharded=False)
    _split4_restore_kernel()[(rows,)](
        score, out, seq_lens, score.stride(1), out.stride(1), float(sm_scale * LOG2E),
        QUERY_LEN=query_len, TILE=_RESTORE_TILE,
    )
    return out


def warm_up_split4_wide(split, device, sm_scale: float = 128**-0.5) -> None:
    """Compiles split4's score-only push at <= 32 and > 32 requests and the restore, with real launches, the same
    calls in the same order on every rank -- before graph capture."""
    ql = split.query_len
    pages = torch.zeros((2, BLK * 72), dtype=torch.uint8, device=device)
    for reqs in sorted({min(32, split.max_rows // ql), split.max_rows // ql}):
        rows = reqs * ql
        iq = torch.zeros((rows, 1, 128), dtype=torch.float8_e4m3fn, device=device)
        table = torch.zeros((reqs, 1), dtype=torch.int32, device=device)
        seq = torch.full((reqs,), ql, dtype=torch.int32, device=device)
        out = torch.empty((1, rows, split.max_width), dtype=torch.float32, device=device)
        split4_wide_scores(iq, pages, table, seq, 0, 1, sm_scale, out, ql, split=split)
    torch.cuda.synchronize(device)


def index_sparse_decode(index_q, index_cache, q, kv_cache, block_table, seq_lens, init_blocks, local_blocks,
                        sm_scale, out, k_scale, v_scale, query_len, max_seq_len, num_kv_heads, quant_out=None):
    """Index top-16 + sparse decode of the decode rows; returns ``topk_idx`` [1, rows, 16] int32 (the ids a top-k
    reusing layer reads). ``quant_out`` (``qout_buffers``): write o_proj's MXFP8 input instead of ``out``.
    With ``_Split4.index`` the route (four GPUs or one) follows the request count, the same on every rank."""
    fused, attn = _decode_fns()
    rows = q.shape[0]
    topk_idx = torch.empty((num_kv_heads, rows, 16), dtype=torch.int32, device=q.device)
    reqs = seq_lens.shape[0]
    if (AK_WIDE and query_len == 8 and AK_WIDE_ONE_SPLIT[0] < rows <= AK_WIDE_ONE_SPLIT[1]
            and ak_wide_takes(reqs, query_len, index_q.shape[1])):
        # ATOM_P1_AK_WIDE at 129-192 rows: split4's top-16, then the attention in one split (the host default is 4
        # splits here: 17.7 / 18.3 / 19.3 -> 16.1 / 16.2 / 16.7 us at 136 / 160 / 192 rows, 175K, cold, MXFP8 out)
        _Split4.index(index_q, index_pages(index_cache), block_table, seq_lens, init_blocks, local_blocks, topk_idx,
                      query_len=query_len, max_seq_len=max_seq_len)
        attn(q, kv_pages(kv_cache, num_kv_heads), topk_idx, block_table, seq_lens, sm_scale, out, k_scale=k_scale,
             v_scale=v_scale, decode_query_len=query_len, quant_out=quant_out, compute_type=_COMPUTE, splits=1)
        return topk_idx
    fused(_Split4.index, index_q, index_pages(index_cache), block_table, seq_lens, init_blocks, local_blocks, q,
          kv_pages(kv_cache, num_kv_heads), topk_idx, block_table, seq_lens, sm_scale, out, k_scale=k_scale,
          v_scale=v_scale, decode_query_len=query_len, max_seq_len=max_seq_len, quant_out=quant_out,
          compute_type=_COMPUTE)
    return topk_idx


def sparse_decode(q, kv_cache, topk_idx, block_table, seq_lens, sm_scale, out, k_scale, v_scale, query_len,
                  num_kv_heads, quant_out=None):
    _, attn = _decode_fns()
    attn(q, kv_pages(kv_cache, num_kv_heads), topk_idx[:, : q.shape[0]], block_table, seq_lens, sm_scale, out,
         k_scale=k_scale, v_scale=v_scale, decode_query_len=query_len, quant_out=quant_out, compute_type=_COMPUTE)


# ---- ATOM_M3_AK_DECODE_QOUT: attention -> o_proj on MXFP8 codes (p1/attn-decode) ----

@functools.cache
def _qout_min_rows() -> int:
    from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import host as decode_host

    return decode_host.MXFP8_OUT_MIN_ROWS


def qout_buffers(rows: int, n: int, k: int, device):
    """``quant_out`` for a ``rows``-row decode feeding an MXFP8 [n, k] GEMM: (codes u8, e8m0 scales u8, tile) in
    ``gemm.prequant_buffers``' layout (vLLM's ``nvfp4_sparse_layer.quant_out_buffers``); None below the attention's
    MXFP8-output minimum (the bf16 output + the GEMM's own quantizer are faster there) and below ``QOUT_MIN_ROWS``
    (p1/attn-decode round 2, cold graph replay on MI355X: 8 rows (k7 G1) -0.04..+0.01 us, i.e. neutral, so G1 keeps
    today's bit-identical path; 16-128 rows -0.7..-2.6 us per layer)."""
    if rows < max(_qout_min_rows(), QOUT_MIN_ROWS):
        return None
    from amd_kernels.gemm.mxfp8_common import gemm

    return gemm.prequant_buffers(rows, n, k, device)


def prequant_linear(quant_out, rows: int, weight: torch.Tensor, ak_weight_scale: torch.Tensor) -> torch.Tensor:
    """o_proj [rows, N] bf16 on the codes the attention wrote: ``mxfp8_gemm_shuffled`` on the (16, 16) tile layout,
    else ``mxfp8_gemm`` on row-major codes (amd-kernels benchmarks/attention/bench_attn_oproj_mxfp8.py
    ``gemm_codes``). Same codes as ``mxfp8_linear``'s quantizer of the bf16 output; the GEMM row may be the
    prequant table's instead of the bf16 one (fp32 summation order only)."""
    from amd_kernels.gemm.mxfp8_common import gemm

    codes, scales, tile = quant_out
    c8 = codes.view(torch.float8_e4m3fn)
    if tile:
        return gemm.mxfp8_gemm_shuffled(c8, scales, weight, ak_weight_scale, torch.bfloat16, rows)
    return gemm.mxfp8_gemm(c8[:rows], scales, weight, ak_weight_scale, torch.bfloat16,
                           activation_scale_preshuffled=True)
