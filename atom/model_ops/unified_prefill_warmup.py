# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Host-side half of the prefix-hit prefill unified_attention path.

The token table, the start-up warm-up shape list and the warm-up gate. Kept out
of `attention_mha` (which imports triton and aiter) so that they need only
torch and run in the CPU unit-test job.
"""

import logging
import time

import torch
from torch import nn

from atom.utils import envs

logger = logging.getLogger("atom")

# Triton specializes an int kernel argument on whether it is a multiple of 16,
# and the token table's row stride is its column count. Rounding the columns up
# keeps every unified_attention launch on one specialization, which halves the
# kernels the start-up warm-up has to compile.
_TOKEN_TABLE_COL_ALIGN = 16

# Prefix length the warm-up launches use: long enough that the KV-split
# heuristics see their full segment caps (128 tiles of 64 at q >= 64).
_UNIFIED_WARMUP_PREFIX = 16384

# Requests per step the small-row warm-up launches use. The request count is a
# kernel argument with its own Triton specializations (1, a multiple of 16, or
# neither), and with the new rows it sets the program count, which picks the KV
# split: each doubling of the programs halves the split. So both multi-request
# kinds step by <= 2x up to 256+ requests, reaching every split count down to
# the unsplit grid for the row counts below. ("Neither" stopped at 12 before,
# so a 31-request, 16-row step compiled a new split on first use.)
_UNIFIED_WARMUP_NUM_SEQS = (
    *(1, 2, 3, 4, 6, 8, 12, 24, 40, 72, 136, 264),  # 1 and "neither"
    *(16, 32, 64, 128, 256),  # multiples of 16
)

# Context of every request but the first in a multi-request launch. Kernel
# selection reads only the longest context (max_seqlen_k), so the others need
# no long prefix; they just have to be prefix hits (context > new rows).
_UNIFIED_WARMUP_SHORT_CTX = 16

# Default of ATOM_PREFILL_UNIFIED_ATTN_MAX_TOTAL: the most new rows, summed over
# the step's requests, that a multi-request prefix-hit step may have and still
# take unified_attention (see unified_prefill_step_qualifies).
_UNIFIED_DEFAULT_MAX_TOTAL = 2048

# Byte cap on one multi-request warm-up launch's q, output and K/V tensors,
# taken before KV sizing (see unified_prefill_warmup_bytes).
_UNIFIED_WARMUP_BYTES = 128 << 20


def token_block_table(
    cu_seqlens_k: torch.Tensor, max_seqlen_k: int, total_kv: int
) -> torch.Tensor:
    """Row i lists the gathered K/V rows of request i, for a block_size=1 cache.

    cu_seqlens_k holds the n requests' start rows. Columns past a request's own
    length are never read (seqused_k stops it); the clamp keeps them inside the
    gathered tensor.
    """
    n_cols = -(-max_seqlen_k // _TOKEN_TABLE_COL_ALIGN) * _TOKEN_TABLE_COL_ALIGN
    cols = torch.arange(n_cols, dtype=cu_seqlens_k.dtype, device=cu_seqlens_k.device)
    return (cu_seqlens_k[:, None] + cols).clamp_(max=total_kv - 1)


def unified_prefill_step_qualifies(
    num_seqs: int, total_rows: int, max_rows: int, max_q: int, max_total: int
) -> bool:
    """Whether a prefix-hit prefill step takes unified_attention (vs FMHA v3).

    A single request qualifies up to max_q new rows. A step with several
    requests qualifies only while its total new rows stay within max_total
    (and each request within max_q): once the step has a few thousand rows,
    FMHA v3 has enough workgroups to fill the device and unified_attention is
    slower. Measured on Qwen3.5-397B TP4 at a 128k prefix (step GPU ms,
    FMHA v3 -> unified): 1 x 2032 rows 136 -> 97, 4 x 496 144 -> 106, but
    8 x 496 194 -> 231 and 4 x 2032 270 -> 346. max_total <= 0 keeps every
    multi-request step on FMHA v3.
    """
    if not 0 < max_rows <= max_q:
        return False
    return num_seqs == 1 or (max_total > 0 and total_rows <= max_total)


def unified_prefill_warmup_bytes(
    q_lens, ctx_lens, num_heads: int, num_kv_heads: int, head_dim: int
) -> int:
    """Bytes of one warm-up launch's synthetic tensors: FP8 K (also passed as
    V) for every context row, FP8 q plus the 16-bit output for every new row.

        (sum(ctx_lens) * num_kv_heads + 3 * sum(q_lens) * num_heads) * head_dim

    The KV-split scratch aiter allocates on top is not counted.
    """
    return (sum(ctx_lens) * num_kv_heads + 3 * sum(q_lens) * num_heads) * head_dim


def unified_prefill_warmup_shapes(
    max_q: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_total: int = _UNIFIED_DEFAULT_MAX_TOTAL,
    prefix: int = _UNIFIED_WARMUP_PREFIX,
    max_bytes: int = _UNIFIED_WARMUP_BYTES,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """(new rows, context rows) per request for each warm-up launch.

    aiter compiles one unified_attention kernel per launch configuration: the
    query block (set by the longest request's new rows), the KV split count
    (set by the total program count and the longest context), and the request
    count's Triton specialization (1, a multiple of 16, or neither). A first use
    whose kernel is not in the Triton disk cache yet costs ~1 s of host time on
    every rank, so the warm-up runs a launch from every bucket the prefix-hit
    prefill path can reach. Rows >= 64 step by 1.25x, narrower than any split
    bucket (adjacent split counts are >= 1.33x apart in program count); the
    short contexts reach the split counts that the context length, not the
    program count, caps.

    Only steps unified_prefill_step_qualifies admits are warmed: single
    requests up to max_q rows, multi-request steps up to max_total rows in all.
    The first request carries the longest context; the others get a short one
    (selection reads only the maximum). In a multi-request launch the other
    requests' new rows are trimmed, if needed, so that the launch stays within
    max_total rows and unified_prefill_warmup_bytes within max_bytes for this
    head geometry; a trimmed launch is still a real step, just one with fewer
    new rows. A multi-request launch that cannot fit (the first request alone
    leaves no row per other request) is dropped: no step of that shape takes
    the path. Single-request launches are never trimmed: a real step of that
    size allocates as much on its own.
    """
    per_row = (num_kv_heads + 3 * num_heads) * head_dim

    def launch(q, n_others, rows, ctx):
        if n_others:
            if max_total <= 0 or q + n_others > max_total:
                return []
            fixed = unified_prefill_warmup_bytes(
                (q,),
                (ctx + n_others * _UNIFIED_WARMUP_SHORT_CTX,),
                num_heads,
                num_kv_heads,
                head_dim,
            )
            rows = min(rows, (max_total - q) // n_others)
            rows = max(1, min(rows, (max_bytes - fixed) // (n_others * per_row)))
        return [
            (
                (q,) + (rows,) * n_others,
                (ctx,) + (rows + _UNIFIED_WARMUP_SHORT_CTX,) * n_others,
            )
        ]

    shapes = []
    big = []
    q = 64.0
    while q < max_q:
        big.append(int(q))
        q *= 1.25
    if max_q >= 64:
        big.append(max_q)
    for q in sorted(set(big)):
        shapes += [
            *launch(q, 0, 0, q + prefix),
            *launch(q, 1, q, q + prefix),
            *launch(q, 1, 2, q + prefix),
            *launch(q, 15, q, q + prefix),
        ]
    for q in (1, 2, 4, 8, 16, 32):
        if q > max_q:
            continue
        for n in _UNIFIED_WARMUP_NUM_SEQS:
            shapes += launch(q, n - 1, q, q + prefix)
        for ctx in (320, 960, 1984):
            shapes += [*launch(q, 0, 0, ctx), *launch(q, 1, q, ctx)]
    return shapes


_warned_no_impl = False


def warmup_unified_prefill_attention(model: nn.Module, dtype: torch.dtype) -> None:
    """Compile the prefix-hit prefill unified_attention kernels at start-up.

    Without this each new launch configuration compiles on the first request
    that needs it, stalling that step by ~1 s on every rank. One impl per
    distinct head geometry is enough: the compiled kernels are per process, not
    per layer. The impls are the modules that carry
    `warmup_unified_fp8_prefill_attention` (attention_mha.PagedAttentionImpl).
    """
    if (
        envs.ATOM_PREFILL_UNIFIED_ATTN_MAX_Q <= 0
        or not envs.ATOM_PREFILL_UNIFIED_ATTN_WARMUP
    ):
        return
    global _warned_no_impl
    done = set()
    found = False
    for module in model.modules():
        warmup = getattr(module, "warmup_unified_fp8_prefill_attention", None)
        if warmup is None:
            continue
        found = True
        geometry = (module.num_heads, module.num_kv_heads, module.head_dim)
        if geometry in done or not module._unified_fp8_prefill_supported():
            continue
        done.add(geometry)
        start = time.perf_counter()
        n = warmup(dtype)
        logger.info(
            "prefix-hit prefill unified_attention warm-up: %d launches "
            "(heads %d/%d, head_dim %d, one request <= %d new rows, "
            "several <= %d in total) in %.1f s",
            n,
            module.num_heads,
            module.num_kv_heads,
            module.head_dim,
            envs.ATOM_PREFILL_UNIFIED_ATTN_MAX_Q,
            envs.ATOM_PREFILL_UNIFIED_ATTN_MAX_TOTAL,
            time.perf_counter() - start,
        )
    if not found and not _warned_no_impl:
        _warned_no_impl = True
        logger.warning(
            "ATOM_PREFILL_UNIFIED_ATTN_MAX_Q=%d but no module has "
            "warmup_unified_fp8_prefill_attention: nothing was warmed up, so "
            "each prefix-hit unified_attention kernel compiles on first use",
            envs.ATOM_PREFILL_UNIFIED_ATTN_MAX_Q,
        )
