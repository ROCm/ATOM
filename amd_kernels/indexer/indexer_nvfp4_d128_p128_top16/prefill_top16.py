# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The prefill top-16 of a step (or a micro-batch) in one call: the one-GPU block scores of every prefill row
(``score_prefill.nvfp4_index_score_prefill``: 128-row tiles on small work, else 512-row tiles) and their top-16 ids
(``topk.index_topk_prefill_indices``: a window tile per ``max_seq_len``). The ids are ``reference.topk_reference``
of the scores. An entry (``amd_kernels.warm_up``): its first call compiles every variant the served sizes reach.
"""

import torch

from amd_kernels.warm_up import entry

from .host_stream import current_stream
from .reference import BLK, D, TOPK, score_width
from .score_prefill.host import nvfp4_index_score_prefill
from .topk.host import index_topk_prefill_indices
from .topk.prefill_kernel import PREFILL_STRIPS

INDEX_HEADS = (1, 2, 4)
MAX_SEQ_LEN = 1 << 20  # the top-k's 8192 blocks
# Warm-up calls: one request of WARM_ROWS new rows per context, one context per top-k window tile (``PREFILL_STRIPS``
# x 64 blocks) and the longest (the last tile over several windows). Up to 212992 tokens that is small work (the
# scorer's 128-row tiles), at MAX_SEQ_LEN the 512-row tiles.
WARM_ROWS = 512
WARM_CONTEXTS = tuple(s * 64 * BLK for s in PREFILL_STRIPS) + (MAX_SEQ_LEN,)


def index_prefill_top16_blocker(
    *, num_index_heads: int, index_head_dim: int, block_size: int, topk_blocks: int
) -> str | None:
    """Why a model's prefill indexer does not fit this entry (None: it fits): gfx950, 1, 2 or 4 index heads per GPU,
    head dim 128, 128-token blocks, top-16."""
    from .gfx950 import on_gfx950

    if not on_gfx950():
        return "needs gfx950"
    if (index_head_dim, block_size, topk_blocks) != (D, BLK, TOPK):
        return (
            f"index head dim {index_head_dim}, block size {block_size}, top-{topk_blocks} "
            f"(needs {D}, {BLK}, top-{TOPK})"
        )
    if num_index_heads not in INDEX_HEADS:
        return f"{num_index_heads} index heads per GPU (needs {INDEX_HEADS})"
    return None


def index_prefill_top16(
    iq: torch.Tensor,  # [rows, H, 128] e4m3, rows contiguous
    index_cache: torch.Tensor,  # NVFP4 index pages [num_blocks, 128, 72] uint8
    block_table: torch.Tensor,  # [batch, >= cdiv(max_seq_len, 128)] int32
    cu_seqlens_q: torch.Tensor,  # [batch + 1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    init_blocks: int,
    local_blocks: int,
    topk_idx: torch.Tensor,  # [H, rows, 16] int32, written
    *,
    max_query_len: int,
    max_seq_len: int,
    num_valid_pages: torch.Tensor,  # [>= rows] int32: row n's causal blocks
    score: torch.Tensor | None = None,
    stream=None,
) -> torch.Tensor:
    """The top-16 block ids of every prefill row (ragged requests, each request's rows after its cached prefix) into
    ``topk_idx``, -1 past a row's block count; returns ``topk_idx``.

    Args:
        score: the scores' workspace ``[H, rows, >= reference.score_width(max_seq_len)]`` fp32 (default: allocated
            per call).
        stream: the launch stream (default the current one).
    """
    rows, heads, _ = iq.shape
    if rows == 0:
        return topk_idx
    if score is None:
        score = torch.empty((heads, rows, score_width(max_seq_len)), dtype=torch.float32, device=iq.device)
    if stream is None:
        stream = current_stream()
    nvfp4_index_score_prefill(
        iq, index_cache, score, block_table, cu_seqlens_q, seq_lens, init_blocks, local_blocks,
        max_query_len, max_seq_len, stream=stream,
    )  # fmt: skip
    index_topk_prefill_indices(
        score, topk_idx, max_seq_len=max_seq_len, block_size=BLK, num_valid_pages=num_valid_pages, stream=stream
    )
    return topk_idx


def _mode(iq, *args, **kwargs):
    return iq.shape[1], iq.device.index


def _sizes(mode):
    return WARM_CONTEXTS


def _call(heads: int, n: int, device):
    """(args, kwargs) of one request of ``min(n, WARM_ROWS)`` new rows at context ``n`` (zeros: the dispatch reads
    the host lengths only; compile-only calls launch nothing)."""
    rows = min(n, WARM_ROWS)
    args = (
        torch.zeros(rows, heads, D, dtype=torch.float8_e4m3fn, device=device),
        torch.zeros(2, BLK, 72, dtype=torch.uint8, device=device),
        torch.zeros(1, -(-n // BLK), dtype=torch.int32, device=device),
        torch.tensor([0, rows], dtype=torch.int32, device=device),
        torch.tensor([n], dtype=torch.int32, device=device),
        0,
        1,
        torch.empty(heads, rows, TOPK, dtype=torch.int32, device=device),
    )
    nvp = torch.ones(rows, dtype=torch.int32, device=device)
    return args, dict(max_query_len=rows, max_seq_len=n, num_valid_pages=nvp)


def _inputs(mode, n, iq, *args, **kwargs):
    return _call(mode[0], n, iq.device)


index_prefill_top16 = entry(
    "indexer/index_prefill_top16_d128_p128", mode=_mode, sizes=_sizes, inputs=_inputs
)(index_prefill_top16)


def warm_up(device, num_index_heads: int) -> None:
    """Before serving: one small call, whose first-call warm-up compiles every variant of ``num_index_heads`` heads
    on ``device`` (``WARM_CONTEXTS``); nothing at a head count the entry does not serve."""
    if num_index_heads not in INDEX_HEADS:
        return
    with torch.inference_mode():
        args, kwargs = _call(num_index_heads, WARM_CONTEXTS[0], torch.device(device))
        index_prefill_top16(*args, **kwargs)
