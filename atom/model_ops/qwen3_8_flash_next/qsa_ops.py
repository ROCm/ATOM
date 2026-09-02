"""Qwen3.8-Flash-Next QSA operators: compressed scoring, selection, and sparse GQA.

Thin launchers over the vendored AITER Triton kernels (see `kernels/`),
following the same three-stage contract the upstream vLLM implementation uses:

  1. `qsa_paged_mqa_logits` scores every causally visible compressed key group
     with `sum_h relu(q_h . k_c) / sqrt(head_dim)`.
  2. `qsa_select_paged_tokens` runs radix top-k over those scores and expands
     the winning groups into logical token ids, appending the incomplete
     causal tail so the newest tokens are never dropped.
  3. `qsa_sparse_paged_gqa` runs online-softmax grouped-query attention over
     exactly those token ids, reading separate paged BF16 K/V caches.

The top-k comes from AITER proper (`aiter.ops.topk.top_k_per_row_prefill`),
which is already present and writes -1 into the unused tail of each row.
"""

import torch
import triton
from aiter.ops.topk import top_k_per_row_prefill

from atom.model_ops.qwen3_8_flash_next.kernels.qsa_expand_indices import (
    _qsa_expand_block_indices_kernel,
)
from atom.model_ops.qwen3_8_flash_next.kernels.qsa_paged_mqa_logits import (
    _qsa_paged_mqa_logits_kernel,
)
from atom.model_ops.qwen3_8_flash_next.kernels.qsa_sparse_paged_gqa import (
    _qsa_sparse_paged_gqa_kernel,
)

# Cap on the FP32 logits buffer; scoring is chunked over query rows to respect
# it, because `columns` grows with the paged-cache capacity.
DEFAULT_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024

_SCORING_BLOCK_N = 32
_EXPAND_BLOCK_N = 256


def _check_vector(name: str, tensor: torch.Tensor, length: int | None = None) -> None:
    if tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be a one-dimensional int32/int64 tensor")
    if length is not None and tensor.shape[0] != length:
        raise ValueError(f"{name} must contain {length} entries")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def qsa_paged_mqa_logits(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    compress_ratio: int = 4,
    score_divisor: float | None = None,
    max_columns: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score compressed key groups.

    `q` is `[tokens, index_heads, head_dim]` and `compressed_k_cache` is
    `[pages, page_size, 1, head_dim]`. Returns FP32 logits `[tokens, columns]`
    and the per-row count of causally visible groups.

    `max_columns` caps how many compressed groups are scored. Column `c`
    addresses group `c` through the page table, so dropping the tail simply
    stops at a group no row can see -- the scores of the ones that remain are
    unchanged. Without it the width is the whole block table, and a 1k-token
    request on a 16k-token engine pays for 16k tokens of scoring and top-k on
    every one of the 12 QSA layers.
    """
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("q must have shape [tokens, heads, head_dim]")
    if (
        compressed_k_cache.ndim != 4
        or compressed_k_cache.shape[2] != 1
        or compressed_k_cache.shape[3] != q.shape[2]
    ):
        raise ValueError(
            "compressed_k_cache must have shape [pages, page_size, 1, head_dim]"
        )
    if q.dtype != compressed_k_cache.dtype:
        raise ValueError("q and compressed_k_cache must have the same dtype")
    if page_table.ndim != 2 or page_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("page_table must be a two-dimensional integer tensor")
    _check_vector("token_to_request", token_to_request, q.shape[0])
    _check_vector("query_positions", query_positions, q.shape[0])
    _check_vector("context_lens", context_lens, page_table.shape[0])

    divisor = q.shape[2] ** 0.5 if score_divisor is None else score_divisor
    if divisor <= 0:
        raise ValueError("score_divisor must be positive")
    columns = page_table.shape[1] * compressed_k_cache.shape[1]
    if max_columns is not None:
        columns = max(1, min(columns, int(max_columns)))
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_groups = torch.zeros(q.shape[0], dtype=torch.int32, device=q.device)
    if q.shape[0] == 0 or columns == 0:
        return logits, visible_groups

    _qsa_paged_mqa_logits_kernel[(q.shape[0], triton.cdiv(columns, _SCORING_BLOCK_N))](
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        visible_groups,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        compressed_k_cache.stride(0),
        compressed_k_cache.stride(1),
        compressed_k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        compressed_k_cache.shape[0],
        page_table.shape[0],
        float(divisor),
        PAGE_SIZE=compressed_k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        COMPRESS_RATIO=compress_ratio,
        BLOCK_N=_SCORING_BLOCK_N,
        BLOCK_D=triton.next_power_of_2(q.shape[2]),
        num_warps=4,
    )
    return logits, visible_groups


def qsa_expand_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    token_to_request: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
    output_width: int | None = None,
) -> torch.Tensor:
    """Expand selected groups to token ids and append the causal tail.

    `output_width` may be narrower than the full `token_topk + ratio - 1` when
    the caller knows the context cannot fill it: column `c` of the result
    depends only on `c` and the row's own counts, so truncating drops columns
    that would have been `-1` padding anyway.
    """
    if compress_ratio <= 0 or token_topk <= 0:
        raise ValueError("compress_ratio and token_topk must be positive")
    if block_indices.ndim != 2 or block_indices.dtype != torch.int32:
        raise ValueError("block_indices must be a two-dimensional int32 tensor")
    if token_topk % compress_ratio:
        raise ValueError("token_topk must be divisible by compress_ratio")
    _check_vector("query_positions", query_positions, block_indices.shape[0])
    _check_vector("token_to_request", token_to_request, block_indices.shape[0])
    _check_vector("context_lens", context_lens)

    block_topk = block_indices.shape[1]
    if block_topk > token_topk // compress_ratio:
        raise ValueError(f"block_indices must have at most {token_topk // compress_ratio} columns")
    if output_width is None:
        output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.shape != (block_indices.shape[0], output_width):
        raise ValueError("out has an invalid shape")
    if block_indices.shape[0] == 0:
        return out

    _qsa_expand_block_indices_kernel[
        (block_indices.shape[0], triton.cdiv(output_width, _EXPAND_BLOCK_N))
    ](
        block_indices,
        query_positions,
        context_lens,
        token_to_request,
        out,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        block_indices.shape[0],
        context_lens.shape[0],
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        OUTPUT_WIDTH=output_width,
        BLOCK_N=_EXPAND_BLOCK_N,
        num_warps=4,
    )
    return out


def qsa_select_paged_tokens(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    token_topk: int,
    compress_ratio: int = 4,
    out: torch.Tensor | None = None,
    logits_workspace_bytes: int = DEFAULT_LOGITS_WORKSPACE_BYTES,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Score, select and expand in one call.

    Returns `[tokens, width]`, where `width` is `budget + ratio - 1` unless
    `max_seq_len` -- the longest sequence in the batch -- shows the context
    cannot fill it, in which case both the scored width and the emitted width
    shrink to what the context can actually reach. Everything dropped would
    have been `-1` padding, so the result is identical column for column, but
    a 1k-token request on a 16k-token engine stops paying 16k-token scoring
    and top-k costs on each of the 12 QSA layers.
    """
    if token_topk % compress_ratio:
        raise ValueError("token_topk must be divisible by compress_ratio")
    rows = q.shape[0]

    columns = page_table.shape[1] * compressed_k_cache.shape[1]
    block_topk = token_topk // compress_ratio
    if block_topk > columns:
        raise ValueError("compressed top-k exceeds paged-cache capacity")
    if max_seq_len is not None:
        reachable = (int(max_seq_len) + compress_ratio - 1) // compress_ratio
        # Round the bucket up to a power of two. `BLOCK_TOPK`, `OUTPUT_WIDTH`
        # and `TOPK` are Triton constexprs, so an exact width would recompile
        # the expand and GQA kernels every few decoded tokens -- which costs
        # far more than the work it saves. Rounding up is always safe: a
        # top-k wider than the visible groups just returns -1 for the extras.
        bucket = 1 << (max(reachable, 1) - 1).bit_length()
        columns = max(1, min(columns, bucket))
        block_topk = max(1, min(block_topk, bucket))
    output_width = block_topk * compress_ratio + compress_ratio - 1

    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    elif out.shape[0] != rows or out.shape[1] < output_width:
        raise ValueError("out has an invalid shape")
    else:
        out = out[:, :output_width]
    if rows == 0:
        return out

    # Chunk over query rows so the FP32 logits buffer stays within budget.
    rows_per_chunk = max(1, logits_workspace_bytes // max(columns * 4, 1))
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        logits, visible_groups = qsa_paged_mqa_logits(
            q[row_slice],
            compressed_k_cache,
            page_table,
            token_to_request[row_slice],
            query_positions[row_slice],
            context_lens,
            compress_ratio,
            max_columns=columns,
        )
        selected_groups = torch.empty(
            (row_end - row_start, block_topk), dtype=torch.int32, device=q.device
        )
        top_k_per_row_prefill(
            logits,
            torch.zeros_like(visible_groups),
            visible_groups,
            selected_groups,
            None,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            block_topk,
        )
        qsa_expand_block_indices(
            selected_groups,
            query_positions[row_slice],
            context_lens,
            token_to_request[row_slice],
            compress_ratio,
            token_topk,
            out[row_slice],
            output_width=output_width,
        )
    return out


def qsa_sparse_paged_gqa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Grouped-query attention restricted to `logical_indices` (-1 = padding)."""
    if q.ndim != 3:
        raise ValueError("q must be [tokens, query_heads, head_dim]")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("K/V caches must have matching [pages, page, heads, dim]")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q, k_cache and v_cache must have the same dtype")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("query heads must form equal groups over KV heads")
    if (
        logical_indices.ndim != 2
        or logical_indices.shape[0] != q.shape[0]
        or logical_indices.dtype != torch.int32
    ):
        raise ValueError("logical_indices must be int32 [tokens, selection_width]")
    if block_table.ndim != 2:
        raise ValueError("block_table must be a two-dimensional integer tensor")
    _check_vector("token_to_request", token_to_request, q.shape[0])

    scale = q.shape[2] ** -0.5 if softmax_scale is None else softmax_scale
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype:
        raise ValueError("out must match q")
    if q.shape[0] == 0:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = max(16, triton.next_power_of_2(group_size))
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    _qsa_sparse_paged_gqa_kernel[(q.shape[0], k_cache.shape[2])](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_request,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        logical_indices.stride(0),
        logical_indices.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        float(scale),
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        NUM_KV_HEADS=k_cache.shape[2],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        BLOCK_M=block_m,
        BLOCK_N=16,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return out
