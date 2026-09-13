# SPDX-License-Identifier: MIT
"""Tiled CSA2 scoring/top-k and compact per-query candidate block selection."""

import torch
import torch.nn.functional as F

from atom.models.deepseek_v41.config import IndexTieBreak


def _merge_topk(best_scores, best_ids, scores, ids, k):
    if best_scores is not None:
        scores = torch.cat((best_scores, scores), dim=-1)
        ids = torch.cat((best_ids, ids), dim=-1)
    count = min(k, scores.shape[-1])
    # Tiles and candidate blocks arrive in the configured priority order.
    # Stable merges preserve that order at exact ties, including ReLU zeros.
    values, selected = scores.sort(dim=-1, descending=True, stable=True)
    values, selected = values[..., :count], selected[..., :count]
    return values, ids.expand_as(scores).gather(-1, selected)


def _ordered_indices(scores, indices):
    valid = torch.isfinite(scores)
    sentinel = torch.iinfo(torch.int64).max
    ordered = indices.long().masked_fill(~valid, sentinel).sort(dim=-1).values
    return torch.where(ordered == sentinel, -1, ordered).int()


class TensorIndexKeys:
    """Contiguous key access with the same tile/gather contract as paged keys."""

    def __init__(self, keys):
        self.keys, self.shape = keys, keys.shape

    def tile(self, start, end):
        return self.keys[:, start:end]

    def gather(self, ids):
        batches = torch.arange(self.shape[0], device=ids.device).view(-1, 1, 1)
        return self.keys[batches, ids]


def select_indices(
    q,
    weights,
    keys,
    visible_lengths,
    *,
    topk,
    candidate_blocks=None,
    make_candidates=False,
    block_size=8,
    topk_blocks=2048,
    query_tile=32,
    key_tile=1024,
    tie_break=IndexTieBreak.SMALL_POSITION,
):
    """Score BF16 QAT values with bounded temporary storage.

    Returns sorted position IDs and, for a candidate source, compact block IDs.
    Reindex gathers only candidate blocks. Equal-score top-k ties follow
    tie_break; returned indices are always sorted in ascending position order.
    Candidate block ties follow the same policy, while the newest block is pinned.
    candidate_blocks contains ascending IDs from the candidate source.
    """
    if isinstance(keys, torch.Tensor):
        keys = TensorIndexKeys(keys)
    prefer_large = IndexTieBreak(tie_break) == IndexTieBreak.LARGE_POSITION
    batch, queries, _, _ = q.shape
    width = keys.shape[1]
    if key_tile % block_size:
        raise ValueError("Index key tiles must align with candidate blocks")
    count = min(topk, width)
    output = torch.full((batch, queries, count), -1, dtype=torch.int32, device=q.device)
    block_count = min(topk_blocks, (width + block_size - 1) // block_size)
    candidates = (
        torch.full(
            (batch, queries, block_count), -1, dtype=torch.int32, device=q.device
        )
        if make_candidates
        else None
    )
    if width == 0:
        return output, candidates
    for q0 in range(0, queries, query_tile):
        q1 = min(q0 + query_tile, queries)
        query, head_weights = q[:, q0:q1], weights[:, q0:q1]
        visible = visible_lengths[q0:q1].view(1, -1, 1)
        best_scores = best_ids = block_scores = block_ids = None
        if candidate_blocks is None:
            key_count = width
            positions = None
        else:
            blocks = candidate_blocks[:, q0:q1].long()
            offsets = torch.arange(block_size, device=q.device)
            positions = (blocks[..., None] * block_size + offsets).flatten(-2)
            positions = positions.masked_fill(
                (blocks[..., None] < 0).expand(*blocks.shape, block_size).flatten(-2),
                -1,
            )
            key_count = positions.shape[-1]
        key_starts = range(0, key_count, key_tile)
        for k0 in reversed(key_starts) if prefer_large else key_starts:
            k1 = min(k0 + key_tile, key_count)
            if positions is None:
                ids = (
                    torch.arange(k0, k1, device=q.device)
                    .view(1, 1, -1)
                    .expand(batch, q1 - q0, -1)
                )
                dots = torch.einsum("bqhd,bkd->bqhk", query, keys.tile(k0, k1))
            else:
                ids = positions[..., k0:k1]
                selected_keys = keys.gather(ids.clamp(0, width - 1))
                dots = torch.einsum("bqhd,bqkd->bqhk", query, selected_keys)
            scores = (dots.relu_() * head_weights.unsqueeze(-1)).sum(dim=2)
            scores = scores.masked_fill(
                (ids < 0) | (ids >= visible) | (ids >= width), -torch.inf
            )
            merge_scores, merge_ids = (
                (scores.flip(-1), ids.flip(-1)) if prefer_large else (scores, ids)
            )
            best_scores, best_ids = _merge_topk(
                best_scores, best_ids, merge_scores, merge_ids, count
            )
            if make_candidates:
                if positions is not None:
                    raise ValueError(
                        "A candidate source must score the complete key range"
                    )
                maxima = (
                    F.pad(scores, (0, -scores.shape[-1] % block_size), value=-torch.inf)
                    .unflatten(-1, (-1, block_size))
                    .amax(-1)
                )
                ids_block = (
                    torch.arange(
                        k0 // block_size,
                        (k1 + block_size - 1) // block_size,
                        device=q.device,
                    )
                    .view(1, 1, -1)
                    .expand_as(maxima)
                )
                newest = (visible - 1) // block_size
                maxima = maxima.masked_fill(
                    (visible > 0) & (ids_block == newest), torch.inf
                )
                if prefer_large:
                    maxima, ids_block = maxima.flip(-1), ids_block.flip(-1)
                block_scores, block_ids = _merge_topk(
                    block_scores, block_ids, maxima, ids_block, block_count
                )
        if best_scores is not None:
            result = _ordered_indices(best_scores, best_ids)
            output[:, q0:q1, : result.shape[-1]] = result
        if make_candidates:
            # +inf pins the newest visible block, whereas -inf denotes padding.
            keep = block_scores > -torch.inf
            candidates[:, q0:q1] = _ordered_indices(
                keep.float().masked_fill(~keep, -torch.inf), block_ids
            )
    return output, candidates
