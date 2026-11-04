# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Target-model probabilities for committed ordinary/speculative output tokens."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenLogprob:
    logprob: float
    rank: int
    top: tuple[tuple[int, float], ...] = ()


def capture_token_logits(logits, batch, metadata=None):
    """Preserve requested target rows before in-place grammar masking.

    Ordinary requests never call this. The snapshot contains only rows for
    logprob requests, including their speculative bonus row, not the full batch.
    """
    requested = [i for i, enabled in enumerate(batch.return_logprobs) if enabled]
    if metadata is None:
        rows = torch.tensor(requested, device=logits.device, dtype=torch.long)
    else:
        counts = metadata.num_draft_tokens_np.tolist()
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        parts = [
            metadata.target_logits_indices[offsets[i] : offsets[i + 1]]
            for i in requested
        ]
        parts.append(metadata.bonus_logits_indices[requested])
        rows = torch.cat(parts).long()
    row_map = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
    row_map[rows] = torch.arange(len(rows), device=logits.device)
    return logits.index_select(0, rows), row_map


def _score_rows(logits, rows, ids, k):
    """Compute normalization and rank with native PyTorch operations.

    Inputs have one flat token dimension, independent of draft width and
    optional snapshots. Only compact probabilities/candidates leave scoring.
    """
    selected = logits[rows].float()
    normalizer = torch.logsumexp(selected, dim=-1)
    token_logits = selected.gather(-1, ids[:, None]).squeeze(-1)
    ranks = (selected > token_logits[:, None]).sum(-1).to(torch.int32) + 1
    values = (token_logits - normalizer)[:, None]
    if k:
        top_values, top_ids = torch.topk(selected, k, dim=-1)
        values = torch.cat((values, top_values - normalizer[:, None]), dim=-1)
        candidates = top_ids.to(torch.int32)
    else:
        candidates = torch.empty(
            (rows.shape[0], 0), device=logits.device, dtype=torch.int32
        )
    return values, candidates, ranks


def gather_token_logprobs(logits, sampled_ids, batch, metadata=None, *, snapshot=None):
    """Return (values, candidate IDs, ranks) for requesting sequences only.

    Values slot 0 holds the sampled token; remaining slots hold top candidates.
    The vocabulary distribution is the raw target distribution, before sampling
    filters. Fixed-width output keeps async token/probability copies aligned.
    Padding positions are computed safely and discarded with their token IDs.
    Rows retain requesting-sequence order and each sampled-ID token width.
    The disabled path never calls here.
    """
    requested = [i for i, enabled in enumerate(batch.return_logprobs) if enabled]
    width = sampled_ids.shape[1] if sampled_ids.ndim == 2 else 1
    ids = sampled_ids.reshape(len(batch.return_logprobs), width)
    device = logits.device
    chosen = torch.tensor(requested, device=device, dtype=torch.long)
    counts = getattr(batch, "top_logprobs", [0] * len(batch.return_logprobs))
    k = min(max((counts[i] for i in requested), default=0), logits.shape[-1])
    shape = (len(requested), width)
    if metadata is None:
        rows = chosen[:, None]
    else:
        cu = metadata.cu_num_draft_tokens.long()
        starts = torch.cat((cu.new_zeros(1), cu[:-1]))
        lengths = cu - starts
        pos = torch.arange(width, device=device)[None, :]
        flat = starts[chosen, None] + pos
        # Append a safe row for zero-draft steps and padded positions.
        target_rows = torch.cat(
            (metadata.target_logits_indices.long(), cu.new_zeros(1))
        )
        rows = torch.where(
            pos < lengths[chosen, None],
            target_rows[flat.clamp(max=target_rows.numel() - 1)],
            metadata.bonus_logits_indices.long()[chosen, None],
        )
    selected_ids = ids[chosen].clamp(min=0).long()
    if snapshot is not None:
        rows = snapshot[1][rows]
        logits = snapshot[0]
    values, candidates, ranks = _score_rows(
        logits, rows.flatten(), selected_ids.flatten(), k
    )
    return values.view(*shape, k + 1), candidates.view(*shape, k), ranks.view(shape)


def unpack_token_logprobs(payload, req_ids, token_ids, requested=None):
    """Decode the matching CPU batch, never the next scheduled batch."""
    if payload is None:
        return None
    values, candidates, ranks = payload
    if requested is not None:
        req_ids = [rid for rid, enabled in zip(req_ids, requested) if enabled]
    if len(values) != len(req_ids):
        raise ValueError("Token/logprob batch identity mismatch")
    return {
        req_id: [
            TokenLogprob(float(v[0]), int(rank), tuple(zip(c, v[1:])))
            for v, c, rank in zip(
                values[i][: len(token_ids[req_id])],
                candidates[i][: len(token_ids[req_id])],
                ranks[i][: len(token_ids[req_id])],
            )
        ]
        for i, req_id in enumerate(req_ids)
    }
