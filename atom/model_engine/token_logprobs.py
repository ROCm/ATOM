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


def gather_token_logprobs(logits, sampled_ids, batch, metadata=None):
    """Return (values, candidate IDs, ranks), retaining the sampled-ID layout.

    Values slot 0 holds the sampled token; remaining slots hold top candidates.
    The vocabulary distribution is the raw target distribution, before sampling
    filters. Fixed-width output keeps async token/probability copies aligned.
    Padding positions are computed safely and discarded with their token IDs.
    Only requesting sequences are selected; the disabled path never calls here.
    """
    requested = [i for i, enabled in enumerate(batch.return_logprobs) if enabled]
    width = sampled_ids.shape[1] if sampled_ids.ndim == 2 else 1
    ids = sampled_ids.reshape(len(batch.return_logprobs), width)
    device = logits.device
    chosen = torch.tensor(requested, device=device, dtype=torch.long)
    counts = getattr(batch, "top_logprobs", [0] * len(batch.return_logprobs))
    k = min(max((counts[i] for i in requested), default=0), logits.shape[-1])
    values = torch.zeros((*ids.shape, k + 1), device=device, dtype=torch.float32)
    candidates = torch.zeros((*ids.shape, k), device=device, dtype=torch.int32)
    ranks = torch.zeros(ids.shape, device=device, dtype=torch.int32)
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
    # No full log-softmax matrix: only the normalizer and selected/top logits
    # survive this operation. These torch operations are correctness-first;
    # reducing the gather/conversion workspace is a separate profiling task.
    selected = logits[rows].float()
    normalizer = torch.logsumexp(selected, dim=-1)
    token_logits = selected.gather(-1, selected_ids[..., None]).squeeze(-1)
    values[chosen, :, 0] = token_logits - normalizer
    ranks[chosen] = (selected > token_logits[..., None]).sum(-1).to(torch.int32) + 1
    if k:
        top_values, top_ids = torch.topk(selected, k, dim=-1)
        values[chosen, :, 1:] = top_values - normalizer[..., None]
        candidates[chosen] = top_ids.to(torch.int32)
    return values, candidates, ranks


def unpack_token_logprobs(payload, req_ids, token_ids):
    """Decode the matching CPU batch, never the next scheduled batch."""
    if payload is None:
        return None
    values, candidates, ranks = payload
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
