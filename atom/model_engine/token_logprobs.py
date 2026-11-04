# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Target-model probabilities for committed ordinary/speculative output tokens."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TokenLogprob:
    logprob: float
    rank: int
    top: tuple[tuple[int, float], ...] = ()


@dataclass
class LogprobRows:
    requests: torch.Tensor
    model_rows: torch.Tensor
    sample_rows: torch.Tensor


def _logprob_rows(batch, metadata, device, width):
    """One compact model-row plan for snapshots and sampled-token scoring.

    All indices stay on the device. Host draft counts supply only the allocation
    size; they are the existing scheduling metadata, never a GPU readback.
    """
    requested = [i for i, enabled in enumerate(batch.return_logprobs) if enabled]
    chosen = getattr(batch, "logprob_request_indices", None)
    if chosen is None:
        # Direct scorer/CPU callers do not run the runner's H2D preparation.
        chosen = torch.tensor(requested, device=device, dtype=torch.long)
    if metadata is None:
        return LogprobRows(
            chosen, chosen, torch.arange(len(requested), device=device)[:, None]
        )
    cu = metadata.cu_num_draft_tokens.long()
    starts = torch.cat((cu.new_zeros(1), cu[:-1]))
    lengths = (cu - starts)[chosen]
    ends = (lengths + 1).cumsum(0)
    compact_starts = torch.cat((ends.new_zeros(1), ends[:-1]))
    count = len(requested) + sum(
        int(metadata.num_draft_tokens_np[i]) for i in requested
    )
    slots = torch.arange(count, device=device)
    owners = torch.searchsorted(ends, slots, right=True)
    positions = slots - compact_starts[owners]
    target = torch.cat((metadata.target_logits_indices.long(), cu.new_zeros(1)))
    flat = starts[chosen[owners]] + positions
    model_rows = torch.where(
        positions < lengths[owners],
        target[flat.clamp(max=target.numel() - 1)],
        metadata.bonus_logits_indices.long()[chosen[owners]],
    )
    sample_rows = compact_starts[:, None] + torch.minimum(
        torch.arange(width, device=device)[None, :], lengths[:, None]
    )
    return LogprobRows(chosen, model_rows, sample_rows)


def capture_token_logits(logits, batch, metadata=None, *, rows=None):
    """Snapshot only requested, real target/bonus rows before grammar masking."""
    if rows is None:
        width = (
            1
            if metadata is None
            else int(max(metadata.num_draft_tokens_np, default=0)) + 1
        )
        rows = _logprob_rows(batch, metadata, logits.device, width)
    return logits.index_select(0, rows.model_rows)


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


def gather_token_logprobs(
    logits, sampled_ids, batch, metadata=None, *, snapshot=None, rows=None
):
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
    if rows is None:
        rows = _logprob_rows(batch, metadata, logits.device, width)
    counts = getattr(batch, "top_logprobs", [0] * len(batch.return_logprobs))
    k = min(max((counts[i] for i in requested), default=0), logits.shape[-1])
    shape = (len(requested), width)
    selected_ids = ids[rows.requests].clamp(min=0).long()
    if snapshot is None:
        score_rows = rows.model_rows[rows.sample_rows]
    else:
        score_rows = rows.sample_rows
        logits = snapshot
    values, candidates, ranks = _score_rows(
        logits, score_rows.flatten(), selected_ids.flatten(), k
    )
    return values.view(*shape, k + 1), candidates.view(*shape, k), ranks.view(shape)


def pack_token_logprobs(values, candidates, ranks):
    """One FP32 transport buffer; integer fields retain their int32 bit patterns."""
    return torch.cat(
        (values, candidates.view(torch.float32), ranks[..., None].view(torch.float32)),
        dim=-1,
    )


def unpack_token_logprobs(payload, req_ids, token_ids, requested=None):
    """Read the completed CPU buffer using its own batch identity and top-K width."""
    if payload is None:
        return None
    if payload.device.type != "cpu":
        raise ValueError("Logprob payload must complete D2H before unpacking")
    k = (payload.shape[-1] - 2) // 2
    # A structured NumPy view converts mixed floats/ints in one tolist call.
    # No numeric cast of IDs: ranks and candidates remain bit-exact, even for
    # int32 values outside FP32's exact integer range.
    dtype = np.dtype(
        [(f"value{i}", np.float32) for i in range(k + 1)]
        + [(f"id{i}", np.int32) for i in range(k)]
        + [("rank", np.int32)]
    )
    records = payload.numpy().view(dtype).reshape(payload.shape[:2]).tolist()
    if requested is not None:
        req_ids = [rid for rid, enabled in zip(req_ids, requested) if enabled]
    if len(records) != len(req_ids):
        raise ValueError("Token/logprob batch identity mismatch")
    return {
        req_id: [
            TokenLogprob(
                record[0],
                record[-1],
                tuple(zip(record[k + 1 : 2 * k + 1], record[1 : k + 1])),
            )
            for record in records[i][: len(token_ids[req_id])]
        ]
        for i, req_id in enumerate(req_ids)
    }
