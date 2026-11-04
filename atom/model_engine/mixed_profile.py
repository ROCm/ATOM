# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Startup profiling of real V4 attention with a small, temporary KV pool."""

import logging
import math

import numpy as np
import torch

from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import Sequence, SequenceType

logger = logging.getLogger("atom")


def mixed_profile_shapes(
    token_budget: int, max_seqs: int, width: int, max_model_len: int
):
    """Cover prefill-heavy, balanced and decode-heavy token layouts."""
    if max_seqs < 2 or token_budget <= width:
        return []
    floor = max(1, token_budget // 8)
    decodes = {
        1,
        min(max_seqs - 1, max(1, token_budget // (2 * width))),
        min(max_seqs - 1, (token_budget - floor) // width),
    }
    return [
        (min(token_budget - n * width, (max_seqs - n) * max_model_len), n, width)
        for n in sorted(decodes)
        if n > 0
    ]


def _batch(runner, prefill_tokens, decode_seqs, width, *, long_context):
    """Valid addresses, with shared read-only zero pages for old KV history.

    Current writes receive private pages and state slots. History values do
    not affect allocation sizes, so they need no model-length KV allocation.
    """
    config = runner.config
    block = runner.block_size
    full_q = getattr(getattr(runner, "drafter", None), "mtp_k", 0) + 1
    lookahead = max(0, full_q - 2)
    max_len = config.max_model_len - lookahead
    prefill_lens = []
    while prefill_tokens:
        n = min(prefill_tokens, max_len)
        prefill_lens.append(n)
        prefill_tokens -= n
    lengths = prefill_lens + [width] * decode_seqs
    if len(lengths) > config.max_num_seqs:
        raise ValueError("mixed profile exceeds max_num_seqs")
    seqs = {}
    next_page = 1
    slots_per_req = runner.pool_plan.entries_per_req[STATE_SLOT_CLASS]
    for row, n in enumerate(lengths):
        prefill = row < len(prefill_lens)
        # The short sweep varies allocation shapes; the long sweep exercises
        # max-context indexer chunks and sparse attention metadata.
        context = max_len if long_context else min(max_len, max(n, 8192))
        seq = Sequence([0] * context, block_size=block, has_per_req_cache=True)
        seq.type = SequenceType.PREFILL if prefill else SequenceType.DECODE
        seq.num_cached_tokens = context - n if prefill else context - full_q
        seq.state_slots = list(range(row * slots_per_req, (row + 1) * slots_per_req))
        n_pages = math.ceil((context + lookahead) / block)
        # At most n + width tokens can write, including a partial first page.
        write_tokens = n if prefill else full_q
        write_pages = min(n_pages, math.ceil((write_tokens + lookahead) / block) + 1)
        seq.block_table = [0] * (n_pages - write_pages) + list(
            range(next_page, next_page + write_pages)
        )
        next_page += write_pages
        seqs[seq.id] = seq
    batch = ScheduledBatch(
        seqs,
        lengths,
        sum(lengths),
        total_tokens_num_prefill=sum(prefill_lens),
        total_tokens_num_decode=decode_seqs * width,
        total_seqs_num=len(lengths),
        total_seqs_num_prefill=len(prefill_lens),
        total_seqs_num_decode=decode_seqs,
        num_spec_step=full_q - 1,
        scheduled_spec_decode_tokens={
            s.id: np.zeros(full_q - 1, dtype=np.int32)
            for s in seqs.values()
            if s.type is SequenceType.DECODE
        },
        is_final_chunk=[True] * len(prefill_lens) + [False] * decode_seqs,
    )
    return batch, next_page


@torch.inference_mode()
def profile_mixed_memory(runner) -> int:
    """Return weights + measured workspace/cache peak, excluding temporary KV.

    This runs before graph capture or request admission, on every TP/DP rank.
    Unlike dummy warmup it executes attention, compressors, MoE and the draft.
    """
    width = getattr(getattr(runner, "drafter", None), "mtp_k", 0) + 1
    # q=1 can fit more requests than full-width verification. DSpark still
    # proposes K drafts per request, so that is a different workspace maximum.
    shapes = [
        shape
        for q in sorted({1, width}, reverse=True)
        for shape in mixed_profile_shapes(
            runner.warmup_max_tokens(),
            runner.config.max_num_seqs,
            q,
            runner.config.max_model_len - max(0, width - 2),
        )
    ]
    if not shapes:
        return torch.cuda.max_memory_allocated()
    # Enough private pages for all current writes plus shared zero history.
    pages = (
        math.ceil(
            (runner.warmup_max_tokens() + runner.max_bs * width) / runner.block_size
        )
        + 2 * runner.max_bs
        + 1
    )
    baseline_peak = torch.cuda.max_memory_allocated()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    max_workspace = 0
    max_non_torch = 0
    with runner.attn_metadata_builder.temporary_profile_cache(pages):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        fixed = torch.cuda.memory_allocated()
        for long_context in (False, True):
            for prefill, n_decode, q in shapes:
                batch, needed = _batch(
                    runner, prefill, n_decode, q, long_context=long_context
                )
                if needed > pages:
                    raise RuntimeError(
                        f"mixed profile needs {needed} pages, allocated {pages}"
                    )
                runner.tokenID_processor.clean()
                torch.cuda.reset_peak_memory_stats()
                runner.forward(batch)
                torch.cuda.synchronize()
                allocated = torch.cuda.max_memory_allocated()
                reserved = torch.cuda.max_memory_reserved()
                free, total = torch.cuda.mem_get_info()
                max_non_torch = max(
                    max_non_torch, total - free - torch.cuda.memory_reserved()
                )
                max_workspace = max(max_workspace, allocated - fixed, reserved - fixed)
                logger.info(
                    "mixed memory profile: prefill=%d decode=%dx%d long_context=%s "
                    "workspace_peak=%.2f GiB reserved_workspace=%.2f GiB",
                    prefill,
                    n_decode,
                    q,
                    long_context,
                    (allocated - fixed) / (1 << 30),
                    (reserved - fixed) / (1 << 30),
                )
                del batch
        runner.tokenID_processor.clean()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    runner._mixed_profile_non_torch = max_non_torch
    logger.info(
        "mixed memory profile complete: temporary KV released; "
        "resident_delta=%.2f GiB workspace_budget=%.2f GiB non_torch_peak=%.2f GiB",
        (torch.cuda.memory_allocated() - baseline) / (1 << 30),
        max_workspace / (1 << 30),
        max_non_torch / (1 << 30),
    )
    return max(baseline_peak, baseline + max_workspace)
