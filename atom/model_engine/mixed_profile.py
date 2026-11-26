# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Startup profiling of real V4 attention with a small, temporary KV pool."""

import logging
import math
from array import array

import numpy as np
import torch

from atom.model_engine.block_manager import draft_kv_lookahead
from atom.model_engine.kv_block import STATE_SLOT_CLASS
from atom.model_engine.mixed_budget import decode_seats, prefill_floor
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import Sequence, SequenceType

logger = logging.getLogger("atom")


class _ProfileTokens:
    """Read-only zero history; materialize only the scheduled token window."""

    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return array("i", [0]) * len(range(*index.indices(self.length)))
        if not -self.length <= index < self.length:
            raise IndexError(index)
        return 0


def mixed_profile_shapes(
    token_budget: int, max_seqs: int, width: int, max_model_len: int
):
    """Cover prefill-heavy, balanced and decode-heavy token layouts."""
    if max_seqs < 2 or token_budget <= width:
        return []
    floor = prefill_floor(token_budget)
    decodes = {
        1,
        min(max_seqs - 1, max(1, token_budget // (2 * width))),
        decode_seats(token_budget, max_seqs, width, prefill=floor),
        # A short prefill leaves its unused token budget to more decode rows.
        decode_seats(token_budget, max_seqs, width),
    }
    return [
        (min(token_budget - n * width, (max_seqs - n) * max_model_len), n, width)
        for n in sorted(decodes)
        if n > 0
    ]


def _profile_spec_tokens(runner):
    drafter = getattr(runner, "drafter", None)
    return 0 if drafter is None else drafter.mtp_k


def _profile_lookahead(runner):
    spec = runner.config.speculative_config
    return draft_kv_lookahead(
        _profile_spec_tokens(runner),
        block_drafter=spec is not None and spec.use_dspark(),
    )


def _batch(runner, prefill_tokens, decode_seqs, width, *, long_context):
    """Valid addresses, with shared read-only zero pages for old KV history.

    Current writes receive private pages and state slots. History values do
    not affect allocation sizes, so they need no model-length KV allocation.
    """
    config = runner.config
    block = runner.block_size
    full_q = _profile_spec_tokens(runner) + 1
    lookahead = _profile_lookahead(runner)
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
        # One row in each segment reaches the context limit. Keep the other
        # histories short: replaying max_seqs * max_model_len keys is needless
        # work for a profile. Query/sequence counts stay at their full shapes;
        # the decode logits plane is already persistent and in the baseline.
        long_row = long_context and row in (0, len(prefill_lens))
        context = max_len if long_row else min(max_len, max(n, 8192))
        seq = Sequence([0], block_size=block, has_per_req_cache=True)
        seq.token_ids = _ProfileTokens(context)
        seq.num_tokens = seq.num_prompt_tokens = seq.num_finalized_tokens = context
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
    # Fresh profile requests have no confidence history to shrink decode.
    # Set its forward width explicitly; draft/KV capacity above stays at K.
    spec_tokens = full_q - 1 if prefill_lens else width - 1
    batch = ScheduledBatch(
        seqs,
        lengths,
        sum(lengths),
        total_tokens_num_prefill=sum(prefill_lens),
        total_tokens_num_decode=decode_seqs * width,
        total_seqs_num=len(lengths),
        total_seqs_num_prefill=len(prefill_lens),
        total_seqs_num_decode=decode_seqs,
        num_spec_step=spec_tokens,
        scheduled_spec_decode_tokens={
            s.id: np.zeros(spec_tokens, dtype=np.int32)
            for s in seqs.values()
            if s.type is SequenceType.DECODE
        },
        is_final_chunk=[True] * len(prefill_lens) + [False] * decode_seqs,
    )
    return batch, next_page


@torch.inference_mode()
def profile_mixed_memory(runner) -> tuple[int, int]:
    """Return (torch peak excluding scratch KV, non-torch peak) in bytes.

    This runs before graph capture or request admission, on every TP/DP rank.
    Unlike dummy warmup it executes attention, compressors, MoE and the draft.
    """
    width = _profile_spec_tokens(runner) + 1
    # q=1 can fit more requests than full-width verification. DSpark still
    # proposes K drafts per request, so that is a different workspace maximum.
    shapes = [
        shape
        for q in sorted({1, width}, reverse=True)
        for shape in mixed_profile_shapes(
            runner.warmup_max_tokens(),
            runner.config.max_num_seqs,
            q,
            runner.config.max_model_len - _profile_lookahead(runner),
        )
    ]
    if runner.config.enable_dp_attention:
        # Only a DP-unified decode enters the sharded logits head. Mixed
        # steps above cannot measure its full-vocabulary collective buffers.
        # Scheduling charges the full draft width even for a shorter verify.
        decode_bs = min(runner.config.max_num_seqs, runner.warmup_max_tokens() // width)
        if decode_bs:
            shapes.extend((0, decode_bs, q) for q in sorted({1, width}))
    if not shapes:
        return torch.cuda.max_memory_allocated(), 0
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
        metrics = runner.gpu_forward_metrics
        runner.gpu_forward_metrics = None
        try:
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
                        max_non_torch,
                        total - free - torch.cuda.memory_reserved(),
                    )
                    max_workspace = max(
                        max_workspace, allocated - fixed, reserved - fixed
                    )
                    logger.info(
                        "mixed memory profile: prefill=%d decode=%dx%d "
                        "long_context=%s max_context=%d "
                        "total_context=%d "
                        "workspace_peak=%.2f GiB reserved_workspace=%.2f GiB",
                        prefill,
                        n_decode,
                        q,
                        long_context,
                        max(batch.context_lens),
                        sum(batch.context_lens),
                        (allocated - fixed) / (1 << 30),
                        (reserved - fixed) / (1 << 30),
                    )
                    del batch
        finally:
            runner.gpu_forward_metrics = metrics
            # Profiling uses synthetic request IDs. Drain their copies before
            # dropping the confidence generations the first real decode reads.
            torch.cuda.synchronize()
            runner.tokenID_processor.clean()
            drafter = getattr(runner, "drafter", None)
            if drafter is not None and drafter.verify_scheduler is not None:
                drafter.verify_scheduler.ell_by_req = {}
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    logger.info(
        "mixed memory profile complete: temporary KV released; "
        "resident_delta=%.2f GiB workspace_budget=%.2f GiB non_torch_peak=%.2f GiB",
        (torch.cuda.memory_allocated() - baseline) / (1 << 30),
        max_workspace / (1 << 30),
        max_non_torch / (1 << 30),
    )
    return max(baseline_peak, baseline + max_workspace), max_non_torch
