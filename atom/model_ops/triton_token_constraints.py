# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared token-ID masking primitives for dense sets and GPU state machines."""

import triton
import triton.language as tl


@triton.jit
def mask_token_columns(row_ptr, columns, forbidden, valid, stride=1):
    """Deny selected columns; allow/force masks supply their complement."""
    tl.store(row_ptr + columns * stride, -float("inf"), mask=valid & forbidden)


@triton.jit
def _mask_logits(
    logits,
    masks,
    rows,
    logits_stride,
    token_stride,
    mask_stride,
    vocab_size,
    BLOCK: tl.constexpr,
):
    mask_row = tl.program_id(0).to(tl.int64)
    row = tl.load(rows + mask_row).to(tl.int64)
    token = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    word = tl.load(
        masks + mask_row * mask_stride + token // 32,
        mask=token < vocab_size,
        other=-1,
    )
    forbidden = ((word >> (token % 32)) & 1) == 0
    mask_token_columns(
        logits + row * logits_stride,
        token,
        forbidden,
        token < vocab_size,
        token_stride,
    )


def apply_packed_bitmask(logits, masks, rows):
    if rows.numel() == 0:
        return
    block = 1024
    _mask_logits[(rows.numel(), triton.cdiv(logits.shape[1], block))](
        logits,
        masks,
        rows,
        logits.stride(0),
        logits.stride(1),
        masks.stride(0),
        logits.shape[1],
        BLOCK=block,
    )


@triton.jit
def _rule_row(
    row, stride, position, cutoff, allow, ids, count, vocab, vb, BLOCK: tl.constexpr
):
    if (cutoff < 0) | (position < cutoff):
        if allow:
            for start in range(vb * BLOCK, vocab, tl.num_programs(1) * BLOCK):
                columns = start + tl.arange(0, BLOCK)
                keep = tl.full((BLOCK,), False, tl.int1)
                for j in range(count):
                    keep = keep | (columns == tl.load(ids + j))
                mask_token_columns(row, columns, ~keep, columns < vocab, stride)
        elif vb == 0:
            for j in range(count):
                token = tl.load(ids + j)
                mask_token_columns(row, token, True, token < vocab, stride)


@triton.jit
def _apply_rules(
    last,
    last_stride,
    token_stride,
    target,
    target_stride,
    target_token_stride,
    rules,
    rule_count,
    positions,
    cu_q,
    cu_drafts,
    targets,
    bonuses,
    vocab,
    HAS_DRAFTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rule = tl.program_id(0)
    vb = tl.program_id(1)
    row = tl.load(rules + rule * 5)
    cutoff = tl.load(rules + rule * 5 + 1)
    allow = tl.load(rules + rule * 5 + 2)
    ids = rules + rule_count * 5 + tl.load(rules + rule * 5 + 3)
    count = tl.load(rules + rule * 5 + 4)
    if HAS_DRAFTS:
        begin = tl.load(cu_drafts + row - 1, mask=row > 0, other=0)
        end = tl.load(cu_drafts + row)
        for r in range(begin, end):
            pos = tl.load(positions + tl.load(targets + r))
            _rule_row(
                target + r.to(tl.int64) * target_stride,
                target_token_stride,
                pos,
                cutoff,
                allow,
                ids,
                count,
                vocab,
                vb,
                BLOCK,
            )
        last_pos = tl.load(positions + tl.load(bonuses + row))
    else:
        last_pos = tl.load(positions + tl.load(cu_q + row + 1) - 1)
    _rule_row(
        last + row * last_stride,
        token_stride,
        last_pos,
        cutoff,
        allow,
        ids,
        count,
        vocab,
        vb,
        BLOCK,
    )


def apply_token_rules(logits, target, rules, count, positions, cu_seqlens, metadata):
    """Logical GPU positions already include deferred accepted-token correction."""
    spec = metadata is not None
    _apply_rules[(count, min(16, triton.cdiv(logits.shape[-1], 1024)))](
        logits,
        logits.stride(0),
        logits.stride(1),
        target if spec else logits,
        target.stride(0) if spec else 0,
        target.stride(1) if spec else 0,
        rules,
        count,
        positions,
        cu_seqlens,
        metadata.cu_num_draft_tokens if spec else cu_seqlens,
        metadata.target_logits_indices if spec else cu_seqlens,
        metadata.bonus_logits_indices if spec else cu_seqlens,
        logits.shape[-1],
        HAS_DRAFTS=spec,
        BLOCK=1024,
    )
