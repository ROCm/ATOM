# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Every per-step decision of which attention mailbox cells the score stage
writes and the merge stage polls (AS, a 64-key chunk each), defined once
(DESIGN_v3 4.9, I3), the way ``index_plan`` does the indexer's.

A token's keys sit in fixed slots: a build's ``sel_slots`` selection slots
first (``selection_slots``: ``index_plan.TOPK`` where its keys include a
selection, else none), slot i valid when i < ``n_sel`` (the selection's
valid ids), then the window's, oldest first, slot ``sel_slots`` + w valid
when w < its window count; a slot's kind is its position, so a pass over a
chunk knows it without a branch a key. Chunks of ``CH`` keys: the score
stage's tasks cover every chunk of a (token, head tile) and write the ones
``chunk_written`` says; the merge stage polls ``polled_chunk`` of each chunk.
Called by the kernel on traced Int32 and by the CPU property test on Python
ints (``index_plan``'s rules: non-negative ``//`` operands only).
"""

from atom.models.deepseek_v41.mono.config import ATTENTION_KEYS_MAX
from atom.mono.dsa_indexer.paged.fused import plan as ip
from atom.mono.plan.arith import sel

CH = 64  # keys a chunk: a score task's, four waves of 16
CHUNKS = ATTENTION_KEYS_MAX // CH  # a (token, head tile)'s chunks
assert CHUNKS * CH == ATTENTION_KEYS_MAX and ip.TOPK % CH == 0


def selection_slots(attn_sel: bool) -> int:
    """A build's selection slots: the top-k's where its keys include a
    selection (``AttnPostBuild.attn_sel``), else none (the DSpark draft's
    window-only attention: every slot the window's)."""
    return ip.TOPK if attn_sel else 0


def window_chunk(sel_slots):
    """The first window chunk: written for every token (a pad row's only one),
    the chunk the merge polls for one that is not."""
    return sel_slots // CH


def window_slots(sel_slots):
    """The window keys a token holds at most."""
    return ATTENTION_KEYS_MAX - sel_slots


def slot_valid(slot, n_sel, count, sel_slots):
    """Whether key slot ``slot`` holds a key: a selection slot below the
    token's ``n_sel`` valid ids, or a window slot below its window ``count``
    (0 for a graph's pad row)."""
    return (slot < n_sel) | ((slot >= sel_slots) & (slot < sel_slots + count))


def chunk_written(chunk, n_sel, count, sel_slots):
    """Whether the score stage writes chunk ``chunk`` of a token: one holding a
    key, and the first window chunk always (its values masked where it holds
    none)."""
    first = window_chunk(sel_slots)
    window = chunk >= first
    return (
        (chunk * CH < n_sel)
        | (window & ((chunk - first) * CH < count))
        | (chunk == first)
    )


def polled_chunk(chunk, n_sel, count, sel_slots):
    """The chunk the merge stage polls for its chunk ``chunk``: itself if
    written, else the first window chunk (its values masked)."""
    return sel(
        chunk_written(chunk, n_sel, count, sel_slots), chunk, window_chunk(sel_slots)
    )


def score_chunks(tokens, tasks, ctas):
    """Chunks a score task (four waves each): two from 12 rows, a CTA's eight
    waves both busy, and wherever a chunk a task (``tasks`` of them) would
    take a second round of the ``ctas``."""
    return 2 if tokens >= 12 or tasks > ctas else 1


def merge_one_test(tokens, merges, ctas):
    """Whether a CTA's merge task is run behind a single test rather than a
    task loop, whose preheader the compiler fills with the merge's hoisted
    argument reloads and spill moves ahead of the task: where a CTA has at
    most one (``merges`` <= ``ctas``), from 4 to 6 rows."""
    return merges <= ctas and 4 <= tokens <= 6


def merge_heads(tokens):
    """Heads a merge task (of a 16-head tile): four up to 6 rows and eight up
    to 12, so a step's merge spreads over more CTAs, the whole tile above."""
    return 4 if tokens <= 6 else 8 if tokens <= 12 else 16
