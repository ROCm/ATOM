# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_MTP_DRAFT_VOCAB: the MTP draft passes' LM head over a subset of the vocabulary.

A draft pass reads the whole LM head (248320 x 2048 for Qwen3.6: 1 GB bf16, 0.5 GB
with ATOM_MTP_DRAFT_LM_HEAD_FP8) to propose one token a sequence. Restricted to a
subset -- id ranges, e.g. the tokenizer's lowest ids plus its special tokens
("0:98790,248044:248070", 98816 ids) -- a pass reads only those rows, and its
sampler only those columns: on MI455X, 40 draft rows, FP8 head with rejection
sampling and the draft presence penalty, 74 -> 49 us a pass (head GEMM 43 -> 20,
the sampler's two kernels 24 -> 16, the two maps below +7). Drafts outside the
subset are never proposed; the verify rows keep the whole head, so only how often
a draft is accepted can change, not the output distribution (with
ATOM_MTP_REJECTION_SAMPLING the draft distribution q is simply zero outside the
subset). With ATOM_MTP_DRAFT_LM_HEAD_FP8 the subset head is the FP8 copy (and the
full FP8 copy is not made).

Draft logits are ``[rows, size]`` over the subset's columns. A column maps to its
vocabulary id through ``lut``: the ids, padded with -1 past the subset, where the
sampler's filler columns (``size + lane``, lane < 2 * its column block, written only
for a row with fewer candidates than k) land, so every map is one in-bounds gather.
The rejection-sampling records are kept in vocabulary ids
(``SpecRejectionSampling.sample_draft``), the draft presence state in column ids
(``DraftPresencePenalty.set_vocab``).

The draft presence penalty reads its mask through ``presence_view``: the full mask
from the first range's start, ``size`` columns wide, when the later ranges are a
small tail (the default spec's 26 special tokens). The first range's columns read
their own bits; a later range's columns read the bits of the ids that follow the
first range (for the default spec, ids 98278..98303), so the specials are penalized
only by accident. That changes the draft distribution q alone -- the records hold
the q each draft was drawn from -- so rejection sampling stays exact, and no per-step
staging runs. Other specs gather each step's rows onto the columns instead
(``gather_columns``, one launch a range)."""

from itertools import pairwise

import torch

# The aiter sampler's widest column block is 64 (MAX_K), its filler columns at most
# 2 * 64 past the last column.
_LUT_PAD = 128
# presence_view: the later ranges hold at most this share of the subset's columns.
_VIEW_TAIL_SHARE = 64


def parse_vocab_ranges(spec: str, vocab_size: int) -> list[tuple[int, int]]:
    """``"0:98278,248044:248070"`` -> ``[(0, 98278), (248044, 248070)]`` (half-open,
    sorted, non-overlapping, inside the vocabulary)."""
    ranges = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = (int(x) for x in part.split(":"))
        assert (
            0 <= a < b <= vocab_size
        ), f"ATOM_MTP_DRAFT_VOCAB range {part} outside [0, {vocab_size})"
        ranges.append((a, b))
    ranges.sort()
    for (a0, b0), (a1, _) in pairwise(ranges):
        assert b0 <= a1, f"ATOM_MTP_DRAFT_VOCAB ranges overlap: {a0}:{b0}, {a1}"
    assert ranges, "ATOM_MTP_DRAFT_VOCAB names no ids"
    return ranges


class DraftVocab:
    def __init__(self, weight: torch.Tensor, ranges, fp8: bool):
        dev = weight.device
        ids = torch.cat([torch.arange(a, b, device=dev) for a, b in ranges])
        self.size = int(ids.numel())
        self.vocab_size = int(weight.shape[0])
        # (vocabulary start, end, first column) of each range
        self.ranges = []
        col = 0
        for a, b in ranges:
            self.ranges.append((int(a), int(b), col))
            col += b - a
        self.ids = ids.to(torch.int32)
        self.lut = torch.cat(
            [self.ids, torch.full((_LUT_PAD,), -1, dtype=torch.int32, device=dev)]
        )
        sub = weight.index_select(0, ids).contiguous()
        if fp8:
            from atom.model_ops.draft_lm_head_fp8 import DraftLMHeadFP8

            assert self.size % 128 == 0, (
                f"ATOM_MTP_DRAFT_VOCAB with the FP8 head: {self.size} ids, "
                "not a multiple of 128 (its block scales)"
            )
            self.fp8 = DraftLMHeadFP8(sub)
            self.weight = None
            del sub
        else:
            self.fp8 = None
            self.weight = sub

    def mm(self, x: torch.Tensor) -> torch.Tensor:
        """bf16 ``[rows, size]`` draft logits over the subset."""
        if self.fp8 is not None:
            return self.fp8.mm(x)
        from aiter.tuned_gemm import tgemm

        return tgemm.mm(x, self.weight, None)

    def to_vocab(self, cols: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Column ids ``cols`` (int32, 1-D) -> vocabulary ids into ``out`` (int32,
        ``cols``' shape; may be ``cols`` itself): one gather launch."""
        return torch.gather(self.lut, 0, cols, out=out)

    def map_cols_(self, cols: torch.Tensor) -> torch.Tensor:
        """Column ids -> vocabulary ids in place, ``cols`` a 2-D (row-strided)
        int32 view such as the sampler records' column lanes: one gather launch."""
        return torch.gather(self.lut.expand(cols.shape[0], -1), 1, cols, out=cols)

    def presence_view(self, mask: torch.Tensor) -> torch.Tensor | None:
        """``mask`` ``[slots, vocab]``'s columns from the first range's start,
        ``size`` wide (module docstring), or None when the later ranges are more
        than a small tail or the view would run past the vocabulary."""
        a0, b0, _ = self.ranges[0]
        tail = self.size - (b0 - a0)
        if a0 + self.size > mask.shape[1] or tail * _VIEW_TAIL_SHARE > self.size:
            return None
        return mask[:, a0 : a0 + self.size]

    def gather_columns(
        self, src: torch.Tensor, rows: torch.Tensor, out: torch.Tensor
    ) -> torch.Tensor:
        """``out[i] = src[rows[i]]`` at the subset's columns: ``src`` ``[R, vocab]``,
        ``rows`` int32 ``[n]``, ``out`` ``[n, size]``; one launch a range."""
        for a, b, col in self.ranges:
            torch.index_select(src[:, a:b], 0, rows, out=out[:, col : col + (b - a)])
        return out
