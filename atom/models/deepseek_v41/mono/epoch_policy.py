# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The V4.1 epoch tags' arithmetic and the target begin's decision, apart from the kernels so a CPU test reaches them
without FlyDSL or AITER (``kernels.mailbox`` and ``epoch_begin`` trace the same expressions).

A pair's tag is (epoch << TAG_BITS) | launch tag, the launch tag a V4.1 kernel's (layer + 1, POST_TAG0 / MOE_TAG0 +
layer + 1, HEAD_TAG, GATHER_TAG + call, CTX_TAG), every one in [1, 1 << TAG_BITS): a live tag's low byte is never 0,
so zeroed memory never matches. A coded pair keeps the tag's low CODED_TAG_BITS and an 8-bit code above them: its
epoch repeats every 1 << (CODED_TAG_BITS - TAG_BITS) epochs, and the begin's periodic full zero (FULL_EVERY, below
that) clears every stale pair first."""

TAG_BITS = 8
CODED_TAG_BITS = 24
CODED_TAG_MASK = (1 << CODED_TAG_BITS) - 1
EPOCH_MASK = (1 << (32 - TAG_BITS)) - 1
FULL_EVERY = 1 << 15
assert FULL_EVERY < 1 << (CODED_TAG_BITS - TAG_BITS)


def epoch_tag(epoch: int, tag: int) -> int:
    """The 32-bit tag (as an unsigned int) of launch tag ``tag`` at ``epoch``."""
    assert 0 < tag < 1 << TAG_BITS, tag
    return (((epoch & EPOCH_MASK) << TAG_BITS) | tag) & 0xFFFFFFFF


def coded_word(tag32: int, code: int) -> int:
    """A coded pair's tag word (unsigned): the tag's low CODED_TAG_BITS, the 8-bit code above."""
    assert 0 <= code < 256, code
    return ((tag32 & CODED_TAG_MASK) | (code << CODED_TAG_BITS)) & 0xFFFFFFFF


def coded_live(word: int, tag32: int) -> bool:
    return ((word ^ tag32) & CODED_TAG_MASK) == 0


def code_of(word: int) -> int:
    return (word >> CODED_TAG_BITS) & 0xFF


def slim_allowed(epoch: int, drafted: int) -> bool:
    """The target begin's decision for the new step's ``epoch``: slim iff a draft pass began after the previous
    target step (``drafted``, the target epoch the draft runner's begin last recorded, is ``epoch - 1``) and
    ``epoch`` is not a periodic full begin's."""
    return drafted == epoch - 1 and epoch % FULL_EVERY != 0
