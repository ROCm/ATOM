# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU and device arithmetic for DSA selection and short-step ownership."""

from dataclasses import dataclass

from atom.mono.dsa_indexer.plan import SCORE_KEYS
from atom.mono.plan.execution import THREADS, WAVES


@dataclass(frozen=True)
class Radix:
    bits: tuple[int, ...]
    # Wider local digits never participate in the distributed mailbox ABI.
    local: bool = False

    def __post_init__(self):
        limit = 11 if self.local else 8
        if (
            sum(self.bits) != 32
            or not all(1 <= b <= limit for b in self.bits)
            or max(self.bits) < 6
        ):
            raise ValueError(
                f"radix requires 32 total bits and a 64..{1 << limit}-bin histogram"
            )

    @property
    def passes(self):
        return len(self.bits)

    @property
    def shifts(self):
        return tuple(32 - sum(self.bits[: p + 1]) for p in range(self.passes))

    @property
    def bins(self):
        return 1 << max(self.bits)

    @property
    def bins_per_thread(self):
        return max(1, self.bins // THREADS)

    @property
    def waves(self):
        return min(WAVES, self.bins // 64)

    @property
    def bins_per_lane(self):
        return self.bins // (64 * self.waves)


DEFAULT_RADIX = Radix((8, 8, 8, 8))
LOCAL_RADIX = Radix((11, 11, 10), local=True)
WIDE_RADIX = Radix((7, 7, 7, 7, 4))
SHORT_WIDE_RADIX = Radix((6, 6, 6, 6, 6, 2))
SHORT_VERIFY_CTX = 8192
SPLIT_MIN = 262144
COUNT_BITS = 22
ARRIVED = 1 << COUNT_BITS
# a packed histogram pair (``IxBuild.packed_hist``): PACK bins' counts of
# PACK_BITS each, then the launch's tag in the high PACK_TAG_BITS
PACK = 3
PACK_BITS = 18


def request_owner(row, qlen, world):
    """A request's query rows stay on one rank."""
    return row // qlen % world


def split_request_keys(context, total, world, *, low_requests=False):
    """Keep short verify local; split a large request that would idle peers."""
    if low_requests:
        return context > SHORT_VERIFY_CTX
    return (context > SPLIT_MIN) & (2 * world * context > total)


def prefix_row(lane, tile):
    """One wave's lane in a 64-row prefix tile."""
    return lane + 64 * tile


def short_flag_count(flag_stride):
    """Only poll/reset score flags reachable by a short-verify request."""
    return min(-(-SHORT_VERIFY_CTX // SCORE_KEYS), flag_stride // 8)
