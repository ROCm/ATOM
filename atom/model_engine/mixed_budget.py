# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Token bounds shared by mixed admission, configuration and startup profiling."""


def prefill_floor(token_budget: int) -> int:
    return max(1, token_budget // 8)


def decode_seats(token_budget: int, max_seqs: int, width: int, prefill=1) -> int:
    """Decode rows that fit beside at least one prefill row."""
    return max(0, min(max_seqs - 1, (token_budget - prefill) // width))
