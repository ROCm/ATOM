# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Rank consensus on a fusion decision (plan §4.3).

A fused kernel's flag chain needs every rank to run the same variant in the
same order. Matching only reads rank-uniform inputs, but we verify once per
bucket, outside any graph replay. The reducer is injectable so the logic runs
on CPU in unit tests.
"""

import hashlib
import logging
from collections.abc import Callable, Sequence

logger = logging.getLogger("atom.gemm_fusion")

# elementwise-MIN all-reduce over a list of ints, one entry per rank-agnostic value
Reducer = Callable[[Sequence[int]], Sequence[int]]


def decision_key(site: str, tokens: int, vid: str, n_fused: int) -> int:
    """Process-independent 31-bit key (python ``hash`` is salted per process)."""
    blob = f"{site}|{tokens}|{vid}|{n_fused}".encode()
    return int.from_bytes(hashlib.blake2b(blob, digest_size=4).digest(), "little") & (
        0x7FFFFFFF
    )


class RankDisagreement(RuntimeError):
    """Some rank hit a contract error while others did not."""


def agree(
    reduce_min: Reducer | None,
    site: str,
    tokens: int,
    vid: str,
    n_fused: int,
    errored: bool = False,
) -> str:
    """Return ``"agree"``, ``"disagree"`` or ``"error"`` (all ranks erred).

    ``reduce_min is None`` means a single rank: trivially agree (or error).
    A mix of errored and non-errored ranks raises :class:`RankDisagreement`
    on every rank, because "covered but contract fails" must never be hidden
    by a fallback on only some ranks.
    """
    key = decision_key(
        site, tokens, "ERR" if errored else vid, 0 if errored else n_fused
    )
    if reduce_min is None:
        return "error" if errored else "agree"
    kmin, nkmax, nerr_max = reduce_min([key, -key, -int(errored)])
    all_err = -nerr_max == 1 and kmin == key
    any_err = -nerr_max == 1
    if any_err and not (all_err and kmin == -nkmax):
        raise RankDisagreement(
            f"gemm_fusion {site} tokens={tokens}: contract error on some ranks only"
        )
    if any_err:
        return "error"
    if kmin != key or -nkmax != key:
        logger.warning(
            "gemm_fusion %s tokens=%d: ranks disagree -> native on all ranks",
            site,
            tokens,
        )
        return "disagree"
    return "agree"


def torch_min_reducer(group=None, device="cuda") -> Reducer:
    """Elementwise-MIN all-reduce of small int64 values over ``group``."""
    import torch
    import torch.distributed as dist

    def _reduce(values):
        t = torch.tensor(list(values), device=device, dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
        return t.tolist()

    return _reduce
