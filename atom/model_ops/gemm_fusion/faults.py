# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Test-only fault seams for evidence (E1/E3/E4/E5/E7). Inert unless
``ATOM_GEMM_FUSION_FAULT`` is set; production code pays one dict lookup.

``ATOM_GEMM_FUSION_FAULT`` = comma list of ``kind[:sp_rank[:arg]]``; ``sp_rank``
empty or ``*`` means every rank. Kinds:

  poison_rank[:r]      check_step behaves as if rank r read a nonzero device
                       poison word (exercises the HOST agree/recover/raise path;
                       real device poison needs a kernel-side forced-timeout build).
  op_raise[:r]         the fused op raises on rank r (E4: one-rank exception).
  rank_native[:r]      rank r takes a native decision where others fuse. This
                       DELIBERATELY violates cross-rank uniformity (E3). Expected
                       outcome: FAIL-STOP. Rank r enters the native RCCL all_to_all while
                       its peers wait in the fused CCO kernel, so the step cannot complete
                       and the step-end digest in check_step is never reached; the run ends
                       at the process-group timeout. E-3 passes when the failure is bounded,
                       every rank exits non-zero, and no wrong output is returned silently.
                       (The digest only sees decision differences in steps that complete.)
  slow_remainder[:r:n] after the native remainder of a 1-stage (large-M) call,
                       rank r spins the GPU ~n microseconds (E5: WAR window).
  recover_skip[:r]     rank r skips recover() (exercises the recovery timeout).
  no_poison_check      disable the per-step poison check (E1 NEGATIVE CONTROL:
                       proves the check has power; never set in production).
"""

import os

ENV = "ATOM_GEMM_FUSION_FAULT"
_cache = None


def _parse():
    out = {}
    for item in filter(None, (os.environ.get(ENV, "")).split(",")):
        parts = item.strip().split(":")
        rank = parts[1] if len(parts) > 1 and parts[1] not in ("", "*") else None
        out[parts[0]] = (None if rank is None else int(rank), parts[2] if len(parts) > 2 else None)
    return out


def reload():
    """Re-read the environment (tests)."""
    global _cache
    _cache = _parse()
    return _cache


def active(kind, sp_rank=0):
    """-> arg string ('' if none) when the fault applies to ``sp_rank``, else None."""
    global _cache
    if _cache is None:
        _cache = _parse()
    ent = _cache.get(kind)
    if ent is None:
        return None
    rank, arg = ent
    if rank is not None and rank != sp_rank:
        return None
    return arg or ""
