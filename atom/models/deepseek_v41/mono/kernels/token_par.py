# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Token parallel: a rank's view of an S-row step as its own S / TP tokens, the
per-token stages running on it unchanged."""

import flydsl.expr as fx

from atom.models.deepseek_v41.mono.kernels.dims import HC, HIDDEN, Q_RANK

# a token's row in each per-token kernel argument, in bytes
TOKEN_ROW_BYTES = {
    "res_in": HC * HIDDEN * 2,
    "res_out": HC * HIDDEN * 2,
    "pend": HIDDEN * 2,
    "normed": HIDDEN * 2,
    "x": HIDDEN * 2,
    "aux": HIDDEN * 2,
    "out": HIDDEN * 2,
    "pre_in": HC * 4,
    "post_in": HC * 4,
    "comb_in": HC * HC * 4,
    "pre_out": HC * 4,
    "post_out": HC * 4,
    "comb_out": HC * HC * 4,
    "pos": 8,
    "ring_rows": 4,
    "qr_out": Q_RANK,
    "qrs_out": Q_RANK // 32,
}


def own_count(s, tp):
    """The tokens a rank owns in a token-parallel step: rank r owns r n .. + n."""
    assert s % tp == 0, f"{s} rows do not split over {tp} ranks"
    return s // tp


def own_rows(s, tp, rank):
    """(first row, rows) ``rank`` owns of a token-parallel s-row step."""
    n = own_count(s, tp)
    return rank * n, n


def own_view(c, names):
    """``c`` seen from this rank's tokens: S their count, ``t0`` the first one's
    row of the step, and each per-token argument in ``names`` moved to it."""
    t0, n = own_rows(c["S"], c["d"].ranks, c["rank"])
    lo = fx.Int64(t0)
    args = dict(c["args"])
    for name in names:
        args[name] = args[name] + lo * TOKEN_ROW_BYTES[name]
    return {**c, "S": n, "args": args, "t0": t0}
