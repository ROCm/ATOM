# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A step's context seen from one of its token groups.

A step wider than ``TOKEN_GROUP`` rows runs its per-token stages a group at a
time on the 16-row code. Each stage moves what it reads: a per-token argument
or buffer by the group's first row, a per-group mailbox region (or a list of
the peers' copies) by whole groups. The stages declare those moves; this is
the one place they are applied.
"""

import flydsl.expr as fx

from atom.models.kimi_k3.mono.config import TOKEN_GROUP, group_rows, groups
from atom.mono.device.sync import shift


def _path(path):
    return (path,) if isinstance(path, str) else path


def _get(c, path):
    for key in path:
        if key not in c:
            return None
        c = c[key]
    return c


def _with(c, path, value):
    """``c`` with the entry at ``path`` replaced (``c`` itself untouched)."""
    head, *rest = path
    return {**c, head: _with(c[head], rest, value) if rest else value}


def group_view(c, g: int, rows=(), regions=()):
    """``c`` seen from token group g: ``S`` its rows; each ``rows`` entry
    (path, bytes a row: a Python or traced integer) a pointer moved the
    group's first row in, each ``regions`` entry (path, bytes a group) a
    mailbox region, or a list of the peers' copies, moved g groups in. A path
    is a key of ``c`` or a tuple of nested keys; one absent from ``c`` is not
    there to move. One group is ``c`` itself."""
    if groups(c["S"]) == 1:
        return c
    t0 = g * TOKEN_GROUP
    v = {**c, "S": group_rows(c["S"], g)}
    for path, row_bytes in rows:
        p = _path(path)
        base = _get(c, p)
        if base is not None:
            at = (
                fx.Int64(t0 * row_bytes)
                if isinstance(row_bytes, int)
                else fx.Int64(t0) * row_bytes
            )
            v = _with(v, p, base + at)
    for path, group_bytes in regions:
        p = _path(path)
        r = _get(c, p)
        if r is not None:
            at = g * group_bytes
            v = _with(
                v, p, [shift(x, at) for x in r] if isinstance(r, list) else shift(r, at)
            )
    return v
