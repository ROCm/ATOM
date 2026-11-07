# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-request work items: a stage's tasks shared among a step's requests in proportion to each request's work.

A stage whose work per request depends on the request (a context-shaped stage: an Indexer's score tiles up to the
request's position, a compressed-KV read) has, in its plain form, a fixed number of tasks a request: the longest
request sets every request's task count, so beside one long request the other requests' tasks idle while the long
one's tasks loop over all its units. ``share`` deals the stage's ``tasks`` among the requests instead: request b gets
``per[b]`` tasks a group (``groups`` groups a request, e.g. an Indexer's row groups) in proportion to its ``units[b]``
(at least one when it has units, none when it has none: a dead request of a ``Program.dyn_rows`` band, a request past a
stage's length cap). Task t then maps (``locate``) to (request b, group, c) and takes the request's units c, c +
per[b], .. (the plain form's loop with ``per[b]`` for its fixed share), so each unit is computed by exactly one task,
with the same arithmetic: the outputs are bit for bit the plain form's.

No queue and no atomic counter: every CTA computes the same shares from the same uniform words (B adds and compares
at the stage's entry, B <= 8 for a band), and a task's loop count is ~ sum(units) / (tasks / groups) whatever the mix
(an atomic ticket pays only for units of unequal cost).

The arithmetic is written over any integer type with ``+ - * //`` and comparisons, and a select ``sel(cond, a, b)``:
traced uniform Int32s in a kernel (``fx_sel``), Python ints in tests and host code (``py_sel``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


def py_sel(cond, a, b):
    return a if cond else b


def fx_sel(cond, a, b):
    import flydsl.expr as fx  # (traced code only: the module imports without FlyDSL)

    return cond.select(*(fx.Int32(v) if isinstance(v, int) else v for v in (a, b)))


@dataclass
class Shares:
    """``share``'s result: ``per[b]`` tasks a group for request b, ``first[b]`` its first task (``first[B]``: the
    tasks used, at most the stage's)."""

    per: list
    first: list


def share(units: Sequence, tasks: int, groups: int = 1, sel: Callable = py_sel, zero=0) -> Shares:
    """Deal ``tasks`` (a Python int: the stage's task count, a multiple of ``groups``) among the requests in proportion
    to ``units[b]``: per[b] = 1 + (slots - live) x units[b] // total for a request with units (slots = tasks / groups,
    live the requests with units), else 0. The per[b] sum to at most ``slots`` (the floors), so ``first[B]`` <=
    ``tasks``; a request with units gets one task a group at least. ``zero``: the integer type's 0 (``fx.Int32(0)``
    traced)."""
    if tasks % groups:
        raise ValueError(f"share: {tasks} tasks do not split over {groups} groups")
    slots = tasks // groups
    if slots < len(units):
        raise ValueError(f"share: {slots} task slots a group for {len(units)} requests (one each at least)")
    total, live = zero, zero
    for u in units:
        total = total + u
        live = live + sel(u > 0, 1, 0)
    spare = slots - live  # (>= 0: live <= len(units) <= slots)
    den = sel(total > 0, total, 1)  # (no units at all: every per[b] is 0, the quotient unused)
    per = [sel(u > 0, 1 + spare * u // den, zero) for u in units]
    first, at = [], zero
    for p in per:
        first.append(at)
        at = at + p * groups
    first.append(at)
    return Shares(per, first)


def locate(s: Shares, t, groups: int = 1, sel: Callable = py_sel, zero=0):
    """Task ``t`` -> (request b, group, c, per, live): the request whose tasks hold t (the last b with first[b] <= t),
    its group and share index c in [0, per), per = per[b], and live = t below first[B] (a task past the used ones:
    nothing to do; b, group and c are then those of the last request, unused)."""
    b, base, per = zero, s.first[0], s.per[0]
    for j in range(1, len(s.per)):
        here = t >= s.first[j]
        b, base, per = sel(here, j, b), sel(here, s.first[j], base), sel(here, s.per[j], per)
    rest = t - base
    one = sel(per > 0, per, 1)
    return b, rest // one if groups > 1 else zero, rest % one, per, t < s.first[-1]
