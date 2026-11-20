# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only proof of ``gemv_tdm_gfx1250.project(late)``'s copy schedule: each
wave's TDM copies replayed in program order, the waits taken as
``s_wait_tensorcnt`` (at most that many of the wave's latest copies still in
flight) with the counts ``_late_waits`` takes (``gemv_layout``'s
``tdm_late_*_ahead``, fewer near the end of K). Every step's slabs have
landed when its barrier passes, every copy lands in a buffer whose last
reader is past that barrier and that no later step needs before it is
refilled, and no wave ever keeps more than three copies in flight."""

import pytest

from atom.models.kimi_k3.mono.kernels.stages import gemv_layout as layout

SLABS = 7168 // layout.TDM_SLAB


class Wave:
    def __init__(self):
        self.issued = []  # tags, in program order
        self.landed = 0  # copies known landed (all but the latest in flight)
        self.most_in_flight = 0

    def issue(self, tag):
        self.issued.append(tag)
        self.most_in_flight = max(self.most_in_flight, len(self.issued) - self.landed)

    def wait(self, count):
        self.landed = max(self.landed, len(self.issued) - count)

    def has_landed(self, tag):
        return self.issued.index(tag) < self.landed


def _effective(ahead, left):
    """The count ``_late_waits`` / ``_late_tail`` end on: ``ahead``, or the
    copies left to follow where fewer do."""
    return min(ahead, left)


def _simulate(group, depth, token_steps, slabs=SLABS):
    steps = slabs // group
    stages = depth + group
    tokens, weights = Wave(), [Wave(), Wave()]  # a task's even and odd slabs
    weight_buffer = {}  # buffer -> slab it holds
    token_buffer = {}
    last_read = {}  # ("w", buffer) / ("t", buffer) -> step that read it

    def issue_weight(slab, step):
        buffer = slab % stages
        held = weight_buffer.get(buffer)
        if held is not None:
            # its slab was read by a step the barrier just passed
            assert last_read.get(("w", buffer), -1) <= step - 1, (slab, step)
            assert held // group <= step - 1
        weight_buffer[buffer] = slab
        weights[slab % 2].issue(("w", slab))

    def issue_tokens(token_step, step):
        buffer = token_step % token_steps
        held = token_buffer.get(buffer)
        if held is not None:
            assert held <= step - 1
        token_buffer[buffer] = token_step
        for g in range(group):
            tokens.issue(("t", token_step, g))

    for token_step in range(min(token_steps - 1, steps)):
        issue_tokens(token_step, -1)
    for slab in range(min(depth, slabs)):
        issue_weight(slab, -1)
    token_ahead = layout.tdm_late_token_ahead(token_steps)
    weight_ahead = layout.tdm_late_weight_ahead(depth, group)
    for s in range(steps):
        tokens.wait(group * _effective(token_ahead, steps - 1 - s))
        mine = [group * s + g for g in range(group)]
        for slab in mine:
            # paired: both waves wait, a slab each; else the slab's own wave
            weights[slab % 2].wait(_effective(weight_ahead, (slabs - 1 - slab) // 2))
        # the barrier: the step's copies have landed
        assert all(tokens.has_landed(("t", s, g)) for g in range(group))
        assert all(weights[slab % 2].has_landed(("w", slab)) for slab in mine)
        if s + token_steps - 1 < steps:
            issue_tokens(s + token_steps - 1, s)
        for slab in mine:
            if slab + depth < slabs:
                issue_weight(slab + depth, s)
        # the step reads its own buffers, which nothing issued above refills
        for slab in mine:
            assert weight_buffer[slab % stages] == slab
            last_read["w", slab % stages] = s
        assert token_buffer[s % token_steps] == s
        last_read["t", s % token_steps] = s
    return [tokens.most_in_flight] + [w.most_in_flight for w in weights]


@pytest.mark.parametrize(
    ("group", "depth"),
    [(2, 6), (2, 4), (2, 2), (1, 6), (1, 5), (1, 4), (1, 3), (1, 2)],
)
@pytest.mark.parametrize("token_steps", (2, 3))
def test_late_copies_land_before_their_step_and_never_overwrite_a_live_buffer(
    group, depth, token_steps
):
    in_flight = _simulate(group, depth, token_steps)
    # a weight wave keeps at most three tensor copies in flight; the token
    # wave as many as the paired rolled loop's (two steps of two)
    assert max(in_flight[1:]) <= 3
    assert in_flight[0] <= max(3, 2 * group)


@pytest.mark.parametrize(("group", "depth"), [(2, 6), (1, 5), (1, 4), (1, 6)])
def test_late_waits_leave_exactly_the_copies_issued_after_the_awaited_one(
    group, depth
):
    """Each wait leaves in flight precisely the wave's copies issued after the
    one it waits for: the barrier never waits on a later copy."""
    stages_issued = {}
    slabs, steps = SLABS, SLABS // group
    for s in range(steps):
        for slab in (group * s + g for g in range(group)):
            # this wave's slabs issued by step s's waits: < depth, or j + depth
            # issued at an earlier step's barrier (slabs of the same parity)
            issued_after = [
                j
                for j in range(slab + 2, slabs, 2)
                if j < depth or (j - depth) // group <= s - 1
            ]
            ahead = layout.tdm_late_weight_ahead(depth, group)
            stages_issued[slab] = len(issued_after)
            assert _effective(ahead, (slabs - 1 - slab) // 2) == len(issued_after)
