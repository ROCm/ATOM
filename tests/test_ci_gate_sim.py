# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A short pass of the heavy-CI gate's protocol simulator (ci_gate_sim.py):
I1 and liveness hold under every combination of the GitHub behaviours the
design does not rely on, and each defence is load-bearing (removing it breaks
I1), so the simulator is shown able to see a violation."""

import ci_gate_sim as sim
import pytest

PER_HYPOTHESIS = 40


def test_i1_and_liveness_hold_under_every_hypothesis():
    tally = sim.run_all(PER_HYPOTHESIS, sim.DEFENCES_ON)

    assert not tally.violations, tally.examples
    assert tally.bugs == 0, tally.examples
    assert tally.outcomes["success"] > 0  # the gate does open


@pytest.mark.parametrize("defence_removed", sorted(sim.DEFENCES_ON))
def test_each_defence_is_needed(defence_removed):
    tally = sim.run_all(PER_HYPOTHESIS * 3, {**sim.DEFENCES_ON, defence_removed: True})

    assert tally.violations
