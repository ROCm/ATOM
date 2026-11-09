# SPDX-License-Identifier: MIT
"""The gfx1250 step kernel's MoE work split (kernels/moe_plan.py) on Python ints: every routed UG unit runs exactly once,
on a CTA that runs routed units, for the striped (wide) and the contiguous (narrow) schedules.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _gfx1250_plan_backend import load

MP = load("moe_plan")
NUG, CTAS, WAVES = 18, 256, 8


def assigned(nch, U):
    ndc = MP.dedicated_ctas(nch, NUG)
    CR = CTAS - ndc
    stripe = MP.striped(nch)
    seen = []
    for b in range(CR):
        u0, u1 = MP.block_lo(b, U, CR), MP.block_lo(b + 1, U, CR)
        for w in range(WAVES):
            wu = u0 + w
            for it in range(max(MP.unit_iters(stripe, U, CR, b, w, u1, wu), 0)):
                seen.append(MP.unit_of(stripe, it, CR, b, w, wu))
    return seen


@pytest.mark.parametrize("nch", [1, 2, 3])
@pytest.mark.parametrize(
    "U", [0, 1, 17, 18 * 6, 2034, 2048, 2049, 2600, 18 * (288 + 16)]
)
def test_units_once(nch, U):
    seen = assigned(nch, U)
    assert sorted(seen) == list(range(U))


def test_schedules():
    assert [MP.striped(n) for n in (1, 2, 3)] == [False, False, True]
    assert [MP.sh_early(n) for n in (1, 2, 3)] == [False, False, MP.SH_EARLY]
    assert MP.dedicated_ctas(3, NUG) == 0  # wide: the shared UG runs early in the front
    assert MP.dedicated_ctas(1, NUG) == NUG  # narrow: one shared unit a dedicated CTA
    assert MP.dedicated_ctas(2, NUG) == 2 * NUG
