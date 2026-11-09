# SPDX-License-Identifier: MIT
"""The gfx1250 step kernel's counters and LDS regions (kernels/layout.py): per-layer slots disjoint and inside
LAYER_BYTES, tail slots ordered, LDS regions ordered."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _gfx1250_plan_backend import load  # noqa: E402

LY = load("layout")


def test_layer_slots_disjoint():
    used = {}
    for name, (off, nb) in LY.layer_slots().items():
        assert off % 4 == 0 and off + nb <= LY.LAYER_BYTES, name
        for b in range(off, off + nb, 4):
            assert b not in used, (name, used.get(b))
            used[b] = name


def test_tail_slots_ordered():
    seq = [LY.HEAD, LY.HEAD_DONE, LY.SAMPLE, LY.VERIFY_DONE, LY.DRAFT_HEAD_DONE, LY.MARKOV, LY.DRAFT_STATE, LY.DRAFT_FRONT,
           LY.TARGET_STATE, LY.TARGET_FRONT]
    assert seq == sorted(seq) and len(set(seq)) == len(seq)


def test_lds_regions():
    assert LY.TP_PAYLOAD < LY.FRONT_LDS < LY.RKEEP

