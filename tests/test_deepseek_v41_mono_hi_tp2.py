# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 hi programs at TP2 on real builds (traced compile-only for gfx950, no GPU launch): the DSpark draft head
over a rank's 64,640 vocab rows takes two 128-id blocks a task (``MarkovHead(blocks=)``), a width whose programs do
not build is left to ATOM's mono decode, and a draft whose heads at several requests do not build runs one request at
a time."""

from pathlib import Path

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")
pytest.importorskip("triton")

from atom.models.deepseek_v41.mono import hi_contract
from atom.models.deepseek_v41.mono.hi.blocks.gfx950 import linear
from atom.models.deepseek_v41.mono.hi.plan.program import Scenario
from atom.models.deepseek_v41.mono.hi.target import MI355X
from atom.models.deepseek_v41.mono.hi.v41 import draft

CONFIG = Path(__file__).parent / "models" / "deepseek_v41" / "fixtures" / "config.json"


@pytest.fixture
def flydsl_cache(monkeypatch, tmp_path):
    # (FlyDSL's runtime cache dir: its compiled programs and the hi fit cache, ``linear.fit_cache``)
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "flydsl"))
    # TP2's derived fits (not in the frozen table) kept out of the process-wide record the other checks read
    monkeypatch.setattr(linear, "DERIVED", {})


@pytest.mark.parametrize("tp, blocks", [(2, 2), (4, 1)])
def test_the_draft_head_takes_a_block_pair_where_a_ranks_vocab_passes_the_cus(
    tp, blocks
):
    """505 blocks of 128 ids at TP2 are more tasks than CUs, so a task takes two blocks (253 tasks); TP4's 253 blocks
    fit one a CU."""
    mk = draft.head_blocks(tp, 5)[-1]
    p = mk.plan(Scenario(1, 5, tp))
    assert p.blocks == blocks
    assert p.mtasks <= MI355X.cus
    assert p.mtasks * p.blocks * 128 >= mk.n


def test_the_tp2_one_request_programs_hold(flydsl_cache):
    """The one-request width at TP2: every program (the two-block draft head among them) builds and holds the
    hand-off contract (its Linear fits held to the frozen table where it has them; TP2's are derived).
    """
    progs = hi_contract.check(hi_contract.config(str(CONFIG)), [6], tp=2)
    assert progs.widths == [6]


def test_tp2_leaves_an_unbuilt_width_to_atom_and_drafts_requests_one_at_a_time(
    flydsl_cache,
):
    """TP2's 18-row window layer spills a VGPR: width 18 is not served (its steps route to ATOM's mono decode), width
    12 is; the two-block draft head does not build at 2 requests, so its draft runs the one-request plan a request at
    a time."""
    progs = hi_contract.check(hi_contract.config(str(CONFIG)), [12, 18], tp=2)
    assert progs.widths == [12]
    assert 18 not in progs.split
    assert progs.dr_plans[2].per_request
