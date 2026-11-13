# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 hi programs at TP2 and their build fallbacks on real builds (traced compile-only for gfx950, no GPU
launch): the DSpark draft head over a rank's 64,640 vocab rows takes two 128-id blocks a task
(``MarkovHead(blocks=)``); a width whose programs are refused is left to ATOM's mono decode; a draft whose heads at
several requests are refused runs the one-request plan a request at a time, keeping only the programs it launches.
The refusals are forced here, so the fallbacks are pinned whatever a FlyDSL version spills."""

from pathlib import Path

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")
pytest.importorskip("triton")

from atom.models.deepseek_v41.mono import hi_compose, hi_contract
from atom.models.deepseek_v41.mono.hi.blocks.gfx950 import linear
from atom.models.deepseek_v41.mono.hi.emit.compile import ResourceError
from atom.models.deepseek_v41.mono.hi.plan.program import PlanError, Scenario
from atom.models.deepseek_v41.mono.hi.target import MI355X
from atom.models.deepseek_v41.mono.hi.v41 import draft

CONFIG = Path(__file__).parent / "models" / "deepseek_v41" / "fixtures" / "config.json"


@pytest.fixture
def flydsl_cache(monkeypatch, tmp_path):
    # (FlyDSL's runtime cache dir: its compiled programs and the hi fit cache, ``linear.fit_cache``)
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "flydsl"))
    # the fits these builds derive kept out of the process-wide record the other checks read
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
    """The one-request width at TP2 in ATOM's default FP4 index layout: every program (the two-block draft head among
    them) builds and holds the hand-off contract."""
    progs = hi_contract.check(
        hi_contract.config(str(CONFIG)), [6], tp=2, index_layout="mfma"
    )
    assert progs.widths == [6]


def test_a_refused_width_is_left_to_atom(flydsl_cache, monkeypatch):
    """Width 12's layer programs refused: 12 is not served (out of ``split``, the step's routing), its programs and
    verify head are not kept, and the caches' batch slots follow the widths served."""
    program = hi_compose.HiPrograms._program

    def refuse_12(self, p, s, *args, **kw):
        if s == 12:
            raise ResourceError("forced")
        return program(self, p, s, *args, **kw)

    monkeypatch.setattr(hi_compose.HiPrograms, "_program", refuse_12)
    progs = hi_contract.check(hi_contract.config(str(CONFIG)), [6, 12], tp=4)
    assert progs.widths == [6] and 12 not in progs.split
    assert {key[1] for key in progs.programs} == {6}
    assert list(progs.head_programs) == [6]
    assert set(map(id, progs._heads.values())) == {id(progs.head_programs[6])}
    assert progs.bmax == 1


def test_a_draft_whose_heads_are_refused_runs_one_request_at_a_time(
    flydsl_cache, monkeypatch
):
    """The draft heads of 2 requests refused: the 2-request draft is the one-request plan a request at a time, and the
    programs that fallback built and does not launch are not kept."""
    head_blocks = draft.head_blocks

    def refuse_batches(tp=4, rows=5, *args, **kw):
        if rows > 5:
            raise PlanError("forced")
        return head_blocks(tp, rows, *args, **kw)

    monkeypatch.setattr(draft, "head_blocks", refuse_batches)
    progs = hi_contract.check(hi_contract.config(str(CONFIG)), [6, 12], tp=4)
    assert progs.dr_plans[2].per_request
    launched = {d.key for plan in progs.dr_plans.values() for d in plan.launches}
    assert set(progs.dr_kerns) == launched
