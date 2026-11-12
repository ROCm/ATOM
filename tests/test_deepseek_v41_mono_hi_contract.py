# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 hi programs' hand-off contracts and memory model on real builds (``hi_contract``): traced compile-only
for gfx950 (no GPU launch), every program of the one-request width (each layer kind, the verify head, the Engram
token-only launch, the DSpark context write and the draft's one-request plan) checked at every served TP size, its
Linear fits derived and held to the frozen table. The engine's builds at load
skip these checks (FlyDSL's disk cache); this is where they run. Every served width:
``python -m atom.models.deepseek_v41.mono.hi_contract CONFIG.json``."""

from pathlib import Path

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")
pytest.importorskip("triton")

from atom.models.deepseek_v41.mono import hi_contract

CONFIG = Path(__file__).parent / "models" / "deepseek_v41" / "fixtures" / "config.json"


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_the_one_request_programs_hold(monkeypatch, tmp_path, tp):
    # (FlyDSL's runtime cache dir: its compiled programs and the hi fit cache, ``linear.fit_cache``)
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "flydsl"))
    progs = hi_contract.check(hi_contract.config(str(CONFIG)), [6], tp=tp)
    assert progs.tp == tp
    kinds = {key[0] for key in progs.kernels()}
    assert {"layer", "head", "ctx", "draft"} <= kinds, kinds
    assert all(
        k.record is not None and k.record.stages for k in progs.kernels().values()
    )
    # the Linear fits derived again (``check`` raises where one differs from the frozen table), every one in it
    assert hi_contract.missing_fits() == {}


def test_a_tp_size_the_programs_are_not_built_for_is_refused():
    with pytest.raises(ValueError, match="TP16"):
        hi_contract.check(hi_contract.config(str(CONFIG)), [6], tp=16)


def test_the_tp2_sampler_scans_every_slice_of_a_rank_row(monkeypatch, tmp_path):
    """At TP2 a rank's row of the verify head's logits (64,640 ids) is scanned in 2 slices (``head.SCAN_SLICES``),
    each slice's specs built where it is polled (``_lg_specs(s=)``): the sampler's two scans (level 1, level 2) poll
    every slice alike, none twice in place of another (greedy rows never sample: only this shows it).
    """
    from collections import Counter

    from atom.models.deepseek_v41.mono.hi.blocks.gfx950 import head as blocks_head
    from atom.models.deepseek_v41.mono.hi.device.gfx950 import mailbox
    from atom.models.deepseek_v41.mono.hi.plan.program import Scenario
    from atom.models.deepseek_v41.mono.hi.runtime.kernel import MonoKernel
    from atom.models.deepseek_v41.mono.hi.v41 import head as H

    for k, v in (("ARCH", "gfx950"), ("FLYDSL_GPU_ARCH", "gfx950"), ("COMPILE_ONLY", "1"),
                 ("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "flydsl"))):  # fmt: skip
        monkeypatch.setenv(k, v)
    built, kept, polled = {}, [], Counter()
    lg_specs, poll = blocks_head._lg_specs, mailbox.poll

    def lg_specs_rec(k, p, r, s=0):
        out = lg_specs(k, p, r, s)
        kept.append(out)  # (held: the specs' ids stay theirs)
        built.update((id(spec), s) for spec in out[0])
        return out

    def poll_rec(specs, *args, **kw):
        polled.update(built[id(spec)] for spec in specs if id(spec) in built)
        return poll(specs, *args, **kw)

    monkeypatch.setattr(blocks_head, "_lg_specs", lg_specs_rec)
    monkeypatch.setattr(mailbox, "poll", poll_rec)
    MonoKernel(
        H.head_program(
            "tp2_sample", Scenario(1, 6, 2), res_out="r_h", sample=True, race=True
        )
    )
    assert (
        set(polled) == set(range(blocks_head.SCAN_SLICES))
        and len(set(polled.values())) == 1
    ), polled
