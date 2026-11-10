# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 hi programs' hand-off contracts and memory model on real builds (``hi_contract``): traced compile-only
for gfx950 (no GPU launch), every program of the one-request width (each layer kind, the verify head, the Engram
token-only launch, the DSpark context write and the draft's one-request plan) checked, its Linear fits derived and held
to the frozen table. The engine's builds at load
skip these checks (FlyDSL's disk cache); this is where they run. Every served width:
``python -m atom.models.deepseek_v41.mono.hi_contract CONFIG.json``."""

from pathlib import Path

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")
pytest.importorskip("triton")

from atom.models.deepseek_v41.mono import hi_contract

CONFIG = Path(__file__).parent / "models" / "deepseek_v41" / "fixtures" / "config.json"


def test_the_one_request_programs_hold(monkeypatch, tmp_path):
    # (FlyDSL's runtime cache dir: its compiled programs and the hi fit cache, ``linear.fit_cache``)
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "flydsl"))
    progs = hi_contract.check(hi_contract.config(str(CONFIG)), [6])
    kinds = {key[0] for key in progs.kernels()}
    assert {"layer", "head", "ctx", "draft"} <= kinds, kinds
    assert all(
        k.record is not None and k.record.stages for k in progs.kernels().values()
    )
    # the Linear fits derived again (``check`` raises where one differs from the frozen table), every one in it
    assert hi_contract.missing_fits() == {}
