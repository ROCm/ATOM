# SPDX-License-Identifier: MIT
"""ATOM_EARLY_PREFILL_SCHEDULE gate (`EngineCore._early_prefill_ok`): a lone
engine and an independent DP rank (ATOM_DP_INDEPENDENT) may schedule the next
prefill behind a decode in flight; a lockstep DP rank, a pipeline head and the
disaggregated prefill / decode engines may not. GPU-free."""

from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.model_engine.engine_core import (
        DecodeEngineCore,
        DPEngineCoreProc,
        EngineCore,
        PrefillEngineCore,
    )
    from atom.model_engine.pp_engine_core import PPEngineCoreProc


class _Scheduler:
    def __init__(self):
        self.prefill_delayer = None
        self.waiting = [object()]
        self._partial_prefill_count = 0
        self.advance_on_schedule = False
        self.kv_connector = None

    def try_schedule_prefill_early(self):
        return None


def _engine(cls):
    proc = cls.__new__(cls)
    proc.kv_transfer_enabled = False
    proc.scheduler = _Scheduler()
    return proc


_DECODE = SimpleNamespace(total_seqs_num_prefill=0)


@pytest.fixture(autouse=True)
def _early_on(monkeypatch):
    monkeypatch.setenv("ATOM_EARLY_PREFILL_SCHEDULE", "1")
    monkeypatch.delenv("ATOM_DP_INDEPENDENT", raising=False)
    # ATOM_DP_INDEPENDENT holds only with replicated experts.
    monkeypatch.setenv("ATOM_DP_REPLICATE_EXPERTS", "1")


def test_lone_engine_schedules_early():
    assert _engine(EngineCore)._early_prefill_ok(_DECODE)


def test_independent_dp_rank_schedules_early(monkeypatch):
    monkeypatch.setenv("ATOM_DP_INDEPENDENT", "1")
    assert _engine(DPEngineCoreProc)._early_prefill_ok(_DECODE)


def test_lockstep_dp_rank_does_not():
    assert not _engine(DPEngineCoreProc)._early_prefill_ok(_DECODE)


@pytest.mark.parametrize("cls", [PPEngineCoreProc, PrefillEngineCore, DecodeEngineCore])
def test_coordinated_engines_do_not(monkeypatch, cls):
    monkeypatch.setenv("ATOM_DP_INDEPENDENT", "1")
    assert not _engine(cls)._early_prefill_ok(_DECODE)


def test_flag_off_or_prefill_in_flight_does_not(monkeypatch):
    monkeypatch.setenv("ATOM_DP_INDEPENDENT", "1")
    proc = _engine(DPEngineCoreProc)
    assert not proc._early_prefill_ok(SimpleNamespace(total_seqs_num_prefill=1))
    monkeypatch.setenv("ATOM_EARLY_PREFILL_SCHEDULE", "0")
    assert not proc._early_prefill_ok(_DECODE)
