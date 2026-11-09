# SPDX-License-Identifier: MIT
import logging
from types import SimpleNamespace

from atom.model_engine import pd_timeline


def _seq():
    return SimpleNamespace(
        id=7,
        arrive_time=100.0,
        kv_transfer_params={"transfer_id": "t-7"},
        num_prompt_tokens=4096,
        num_cached_tokens=1024,
    )


def test_marks_keep_first_unless_last_and_emit_once(monkeypatch, caplog):
    monkeypatch.setattr(pd_timeline, "_ENABLED", True)
    clock = iter([100.25, 100.5, 101.0])
    monkeypatch.setattr(pd_timeline, "time", SimpleNamespace(time=lambda: next(clock)))
    seq = _seq()
    pd_timeline.mark(seq, "sched0")
    pd_timeline.mark(seq, "sched0")
    pd_timeline.mark(seq, "sched1", last=True)
    pd_timeline.mark(seq, "sched1", last=True)
    with caplog.at_level(logging.INFO, logger="atom"):
        pd_timeline.emit(seq, "P")
        pd_timeline.emit(seq, "P")
    lines = [r.getMessage() for r in caplog.records if "[PD-TL]" in r.getMessage()]
    assert lines == [
        (
            "[PD-TL] role=P rid=7 tid=t-7 isl=4096 cached=1024 t0=100.000 "
            "sched0=250.0 sched1=1000.0"
        )
    ]


def test_disabled_marks_nothing(monkeypatch, caplog):
    monkeypatch.setattr(pd_timeline, "_ENABLED", False)
    seq = _seq()
    pd_timeline.mark(seq, "ready")
    with caplog.at_level(logging.INFO, logger="atom"):
        pd_timeline.emit(seq, "D")
    assert "_pd_timeline" not in seq.__dict__
    assert not [r for r in caplog.records if "[PD-TL]" in r.getMessage()]
