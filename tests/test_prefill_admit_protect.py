"""--prefill-admit-policy protect (burst after a prefill, then hold while a
young decode runs). Single rank, no GPU / distributed group."""

import pytest

from atom.model_engine.prefill_delayer import PrefillDelayer


def _delayer(**kw):
    kw.setdefault("protect_tokens", 200)
    d = PrefillDelayer(
        dp_size=1,
        cpu_group=None,
        max_num_batched_tokens=16384,
        admit_policy="protect",
        **kw,
    )
    # consume the unconditional warmup FIRE
    assert d.should_allow_prefill(prefillable=False, pending_tokens=0).allow
    return d


def _tick(d, young, prefillable=True, running=4, **kw):
    return d.should_allow_prefill(
        prefillable=prefillable,
        pending_tokens=500 if prefillable else 0,
        running_decode_batch=running,
        youngest_decode_tokens=young,
        **kw,
    ).allow


def test_holds_while_young_then_fires_when_aged():
    d = _delayer()
    assert not _tick(d, young=10)
    assert not _tick(d, young=199)
    assert _tick(d, young=200)
    assert d._stat_fire_aged == 1
    assert d._stat_hold_protect == 2


def test_no_admission_right_after_prefill_while_young():
    d = _delayer()
    d.notify_prefill_executed(1000)
    assert not _tick(d, young=1)
    assert d._stat_fire_chunk == 0


def test_chunked_prefill_continues_while_young():
    d = _delayer()
    d.notify_prefill_executed(16384)
    assert _tick(d, young=1, has_partial=True)
    assert _tick(d, young=4, has_partial=True)
    assert d._stat_fire_chunk == 2
    # the last chunk ran: new work waits for the young decodes again
    assert not _tick(d, young=7)


def test_must_fire_bounds_beat_protection():
    d = _delayer(max_queue_ms=100.0)
    assert _tick(d, young=1, oldest_waiting_age_ms=150.0)
    assert d._stat_fire_queue_ms == 1
    assert _tick(d, young=1, running=0)  # no decode to protect
    d = _delayer(ttft_max_ticks=3)
    held = [_tick(d, young=1) for _ in range(4)]
    assert held == [False, False, False, True]
    assert d._stat_fire_ttft == 1


def test_interval_ignored_and_bad_config_rejected():
    d = _delayer(prefill_decode_interval=8)
    assert d.prefill_decode_interval == 0
    with pytest.raises(ValueError):
        _delayer(protect_tokens=0)
    with pytest.raises(ValueError):
        PrefillDelayer(
            dp_size=2,
            cpu_group=object(),
            max_num_batched_tokens=16384,
            admit_policy="protect",
            protect_tokens=200,
        )
