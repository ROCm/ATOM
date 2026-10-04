"""CPU-only tests for the opt-in EngineCore decode-step stats reporter."""

import logging
from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.model_engine import engine_core


def _batch(
    *,
    prefill_tokens: int = 0,
    decode_tokens: int = 0,
    prefill_seqs: int = 0,
    decode_seqs: int = 0,
):
    return SimpleNamespace(
        total_tokens_num_prefill=prefill_tokens,
        total_tokens_num_decode=decode_tokens,
        total_seqs_num_prefill=prefill_seqs,
        total_seqs_num_decode=decode_seqs,
    )


def _new_core(monkeypatch, clock):
    monkeypatch.setattr(engine_core.time, "monotonic", lambda: clock[0])
    core = engine_core.EngineCore.__new__(engine_core.EngineCore)
    core.label = "test core"
    core.scheduler = SimpleNamespace(
        get_request_counts=lambda: (6, 7),
        _kv_usage=lambda: 0.375,
    )
    config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_rank=3))
    core._init_engine_stats_reporter(config)
    return core


def _stats_lines(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("[EngineStats DP")
    ]


def test_reporter_is_off_by_default_and_ignores_inert_interval(monkeypatch, caplog):
    monkeypatch.delenv("ATOM_LOG_ENGINE_STATS", raising=False)
    monkeypatch.setenv("ATOM_ENGINE_STATS_INTERVAL", "not-an-integer")

    with caplog.at_level(logging.WARNING, logger="atom"):
        core = _new_core(monkeypatch, [10.0])
        core._record_engine_stats(_batch(decode_tokens=4, decode_seqs=4))

    assert core._engine_stats_enabled is False
    assert core._engine_stats_interval is None
    assert core._engine_stats_decode_steps == 0
    assert not caplog.records


@pytest.mark.parametrize("value", ["", "not-an-integer", "0", "-2"])
def test_invalid_enabled_interval_warns_and_uses_default(monkeypatch, caplog, value):
    monkeypatch.setenv("ATOM_LOG_ENGINE_STATS", "1")
    monkeypatch.setenv("ATOM_ENGINE_STATS_INTERVAL", value)

    with caplog.at_level(logging.WARNING, logger="atom"):
        core = _new_core(monkeypatch, [10.0])

    assert core._engine_stats_interval == 40
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert f"ATOM_ENGINE_STATS_INTERVAL={value!r}" in message
    assert "using default 40" in message


def test_decode_cadence_counters_and_emitted_fields_use_monotonic_time(
    monkeypatch, caplog
):
    monkeypatch.setenv("ATOM_LOG_ENGINE_STATS", "1")
    monkeypatch.setenv("ATOM_ENGINE_STATS_INTERVAL", "2")
    clock = [100.0]
    core = _new_core(monkeypatch, clock)

    with caplog.at_level(logging.INFO, logger="atom"):
        core._record_engine_stats(_batch(prefill_tokens=10, prefill_seqs=1))
        core._record_engine_stats(
            _batch(
                prefill_tokens=4,
                decode_tokens=3,
                prefill_seqs=2,
                decode_seqs=3,
            )
        )
        core._record_engine_stats(_batch(decode_tokens=2, decode_seqs=2))
        assert _stats_lines(caplog) == []

        clock[0] = 104.0
        core._record_engine_stats(_batch(decode_tokens=5, decode_seqs=5))

        clock[0] = 105.0
        core._record_engine_stats(_batch(decode_tokens=4, decode_seqs=4))
        core._record_engine_stats(_batch(prefill_tokens=6, prefill_seqs=1))
        assert len(_stats_lines(caplog)) == 1

        clock[0] = 108.0
        core._record_engine_stats(_batch(decode_tokens=8, decode_seqs=8))

    assert _stats_lines(caplog) == [
        (
            "[EngineStats DP3] decode_step=2 running_reqs=6 waiting_reqs=7 "
            "kv_cache_util=37.5% decode_bs=5 decode_tput=2.5tok/s "
            "prefill_tput=3.5tok/s | interval=2steps/4.00s "
            "prefill_tokens=14 prefill_reqs=3"
        ),
        (
            "[EngineStats DP3] decode_step=4 running_reqs=6 waiting_reqs=7 "
            "kv_cache_util=37.5% decode_bs=8 decode_tput=3.0tok/s "
            "prefill_tput=1.5tok/s | interval=2steps/4.00s "
            "prefill_tokens=6 prefill_reqs=1"
        ),
    ]
    assert core._engine_stats_decode_steps == 4
    assert core._engine_stats_interval_decode_tokens == 0
    assert core._engine_stats_interval_prefill_tokens == 0
    assert core._engine_stats_interval_prefill_seqs == 0
    assert core._engine_stats_last_time == 108.0
