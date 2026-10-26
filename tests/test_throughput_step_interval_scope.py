# SPDX-License-Identifier: MIT
"""`--throughput-log-interval-steps` is refused where nothing feeds it.

Only `EngineCore.busy_loop` and its DP override call `bump_step`. The P/D
engines and PP replace those loops entirely, so the counter there would stay
0 and `window_expired` would never open the window again -- the engine would
announce a logging mode and then print nothing, which is strictly worse than
either supporting it or refusing it. `Config.__post_init__` falls back to the
clock and warns instead.
"""

from __future__ import annotations

import pytest

from atom.config import Config


def _resolve(**kw) -> Config:
    """Drive only the step-interval clause, not the whole __post_init__.

    `Config.__post_init__` reaches the model loader and the GPU; this test is
    about one guard, so it is called directly on a bare instance.
    """
    cfg = Config.__new__(Config)
    cfg.throughput_log_interval = kw.pop("throughput_log_interval", 10.0)
    cfg.throughput_log_interval_steps = kw.pop("throughput_log_interval_steps", 0)
    cfg.pipeline_parallel_size = kw.pop("pipeline_parallel_size", 1)
    cfg.enable_rapidserve = kw.pop("enable_rapidserve", False)
    cfg.kv_transfer_config = kw.pop("kv_transfer_config", {})
    assert not kw, f"unused: {kw}"
    cfg._resolve_throughput_step_interval()
    return cfg


def test_base_and_dp_keep_the_step_interval():
    """Nothing to refuse: these are the loops that call bump_step."""
    assert (
        _resolve(throughput_log_interval_steps=40).throughput_log_interval_steps == 40
    )


def test_unset_stays_unset():
    assert _resolve().throughput_log_interval_steps == 0


@pytest.mark.parametrize("steps", [-1, -40])
def test_a_negative_interval_is_normalised_to_off(steps):
    assert (
        _resolve(throughput_log_interval_steps=steps).throughput_log_interval_steps == 0
    )


@pytest.mark.parametrize(
    "topology",
    [
        {"pipeline_parallel_size": 2},
        {"enable_rapidserve": True},
        {"kv_transfer_config": {"kv_role": "kv_producer"}},
        {"kv_transfer_config": {"kv_role": "kv_consumer"}},
    ],
    ids=["pp", "rapidserve", "pd_prefill", "pd_decode"],
)
def test_unsupported_topologies_fall_back_to_the_clock(topology, caplog):
    with caplog.at_level("WARNING"):
        cfg = _resolve(throughput_log_interval_steps=40, **topology)
    assert cfg.throughput_log_interval_steps == 0, "would have logged nothing at all"
    assert "throughput_log_interval_steps=40 ignored" in caplog.text


def test_a_still_unparsed_kv_transfer_config_does_not_crash():
    """The guard runs after the JSON string is parsed -- pin that it must.

    `kv_transfer_config` reaches `__post_init__` as a string from the CLI and
    is turned into a dict partway through. A guard placed before that point
    calls `.get` on a `str` and takes the whole server down at construction,
    which is how this clause was first written.
    """
    cfg = _resolve(
        throughput_log_interval_steps=40,
        kv_transfer_config={"kv_role": "kv_producer"},
    )
    assert cfg.throughput_log_interval_steps == 0
    with pytest.raises(AttributeError):
        # Documents the failure the ordering avoids, so moving the call back
        # above the parse fails here rather than in production.
        _resolve(
            throughput_log_interval_steps=40,
            kv_transfer_config='{"kv_role": "kv_producer"}',
        )


def test_the_warning_names_the_clock_it_fell_back_to():
    """A warning that does not say what happens instead is half a warning."""
    import logging

    logger = logging.getLogger("atom")
    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    logger.addHandler(handler)
    try:
        _resolve(
            throughput_log_interval_steps=40,
            throughput_log_interval=2.5,
            pipeline_parallel_size=4,
        )
    finally:
        logger.removeHandler(handler)
    assert any("2.5s clock" in m for m in records), records
