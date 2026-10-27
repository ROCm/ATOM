# SPDX-License-Identifier: MIT
"""`--throughput-log-interval-steps` is refused outside its validated scope.

PP is the hard case: `PPEngineCoreProc` replaces `busy_loop` outright, so
`bump_step` is never reached and the line would go silent after announcing a
logging mode. RapidServe and P/D are a scope choice -- `PrefillEngineCore` and
`DecodeEngineCore` inherit the base loop and override only
`_process_engine_step`, so the counter would advance there. Either way
`Config.__post_init__` falls back to the clock and warns.

The predicate has to tell a disaggregated role apart from an LMCache cache
tier, which also travels in `kv_transfer_config`.
"""

from __future__ import annotations

import pytest

from atom.config import Config


def _resolve(**kw) -> Config:
    """Drive the one clause: `__post_init__` reaches the loader and the GPU."""
    cfg = Config.__new__(Config)
    cfg.throughput_log_interval = kw.pop("throughput_log_interval", 10.0)
    cfg.throughput_log_interval_steps = kw.pop("throughput_log_interval_steps", 0)
    cfg.pipeline_parallel_size = kw.pop("pipeline_parallel_size", 1)
    cfg.enable_rapidserve = kw.pop("enable_rapidserve", False)
    cfg.kv_transfer_config = kw.pop("kv_transfer_config", {})
    cfg.enable_log_stats = kw.pop("enable_log_stats", True)
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
        # A real launch always names the connector; the role alone is not a
        # config the factory could build anything from.
        {"kv_transfer_config": {"kv_connector": "mooncake", "kv_role": "kv_producer"}},
        {"kv_transfer_config": {"kv_connector": "mooncake", "kv_role": "kv_consumer"}},
        # A MultiConnector prefill node carries the role one level down, with
        # nothing at the top -- see the multi/ README's launch command.
        {
            "kv_transfer_config": {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                    {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                ],
            }
        },
        # No explicit role: a transfer connector defaults to kv_producer.
        {"kv_transfer_config": {"kv_connector": "mooncake"}},
    ],
    ids=["pp", "rapidserve", "pd_prefill", "pd_decode", "pd_multi", "pd_default_role"],
)
def test_unsupported_topologies_fall_back_to_the_clock(topology, caplog):
    with caplog.at_level("WARNING"):
        cfg = _resolve(throughput_log_interval_steps=40, **topology)
    assert cfg.throughput_log_interval_steps == 0, "would have logged nothing at all"
    assert "throughput_log_interval_steps=40 ignored" in caplog.text


@pytest.mark.parametrize(
    "kv_transfer_config",
    [
        {"kv_role": "offload", "kv_connector": "lmcache_offload"},
        {"kv_connector": "lmcache_offload"},
        # `kv_producer` / `kv_consumer` are legal offload DIRECTIONS (save
        # only / load only) -- see `_VALID_KV_ROLES`. The role does not make
        # the engine disaggregated; the connector name does.
        {"kv_connector": "lmcache_offload", "kv_role": "kv_producer"},
        {"kv_connector": "lmcache_offload", "kv_role": "kv_consumer"},
    ],
    ids=["offload_role", "no_role", "offload_save_only", "offload_load_only"],
)
def test_a_cache_tier_is_not_a_disaggregated_engine(kv_transfer_config, caplog):
    """`kv_transfer_config` carries LMCache offload too, on an ordinary engine.

    That engine runs the loop that counts, so refusing it would turn the option
    off for a topology it works on -- and name a P/D engine while doing it.
    """
    with caplog.at_level("WARNING"):
        cfg = _resolve(
            throughput_log_interval_steps=40,
            kv_transfer_config=kv_transfer_config,
        )
    assert cfg.throughput_log_interval_steps == 40
    assert "ignored" not in caplog.text


@pytest.mark.parametrize(
    "connectors",
    [
        {"kv_connector": "mooncake", "kv_role": "kv_producer"},  # dict, not list
        "mooncake",
        [None],
        ["mooncake"],
        [],
    ],
    ids=["dict", "string", "null_entry", "string_entry", "empty"],
)
def test_a_malformed_connectors_blob_does_not_raise(connectors):
    """The blob is user-supplied CLI JSON, and this clause degrades, not raises.

    Writing `connectors` as a dict is an easy hand slip. Raising here is an
    `AttributeError` out of `Config.__post_init__` -- the server refuses to
    start, blamed on a logging option, and before the KV config's own
    validation has had a chance to report the real problem.
    """
    cfg = _resolve(
        throughput_log_interval_steps=40,
        kv_transfer_config={
            "kv_connector": "multi",
            "connectors": connectors,
        },
    )
    assert cfg.throughput_log_interval_steps == 40


def test_the_line_being_off_is_not_worth_a_warning(caplog):
    """The fallback a warning would name is the clock, equally silent here."""
    with caplog.at_level("WARNING"):
        cfg = _resolve(throughput_log_interval_steps=40, enable_log_stats=False)
    assert cfg.throughput_log_interval_steps == 0
    assert caplog.text == ""


def test_the_guard_runs_after_kv_transfer_config_is_parsed():
    """It reads `kv_role` off a dict; the CLI hands `__post_init__` a string.

    Asserted on `__post_init__`'s own source rather than by calling it, which
    needs the loader and the GPU. An earlier draft asserted the `AttributeError`
    from passing a string to `_resolve` directly -- but `_resolve` never runs
    `__post_init__`, so moving the call back above the parse left it green.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(Config.__post_init__)))
    nodes = list(ast.walk(tree))

    guard = next(
        n.lineno
        for n in nodes
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_resolve_throughput_step_interval"
    )
    # Every rebind of the field, not just one parser: the JSON path has a
    # `literal_eval` fallback, and either may be the one that is kept.
    parse = max(
        n.lineno
        for n in nodes
        if isinstance(n, ast.Attribute)
        and isinstance(n.ctx, ast.Store)
        and n.attr == "kv_transfer_config"
    )
    assert guard > parse, (
        "_resolve_throughput_step_interval reads kv_transfer_config.kv_role, "
        "so it must run after the CLI string has been parsed into a dict"
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
