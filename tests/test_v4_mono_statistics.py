# SPDX-License-Identifier: MIT
from copy import deepcopy
from types import SimpleNamespace

import pytest

from atom.benchmarks.mono_stats import COUNTERS, execution_delta
from atom.mono.runtime.statistics import serving_statistics


def test_engine_snapshot_keeps_worker_errors_and_does_not_hide_ranks():
    from atom.model_engine.collective_rpc import COLLECTIVE_RPC_METHODS, RpcResult
    from atom.model_engine.llm_engine import LLMEngine

    calls = []

    def rpc(method, **kwargs):
        calls.append((method, kwargs))
        return [
            RpcResult("snapshot", 0, value={"enabled": True}),
            RpcResult("snapshot", 1, error="worker timed out"),
        ]

    engine = SimpleNamespace(core_mgr=SimpleNamespace(collective_rpc=rpc))
    result = LLMEngine.get_mono_statistics(engine, timeout=7)
    assert "get_mono_statistics" in COLLECTIVE_RPC_METHODS
    assert calls == [("get_mono_statistics", {"timeout": 7})]
    assert result["ranks"][1] == {"rank": 1, "stats": None, "error": "worker timed out"}


def _snapshot(policy="auto", steps=0, fallback=0):
    return {
        "version": 1,
        "ranks": [
            {
                "rank": rank,
                "error": None,
                "stats": {
                    "policy": policy,
                    "installed": policy != "off",
                    "enabled": policy != "off",
                    "profile": "ordinary",
                    **dict.fromkeys(COUNTERS, 0),
                    "decode_attempts": steps + fallback,
                    "selected_steps": steps,
                    "graph_replays": steps,
                    "fallback_steps": fallback,
                    "fallback_reasons": {"width": fallback},
                },
            }
            for rank in range(8)
        ],
    }


def test_host_statistics_off_and_disabled():
    assert serving_statistics(None, "off", 3) == {
        "rank": 3,
        "policy": "off",
        "installed": False,
        "enabled": False,
    }
    session = SimpleNamespace(
        disabled=True, serving_status=lambda: {"selected_steps": 0}
    )
    assert not serving_statistics(session, "auto", 0)["enabled"]


def test_measured_delta_excludes_warmup_and_reports_fallback():
    evidence = execution_delta(
        _snapshot(steps=5), _snapshot(steps=15, fallback=2), "ordinary", "auto"
    )
    assert evidence["mono_hit_rate"] == 10 / 12
    assert evidence["ranks"][0]["selected_steps"] == 10
    assert evidence["ranks"][0]["fallback_reasons"] == {"width": 2}


def test_off_control_is_explicit():
    assert execution_delta(_snapshot("off"), _snapshot("off"), "ordinary", "off")[
        "measurement_valid"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_rank",
        "duplicate_rank",
        "error",
        "disabled",
        "profile",
        "reset",
        "only_capture",
        "only_warmup",
        "rank_mismatch",
        "unaccounted",
        "not_executed",
    ],
)
def test_invalid_execution_fails_closed(mutation):
    before, after = _snapshot(steps=5), _snapshot(steps=15)
    row = after["ranks"][0]
    if mutation == "missing_rank":
        after["ranks"].pop()
    elif mutation == "duplicate_rank":
        row["rank"] = 1
    elif mutation == "error":
        row["error"] = "worker timeout"
    elif mutation == "disabled":
        row["stats"]["enabled"] = False
    elif mutation == "profile":
        row["stats"]["profile"] = "mtp3_target"
    elif mutation == "reset":
        row["stats"]["selected_steps"] = 0
    elif mutation == "only_capture":
        row["stats"]["graph_replays"] = 5
    elif mutation == "only_warmup":
        after = deepcopy(before)
    elif mutation == "rank_mismatch":
        row["stats"].update(decode_attempts=16, selected_steps=16, graph_replays=16)
    elif mutation == "unaccounted":
        row["stats"]["decode_attempts"] += 1
    elif mutation == "not_executed":
        row["stats"]["eager_forwards"] = -1
    with pytest.raises(ValueError):
        execution_delta(before, after, "ordinary", "auto")


def test_off_control_cannot_install_mono():
    after = _snapshot("off")
    after["ranks"][0]["stats"]["installed"] = True
    with pytest.raises(ValueError, match="off control"):
        execution_delta(_snapshot("off"), after, "ordinary", "off")


def test_fixed_width_coverage_gate_and_mixed_reporting_are_separate():
    before, after = _snapshot(), _snapshot(steps=92, fallback=8)
    assert execution_delta(before, after, "ordinary", "auto")["mono_hit_rate"] == 0.92
    with pytest.raises(ValueError, match="below"):
        execution_delta(before, after, "ordinary", "auto", min_hit_rate=0.95)
    assert execution_delta(
        before, _snapshot(steps=95, fallback=5), "ordinary", "auto", min_hit_rate=0.95
    )["measurement_valid"]


@pytest.mark.parametrize("minimum", [-1, 2, float("nan")])
def test_invalid_coverage_threshold_is_refused(minimum):
    with pytest.raises(ValueError, match="minimum"):
        execution_delta(
            _snapshot(), _snapshot(steps=1), "ordinary", "auto", min_hit_rate=minimum
        )


@pytest.mark.parametrize("mutation", ["missing", "negative", "reset", "rank_mismatch"])
def test_fallback_reasons_must_account_for_every_rank(mutation):
    before, after = _snapshot(fallback=2), _snapshot(steps=10, fallback=4)
    reasons = after["ranks"][0]["stats"]["fallback_reasons"]
    if mutation == "missing":
        reasons["width"] = 3
    elif mutation == "negative":
        reasons["width"] = -1
    elif mutation == "reset":
        del reasons["width"]
    else:
        reasons.update(width=2, contract=2)
    with pytest.raises(ValueError):
        execution_delta(before, after, "ordinary", "auto")


def test_validation_environment_is_registered(monkeypatch):
    from atom.benchmarks.mono_stats import configured_policy, configured_profile
    from atom.utils import envs

    monkeypatch.setenv("ATOM_V4_MONO_PROFILE", "mtp3_target")
    monkeypatch.setenv("ATOM_V4_MONO_POLICY", "off")
    assert configured_profile() == "mtp3_target"
    assert envs.ATOM_V4_MONO_POLICY == "off"
    assert configured_policy() == envs.ATOM_V4_MONO_POLICY
    assert configured_profile() == envs.ATOM_V4_MONO_PROFILE
    monkeypatch.delenv("ATOM_V4_MONO_PROFILE")
    monkeypatch.delenv("ATOM_V4_MONO_POLICY")
    assert configured_profile() == envs.ATOM_V4_MONO_PROFILE == ""
    assert configured_policy() == envs.ATOM_V4_MONO_POLICY == "auto"


def test_saved_snapshot_can_be_verified_after_teardown(tmp_path, monkeypatch):
    import json
    import sys

    from atom.benchmarks import mono_stats

    before, after, result = (
        tmp_path / name for name in ("before.json", "after.json", "result.json")
    )
    before.write_text(json.dumps(_snapshot("off")))
    after.write_text(json.dumps(_snapshot("off")))
    result.write_text(json.dumps({"n-samples": {"gsm8k": {"effective": 1319}}}))
    monkeypatch.setenv("ATOM_V4_MONO_PROFILE", "ordinary")
    monkeypatch.setenv("ATOM_V4_MONO_POLICY", "off")
    monkeypatch.setattr(
        mono_stats, "snapshot", lambda _: pytest.fail("unexpected live RPC")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mono_stats",
            "verify",
            "--before",
            str(before),
            "--after",
            str(after),
            "--output",
            str(tmp_path / "verified.json"),
            "--result",
            str(result),
            "--min-samples",
            "1319",
        ],
    )
    mono_stats.main()
    assert json.loads(result.read_text())["atom_ci_metadata"]["mono_execution"][
        "measurement_valid"
    ]
