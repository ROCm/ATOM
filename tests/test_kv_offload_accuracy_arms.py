"""CPU-only validity checks for the service-level offload probe."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    import sys
    import urllib.request

    def unexpected_request(*args, **kwargs):
        raise AssertionError("import must not contact a server")

    monkeypatch.setattr(sys, "argv", ["kv_offload_accuracy_arms.py"])
    monkeypatch.setattr(urllib.request, "urlopen", unexpected_request)
    path = Path(__file__).resolve().parents[1] / "tools/kv_offload_accuracy_arms.py"
    spec = importlib.util.spec_from_file_location("accuracy_arms", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(probe, **values):
    return {name: values.get(name, 0.0) for name in probe.METRICS}


def test_import_does_not_contact_server(probe):
    assert callable(probe.main)


def test_missing_metric_is_not_a_zero_hit(probe):
    with pytest.raises(ValueError, match="missing"):
        probe.parse_metrics('vllm:external_prefix_cache_queries_total{model="m"} 3\n')


def test_metrics_sum_labels_but_not_created_or_similar_names(probe):
    text = "\n".join(
        f'vllm:{name}{{engine="{rank}"}} {rank + 1}'
        for name in probe.METRICS
        for rank in range(2)
    )
    text += "\nvllm:prefix_cache_hits_total_created 999\n"
    assert probe.parse_metrics(text) == {name: 3.0 for name in probe.METRICS}


def test_nonfinite_metrics_are_invalid(probe):
    text = "\n".join(f"vllm:{name} NaN" for name in probe.METRICS)
    with pytest.raises(ValueError, match="finite"):
        probe.parse_metrics(text)


def test_counter_reset_is_invalid(probe):
    before = metrics(probe, external_prefix_cache_hits_total=128)
    with pytest.raises(ValueError, match="decreased"):
        probe.metric_delta(before, metrics(probe))


def test_supply_uses_all_prompt_tokens_not_only_hbm_miss_tail(probe):
    result = probe.summarize(
        [(True, 1000), (True, 1000)],
        metrics(
            probe,
            external_prefix_cache_queries_total=400,
            external_prefix_cache_hits_total=300,
            prefix_cache_queries_total=2000,
            prefix_cache_hits_total=1600,
        ),
    )
    assert result["tier_share"] == 15
    assert result["hbm_share"] == 80
    assert result["reuse_share"] == 95
    assert result["prompt_tokens"] == 2000


def test_reuse_exceeding_prompt_budget_is_invalid(probe):
    with pytest.raises(ValueError, match="exceeds"):
        probe.summarize(
            [(True, 1000)],
            metrics(
                probe,
                external_prefix_cache_hits_total=500,
                prefix_cache_hits_total=600,
            ),
        )


def test_same_questions_are_used_at_each_concurrency(probe):
    assert probe.make_data(64, 16, 20261011) == probe.make_data(64, 16, 20261011)
    codes, _, sectors = probe.make_data(64, 16, 20261011)
    assert len(codes) == 64 and len(set(sectors)) == 48


def test_warm_accuracy_regression_fails_even_with_good_hit_rate(probe):
    args = SimpleNamespace(
        questions=16, min_tier_share=50, min_hbm_share=0, min_reuse_share=50
    )
    blind = {"correct": 0, "tier_share": 0, "hbm_share": 0, "reuse_share": 0}
    cold = {"correct": 16, "tier_share": 0, "hbm_share": 0, "reuse_share": 0}
    warm = {"correct": 15, "tier_share": 95, "hbm_share": 0, "reuse_share": 95}
    with pytest.raises(ValueError, match="accuracy"):
        probe.validate(args, blind, cold, warm)


@pytest.mark.parametrize("tier_tokens,hbm_tokens", [(900, 0), (0, 900), (300, 600)])
def test_concurrent_cli_records_actual_supply(
    probe, monkeypatch, tmp_path, tier_tokens, hbm_tokens
):
    import json
    import re
    import threading

    counters = metrics(probe)
    lock = threading.Lock()
    calls = []

    def ask(args, doc, sector, *, with_doc, nonce=None):
        text = "unknown"
        if with_doc:
            text = re.search(rf"Record {sector:04d}:.* is (.*?)\.", doc)[1]
        with lock:
            calls.append((with_doc, nonce))
            counters["prefix_cache_queries_total"] += 1000
            counters["external_prefix_cache_queries_total"] += 1000
            if with_doc and nonce is None:
                counters["external_prefix_cache_hits_total"] += tier_tokens
                counters["prefix_cache_hits_total"] += hbm_tokens
        return {
            "text": text,
            "prompt_tokens": 1000,
            "sector": sector,
            "finish_reason": "stop",
        }

    monkeypatch.setattr(probe, "ask", ask)
    monkeypatch.setattr(probe, "tier", lambda url: dict(counters))
    output = tmp_path / "result.json"
    rc = probe.main(
        [
            "8021",
            "16",
            "64",
            "96",
            "test-model",
            "--concurrency",
            "4",
            "--min-tier-share",
            str(tier_tokens / 10),
            "--min-hbm-share",
            str(hbm_tokens / 10),
            "--out",
            str(output),
        ]
    )
    assert rc == 0
    data = json.loads(output.read_text())
    assert data["status"] == "passed"
    assert len(calls) == 49
    assert len({nonce for _, nonce in calls if nonce is not None}) == 16
    assert data["arms"]["cold"]["reuse_share"] == 0
    warm = data["arms"]["warm"]
    assert warm["correct"] == 16 and len(warm["responses"]) == 16
    assert warm["tier_hits"] == tier_tokens * 16
    assert warm["hbm_hits"] == hbm_tokens * 16
    assert warm["prompt_tokens"] == 16000


def test_cold_hbm_reuse_invalidates_control(probe):
    args = SimpleNamespace(
        questions=16, min_tier_share=0, min_hbm_share=50, min_reuse_share=50
    )
    blind = {"correct": 0, "tier_share": 0, "hbm_share": 0, "reuse_share": 0}
    cold = {"correct": 16, "tier_share": 0, "hbm_share": 90, "reuse_share": 90}
    warm = {"correct": 16, "tier_share": 0, "hbm_share": 90, "reuse_share": 90}
    with pytest.raises(ValueError, match="cold"):
        probe.validate(args, blind, cold, warm)
