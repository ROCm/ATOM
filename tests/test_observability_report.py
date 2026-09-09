import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

_path = (
    Path(__file__).resolve().parents[1]
    / ".github/scripts/atomesh/observability/export_report.py"
)
_spec = importlib.util.spec_from_file_location("export_report", _path)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


@pytest.mark.parametrize(
    "fraction", ["2", "27", "274", "2741", "27414", "274144", "274144682"]
)
@pytest.mark.parametrize("zone", ["Z", "+00:00"])
def test_prometheus_scrape_timestamps_accept_variable_fractional_precision(
    fraction, zone
):
    timestamp = report.timestamp(f"2026-09-09T09:03:10.{fraction}{zone}")
    expected = report.timestamp("2026-09-09T09:03:10Z") + float("0." + fraction)
    assert abs(timestamp - expected) < 0.000002


def test_prometheus_export_uses_real_response_values_and_keeps_missing_points(
    monkeypatch, tmp_path
):
    queries = []

    def response(request, timeout):
        params = parse_qs(urlsplit(request.full_url).query)
        queries.append(params["query"][0])
        assert params["start"] == ["100"] and params["end"] == ["110"]
        payload = {
            "status": "success",
            "data": {
                "result": [{"values": [[100, "NaN"], [105, "6.25"], [110, "+Inf"]]}]
            },
        }
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(report.urllib.request, "urlopen", response)
    output = tmp_path / "report.html"
    data = report.generate_report("http://prometheus.example", 100, 110, output)
    assert len(queries) == sum(
        len(list(report.panel_queries(p, 60))) * (1 if p["role"] == "overall" else 2)
        for p in data["panels"]
    )
    assert any('role="prefill",streaming="false"' in q for q in queries)
    assert any("histogram_quantile(0.9," in q for q in queries)
    for panel in data["panels"]:
        for points in [
            *panel["series"].values(),
            *panel.get("block_counts", {}).values(),
            *panel.get("cache_counts", {}).values(),
        ]:
            assert points == [[100.0, None], [105.0, 6.25], [110.0, None]]
    text = output.read_text()
    embedded = text.split('<script id="report-data" type="application/json">')[1].split(
        "</script>"
    )[0]
    assert json.loads(embedded) == data
    assert data["meta"]["kind"] == "prometheus"
    assert "__REPORT_DATA__" not in text


def test_json_interface_escapes_script_content_without_mutating_input(tmp_path):
    data = report.demo_data()
    data["meta"]["title"] = '</script><script>alert("test")</script>'
    before = json.dumps(data)
    output = tmp_path / "report.html"
    report.write_report(data, output)
    assert json.dumps(data) == before
    assert data["meta"]["title"] not in output.read_text()
    embedded = (
        output.read_text()
        .split('<script id="report-data" type="application/json">')[1]
        .split("</script>")[0]
    )
    assert json.loads(embedded)["meta"]["title"] == data["meta"]["title"]


def test_failed_source_does_not_overwrite_previous_report(monkeypatch, tmp_path):
    def failed(*args, **kwargs):
        raise OSError("source unavailable")

    monkeypatch.setattr(report.urllib.request, "urlopen", failed)
    output = tmp_path / "report.html"
    output.write_text("previous report")
    with pytest.raises(RuntimeError, match="All Prometheus queries failed"):
        report.generate_report("http://prometheus.example", 100, 110, output)
    assert output.read_text() == "previous report"


@pytest.mark.parametrize("value", [-1, float("nan"), "6.25"])
def test_invalid_json_values_are_rejected(value, tmp_path):
    data = report.demo_data()
    data["panels"][0]["series"]["p99"][0][1] = value
    with pytest.raises(ValueError, match="Metric values"):
        report.write_report(data, tmp_path / "report.html")


def test_scheduler_queries_keep_units_and_gauge_semantics():
    panels = {p["id"]: p for p in report.panels_for("pd")}
    batch = report.query_for(panels["decode_batch_size"], "mean", 60)
    assert batch.startswith("1 * ") and "rate(atom:decode_batch_size_sum" in batch
    queue = report.query_for(panels["decode_queues"], "waiting", 60)
    assert 'state="waiting"' in queue and "rate(" not in queue
    kv = report.query_for(panels["prefill_kv_blocks"], "used", 60)
    assert kv.startswith("100 * sum(") and 'state="total"' in kv
    assert "rate(" not in kv
    used_count = report.block_count_query_for(panels["prefill_kv_blocks"], "used")
    total_count = report.block_count_query_for(panels["decode_kv_blocks"], "total")
    assert (
        used_count
        == 'sum(atom:scheduler_kv_cache_blocks{job="atom",role="prefill",state="used"})'
    )
    assert (
        total_count
        == 'sum(atom:scheduler_kv_cache_blocks{job="atom",role="decode",state="total"})'
    )
    transfer = report.query_for(panels["pd_kv_transfer"], "p99", 60)
    assert 'role="decode"' in transfer and "histogram_quantile(0.99," in transfer
    assert panels["decode_batch_size"]["unit"] == "requests"
    assert panels["prefill_kv_blocks"]["unit"] == "%"
    assert "pd_kv_transfer" not in {p["id"] for p in report.panels_for("standalone")}


def test_grouped_queries_preserve_instances_and_weight_ratios():
    panels = {p["id"]: p for p in report.panels_for("pd")}
    cache = report.query_for(panels["prefill_cache_hit"], "reuse", 60, by_instance=True)
    assert cache.count("sum by (instance)") == 3
    assert "increase(atom:prefix_cache_cached_tokens_total" in cache
    assert "increase(atom:prefix_cache_offload_tokens_total" in cache
    assert "increase(atom:prefix_cache_full_tokens_total" in cache
    assert "avg(" not in cache
    latency = report.query_for(
        panels["decode_gpu_forward"], "p99", 60, by_instance=True
    )
    assert "histogram_quantile(0.99, sum by (instance, le)" in latency
    aggregate = report.query_for(panels["decode_gpu_forward"], "p99", 60)
    assert "sum by (le)" in aggregate and "instance" not in aggregate
    assert panels["decode_context_tokens"]["unit"] == "tokens"
    assert panels["prefill_batch_tokens"]["metric"] == "atom:prefill_batch_tokens"


def test_instance_queries_retain_missing_values_and_do_not_average_percentiles(
    monkeypatch, tmp_path
):
    def response(request, timeout):
        query = parse_qs(urlsplit(request.full_url).query)["query"][0]
        values = [{"metric": {}, "values": [[100, "12"], [105, "18"]]}]
        if "by (instance" in query:
            values = [
                {"metric": {"instance": host}, "values": [[100, value], [105, "NaN"]]}
                for host, value in [("node-a:8010", "10"), ("node-b:8010", "30")]
            ]
        return io.BytesIO(
            json.dumps({"status": "success", "data": {"result": values}}).encode()
        )

    monkeypatch.setattr(report.urllib.request, "urlopen", response)
    data = report.generate_report("http://fixture", 100, 110, tmp_path / "report.html")
    panel = next(p for p in data["panels"] if p["id"] == "prefill_ttft")
    assert panel["series"]["p99"][0][1] == 12
    assert panel["instances"]["node-a:8010"]["series"]["p99"] == [
        [100, 10],
        [105, None],
    ]
    assert panel["instances"]["node-b:8010"]["series"]["p99"][0][1] == 30
    assert {"role": "prefill", "instance": "node-b:8010"} in data["meta"]["instances"]
    cache = next(p for p in data["panels"] if p["id"] == "prefill_cache_hit")
    assert cache["instances"]["node-a:8010"]["cache_counts"]["gpu"][0][1] == 10
    assert cache["instances"]["node-a:8010"]["cache_counts"]["lmcache"][0][1] == 10
    data["panels"][0]["instances"]["bad:1"] = {"series": {"p99": [[100, -1]]}}
    with pytest.raises(ValueError, match="Metric values"):
        report.write_report(data, tmp_path / "bad.html")


def test_archived_gpu_only_cache_report_keeps_its_original_meaning(tmp_path):
    data = report.demo_data()
    panel = next(p for p in data["panels"] if p.get("kind") == "cache")
    panel.pop("cache_breakdown")
    panel.pop("instances")
    panel["series"] = {"hit": panel["series"]["gpu"]}
    panel["cache_counts"] = {
        "cached": panel["cache_counts"]["gpu"],
        "prompt": panel["cache_counts"]["prompt"],
    }
    report.write_report(data, tmp_path / "legacy.html")
    assert report.statistics_for(panel) == ("hit",)
    query = report.query_for(panel, "hit", 60)
    assert "offload" not in query
    assert set(panel["series"]) == {"hit"}


@pytest.mark.skipif(
    not os.environ.get("ATOMESH_TEST_PROMTOOL_BIN"),
    reason="Set ATOMESH_TEST_PROMTOOL_BIN for PromQL tier coverage checks",
)
@pytest.mark.parametrize("missing", [True, False])
def test_cache_reuse_queries_distinguish_missing_tier_from_zero(tmp_path, missing):
    panel = next(p for p in report.panels_for("pd") if p["id"] == "prefill_cache_hit")
    series = []
    for instance, gpu, offload, prompt in (
        ("a:8010", 8, 1, 10),
        ("b:8010", 20, 0, 100),
    ):
        for suffix, increment in (
            ("cached", gpu),
            ("offload", offload),
            ("full", prompt),
        ):
            if missing and instance == "b:8010" and suffix == "offload":
                continue
            series.append(
                {
                    "series": f'atom:prefix_cache_{suffix}_tokens_total{{job="atom",role="prefill",instance="{instance}"}}',
                    "values": f"0+{increment}x8",
                }
            )
    checks = []
    for grouped in (False, True):
        expected = [{"labels": '{instance="a:8010"}', "value": 90}] if grouped else []
        if not missing:
            expected = (
                [*expected, {"labels": '{instance="b:8010"}', "value": 20}]
                if grouped
                else [{"labels": "{}", "value": 100 * 29 / 110}]
            )
        checks.append(
            {
                "expr": report.query_for(panel, "reuse", 60, by_instance=grouped),
                "eval_time": "120s",
                "exp_samples": expected,
            }
        )
    fixture = tmp_path / "cache-coverage.json"
    fixture.write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "interval": "15s",
                        "input_series": series,
                        "promql_expr_test": checks,
                    }
                ]
            }
        )
    )
    result = subprocess.run(
        [os.environ["ATOMESH_TEST_PROMTOOL_BIN"], "test", "rules", str(fixture)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
