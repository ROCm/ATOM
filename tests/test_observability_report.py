import importlib.util
import io
import json
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
    assert len(queries) == 20
    assert any('role="prefill",streaming="false"' in q for q in queries)
    assert any("histogram_quantile(0.9," in q for q in queries)
    for panel in data["panels"]:
        for points in panel["series"].values():
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
    with pytest.raises(ValueError, match="Latency"):
        report.write_report(data, tmp_path / "report.html")
