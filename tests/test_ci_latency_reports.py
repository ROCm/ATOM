import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry, Histogram, generate_latest

SCRIPTS = Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/observability"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def collector(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return load_module("collect_metrics")


def test_targets_preserve_distinct_hosts_ports_and_roles(collector):
    config = collector.scrape_config(
        ["10.0.0.1:8010", "10.0.0.1:8011"],
        ["10.0.0.2:8020", "10.0.0.3:8020"],
        "127.0.0.1:30100",
    )
    api, mesh = config["scrape_configs"]
    assert api["static_configs"][0] == {
        "targets": ["10.0.0.1:8010", "10.0.0.1:8011"],
        "labels": {"observer": "api", "role": "prefill"},
    }
    assert api["static_configs"][1]["targets"] == ["10.0.0.2:8020", "10.0.0.3:8020"]
    assert mesh["static_configs"][0]["targets"] == ["127.0.0.1:30100"]
    with pytest.raises(ValueError):
        collector.scrape_config(["user:secret@host:8010"], ["host:8020"], "host:29100")


@pytest.mark.parametrize("exit_code", [0, 17])
def test_collector_setup_failure_preserves_benchmark_exit_and_diagnostic_report(
    collector, monkeypatch, tmp_path, exit_code
):
    def unavailable(_directory):
        raise OSError("test download unavailable")

    monkeypatch.setattr(collector, "ensure_prometheus", unavailable)
    marker = tmp_path / "executed"
    args = argparse.Namespace(
        output=tmp_path / "report",
        model="test",
        prefill=["127.0.0.1:8010"],
        decode=["127.0.0.1:8020"],
        mesh="127.0.0.1:29100",
        command=[
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).touch(); sys.exit(int(sys.argv[2]))",
            str(marker),
            str(exit_code),
        ],
    )
    assert collector.run(args) == exit_code
    assert marker.exists()
    status = json.loads((args.output / "status.json").read_text())
    assert status["status"] == "unavailable"
    assert status["benchmark_exit_code"] == exit_code
    assert "download unavailable" in status["errors"][0]
    data = json.loads((args.output / "report-data.json").read_text())
    assert data["meta"]["kind"] == "recorded"
    assert all(not points for p in data["panels"] for points in p["series"].values())
    assert (args.output / "report.html").is_file()


def make_report(results, cell, job, name="aiperf-model-pd-c4"):
    folder = (
        results / cell / f"slurm_job-{job}" / "benchmark_results" / name / "metrics"
    )
    folder.mkdir(parents=True)
    (folder / "report.html").write_text(f"report for {cell}/{job}")
    (folder / "status.json").write_text(
        json.dumps({"status": "complete", "benchmark_exit_code": 0})
    )
    return folder


def test_staging_selects_current_cell_and_job_and_links_the_artifact(tmp_path):
    stage = load_module("stage_reports")
    results, output = tmp_path / "results", tmp_path / "staged"
    make_report(results, "cell-a", 100)
    make_report(results, "cell-a", 101)
    make_report(results, "cell-b", 101)
    (results / "cell-a.slurm-job-id").write_text("101\n")
    manifest = stage.stage_reports(results, "cell-a", output)
    assert len(manifest["reports"]) == 1
    html = output / manifest["reports"][0]["html"]
    assert html.read_text() == "report for cell-a/101"
    summary = tmp_path / "summary.md"
    stage.write_summary(
        manifest, "https://github.com/org/repo/actions/runs/123/artifacts/456", summary
    )
    assert "artifacts/456" in summary.read_text() and "101" in summary.read_text()
    with pytest.raises(FileExistsError):
        stage.stage_reports(results, "cell-a", output)


def test_missing_job_never_publishes_historical_reports(tmp_path):
    stage = load_module("stage_reports")
    results = tmp_path / "results"
    make_report(results, "cell-a", 100)
    manifest = stage.stage_reports(results, "cell-a", tmp_path / "staged")
    assert manifest["reports"] == [] and manifest["notes"]
    assert not list((tmp_path / "staged").rglob("*.html"))


def test_interrupted_status_keeps_report_and_logs_available(tmp_path):
    stage = load_module("stage_reports")
    results = tmp_path / "results"
    folder = make_report(results, "cell-a", 101)
    (folder / "status.json").write_text('{"status":')
    (folder / "prometheus.log").write_text("interrupted collector")
    (results / "cell-a.slurm-job-id").write_text("101")
    output = tmp_path / "staged"
    manifest = stage.stage_reports(results, "cell-a", output)
    report = manifest["reports"][0]
    assert report["status"] == "unavailable" and report["errors"]
    assert (output / report["html"]).is_file()
    assert (
        output / report["name"] / "prometheus.log"
    ).read_text() == "interrupted collector"


@pytest.mark.parametrize("has_lock", [False, True])
def test_mesh_build_uses_writable_copy_and_archives_resolved_dependencies(
    tmp_path, has_lock
):
    source = tmp_path / "read only source"
    source.mkdir()
    (source / "Cargo.toml").write_text("[package]\n")
    (source / "target").mkdir()
    (source / "target" / "old-artifact").touch()
    if has_lock:
        (source / "Cargo.lock").write_text("existing lock")
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    cargo = binary_dir / "cargo"
    cargo.write_text(
        "#!" + sys.executable + "\n"
        "import sys\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "source=Path(args[args.index('--manifest-path')+1]).parent\n"
        "target=Path(args[args.index('--target-dir')+1])\n"
        "assert not (source/'target').exists()\n"
        "assert ('--locked' in args)==(source/'Cargo.lock').exists()\n"
        "(source/'Cargo.lock').write_text('resolved dependencies')\n"
        "(target/'ci').mkdir(parents=True)\n"
        "(target/'ci/atomesh').touch()\n"
    )
    cargo.chmod(0o755)
    logs = tmp_path / "logs"
    result = subprocess.run(
        ["bash", str(SCRIPTS / "build_mesh.sh"), str(source), str(logs)],
        env={
            **os.environ,
            "PATH": str(binary_dir) + os.pathsep + os.environ["PATH"],
            "TMPDIR": str(tmp_path),
            "ATOMESH_MESH_TARGET_DIR": str(tmp_path / "cache"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(result.stdout.strip()).is_file()
    assert (logs / "mesh-Cargo.lock").read_text() == "resolved dependencies"
    assert (source / "Cargo.lock").exists() == has_lock
    if has_lock:
        assert (source / "Cargo.lock").read_text() == "existing lock"


def test_partial_report_remains_downloadable_and_path_traversal_is_rejected(tmp_path):
    stage = load_module("stage_reports")
    results = tmp_path / "results"
    report = make_report(results, "cell-a", 101)
    (report / "status.json").write_text(
        json.dumps({"status": "partial", "benchmark_exit_code": 7})
    )
    (results / "cell-a.slurm-job-id").write_text("101")
    manifest = stage.stage_reports(results, "cell-a", tmp_path / "staged")
    assert manifest["reports"][0]["status"] == "partial"
    assert manifest["reports"][0]["html"]
    with pytest.raises(ValueError):
        stage.stage_reports(results, "../cell-a", tmp_path / "bad")


@pytest.mark.skipif(
    not os.environ.get("ATOMESH_TEST_PROMETHEUS_BIN"),
    reason="Set ATOMESH_TEST_PROMETHEUS_BIN for the real Prometheus integration",
)
def test_real_prometheus_exports_all_panels_after_failed_benchmark_and_stops(tmp_path):
    registry = CollectorRegistry()
    ttft = Histogram(
        "atom:time_to_first_token_seconds", "fixture", ["streaming"], registry=registry
    )
    prefill, decode = ttft.labels("false"), ttft.labels("true")
    itl = Histogram("atom:inter_token_latency_seconds", "fixture", registry=registry)
    mesh = Histogram(
        "mesh_router_ttft_seconds",
        "fixture",
        ["router_type", "backend_type"],
        registry=registry,
    ).labels("http", "pd")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/tick":
                prefill.observe(0.02)
                decode.observe(0.01)
                mesh.observe(0.035)
                itl.observe(0.006)
            body = generate_latest(registry)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = f"127.0.0.1:{server.server_port}"
    benchmark = tmp_path / "benchmark.py"
    benchmark.write_text(
        "import sys,time,urllib.request\n"
        "for _ in range(12):\n"
        "    urllib.request.urlopen(sys.argv[1]+'/tick',timeout=5).close()\n"
        "    time.sleep(1)\n"
        "sys.exit(7)\n"
    )
    output = tmp_path / "report"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "collect_metrics.py"),
                "--output",
                str(output),
                "--model",
                "Synthetic HTTP fixture",
                "--prefill",
                target,
                "--decode",
                target,
                "--mesh",
                target,
                "--",
                sys.executable,
                str(benchmark),
                "http://" + target,
            ],
            env={
                **os.environ,
                "ATOMESH_PROMETHEUS_BIN": os.environ["ATOMESH_TEST_PROMETHEUS_BIN"],
            },
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert completed.returncode == 7, completed.stdout + completed.stderr
        status = json.loads((output / "status.json").read_text())
        assert status["status"] == "partial" and status["errors"] == []
        data = json.loads((output / "report-data.json").read_text())
        assert all(
            any(value is not None and value > 0 for _, value in series)
            for panel in data["panels"]
            for series in panel["series"].values()
        )
        assert "Server is ready" in (output / "prometheus.log").read_text()
        assert "See you next time!" in (output / "prometheus.log").read_text()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
