"""CI collection contract; optional integration with pinned real VM binaries."""

import gzip
import importlib.util
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.entrypoints.openai.metrics import AtomMetricsExporter
from atom.entrypoints.openai.request_metrics import RequestObservation
from atom.model_engine.scheduling_metrics import SchedulingMetrics

SCRIPT = Path(__file__).parents[1] / ".github/scripts/atomesh/observability.py"
spec = importlib.util.spec_from_file_location("ci_observability", SCRIPT)
obs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(obs)


def test_gpu_exporter_reads_cpu_sysfs_and_ignores_non_gpu_entries(tmp_path):
    device = tmp_path / "card0/device"
    hwmon = device / "hwmon/hwmon0"
    hwmon.mkdir(parents=True)
    for name, value in {
        "vendor": "0x1002",
        "gpu_busy_percent": "42",
        "mem_info_vram_used": "1234",
    }.items():
        (device / name).write_text(value)
    (hwmon / "temp1_input").write_text("55000")
    body = obs.gpu_metrics(tmp_path).decode()
    assert 'atom_ci_gpu_busy_ratio{gpu="card0"} 0.42' in body
    assert 'atom_ci_gpu_temperature_celsius{gpu="card0"} 55.0' in body
    assert 'atom_ci_gpu_vram_used_bytes{gpu="card0"} 1234.0' in body


def test_prepare_separates_fast_and_full_scrapes_without_duplicate_families(tmp_path):
    obs.prepare(
        SimpleNamespace(
            directory=tmp_path,
            target=[
                "prefill=http://node:8010/metrics",
                "decode=http://node:8020/metrics",
            ],
            run_id="42",
            model="m",
            topology="cpp4-dcp4",
            case="c48",
            phase="benchmark",
            events="1",
        )
    )
    config = json.loads((tmp_path / "scrape.json").read_text())
    full = [
        job for job in config["scrape_configs"] if job["metrics_path"] == "/metrics"
    ]
    fast = [
        job
        for job in config["scrape_configs"]
        if job["metrics_path"] == "/metrics/scheduling"
    ]
    assert len(full) == len(fast) == 2
    assert all(job["scrape_interval"] == "100ms" for job in fast)
    assert all(job["metric_relabel_configs"][0]["action"] == "drop" for job in full)
    assert config["global"]["scrape_interval"] == "1s"


def test_pages_publish_only_compressed_html_and_escape_destinations(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "obs_pages", SCRIPT.with_name("observability_pages.py")
    )
    pages = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pages)
    source = tmp_path / "artifacts/job/observability/benchmark"
    source.mkdir(parents=True)
    page = '<!doctype html><html lang="en"><title>Test report</title></html>'
    (source / "index.html").write_text(page)
    (source / "run.json").write_text(
        json.dumps({"case": "case c48", "phase": "benchmark"})
    )
    (source / "events.jsonl").write_text('{"request_id":"private-detail"}\n')
    dest = tmp_path / "site"
    path, count = pages.stage_reports(tmp_path / "artifacts", dest, "123", "2")
    assert count == 1
    target = dest / path / "case-c48/benchmark"
    assert gzip.decompress((target / "report.html.gz").read_bytes()).decode() == page
    assert "DecompressionStream" in (target / "index.html").read_text()
    assert not list(dest.rglob("*.jsonl"))
    assert "case-c48/benchmark/" in (dest / path / "index.html").read_text()
    with pytest.raises(ValueError):
        pages.stage_reports(tmp_path / "artifacts", dest, "../123", "1")


BIN = Path(os.environ.get("ATOM_OBSERVABILITY_TEST_BIN", "/tmp/atom-observability-bin"))


@pytest.mark.skipif(
    not (BIN / "vmagent-prod").exists(),
    reason="set ATOM_OBSERVABILITY_TEST_BIN to pinned VM/vmagent binaries",
)
def test_real_vmagent_storage_export_html_and_failure_detection(tmp_path):
    servers, threads, processes, exporters = [], [], [], []
    stop = threading.Event()

    def serve(render):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(render(self.path))

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        servers.append(server)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        threads.append(thread)
        return f"http://127.0.0.1:{server.server_port}/metrics"

    targets = []
    try:
        for role in ("prefill", "decode"):
            exporter = AtomMetricsExporter()
            exporter.requests.configure(
                model="test",
                role=role,
                events_path=tmp_path / "events" / f"{role}.jsonl",
                run_id="integration",
            )
            exporter.update({"enabled": True, "requests_running": 1})
            stats = SchedulingMetrics()
            exporter.scheduling_provider = lambda stats=stats: {"0": stats.snapshot()}
            exporters.append(exporter)

            def render(path, exporter=exporter):
                return (
                    exporter.render_scheduling()
                    if path == "/metrics/scheduling"
                    else exporter.render()
                )

            targets.append(f"{role}={serve(render)}")

            def generate(exporter=exporter, stats=stats, role=role):
                i = 0
                while not stop.wait(0.25):
                    observation = RequestObservation(exporter.requests, f"{role}-{i}")
                    observation.on_output(
                        SimpleNamespace(
                            output_tokens=[1], finished=False, finish_reason=None
                        )
                    )
                    observation.on_output(
                        SimpleNamespace(
                            output_tokens=[2, 3] if role == "decode" else [],
                            finished=True,
                            finish_reason="stop",
                        )
                    )
                    seq = SimpleNamespace()
                    stats.enqueue(seq)
                    stats.duration.observe(0.001)
                    stats.execute(
                        SimpleNamespace(
                            req_ids=[1],
                            total_seqs_num_prefill=1 if role == "prefill" else 0,
                            num_scheduled_tokens=[128 if role == "prefill" else 3],
                            context_lens=[4096],
                        ),
                        {1: seq},
                    )
                    snapshot = stats.snapshot()
                    snapshot["snapshot_timestamp_seconds"] = time.time()
                    exporter.scheduling_provider = lambda snapshot=snapshot: {
                        "0": snapshot
                    }
                    i += 1

            thread = threading.Thread(target=generate, daemon=True)
            thread.start()
            threads.append(thread)
        targets.append("router=" + serve(lambda _: b"mesh_router_requests_total 10\n"))
        targets.append(
            "gpu=" + serve(lambda _: b'atom_ci_gpu_vram_used_bytes{gpu="card0"} 1234\n')
        )
        obs.prepare(
            SimpleNamespace(
                directory=tmp_path,
                target=targets,
                run_id="integration",
                model="test",
                topology="synthetic-cpp4-dcp4",
                case="synthetic-c48",
                phase="benchmark",
                events="1",
            )
        )
        meta = json.loads((tmp_path / "run.json").read_text())
        base = f"http://127.0.0.1:{meta['vm_port']}"
        with (tmp_path / "vm.log").open("wb") as log:
            vm = subprocess.Popen(
                [
                    str(BIN / "victoria-metrics-prod"),
                    f"-httpListenAddr=127.0.0.1:{meta['vm_port']}",
                    f"-storageDataPath={tmp_path / 'vmdata'}",
                    "-memory.allowedBytes=128MiB",
                    "-search.latencyOffset=0",
                    "-search.maxPointsPerTimeseries=1000000",
                ],
                stdout=log,
                stderr=log,
            )
        processes.append(vm)
        for _ in range(50):
            try:
                obs.fetch(base + "/health")
                break
            except OSError:
                time.sleep(0.1)
        else:
            pytest.fail((tmp_path / "vm.log").read_text())
        obs.preflight(SimpleNamespace(directory=tmp_path))
        with (tmp_path / "vmagent.log").open("wb") as log:
            agent = subprocess.Popen(
                [
                    str(BIN / "vmagent-prod"),
                    "-httpListenAddr=127.0.0.1:0",
                    "-memory.allowedBytes=128MiB",
                    f"-promscrape.config={tmp_path / 'scrape.json'}",
                    f"-remoteWrite.url={base}/api/v1/write",
                    f"-remoteWrite.tmpDataPath={tmp_path / 'buffer'}",
                    "-remoteWrite.flushInterval=1s",
                ],
                stdout=log,
                stderr=log,
            )
        processes.append(agent)
        time.sleep(12)
        assert agent.poll() is None, (tmp_path / "vmagent.log").read_text()
        agent.terminate()
        assert agent.wait(timeout=20) == 0
        stop.set()
        for exporter in exporters:
            exporter.requests.close()
        obs.finish(SimpleNamespace(directory=tmp_path))
        report = json.loads((tmp_path / "validation.json").read_text())
        assert report["status"] == "pass", report["errors"]
        assert report["series"]["decode/executed_batch_size_last@0.95"]
        assert report["request_events"]["decode"]["output_chunks"] > 0
        assert (tmp_path / "timeseries.jsonl.gz").stat().st_size > 0
        assert "/*REPORT_DATA*/null" not in (tmp_path / "index.html").read_text()
        print(f"Synthetic integration report: {tmp_path / 'index.html'}")
        exporters[0].requests.dropped.inc()
        with pytest.raises(SystemExit):
            obs.finish(SimpleNamespace(directory=tmp_path))
        assert "dropped" in (tmp_path / "validation.json").read_text()
    finally:
        stop.set()
        for exporter in exporters:
            exporter.requests.close()
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)
