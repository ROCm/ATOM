"""Hardware telemetry tests use fake sysfs; no GPU or inference dependencies."""

import importlib
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import pytest
from prometheus_client.parser import text_string_to_metric_families

SCRIPTS = Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/observability"


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return tuple(
        importlib.import_module(name)
        for name in (
            "hardware_exporter",
            "hardware_report",
            "export_report",
            "collect_metrics",
        )
    )


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(value))


def gpu(tmp_path, bdf="0000:05:00.0", card="card9"):
    root = tmp_path / "drm"
    device = tmp_path / "devices" / bdf
    hwmon = device / "hwmon/hwmon12"
    for name, value in {
        "name": "amdgpu",
        "freq8_label": "sclk",
        "freq8_input": 1234000000,
        "freq2_label": "mclk",
        "freq2_input": 2000000000,
        "temp3_label": "junction",
        "temp3_input": 63500,
        "temp4_label": "mem",
        "temp4_input": 41000,
        "power1_label": "PPT",
        "power1_input": 350000000,
        "power1_cap": 1400000000,
    }.items():
        write(hwmon / name, value)
    for name, value in {
        "gpu_busy_percent": 0,
        "mem_busy_percent": 45,
        "mem_info_vram_used": 2**30,
        "mem_info_vram_total": 8 * 2**30,
    }.items():
        write(device / name, value)
    (root / card).mkdir(parents=True)
    (root / card / "device").symlink_to(device, target_is_directory=True)
    return root, device, hwmon


def parse(sampler):
    body, fresh = sampler.snapshot()
    assert fresh
    return {
        family.name: family.samples
        for family in text_string_to_metric_families(body.decode())
    }


def test_units_labels_selection_and_drm_alias_deduplication(tmp_path, modules):
    exporter, *_ = modules
    root, device, _ = gpu(tmp_path)
    gpu(tmp_path, "0000:15:00.0", "card1")
    (root / "card17").mkdir()
    (root / "card17/device").symlink_to(device, target_is_directory=True)
    sampler = exporter.Sampler(
        root, hostname='node"a', interval=1, pci_roles={"0000:05:00.0": "prefill"}
    )
    sampler.sample()
    values = parse(sampler)
    assert len(values["atom_gpu_info"]) == 1
    assert values["atom_gpu_info"][0].labels["pci_bdf"] == "0000:05:00.0"
    assert values["atom_gpu_info"][0].labels["hostname"] == 'node"a'
    assert values["atom_gpu_sclk_mhz"][0].value == 1234
    assert values["atom_gpu_junction_celsius"][0].value == 63.5
    assert values["atom_gpu_power_watts"][0].value == 350
    assert values["atom_gpu_vram_used_bytes"][0].value == 2**30
    assert values["atom_gpu_busy_percent"][0].value == 0


def test_missing_and_invalid_sensors_never_reuse_old_values(tmp_path, modules):
    exporter, *_ = modules
    root, device, hwmon = gpu(tmp_path)
    sampler = exporter.Sampler(root, hostname="node", interval=1)
    sampler.sample()
    (hwmon / "freq8_input").unlink()
    write(hwmon / "temp3_input", "nan")
    write(device / "gpu_busy_percent", 101)
    sampler.sample()
    values = parse(sampler)
    assert "atom_gpu_sclk_mhz" not in values
    assert "atom_gpu_junction_celsius" not in values
    assert "atom_gpu_busy_percent" not in values
    available = {
        x.labels["sensor"]: x.value for x in values["atom_gpu_sensor_available"]
    }
    assert available["sclk"] == available["junction"] == available["busy"] == 0
    assert available["mclk"] == 1
    sampler.updated -= 10
    assert sampler.snapshot()[1] is False


def test_registration_merges_shared_cards_and_reports_failed_workers(tmp_path, modules):
    exporter, *_ = modules
    for name, role in (("p", "prefill"), ("d", "decode")):
        write(
            tmp_path / (name + ".json"),
            json.dumps({"role": role, "devices": ["0000:05:00.0"]}),
        )
    write(tmp_path / "failed.json", "{}")
    write(tmp_path / "in-progress.tmp", "{")
    roles, errors = exporter.registrations(tmp_path)
    assert roles == {"0000:05:00.0": "decode+prefill"}
    assert errors == 1


def test_missing_registered_device_is_visible_as_unavailable(tmp_path, modules):
    exporter, *_ = modules
    sampler = exporter.Sampler(
        tmp_path, hostname="node", interval=1, pci_roles={"0000:05:00.0": "decode"}
    )
    sampler.sample()
    values = parse(sampler)
    assert len(values["atom_gpu_info"]) == 1
    assert all(s.value == 0 for s in values["atom_gpu_sensor_available"])


def vector(name, values, **labels):
    return {
        "metric": {
            "__name__": name,
            "hostname": "node-a",
            "pci_bdf": "0000:05:00.0",
            "gpu_role": "prefill",
            **labels,
        },
        "values": values,
    }


def test_report_retains_peaks_gaps_and_excludes_out_of_run_samples(modules):
    _, report, export, _ = modules
    vectors = [
        vector(
            "atom_gpu_sclk_mhz",
            [[99, 9999], [101, 100], [102, 2000], [104, 100], [111, 300], [121, 9999]],
        )
    ]
    notes = []
    panels = report.panels_from_vectors(
        vectors, 100, 120, step=5, interval=1, diagnostics=notes
    )
    clock = panels[0]
    assert clock["series"]["max"] == [[105, 2000], [110, None], [115, 300], [120, None]]
    assert clock["sample_counts"] == [[105, 3], [110, 0], [115, 1], [120, 0]]
    summary = next(iter(clock["instances"].values()))["summary"]
    assert summary["mean"] == 625
    assert summary["coverage_percent"] == 20
    assert summary["min"] == 100 and summary["max"] == 2000
    export.validate_data(
        {"meta": {"start": 100, "end": 120, "step": 5, "window": 60}, "panels": panels}
    )


def test_energy_does_not_bridge_missing_scrapes_and_devices_not_pooled(modules):
    _, report, *_ = modules
    vectors = [
        vector("atom_gpu_power_watts", [[101, 100], [102, 200], [108, 300]]),
        vector(
            "atom_gpu_power_watts", [[101, 500], [102, 500]], pci_bdf="0000:15:00.0"
        ),
    ]
    panels = report.panels_from_vectors(
        vectors, 100, 110, step=5, interval=1, diagnostics=[]
    )
    panel = next(p for p in panels if p["id"] == "hardware_power")
    a, b = [v["summary"] for v in panel["instances"].values()]
    assert a["energy_joules"] == 150 and a["integrated_seconds"] == 1
    assert b["energy_joules"] == 500
    with pytest.raises(ValueError, match="Duplicate"):
        report.panels_from_vectors(
            vectors + vectors, 100, 110, step=5, interval=1, diagnostics=[]
        )


def test_hardware_targets_and_interval_are_independent(modules):
    *_, collector = modules
    config = collector.scrape_config(
        ["p:8010"],
        ["d:8020"],
        "m:29100",
        hardware=["p:29108", "p:29108", "d:29108"],
        hardware_interval=0.1,
    )
    assert config["global"]["scrape_interval"] == "1s"
    hardware = config["scrape_configs"][-1]
    assert hardware["static_configs"][0]["targets"] == ["d:29108", "p:29108"]
    assert hardware["scrape_interval"] == "100ms"
    with pytest.raises(ValueError):
        collector.scrape_config(
            ["p:8010"], ["d:8020"], "m:29100", hardware=["bad:80/path"]
        )


def test_http_endpoint_and_stale_health(tmp_path, modules):
    exporter, *_ = modules
    root, _, _ = gpu(tmp_path)
    sampler = exporter.Sampler(root, hostname="node", interval=1)
    sampler.sample()
    with ThreadingHTTPServer(("127.0.0.1", 0), exporter.handler_for(sampler)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/metrics") as response:
                assert b"atom_gpu_sclk_mhz" in response.read()
            sampler.updated -= 10
            from urllib.error import HTTPError

            with pytest.raises(HTTPError) as exc:
                urlopen(f"http://127.0.0.1:{server.server_port}/health")
            assert exc.value.code == 503
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_reject_invalid_interval(value, modules):
    exporter, *_ = modules
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        exporter.positive_seconds(value)


def test_real_prometheus_hardware_report_preserves_benchmark_exit(tmp_path, modules):
    import os
    import subprocess
    import sys

    binary = os.environ.get("ATOMESH_TEST_PROMETHEUS_BIN")
    if not binary:
        pytest.skip("Set ATOMESH_TEST_PROMETHEUS_BIN for the real collector test")
    exporter, _, report, _ = modules
    root, _, _ = gpu(tmp_path)
    sampler = exporter.Sampler(root, hostname="fixture", interval=0.1)
    sampler.sample()
    stopped = threading.Event()

    def sample_loop():
        while not stopped.wait(0.1):
            sampler.sample()

    with ThreadingHTTPServer(("127.0.0.1", 0), exporter.handler_for(sampler)) as server:
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        sampling_thread = threading.Thread(target=sample_loop, daemon=True)
        server_thread.start()
        sampling_thread.start()
        target = f"127.0.0.1:{server.server_port}"
        output = tmp_path / "report"
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "collect_metrics.py"),
                    "--output",
                    str(output),
                    "--model",
                    "hardware fixture",
                    "--prefill",
                    target,
                    "--decode",
                    target,
                    "--mesh",
                    target,
                    "--hardware",
                    target,
                    "--hardware",
                    "127.0.0.1:1",
                    "--hardware-scrape-interval-seconds",
                    ".1",
                    "--",
                    sys.executable,
                    "-c",
                    "import time; time.sleep(2); raise SystemExit(7)",
                ],
                env={**os.environ, "ATOMESH_PROMETHEUS_BIN": binary},
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 7, result.stdout + result.stderr
            data = json.loads((output / "report-data.json").read_text())
            status = json.loads((output / "status.json").read_text())
            assert status["benchmark_exit_code"] == 7
            assert status["status"] == "partial"
            assert any("failed a scrape" in e for e in status["errors"])
            hardware = [p for p in data["panels"] if p.get("kind") == "hardware"]
            assert len(hardware) == len(exporter.METRICS)
            clock = next(p for p in hardware if p["id"] == "hardware_sclk")
            summary = next(iter(clock["instances"].values()))["summary"]
            assert summary["mean"] == 1234 and summary["samples"] >= 10
            raw = json.loads((output / "hardware-samples.json").read_text())
            assert raw
            assert all(
                status["start"] < float(t) <= status["benchmark_end"]
                for v in raw
                for t, _ in v["values"]
            )
            assert data["meta"]["hardware"]["end"] == status["benchmark_end"]
            report.validate_data(data)
            html = (output / "report.html").read_text()
            assert "Hardware" in html and "Coverage %" in html
            assert "See you next time!" in (output / "prometheus.log").read_text()
        finally:
            stopped.set()
            server.shutdown()
            sampling_thread.join()
            server_thread.join()
