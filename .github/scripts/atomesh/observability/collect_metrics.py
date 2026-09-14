"""Run a benchmark with an isolated Prometheus collector and export its report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

import export_report

PROMETHEUS_VERSION = "3.5.0"
REPORT_STEP = 5


def save_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def scrape_config(prefill: list[str], decode: list[str], mesh: str) -> dict:
    """Accept the server script's resolved addresses, including per-worker ports."""

    def target(address):
        parsed = urlsplit("http://" + address)
        if not parsed.hostname or not parsed.port or parsed.path or parsed.query:
            raise ValueError(f"Invalid metrics target: {address}")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError(f"Invalid metrics target: {address}")
        return address

    return {
        "global": {"scrape_interval": "5s", "scrape_timeout": "4s"},
        "scrape_configs": [
            {
                "job_name": "atom",
                "static_configs": [
                    {
                        "targets": [target(t) for t in addresses],
                        "labels": {"observer": "api", "role": role},
                    }
                    for role, addresses in (("prefill", prefill), ("decode", decode))
                ],
            },
            {
                "job_name": "atom-mesh",
                "static_configs": [
                    {
                        "targets": [target(mesh)],
                        "labels": {"observer": "mesh", "role": "router"},
                    }
                ],
            },
        ],
    }


def ensure_prometheus(directory: Path) -> str:
    override = os.environ.get("ATOMESH_PROMETHEUS_BIN")
    if override:
        binary = shutil.which(override)
        if not binary:
            raise FileNotFoundError(f"Prometheus executable not found: {override}")
        return binary
    installed = shutil.which("prometheus")
    if installed:
        return installed
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine())
    if platform.system() != "Linux" or arch is None:
        raise RuntimeError("Set ATOMESH_PROMETHEUS_BIN on this platform")
    stem = f"prometheus-{PROMETHEUS_VERSION}.linux-{arch}"
    filename = stem + ".tar.gz"
    base = f"https://github.com/prometheus/prometheus/releases/download/v{PROMETHEUS_VERSION}/"
    with urlopen(base + "sha256sums.txt", timeout=30) as response:
        checksums = response.read().decode()
    expected = next(
        line.split()[0]
        for line in checksums.splitlines()
        if line.split()[-1].lstrip("*") == filename
    )
    archive_path = directory / filename
    digest = hashlib.sha256()
    with (
        urlopen(base + filename, timeout=60) as response,
        archive_path.open("wb") as output,
    ):
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError("Prometheus archive checksum mismatch")
    binary = directory / "prometheus"
    with tarfile.open(archive_path) as archive:
        member = archive.getmember(stem + "/prometheus")
        if not member.isfile():
            raise RuntimeError("Prometheus binary is not a regular file")
        with archive.extractfile(member) as source, binary.open("wb") as output:
            shutil.copyfileobj(source, output)
    binary.chmod(0o755)
    return str(binary)


def get_json(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        return json.load(response)


def wait_for_prometheus(process, log: Path, target_count: int) -> str:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Prometheus exited with code {process.returncode}")
        match = re.search(r"address=127\.0\.0\.1:([1-9]\d*)", log.read_text())
        if match:
            url = f"http://127.0.0.1:{match[1]}"
            try:
                targets = get_json(url + "/api/v1/targets")["data"]["activeTargets"]
                if len(targets) == target_count and all(
                    t["health"] == "up" for t in targets
                ):
                    return url
            except (OSError, ValueError, KeyError):
                pass
        time.sleep(0.5)
    raise RuntimeError(
        "Prometheus or its metrics targets did not become ready in 45 seconds"
    )


def stop_process(process) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def wait_for_final_scrape(
    process,
    url: str,
    start: float,
    benchmark_end: float,
    target_count: int,
    interrupted: Callable[[], bool],
) -> float:
    """Wait for post-benchmark scrapes and a query step that includes them."""
    deadline = time.monotonic() + 30
    query_end = None
    while time.monotonic() < deadline:
        if interrupted():
            return time.time()
        if process.poll() is not None:
            raise RuntimeError("Prometheus exited before the final scrape")
        if query_end is None:
            try:
                targets = get_json(url + "/api/v1/targets")["data"]["activeTargets"]
                if len(targets) == target_count and all(
                    t["health"] == "up"
                    and export_report.timestamp(t["lastScrape"]) > benchmark_end
                    for t in targets
                ):
                    # query_range evaluates start + n * step, not necessarily
                    # its end parameter. Wait until that grid includes the
                    # completed scrapes, without requesting future data points.
                    query_end = (
                        start
                        + math.ceil((time.time() - start) / REPORT_STEP) * REPORT_STEP
                    )
            except (OSError, ValueError, KeyError):
                pass
        if query_end is not None and time.time() >= query_end:
            return query_end
        time.sleep(0.2)
    raise RuntimeError("Final metrics scrapes did not complete in 30 seconds")


def empty_report(start, end, model, notes):
    panels = export_report.panels_for("pd")
    for panel in panels:
        panel["series"] = {key: [] for key in export_report.STATISTICS}
    return {
        "meta": {
            "title": "Agentic PD latency report",
            "model": model,
            "start": start,
            "end": max(end, start + 1),
            "step": REPORT_STEP,
            "window": 60,
            "kind": "recorded",
            "notes": notes,
        },
        "panels": panels,
    }


def finalize_report(data: dict, status: dict, notes: list[str]) -> None:
    """Finalize collection diagnostics and report metadata in one place."""
    missing = [
        p["id"]
        for p in data["panels"]
        if not any(
            any(v is not None for _, v in points) for points in p["series"].values()
        )
    ]
    if missing:
        status["errors"].append("No samples for: " + ", ".join(missing))
    status["errors"] = list(dict.fromkeys(status["errors"]))
    status["status"] = (
        "partial" if status["errors"] or status["benchmark_exit_code"] else "complete"
    )
    if len(missing) == len(data["panels"]):
        status["status"] = "unavailable"
    data["meta"]["notes"] = list(dict.fromkeys([*notes, *status["errors"]]))


def publish_report(output: Path, data: dict, status: dict) -> None:
    """Publish once, retaining benchmark status even when an artifact fails."""
    status["publication_errors"] = []
    for name, write in (
        ("report.html", lambda path: export_report.write_report(data, path)),
        ("report-data.json", lambda path: save_json(path, data)),
        ("status.json", lambda path: save_json(path, status)),
    ):
        try:
            write(output / name)
        except Exception as exc:  # noqa: BLE001
            # Publication is best effort after the benchmark has completed.
            # Keep errors out of collection diagnostics and try other artifacts.
            message = f"Could not publish {name}: {exc}"
            status["publication_errors"].append(message)
            if status["status"] == "complete":
                status["status"] = "partial"
            print(f"[metrics] {message}", flush=True)


def run(args) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = scrape_config(args.prefill, args.decode, args.mesh)
    # JSON is valid YAML and avoids a YAML dependency inside the model image.
    save_json(output / "prometheus.yml", config)
    status = {
        "status": "collecting",
        "benchmark_exit_code": None,
        "model": args.model,
        "errors": [],
    }
    save_json(output / "status.json", status)
    child, collector, prometheus_url = None, None, None
    received_signal, signal_time = None, None
    start = time.time()
    benchmark_rc = 1

    def interrupted(signum, _frame):
        nonlocal received_signal, signal_time
        received_signal, signal_time = signum, time.monotonic()
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)

    original_handlers = {
        sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        # TSDB and downloaded binaries stay on the node's local disk, not /run_logs.
        with (
            tempfile.TemporaryDirectory(prefix="atom-ci-metrics-") as local,
            (output / "prometheus.log").open("w") as log,
        ):
            try:
                binary = ensure_prometheus(Path(local))
                collector = subprocess.Popen(
                    [
                        binary,
                        f"--config.file={output / 'prometheus.yml'}",
                        f"--storage.tsdb.path={local}/data",
                        "--storage.tsdb.retention.time=24h",
                        "--web.listen-address=127.0.0.1:0",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                prometheus_url = wait_for_prometheus(
                    collector,
                    output / "prometheus.log",
                    len(args.prefill) + len(args.decode) + 1,
                )
                # Establish counter baselines before the next benchmark begins.
                time.sleep(6)
                save_json(
                    output / "targets-before.json",
                    get_json(prometheus_url + "/api/v1/targets"),
                )
            except (
                OSError,
                ValueError,
                KeyError,
                RuntimeError,
                StopIteration,
                tarfile.TarError,
            ) as exc:
                status["errors"].append(f"Collection setup failed: {exc}")
                print(f"[metrics] {status['errors'][-1]}", flush=True)

            start = time.time()
            try:
                if received_signal is None:
                    child = subprocess.Popen(args.command, start_new_session=True)
                    while child.poll() is None:
                        if signal_time and time.monotonic() - signal_time > 20:
                            os.killpg(child.pid, signal.SIGKILL)
                        time.sleep(0.2)
                    benchmark_rc = child.returncode
                else:
                    benchmark_rc = 128 + received_signal
            except OSError as exc:
                status["errors"].append(f"Benchmark could not start: {exc}")
                benchmark_rc = 127
            end = time.time()
            # Keep the command's end separate from the report's collection end.
            status.update(
                start=start,
                end=end,
                benchmark_end=end,
                benchmark_exit_code=benchmark_rc,
            )
            collection_end = end
            notes = [
                (
                    "Collected during the complete AIPerf invocation, including its warmup and drain. "
                    "Each point summarizes the preceding 60 seconds; percentiles are histogram estimates."
                )
            ]
            if benchmark_rc:
                notes.append(
                    f"Benchmark exited with code {benchmark_rc}; available samples are retained."
                )
            data = empty_report(start, end, args.model, [])
            try:
                if prometheus_url is None:
                    raise RuntimeError("No Prometheus collector is available")
                try:
                    collection_end = wait_for_final_scrape(
                        collector,
                        prometheus_url,
                        start,
                        end,
                        len(args.prefill) + len(args.decode) + 1,
                        lambda: received_signal is not None,
                    )
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    # Export whatever is available even if a target stays down.
                    collection_end = time.time()
                    status["errors"].append(f"Final scrape incomplete: {exc}")
                data = export_report.collect_report(
                    prometheus_url,
                    start,
                    max(collection_end, start + 1),
                    step=REPORT_STEP,
                    model=args.model,
                    title="Agentic PD latency report",
                    diagnostics=status["errors"],
                )
                targets = get_json(prometheus_url + "/api/v1/targets")
                save_json(output / "targets-after.json", targets)
                up_query = (
                    "min_over_time(up[" + str(max(1, math.ceil(end - start))) + "s])"
                )
                health = get_json(
                    prometheus_url
                    + "/api/v1/query?"
                    + urlencode({"query": up_query, "time": end})
                )
                save_json(output / "scrape-health.json", health)
                if any(
                    float(item["value"][1]) < 1 for item in health["data"]["result"]
                ):
                    status["errors"].append(
                        "One or more metrics targets failed a scrape during the run"
                    )
            except (
                OSError,
                ValueError,
                KeyError,
                RuntimeError,
                StopIteration,
                tarfile.TarError,
            ) as exc:
                status["errors"].append(f"Report export failed: {exc}")
            finally:
                stop_process(collector)

            status["collection_end"] = collection_end
            data["meta"].update(
                end=max(collection_end, start + 1),
                benchmark_end=end,
                collection_end=collection_end,
            )
            notes.append(
                f"The report includes {max(0, collection_end - end):.3f} seconds "
                "of post-benchmark collection to capture final observations. "
                "This interval is excluded from the benchmark duration and may "
                "include other traffic after the benchmark."
            )
            finalize_report(data, status, notes)
            publish_report(output, data, status)
            print(
                f"[metrics] {status['status']}: {output / 'report.html'}",
                flush=True,
            )
    finally:
        stop_process(collector)
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)
    return (
        128 + received_signal
        if received_signal
        else (128 - benchmark_rc if benchmark_rc < 0 else benchmark_rc)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prefill", action="append", required=True)
    parser.add_argument("--decode", action="append", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    if not args.command:
        parser.error("A benchmark command is required after --")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
