"""CI observability: scrape config, GPU exporter, VM export and offline report."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import socket
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families


def fetch(url, params=None):
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def samples(text):
    return [
        s for family in text_string_to_metric_families(text) for s in family.samples
    ]


def prepare(args):
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        vm_port = sock.getsockname()[1]
    targets = []
    for value in args.target:
        role, url = value.split("=", 1)
        parsed = urllib.parse.urlsplit(url)
        if (
            role not in ("prefill", "decode", "router", "gpu")
            or parsed.scheme != "http"
            or not parsed.netloc
        ):
            raise ValueError(f"Invalid target: {value}")
        target = {"role": role, "url": url, "instance": parsed.netloc}
        if target not in targets:
            targets.append(target)
    labels = {
        "run_id": args.run_id,
        "model": args.model,
        "topo": args.topology,
        "case": args.case,
        "phase": args.phase,
    }
    config = {
        "global": {
            "scrape_interval": "1s",
            "scrape_timeout": "900ms",
            "external_labels": labels,
        },
        "scrape_configs": [
            {
                "job_name": f"{target['role']}-{i}",
                # Existing model/role labels win; stored target role agrees with new histograms.
                "honor_labels": True,
                "metrics_path": urllib.parse.urlsplit(target["url"]).path,
                "static_configs": [
                    {
                        "targets": [target["instance"]],
                        "labels": {"role": target["role"]},
                    }
                ],
            }
            for i, target in enumerate(targets)
        ],
    }
    for job, target in zip(list(config["scrape_configs"]), targets):
        if target["role"] in ("prefill", "decode"):
            job["metric_relabel_configs"] = [
                {
                    "source_labels": ["__name__"],
                    "regex": "atom:(scheduler|scheduling|executed)_.*",
                    "action": "drop",
                }
            ]
            config["scrape_configs"].append(
                {
                    "job_name": job["job_name"] + "-scheduling",
                    "honor_labels": True,
                    "scrape_interval": "100ms",
                    "scrape_timeout": "90ms",
                    "metrics_path": "/metrics/scheduling",
                    "static_configs": job["static_configs"],
                }
            )
    (directory / "scrape.json").write_text(json.dumps(config, indent=2))
    (directory / "run.json").write_text(
        json.dumps(
            {
                **labels,
                "vm_port": vm_port,
                "targets": targets,
                "start": time.time(),
                "events_enabled": args.events == "1",
            },
            indent=2,
        )
    )


def preflight(args):
    meta = json.loads((args.directory / "run.json").read_text())
    errors = []
    for target in meta["targets"]:
        try:
            body = fetch(target["url"]).decode()
            names = {s.name for s in samples(body)}
            if target["role"] in ("prefill", "decode"):
                required = {
                    "atom:requests_running",
                    "atom:ttft_seconds_count",
                    "atom:e2e_latency_seconds_count",
                }
                if not required <= names:
                    raise ValueError(
                        f"Missing metrics: {sorted(required - names)}; check CI checkout is served"
                    )
                fast_body = fetch(target["url"] + "/scheduling").decode()
                fast_names = {sample.name for sample in samples(fast_body)}
                if (
                    not {
                        "atom:scheduler_duration_seconds_count",
                        "atom:executed_batch_size_count",
                    }
                    <= fast_names
                ):
                    raise ValueError("Missing lightweight scheduler metrics")
            elif target["role"] == "gpu" and "atom_ci_gpu_vram_used_bytes" not in names:
                raise ValueError("No AMD GPU sysfs metrics found")
            elif target["role"] == "router" and not body.strip():
                raise ValueError("Empty MESH metrics response")
            (
                args.directory
                / f"before-{target['role']}-{target['instance'].replace(':', '_')}.prom"
            ).write_text(body)
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{target['role']} {target['instance']}: {exc}")
    (args.directory / "preflight.json").write_text(
        json.dumps({"errors": errors}, indent=2)
    )
    if errors:
        raise RuntimeError("; ".join(errors))


def queries(selector, quantile="0.95"):
    result = {}
    for role in ("prefill", "decode"):
        match = selector + f',role="{role}"'
        for metric in ("ttft", "tpot", "e2e_latency", "output_chunk_interval"):
            result[f"{role}/{metric}"] = (
                f"histogram_quantile({quantile}, sum by (le) (rate(atom:{metric}_seconds_bucket{{{match}}}[30s])))"
            )
        for metric in (
            "requests_running",
            "requests_waiting",
            "requests_parked_kv_load",
            "kv_cache_usage_ratio",
            "prefix_cache_hit_ratio",
            "lmcache_loads_pending",
            "lmcache_saves_pending",
        ):
            result[f"{role}/{metric}"] = f"max(atom:{metric}{{{match}}})"
        for metric in (
            "prompt_tokens",
            "generation_tokens",
            "preemptions",
            "lmcache_loaded_tokens",
            "lmcache_saved_tokens",
        ):
            result[f"{role}/{metric}/s"] = (
                f"sum(rate(atom:{metric}_total{{{match}}}[30s]))"
            )
    for role in ("prefill", "decode"):
        match = selector + f',role="{role}"'
        for metric in ("duration", "queue"):
            result[f"{role}/scheduler_{metric}"] = (
                f"histogram_quantile({quantile}, sum by (le, rank) (rate(atom:scheduler_{metric}_seconds_bucket{{{match}}}[1s])))"
            )
        for metric in ("batch_size", "query_tokens", "context_tokens"):
            result[f"{role}/executed_{metric}"] = (
                f"histogram_quantile({quantile}, sum by (le, rank, stage) (rate(atom:executed_{metric}_bucket{{{match}}}[1s])))"
            )
        for metric in (
            "batch_size",
            "query_tokens",
            "query_tokens_min",
            "query_tokens_max",
            "query_tokens_mean",
            "context_tokens_min",
            "context_tokens_max",
            "context_tokens_mean",
        ):
            result[f"{role}/executed_{metric}_last"] = (
                f"atom:executed_{metric}_last{{{match}}}"
            )
        result[f"{role}/scheduling_last_execution_age"] = (
            f"time() - atom:scheduling_last_execution_timestamp_seconds{{{match}}}"
        )
        result[f"{role}/scheduling_snapshot_age"] = (
            f"time() - atom:scheduling_snapshot_timestamp_seconds{{{match}}}"
        )
        for metric in (
            "requests_waiting",
            "requests_running",
            "requests_partial_prefill",
            "requests_parked_kv_load",
        ):
            result[f"{role}/scheduling_{metric}"] = (
                f"atom:scheduling_{metric}{{{match}}}"
            )
        result[f"{role}/mtp_acceptance_rate"] = f"atom:mtp_acceptance_rate{{{match}}}"
    result["collection/scrape_duration_seconds"] = (
        "scrape_duration_seconds{" + selector + "}"
    )
    for metric in ("ttft", "tpot", "request_duration"):
        result[f"router/{metric}"] = (
            f"histogram_quantile({quantile}, sum by (le) (rate(mesh_router_{metric}_seconds_bucket{{{selector}}}[30s])))"
        )
    for metric in (
        "busy_ratio",
        "vram_used_bytes",
        "temperature_celsius",
        "power_watts",
    ):
        result[f"gpu/{metric}"] = f"atom_ci_gpu_{metric}{{{selector}}}"
    return result


def clean_values(value):
    if isinstance(value, dict):
        return {k: clean_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_values(v) for v in value]
    if value in ("NaN", "+Inf", "-Inf"):
        return None
    return value


def finish(args):
    directory = args.directory
    meta = json.loads((directory / "run.json").read_text())
    meta["end"] = time.time()
    (directory / "run.json").write_text(json.dumps(meta, indent=2))
    base = f"http://127.0.0.1:{meta['vm_port']}"
    selector = f'run_id={json.dumps(meta["run_id"])},case={json.dumps(meta["case"])},phase={json.dumps(meta["phase"])}'
    report = {"run": meta, "errors": [], "warnings": [], "series": {}, "scrapes": []}
    for target in meta["targets"]:
        try:
            t0 = time.monotonic()
            body = fetch(target["url"]).decode()
            elapsed = time.monotonic() - t0
            snapshot = samples(body)
            (
                directory
                / f"after-{target['role']}-{target['instance'].replace(':', '_')}.prom"
            ).write_text(body)
            if target["role"] in ("prefill", "decode"):
                counts = [
                    s.value
                    for s in snapshot
                    if s.name == "atom:e2e_latency_seconds_count"
                    and s.labels.get("role") == target["role"]
                ]
                if not counts or sum(counts) <= 0:
                    report["errors"].append(
                        f"No completed sequence histogram on {target['instance']} ({target['role']})"
                    )
                fast_samples = samples(fetch(target["url"] + "/scheduling").decode())
                if not any(
                    sample.name == "atom:executed_batch_size_count"
                    and sample.labels.get("stage") == target["role"]
                    and sample.value > 0
                    for sample in fast_samples
                ):
                    report["errors"].append(
                        f"No executed batch histogram for {target['role']} on {target['instance']}"
                    )
                for name in (
                    "atom:request_events_dropped_total",
                    "atom:request_events_write_errors_total",
                ):
                    if any(s.name == name and s.value > 0 for s in snapshot):
                        report["errors"].append(f"{target['instance']}: {name} > 0")
            report["scrapes"].append(
                {**target, "duration_seconds": elapsed, "sample_count": len(snapshot)}
            )
        except (OSError, ValueError, KeyError) as exc:
            report["errors"].append(f"Final scrape {target['url']}: {exc}")
    try:
        # Native VM JSON-line export can be reimported after node reclamation.
        with (
            urllib.request.urlopen(
                base
                + "/api/v1/export?"
                + urllib.parse.urlencode({"match[]": "{" + selector + "}"}),
                timeout=120,
            ) as source,
            gzip.open(directory / "timeseries.jsonl.gz", "wb") as dest,
        ):
            shutil.copyfileobj(source, dest)
        availability = json.loads(
            fetch(
                base + "/api/v1/query",
                {"query": f"avg_over_time(up{{{selector}}}[24h])"},
            )
        )
        up = availability.get("data", {}).get("result", [])
        report["availability"] = up
        for target in meta["targets"]:
            found = [x for x in up if x["metric"].get("instance") == target["instance"]]
            if not found or any(float(x["value"][1]) < 0.95 for x in found):
                report["errors"].append(
                    f"Scrape availability below 95%: {target['instance']}"
                )
        for q in ("0.5", "0.9", "0.95", "0.99"):
            for name, query in queries(selector, q).items():
                if q != "0.95" and not query.startswith("histogram_quantile"):
                    continue
                data = json.loads(
                    fetch(
                        base + "/api/v1/query_range",
                        {
                            "query": query,
                            "start": meta["start"],
                            "end": meta["end"],
                            "step": (
                                0.1
                                if any(
                                    tag in name
                                    for tag in (
                                        "scheduler_",
                                        "scheduling_",
                                        "executed_",
                                    )
                                )
                                else 1
                            ),
                        },
                    )
                )
                if data.get("status") != "success":
                    raise ValueError(data)
                report["series"][f"{name}@{q}"] = clean_values(data["data"]["result"])
        for role in ("prefill", "decode"):
            data = report["series"].get(f"{role}/ttft@0.95", [])
            if not any(
                v is not None for series in data for _, v in series.get("values", [])
            ):
                report["errors"].append(f"No stored TTFT curve for {role}")
        if not report["series"].get("router/ttft@0.95"):
            report["warnings"].append(
                "This MESH HTTP backend did not export TTFT; no router TTFT is fabricated. ATOM TTFT and client AIPerf remain separate measurements."
            )
    except (OSError, ValueError, KeyError) as exc:
        report["errors"].append(f"Storage/query/export: {exc}")
    if meta.get("events_enabled"):
        try:
            report["request_events"] = summarize_events(directory / "events")
            for role in ("prefill", "decode"):
                if not report["request_events"].get(role, {}).get("completed", 0):
                    report["errors"].append(f"No completed request events for {role}")
        except (OSError, ValueError, KeyError) as exc:
            report["errors"].append(f"Request events: {exc}")
    report["status"] = "fail" if report["errors"] else "pass"
    (directory / "validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    template = Path(__file__).with_name("observability.html").read_text()
    payload = json.dumps(report, allow_nan=False).replace("<", "\\u003c")
    (directory / "index.html").write_text(
        template.replace("/*REPORT_DATA*/null", payload)
    )
    print(
        json.dumps({k: report[k] for k in ("status", "errors", "warnings")}, indent=2)
    )
    if report["errors"]:
        raise SystemExit(1)


def summarize_events(directory):
    result = {}
    values = {}
    for path in sorted(directory.glob("*.jsonl")):
        with path.open() as handle:
            for line in handle:
                if not line.endswith("\n"):
                    continue  # A live writer may be midway through its last line.
                event = json.loads(line)
                role = event["role"]
                row = result.setdefault(
                    role,
                    {"completed": 0, "output_chunks": 0, "example_request_ids": []},
                )
                if event["event"] == "output_chunk":
                    row["output_chunks"] += 1
                elif event["event"] == "request_finished":
                    row["completed"] += 1
                    if len(row["example_request_ids"]) < 20:
                        row["example_request_ids"].append(event["request_id"])
                    for name in ("ttft", "tpot", "latency"):
                        if event[name] is not None:
                            values.setdefault((role, name), []).append(event[name])
    for (role, name), samples_ in values.items():
        samples_.sort()

        def quantile(q, samples_=samples_):
            index = (len(samples_) - 1) * q
            low, high = math.floor(index), math.ceil(index)
            return samples_[low] + (samples_[high] - samples_[low]) * (index - low)

        result[role][name] = {str(q): quantile(q) for q in (0.5, 0.9, 0.95, 0.99)}
    return result


def gpu_metrics(root=Path("/sys/class/drm")):
    """Read AMD DRM/hwmon counters without loading a GPU runtime or using GPU memory."""
    rows = []
    names = {
        "gpu_busy_percent": ("busy_ratio", 0.01),
        "mem_info_vram_used": ("vram_used_bytes", 1),
        "mem_info_vram_total": ("vram_total_bytes", 1),
    }
    for card in sorted(root.glob("card[0-9]*")):
        if not card.name[4:].isdigit():
            continue
        device = card / "device"
        try:
            if (device / "vendor").read_text().strip() != "0x1002":
                continue
        except OSError:
            continue
        values = {}
        for file, (name, scale) in names.items():
            try:
                values[name] = float((device / file).read_text()) * scale
            except (OSError, ValueError):
                pass
        for hwmon in device.glob("hwmon/hwmon*"):
            for file, name, scale in (
                ("temp1_input", "temperature_celsius", 0.001),
                ("power1_average", "power_watts", 0.000001),
            ):
                try:
                    values[name] = float((hwmon / file).read_text()) * scale
                except (OSError, ValueError):
                    pass
        for name, value in values.items():
            rows.append(f'atom_ci_gpu_{name}{{gpu="{card.name}"}} {value}')
    types = [
        f"# TYPE {name} gauge" for name in sorted({row.split("{")[0] for row in rows})
    ]
    return ("\n".join(types + rows) + "\n").encode()


def gpu(args):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/metrics":
                self.send_error(404)
                return
            body = gpu_metrics()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "preflight", "finish"):
        child = commands.add_parser(name)
        child.add_argument("--directory", type=Path, required=True)
        if name == "prepare":
            for key in ("run-id", "model", "topology", "case", "phase"):
                child.add_argument("--" + key, required=True)
            child.add_argument("--target", action="append", required=True)
            child.add_argument("--events", choices=("0", "1"), default="1")
    commands.add_parser("gpu").add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
