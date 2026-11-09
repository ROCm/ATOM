#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Preview the dashboard locally with published data and optional extra perf points.

Example:
    python tools/preview_benchmark_dashboard.py --refresh

Only reads public dashboard assets. Extra points stay in the preview directory;
this script does not publish data or require GitHub credentials.
"""

import argparse
import json
import math
import re
import tempfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = "https://rocm.github.io/ATOM/benchmark-dashboard/"
ASSETS = {
    name: PUBLIC + name
    for name in (
        "data.js",
        "models_map.js",
        "atomesh_summary.js",
        "atom_logo.png",
        "atom_logo_mini.png",
    )
}
ASSETS.update(
    {
        "chart.umd.min.js": "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js",
        "chartjs-plugin-datalabels.min.js": "https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-datalabels/2.2.0/chartjs-plugin-datalabels.min.js",
    }
)


def load_sweep(file, run_url=None):
    """Read ATOM AgentX records from an InferenceX agg_bmk.json artifact.

    The artifact is an array, not a dashboard-specific points wrapper. Run URLs
    are supplied separately because InferenceX attaches run context at ingestion.
    Keep the recorded P90 values; do not derive them from rounded mean latencies.
    """
    records = json.loads(file.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list):
        raise TypeError(f"{file}: expected the JSON array from agg_bmk.json")
    run_pattern = re.compile(
        r"https://github\.com/([\w.-]+)/([\w.-]+)/actions/runs/([1-9]\d*)"
        r"(?:/attempts/([1-9]\d*))?/?$"
    )

    def positive(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value if math.isfinite(value) and value > 0 else None

    def metric_value(section, *keys):
        for key in keys:
            if section is None:
                return None
            if not isinstance(section, dict):
                raise TypeError(f"{file}: expected an object before metric {key}")
            section = section.get(key)
        return positive(section)

    points = []
    skipped = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"{file}: record {index + 1} must be an object")
        if record.get("framework") not in {"atom", "atom-disagg", "atomesh"}:
            skipped += 1
            continue
        metrics = record.get("request_metrics")
        if not isinstance(metrics, dict):
            raise TypeError(f"{file}: ATOM record {index + 1} lacks request_metrics")
        url = record.get("run_url") or run_url or ""
        match = run_pattern.fullmatch(url)
        if not match:
            raise ValueError(
                "Sweep results need their original GitHub Actions run URL; "
                "supply --run-url https://github.com/OWNER/REPO/actions/runs/RUN_ID"
            )
        owner, repo, run_id, url_attempt = match.groups()
        supplied = run_pattern.fullmatch(run_url) if run_url else None
        if run_url and (not supplied or supplied.groups()[:3] != match.groups()[:3]):
            raise ValueError("--run-url conflicts with the sweep record's run_url")
        attempts = [
            value
            for value in (
                record.get("run_attempt"),
                url_attempt,
                supplied[4] if supplied else None,
            )
            if value is not None
        ]
        if any(not str(value).isdigit() or int(value) < 1 for value in attempts):
            raise ValueError(f"Invalid sweep run_attempt: {attempts!r}")
        if len({int(value) for value in attempts}) > 1:
            raise ValueError("Run URL attempt conflicts with sweep run_attempt")
        attempt = int(attempts[0]) if attempts else None
        if attempt:
            url = f"https://github.com/{owner}/{repo}/actions/runs/{run_id}/attempts/{attempt}"
        model = record.get("model")
        hardware = record.get("hw")
        concurrency = positive(record.get("conc"))
        if not isinstance(model, str) or not isinstance(hardware, str):
            raise TypeError(f"{file}: model and hw must be strings")
        if not model or not hardware or concurrency is None:
            raise ValueError(f"{file}: record {index + 1} needs model, hw and conc")
        # Only ATOM results enter the ATOM series. Mixed-engine sweeps are common.
        hardware_match = re.search(r"\b(mi\d+x?)\b", hardware, re.IGNORECASE)
        hardware = hardware_match[1].lower() if hardware_match else hardware
        dataset = record.get("dataset") or {}
        if not isinstance(dataset, dict):
            raise TypeError(f"{file}: dataset must be an object")
        prefill = positive(record.get("prefill_num_workers"))
        decode = positive(record.get("decode_num_workers"))
        points.append(
            {
                "source": "ATOMesh",
                "source_label": f"Sweep · {repo} #{run_id}"
                + (f" / attempt {attempt}" if attempt else ""),
                "source_url": url,
                "source_status": "Imported sweep",
                "model": model,
                "hardware": hardware,
                "precision": record.get("precision", ""),
                "benchmark_kind": "aiperf_agentic",
                "scenario": record.get("scenario_type", "agentic-coding"),
                "public_dataset": dataset.get("loader")
                or dataset.get("hf_dataset_name", ""),
                "concurrency": concurrency,
                "total_gpu": positive(record.get("num_gpus")),
                "prefill_workers": prefill,
                "decode_workers": decode,
                "prefill_tp": record.get("prefill_tp"),
                "decode_tp": record.get("decode_tp"),
                "prefill_dcp": record.get("prefill_dcp_size"),
                "decode_dcp": record.get("decode_dcp_size"),
                "num_prefill_gpu": record.get("num_prefill_gpu"),
                "num_decode_gpu": record.get("num_decode_gpu"),
                "config_label": record.get("recipe_fingerprint")
                or record.get("config_label")
                or f"sweep-point-{index + 1}",
                "speculative_method": record.get("spec_algorithm", ""),
                "image": record.get("image", ""),
                "interactivity_p90_itl": metric_value(
                    metrics, "latency", "intvty", "p90"
                ),
                "interactivity": metric_value(
                    metrics, "latency", "e2e_norm_intvty", "p90"
                ),
                "interactivity_method": "p90_e2e_normalized",
                "total_tput": metric_value(
                    metrics, "throughput", "total", "tokens_per_second"
                ),
                "input_tput": metric_value(
                    metrics, "throughput", "input", "tokens_per_second"
                ),
                "output_tput": metric_value(
                    metrics, "throughput", "output", "tokens_per_second"
                ),
                "tput_per_gpu": metric_value(
                    metrics, "throughput", "per_gpu", "total_tput_tps"
                ),
                "run_url": url,
            }
        )
    if skipped:
        print(f"Skipped {skipped} non-ATOM sweep records", flush=True)
    if not points:
        raise ValueError(f"{file}: no ATOM AgentX results found")
    return points


def prepare(output, points_files, refresh=False, *, sweep_file=None, run_url=None):
    points = load_sweep(sweep_file, run_url) if sweep_file else []
    for file in points_files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("points"), list):
            raise TypeError(f"{file}: expected an object containing a points array")
        points.extend(payload["points"])

    output.mkdir(parents=True, exist_ok=True)
    for name, url in ASSETS.items():
        target = output / name
        if refresh or not target.exists():
            print(f"Downloading {name}", flush=True)
            with urlopen(url, timeout=90) as response:
                payload = response.read()
            target.write_bytes(payload)

    html = (ROOT / ".github/dashboard/index.html").read_text(encoding="utf-8")
    for name, url in ASSETS.items():
        html = html.replace(url, name)
    if points:
        # Escaping '<' keeps JSON text from terminating its script element.
        extra = json.dumps({"points": points}, ensure_ascii=True).replace(
            "<", "\\u003c"
        )
        script = "<script>window.ATOM_LOCAL_BENCHMARK = " + extra + ";</script>\n"
        html = html.replace(
            '<script src="data.js"></script>',
            script + '<script src="data.js"></script>',
        )
    (output / "index.html").write_text(html, encoding="utf-8")
    params = {
        "mode": "Disaggregated",
        "hardware": "mi355x",
        "intvty": "p90_itl",
        "explorerSource": "ATOM CI",
    }
    if points:
        params.update(
            explorerModel=points[0]["model"],
            explorerSource=points[0].get("source_label", "ATOM CI"),
            hardware=points[0].get("hardware", "mi355x"),
        )
    return urlencode(params)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, action="append", default=[])
    parser.add_argument(
        "--sweep",
        type=Path,
        help="Import ATOM AgentX records from an agg_bmk.json file",
    )
    parser.add_argument(
        "--run-url", help="Original GitHub Actions run URL for the sweep artifact"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "atom-dashboard-preview" / "site",
    )
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument(
        "--refresh", action="store_true", help="Refresh cached public assets"
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Prepare files without starting a server",
    )
    args = parser.parse_args()
    if args.run_url and not args.sweep:
        parser.error("--run-url requires --sweep")
    try:
        fragment = prepare(
            args.output,
            args.points,
            args.refresh,
            sweep_file=args.sweep,
            run_url=args.run_url,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Preview: http://127.0.0.1:{args.port}/#{fragment}", flush=True)
    print(f"Files: {args.output.resolve()}", flush=True)
    if not args.no_serve:
        handler = partial(
            SimpleHTTPRequestHandler, directory=str(args.output.resolve())
        )
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
