#!/usr/bin/env python3
"""Generate regression re-run actions from a regression report and model registry.

Reads regression_report.json (from summarize.py) and models.json (model config
registry), then outputs a pipe-delimited action file consumed by the CI workflow.

Usage:
    python3 regression_rerun.py REPORT_JSON MODELS_JSON [-o OUTPUT] [--top-n 3]

Output format (one action per line):
    DOWNLOAD|<hf_path>
    LAUNCH|<hf_path>|<server_args>
    BENCHMARK|<hf_path>|<ISL>|<OSL>|<CONC>|<bench_args>
    STOP
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _find_model_config(regression_entry, models):
    """Match a regression entry to a models.json config.

    Matches by model_id (HF path) + variant suffix, NOT by display name,
    because display name formats differ between summarize.py output
    (e.g. "DeepSeek-R1-0528-mtp3") and models.json (e.g. "DeepSeek-R1-0528 MTP3").
    """
    model_id = regression_entry.get("model_id", "")
    display = regression_entry.get("model", "")

    # Detect variant from display name (e.g. "...-mtp3" -> suffix "-mtp3")
    variant_match = re.search(r"-(mtp\d*)$", display, re.IGNORECASE)
    variant_suffix = f"-{variant_match.group(1).lower()}" if variant_match else ""

    # Primary match: model_id (HF path) + suffix
    if model_id:
        for m in models:
            if m["path"] == model_id and m.get("suffix", "") == variant_suffix:
                return m

    # Fallback: match by path basename (for old reports without model_id)
    display_base = re.sub(r"-(mtp\d*)$", "", display, flags=re.IGNORECASE)
    for m in models:
        path_basename = m["path"].split("/")[-1]
        if path_basename == display_base and m.get("suffix", "") == variant_suffix:
            return m

    return None


def _regression_severity(r):
    """Sort key: most severe regression first (largest throughput drop)."""
    metrics = r.get("metrics", {})
    throughput_pct = metrics.get("output_throughput", {}).get("pct", 0)
    tpot_pct = metrics.get("mean_tpot_ms", {}).get("pct", 0)
    # Throughput drop is negative, TPOT increase is positive — both are bad
    return min(throughput_pct, -tpot_pct)


def generate_actions(report_path, models_path, top_n=3):
    """Generate action lines from regression report and model registry."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    models = json.loads(Path(models_path).read_text(encoding="utf-8"))

    regressions = report.get("regressions", [])
    if not regressions:
        return []

    # Sort by severity and take top N
    regressions.sort(key=_regression_severity)
    regressions = regressions[:top_n]

    # Group by model config (same server instance can run multiple benchmarks)
    groups = {}
    for r in regressions:
        config = _find_model_config(r, models)
        if config is None:
            print(
                f"WARNING: No model config found for {r.get('model', '?')}, skipping",
                file=sys.stderr,
            )
            continue

        # Group key: path + suffix (determines unique server instance)
        key = (config["path"], config.get("suffix", ""))
        if key not in groups:
            groups[key] = {
                "config": config,
                "benchmarks": [],
            }
        groups[key]["benchmarks"].append(r)

    # Generate actions
    actions = []
    seen_downloads = set()
    for (path, _suffix), group in groups.items():
        config = group["config"]

        # DOWNLOAD (deduplicate by HF path)
        if path not in seen_downloads:
            actions.append(f"DOWNLOAD|{path}")
            seen_downloads.add(path)

        # LAUNCH with profiler args appended
        server_args = config.get("args", "")
        profiler_args = "--torch-profiler-dir /app/trace --mark-trace"
        full_args = f"{server_args} {profiler_args}".strip()
        actions.append(f"LAUNCH|{path}|{full_args}")

        # BENCHMARK for each regressed config
        bench_args = config.get("bench_args", "")
        for r in group["benchmarks"]:
            actions.append(
                f"BENCHMARK|{path}|{r['isl']}|{r['osl']}|{r['conc']}|{bench_args}"
            )

        actions.append("STOP")

    return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Path to regression_report.json")
    parser.add_argument("models", help="Path to models.json")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Max regressed configs to re-run (default: 3)",
    )
    args = parser.parse_args()

    actions = generate_actions(args.report, args.models, args.top_n)

    if not actions:
        print("No regressions to re-run", file=sys.stderr)
        return

    output = "\n".join(actions) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Wrote {len(actions)} actions to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
