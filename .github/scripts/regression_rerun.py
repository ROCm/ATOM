#!/usr/bin/env python3
"""Generate regression re-run configs from a regression report and model registry.

Reads regression_report.json (from summarize.py) and models.json (model config
registry), selects the top-N most regressed configs, resolves their full model
config (server args, bench args), and outputs a flat config list.

The output is consumed by the CI workflow's "Run configs with profiler" step,
which uses the same atom_test.sh launch/benchmark pattern as the benchmark job.

Usage:
    python3 regression_rerun.py REPORT_JSON MODELS_JSON [-o OUTPUT] [--top-n 3]

Output format (pipe-delimited, one config per line, sorted by model+args for grouping):
    <model_path>|<server_args>|<bench_args>|<isl>|<osl>|<conc>|<env_vars>|<prefix>
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


def generate_configs(report_path, models_path, top_n=3):
    """Generate flat config list from regression report and model registry."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    models = json.loads(Path(models_path).read_text(encoding="utf-8"))

    regressions = report.get("regressions", [])
    if not regressions:
        return []

    # Sort by severity and take top N
    regressions.sort(key=_regression_severity)
    regressions = regressions[:top_n]

    configs = []
    for r in regressions:
        model_cfg = _find_model_config(r, models)
        if model_cfg is None:
            print(
                f"WARNING: No model config found for {r.get('model', '?')}, skipping",
                file=sys.stderr,
            )
            continue

        isl, osl, conc = r.get("isl"), r.get("osl"), r.get("conc")
        if None in (isl, osl, conc):
            print(
                f"WARNING: skipping {r.get('model', '?')} — missing isl/osl/conc",
                file=sys.stderr,
            )
            continue

        configs.append(
            {
                "model_path": model_cfg["path"],
                "server_args": model_cfg.get("args", ""),
                "bench_args": model_cfg.get("bench_args", ""),
                "isl": isl,
                "osl": osl,
                "conc": conc,
                "env_vars": model_cfg.get("env_vars", ""),
                "prefix": model_cfg.get("prefix", ""),
            }
        )

    # Sort by model_path + server_args so same-server configs are adjacent
    # (allows the YAML loop to group server launches)
    configs.sort(key=lambda c: (c["model_path"], c["server_args"]))
    return configs


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

    configs = generate_configs(args.report, args.models, args.top_n)

    if not configs:
        print("No regressions to re-run", file=sys.stderr)
        return

    lines = []
    for c in configs:
        lines.append(
            f"{c['model_path']}|{c['server_args']}|{c['bench_args']}"
            f"|{c['isl']}|{c['osl']}|{c['conc']}"
            f"|{c['env_vars']}|{c['prefix']}"
        )

    output = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Wrote {len(lines)} configs to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output)


if __name__ == "__main__":
    main()
