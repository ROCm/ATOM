#!/usr/bin/env python3
"""Generate regression re-run configs from a regression report and model registry.

Reads regression_report.json (from summarize.py) and models.json (model config
registry), selects the top-N most regressed configs, resolves their full model
config (server args, bench args), and outputs configs for CI.

Two output modes:
  --output (-o)        Flat pipe-delimited text (legacy, one config per line)
  --output-matrix      JSON array grouped by model for GitHub Actions matrix

Usage:
    # Flat output (legacy)
    python3 regression_rerun.py REPORT MODELS -o configs.txt [--top-n 3]

    # Matrix output (grouped by model for parallel CI)
    python3 regression_rerun.py REPORT MODELS --output-matrix matrix.json [--top-n 3]
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


def generate_matrix(report_path, models_path, top_n=3):
    """Generate a GitHub Actions matrix JSON grouped by model server key.

    Each matrix cell represents a unique (model_path, server_args, env_vars)
    group. Configs within the same group share a server and run sequentially.
    Different groups run in parallel as separate matrix cells.

    Returns a list of dicts, each with:
      - model_path, server_args, env_vars, runner, prefix: scalar fields
      - configs: JSON string of [{isl, osl, conc, bench_args, prefix}, ...]
    """
    configs = generate_configs(report_path, models_path, top_n)
    if not configs:
        return []

    # Group by server key (same key = same server launch)
    groups = {}
    for c in configs:
        key = (c["model_path"], c["server_args"], c["env_vars"])
        groups.setdefault(key, []).append(c)

    # Load models.json to look up runner field
    models = json.loads(Path(models_path).read_text(encoding="utf-8"))

    matrix = []
    for (model_path, server_args, env_vars), group_configs in groups.items():
        # Find runner from models.json (first matching model)
        runner = "atom-mi355-8gpu.predownload"
        for m in models:
            if m["path"] == model_path and m.get("args", "") == server_args:
                runner = m.get("runner", runner)
                break

        # Use prefix from first config (all configs in group share model)
        prefix = group_configs[0]["prefix"]

        matrix.append(
            {
                "model_path": model_path,
                "server_args": server_args,
                "env_vars": env_vars,
                "runner": runner,
                "prefix": prefix,
                "configs": json.dumps(
                    [
                        {
                            "isl": c["isl"],
                            "osl": c["osl"],
                            "conc": c["conc"],
                            "bench_args": c["bench_args"],
                            "prefix": c["prefix"],
                        }
                        for c in group_configs
                    ]
                ),
            }
        )

    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Path to regression_report.json")
    parser.add_argument("models", help="Path to models.json")
    parser.add_argument("-o", "--output", help="Flat pipe-delimited output file")
    parser.add_argument(
        "--output-matrix",
        help="GitHub Actions matrix JSON output file (grouped by model)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Max regressed configs to re-run (default: 3)",
    )
    args = parser.parse_args()

    if args.output_matrix:
        matrix = generate_matrix(args.report, args.models, args.top_n)
        output = json.dumps(matrix)
        Path(args.output_matrix).write_text(output, encoding="utf-8")
        print(
            f"Wrote {len(matrix)} matrix cells to {args.output_matrix}",
            file=sys.stderr,
        )
        return

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
