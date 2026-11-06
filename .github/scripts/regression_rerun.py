#!/usr/bin/env python3
"""Generate regression re-run configs from a regression report and model registry.

Reads regression_report.json (from summarize.py) and models.json (model config
registry), resolves each regressed config to its catalog variant (server args,
bench args), and outputs the top-N most regressed resolved configs for CI.

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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import load_variants


def _normalize_model_path(path):
    """Strip the /models/ mount that CI uses for pre-downloaded checkpoints.

    With MODEL_HOST_ROOT set, benchmark-tmpl.yml serves "/models/<path>", so a
    result's model_id is e.g. "/models/deepseek-ai/DeepSeek-V4-Pro" while the
    catalog path is "deepseek-ai/DeepSeek-V4-Pro".
    """
    return path.removeprefix("/models/").rstrip("/")


def _find_model_config(regression_entry, models):
    """Match a regression entry to one load_variants() record.

    summarize.py reports a result's benchmark_model_name, which is the catalog
    display of the variant (e.g. "DeepSeek-V4-Pro DPA DSpark"), as "model", and
    the served checkpoint (e.g. "/models/deepseek-ai/DeepSeek-V4-Pro-DSpark",
    a variant-level path) as "model_id". Results without benchmark_model_name
    are reported as "<checkpoint basename>[-mtp<N>]"; those match basename +
    suffix. Variants sharing a display are told apart by their concurrency
    band. Returns None unless exactly one record matches.
    """
    model_id = regression_entry.get("model_id", "")
    display = regression_entry.get("model", "")

    if model_id:
        model_id = _normalize_model_path(model_id)
        models = [m for m in models if _normalize_model_path(m["path"]) == model_id]

    matches = [m for m in models if m["display"] == display]
    if not matches:
        matches = [
            m
            for m in models
            if (m["path"].split("/")[-1] + m.get("suffix", "")).lower()
            == display.lower()
        ]

    conc = regression_entry.get("conc")
    if len(matches) > 1 and conc is not None:
        matches = [m for m in matches if m["conc_min"] <= conc <= m["conc_max"]]
    return matches[0] if len(matches) == 1 else None


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
    models = load_variants(models_path)

    regressions = report.get("regressions", [])
    if not regressions:
        return []

    # Sort by severity; top N is taken after resolution so an entry that has
    # no catalog config does not use up a rerun slot.
    regressions.sort(key=_regression_severity)

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
                # prefix + suffix, as in the benchmark RESULT_FILENAME, so two
                # variants of one model get distinct trace dirs and artifacts
                "prefix": model_cfg.get("prefix", "") + model_cfg.get("suffix", ""),
            }
        )

    configs = configs[:top_n]

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
    models = load_variants(models_path)

    # The cell prefix names the per-cell artifacts, so it must be unique. Each
    # group keeps the prefix of its first config (all configs in a group share
    # the server settings) unless an earlier group already has it (e.g. one
    # variant with different args per concurrency band); that group then gets
    # the first "<prefix>-<n>" that no other group uses or would use.
    cell_prefixes = []
    taken = {g[0]["prefix"] for g in groups.values()}
    for group_configs in groups.values():
        prefix = group_configs[0]["prefix"]
        if prefix in cell_prefixes:
            n = 2
            while f"{prefix}-{n}" in taken:
                n += 1
            prefix = f"{prefix}-{n}"
            taken.add(prefix)
        cell_prefixes.append(prefix)

    matrix = []
    for ((model_path, server_args, env_vars), group_configs), prefix in zip(
        groups.items(), cell_prefixes
    ):
        # Find runner from models.json (first matching model)
        runner = "linux-atom-pitt-mi350x-8"
        for m in models:
            if m["path"] == model_path and m.get("args", "") == server_args:
                runner = m.get("runner", runner)
                break

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
