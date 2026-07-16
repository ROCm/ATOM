#!/usr/bin/env python3
"""Benchmark model catalog: the single source of truth for *what* to benchmark.

`models.json` is authored in a structured form:

    {
      "default_scenarios": [
        {"isl": 1024, "osl": 1024,
         "concurrency": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
         "random_range_ratio": 0.8},
        ...
      ],
      "models": [
        {
          "display": "DeepSeek-V4-Pro",
          "path": "deepseek-ai/DeepSeek-V4-Pro",
          "prefix": "deepseek-v4-pro",
          "runner": "atom-mi355-8gpu.predownload",
          "env_vars": "AITER_BF16_FP8_MOE_BOUND=0\nATOM_MOE_GU_ITLV=1",
          "config": {"tp": 8, "kv_cache_dtype": "fp8",
                     "extra_args": "--hf-overrides '...'"},  // shared by all variants
          "variants": [
            {"label": "", "suffix": "", "conc_max": 256},
            {"label": "MTP3", "suffix": "-mtp3",
             "extra_args": "--method mtp --num-speculative-tokens 3",
             "bench_args": "--use-chat-template", "conc_min": 4, "conc_max": 256},
            {"label": "DPA", "suffix": "-dpa",
             "extra_args": "--enable-dp-attention",
             "conc_min": 64, "conc_max": 1024}
          ]
        },
        ...
      ]
    }

A *variant* (base / MTP / DP-attention / ...) is a dimension of the same model,
not a duplicated top-level entry. Each variant runs a set of *scenarios*
(isl/osl + concurrency list). Scenarios resolve in this order:

    variant.scenarios  ->  model.scenarios  ->  catalog.default_scenarios

and are then filtered by the variant's `conc_min`/`conc_max` band (the
declarative replacement for the old workflow `exclude` block).

Three public entry points keep every consumer in sync:

- `load_variants(path)`  -> flat per-variant dicts (server args, suffix, ...).
  Used by the dashboard display-name map and regression rerun.
- `build_cells(path, ...)` -> fully-expanded benchmark cells (variant x scenario
  x concurrency). Each cell self-describes one server+benchmark run.
- `build_cell_configs(path, ...)` -> cells regrouped by (variant x scenario) into
  the first-level matrix configs the GPU `benchmark` job iterates; each config
  carries a concurrency list the reusable template fans out over.
- `validate_dispatch_inputs(path, keys)` -> assert the workflow_dispatch model
  checkboxes stay in sync with the catalog prefixes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Concurrency band defaults when a variant does not constrain itself. conc_max
# mirrors the legacy workflow rule "cap others at 256" (DP-attention raises it).
DEFAULT_CONC_MAX = 256
DEFAULT_CONC_MIN = 0
DEFAULT_RATIO = 0.8


def _load_catalog(path: str | Path) -> dict[str, Any]:
    """Read models.json, tolerating both the structured object and a bare list."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):  # backwards-compat: legacy flat array
        return {"default_scenarios": [], "models": data}
    return data


def build_args(config: dict[str, Any], variant: dict[str, Any]) -> str:
    """Compose the server CLI arg string from the catalog.

    Only the common basics are structured fields; everything model- or
    variant-specific (MTP, DP-attention, sparse-attention overrides, memory
    utilization, ...) is passed verbatim via `extra_args`. Fixed order:

        --kv_cache_dtype <dtype> [-tp <n>] [--trust-remote-code]
        [<config.extra_args>] [<variant.extra_args>]
    """
    parts: list[str] = [f"--kv_cache_dtype {config.get('kv_cache_dtype', 'fp8')}"]
    if config.get("tp") is not None:
        parts.append(f"-tp {config['tp']}")
    if config.get("trust_remote_code"):
        parts.append("--trust-remote-code")
    if config.get("extra_args"):
        parts.append(config["extra_args"])
    if variant.get("extra_args"):
        parts.append(variant["extra_args"])

    return " ".join(parts)


def build_env_vars(model: dict[str, Any], variant: dict[str, Any]) -> str:
    """Compose model- and variant-level environment variables."""
    parts = [p for p in (model.get("env_vars", ""), variant.get("env_vars", "")) if p]
    return "\n".join(parts)


def _iter_variants(catalog: dict[str, Any]):
    """Yield (model, variant) pairs, defaulting to a single base variant."""
    for model in catalog["models"]:
        for variant in model.get("variants") or [{"label": "", "suffix": ""}]:
            yield model, variant


def _variant_record(model: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    """Flatten one (model, variant) into the per-variant dict consumers expect."""
    label = variant.get("label", "")
    return {
        "display": model["display"] + (f" {label}" if label else ""),
        "path": model["path"],
        "prefix": model["prefix"],
        "args": build_args(model.get("config", {}), variant),
        "bench_args": variant.get("bench_args", ""),
        "suffix": variant.get("suffix", ""),
        "runner": model["runner"],
        "env_vars": build_env_vars(model, variant),
        "conc_min": variant.get("conc_min", DEFAULT_CONC_MIN),
        "conc_max": variant.get("conc_max", DEFAULT_CONC_MAX),
    }


def load_variants(path: str | Path) -> list[dict[str, Any]]:
    """Return the flat per-variant list (server args, suffix, conc band, ...)."""
    catalog = _load_catalog(path)
    return [_variant_record(m, v) for m, v in _iter_variants(catalog)]


def _resolve_scenarios(
    model: dict[str, Any],
    variant: dict[str, Any],
    default_scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pick the scenario list for a variant and filter it by its conc band."""
    scenarios = variant.get("scenarios") or model.get("scenarios") or default_scenarios
    cmin = variant.get("conc_min", DEFAULT_CONC_MIN)
    cmax = variant.get("conc_max", DEFAULT_CONC_MAX)
    resolved: list[dict[str, Any]] = []
    for sc in scenarios:
        concs = [c for c in sc["concurrency"] if cmin <= c <= cmax]
        if concs:
            resolved.append({**sc, "concurrency": concs})
    return resolved


def _scenarios_from_param_lists(
    param_lists: str, conc_min: int, conc_max: int
) -> list[dict[str, Any]]:
    """Parse a workflow_dispatch `param_lists` string into scenario dicts.

    Format: "isl,osl,conc,ratio" sets separated by ';'. The concurrency band of
    the variant still applies (a set whose conc is out of band is dropped),
    matching the legacy behaviour where `exclude` pruned dispatch runs too.
    """
    resolved: list[dict[str, Any]] = []
    for chunk in param_lists.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        isl, osl, conc, ratio = (p.strip() for p in chunk.split(","))
        conc_i = int(conc)
        if not (conc_min <= conc_i <= conc_max):
            continue
        resolved.append(
            {
                "isl": int(isl),
                "osl": int(osl),
                "concurrency": [conc_i],
                "random_range_ratio": float(ratio),
            }
        )
    return resolved


def _fmt_ratio(ratio: Any) -> str:
    """Render a ratio the way the legacy RESULT_FILENAME did (0.8 -> '0.8')."""
    f = float(ratio)
    return str(int(f)) if f == int(f) else str(f)


def build_cells(
    path: str | Path,
    param_lists: str | None = None,
    model_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Expand the catalog into fully-resolved benchmark cells.

    Each cell self-describes one run:
        display, prefix, suffix, model_path, server_args, bench_args, env_vars,
        runner, isl, osl, conc, ratio, result_filename

    `param_lists` (workflow_dispatch) overrides the catalog scenarios with an
    explicit grid; otherwise per-variant/default scenarios are used. `model_filter`
    keeps only models whose prefix is in the set (None = all).
    """
    catalog = _load_catalog(path)
    default_scenarios = catalog.get("default_scenarios", [])
    cells: list[dict[str, Any]] = []
    for model, variant in _iter_variants(catalog):
        if model_filter is not None and model["prefix"] not in model_filter:
            continue
        rec = _variant_record(model, variant)
        if param_lists:
            scenarios = _scenarios_from_param_lists(
                param_lists, rec["conc_min"], rec["conc_max"]
            )
        else:
            scenarios = _resolve_scenarios(model, variant, default_scenarios)
        for sc in scenarios:
            ratio = sc.get("random_range_ratio", DEFAULT_RATIO)
            ratio_str = _fmt_ratio(ratio)
            for conc in sc["concurrency"]:
                cells.append(
                    {
                        "display": rec["display"],
                        "prefix": rec["prefix"],
                        "suffix": rec["suffix"],
                        "model_path": rec["path"],
                        "server_args": rec["args"],
                        "bench_args": rec["bench_args"],
                        "env_vars": rec["env_vars"],
                        "runner": rec["runner"],
                        "isl": sc["isl"],
                        "osl": sc["osl"],
                        "conc": conc,
                        "ratio": ratio,
                        "result_filename": (
                            f"{rec['prefix']}{rec['suffix']}-"
                            f"{sc['isl']}-{sc['osl']}-{conc}-{ratio_str}"
                        ),
                    }
                )
    return cells


def scenario_tag(isl: int, osl: int) -> str:
    """Short scenario key for an (isl, osl) pair, e.g. 1024/1024 -> ``1k1k``.

    Falls back to ``<isl>_<osl>`` for lengths that are not whole 1024 multiples
    so the tag stays unambiguous.
    """

    def _fmt(n: int) -> str:
        return f"{n // 1024}k"

    if isl >= 1024 and osl >= 1024 and isl % 1024 == 0 and osl % 1024 == 0:
        return f"{_fmt(isl)}{_fmt(osl)}"
    return f"{isl}_{osl}"


def build_cell_configs(
    path: str | Path,
    param_lists: str | None = None,
    model_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Group cells into first-level matrix configs: one per (variant, scenario).

    A single GitHub Actions matrix tops out at 256 entries, which one-job-per-cell
    overflows once the catalog passes 256 cells. Instead the workflow drives a
    two-level fan-out (mirrors InferenceX run-sweep): the top ``benchmark`` job
    matrixes over these configs (model variant × scenario), and the reusable
    ``benchmark-tmpl.yml`` it calls matrixes over each config's ``concurrency``
    list. Both matrices stay far below 256 while every (cell = config × conc)
    still runs as its own parallel job.

    Each config carries the per-server-launch fields plus the scenario shape and
    a JSON-encoded ``concurrency`` list (the second-level matrix). ``result_filename``
    is rebuilt per concurrency inside the template from
    ``{prefix}{suffix}-{isl}-{osl}-{conc}-{ratio_str}`` (unchanged naming contract).
    """
    cells = build_cells(path, param_lists=param_lists, model_filter=model_filter)

    configs: dict[tuple, dict[str, Any]] = {}
    for c in cells:
        key = (
            c["prefix"],
            c["suffix"],
            c["model_path"],
            c["server_args"],
            c["env_vars"],
            c["isl"],
            c["osl"],
            c["ratio"],
        )
        cfg = configs.get(key)
        if cfg is None:
            cfg = {
                "display": c["display"],
                "prefix": c["prefix"],
                "suffix": c["suffix"],
                "model_path": c["model_path"],
                "server_args": c["server_args"],
                "bench_args": c["bench_args"],
                "env_vars": c["env_vars"],
                "runner": c["runner"],
                "isl": c["isl"],
                "osl": c["osl"],
                "ratio": c["ratio"],
                "ratio_str": _fmt_ratio(c["ratio"]),
                "scenario": scenario_tag(c["isl"], c["osl"]),
                "_conc": [],
            }
            configs[key] = cfg
        cfg["_conc"].append(c["conc"])

    out: list[dict[str, Any]] = []
    for cfg in configs.values():
        conc = sorted(cfg.pop("_conc"))
        cfg["concurrency"] = json.dumps(conc)
        out.append(cfg)
    return out


def validate_dispatch_inputs(path: str | Path, input_keys: set[str]) -> list[str]:
    """Check workflow_dispatch boolean keys stay in sync with catalog prefixes.

    Returns a list of human-readable problems (empty == in sync). Only model
    boolean toggles are expected to match prefixes; callers pass the relevant
    subset of input keys.
    """
    prefixes = {m["prefix"] for m in _load_catalog(path)["models"]}
    problems: list[str] = []
    missing = prefixes - input_keys
    extra = input_keys - prefixes
    if missing:
        problems.append(
            f"catalog prefixes with no workflow_dispatch input: {sorted(missing)}"
        )
    if extra:
        problems.append(
            f"workflow_dispatch inputs with no catalog prefix: {sorted(extra)}"
        )
    return problems


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--cells":
        # --cells <models.json> [param_lists] [comma,sep,prefixes]
        path = args[1]
        param_lists = args[2] if len(args) > 2 and args[2] else None
        mfilter = set(args[3].split(",")) if len(args) > 3 and args[3] else None
        print(json.dumps(build_cells(path, param_lists, mfilter)))
    elif args and args[0] == "--variants":
        print(json.dumps(load_variants(args[1]), indent=2, ensure_ascii=False))
    else:
        path = args[0] if args else ".github/benchmark/models.json"
        print(json.dumps(load_variants(path), indent=2, ensure_ascii=False))
