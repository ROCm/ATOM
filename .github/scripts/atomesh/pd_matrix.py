#!/usr/bin/env python3
"""Expand ATOMesh real P/D benchmark YAML into workflow matrix cells."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


def deep_merge(*items: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        for key, value in (item or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
    return merged


def normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").lower()


def format_display_topology(
    topology: str,
    suite_cfg: dict[str, Any],
    prefill_cfg: dict[str, Any],
    decode_cfg: dict[str, Any],
) -> str:
    if "display_topology" in suite_cfg:
        return str(suite_cfg["display_topology"])

    parts = [
        part.upper()
        for part in re.split(r"[_-]+", topology)
        if part and not re.fullmatch(r"tp\d*", part, re.IGNORECASE)
    ]
    prefill_tp = prefill_cfg.get("tp")
    decode_tp = decode_cfg.get("tp")
    if prefill_tp == decode_tp and prefill_tp is not None:
        parts.append(f"TP{prefill_tp}")

    return "-".join(parts)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def role_env(
    defaults: dict[str, Any],
    backend_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    suite_cfg: dict[str, Any],
    role: str,
) -> dict[str, str]:
    env = deep_merge(
        defaults.get("env", {}).get("common", {}),
        backend_cfg.get("env", {}).get("common", {}),
        model_cfg.get("env", {}).get("common", {}),
        suite_cfg.get("env", {}).get("common", {}),
        backend_cfg.get("env", {}).get(role, {}),
        model_cfg.get("env", {}).get(role, {}),
        suite_cfg.get("env", {}).get(role, {}),
    )
    return {str(key): str(value) for key, value in env.items()}


def build_cell(
    *,
    cfg: dict[str, Any],
    model_name: str,
    model_cfg: dict[str, Any],
    suite_name: str,
    suite_cfg: dict[str, Any],
    override_image: str | None,
) -> dict[str, Any]:
    defaults = cfg.get("defaults", {})
    backend_name = str(model_cfg.get("backend", "atom"))
    backend_cfg = cfg.get("backends", {}).get(backend_name)
    if backend_cfg is None:
        raise ValueError(
            f"Model {model_name} references unknown backend {backend_name}"
        )
    if backend_name != "atom":
        raise ValueError(
            f"Only atom backend is currently supported, got {backend_name}"
        )

    nodes = [str(node) for node in normalize_list(suite_cfg.get("nodes"))]
    if len(nodes) < 2:
        raise ValueError(
            f"{suite_cfg.get('name', model_name)} needs at least two nodes"
        )

    prefill_cfg = deep_merge(
        backend_cfg.get("service", {}).get("prefill", {}),
        suite_cfg.get("prefill", {}),
    )
    decode_cfg = deep_merge(
        backend_cfg.get("service", {}).get("decode", {}),
        suite_cfg.get("decode", {}),
    )
    router_cfg = deep_merge(
        backend_cfg.get("service", {}).get("router", {}),
        suite_cfg.get("router", {}),
    )
    server_args = deep_merge(
        model_cfg.get("server", {}).get("common_args", {}),
        suite_cfg.get("server", {}).get("common_args", {}),
    )
    runner_cfg = deep_merge(defaults.get("runner", {}), suite_cfg.get("runner", {}))
    benchmark_cfg = deep_merge(
        defaults.get("benchmark", {}),
        suite_cfg.get("benchmark", {}),
    )
    accuracy_cfg = deep_merge(
        model_cfg.get("accuracy", {}), suite_cfg.get("accuracy", {})
    )

    concurrency = [int(value) for value in normalize_list(suite_cfg.get("concurrency"))]
    isl = [int(value) for value in normalize_list(suite_cfg.get("isl"))]
    if not concurrency or not isl:
        raise ValueError(
            f"{suite_cfg.get('name', model_name)} must define isl and concurrency"
        )

    topology = str(suite_cfg["topology"])
    display_topology = format_display_topology(
        topology, suite_cfg, prefill_cfg, decode_cfg
    )
    cell_id = slug(f"{model_name}-{suite_cfg.get('name', topology)}-{suite_name}")
    image = override_image or str(backend_cfg.get("image"))
    return {
        "id": cell_id,
        "suite": suite_name,
        "name": str(suite_cfg.get("name", cell_id)),
        "model": model_name,
        "backend": backend_name,
        "image": image,
        "model_path": str(model_cfg["model_path"]),
        "precision": str(model_cfg.get("precision", "")),
        "topology": topology,
        "display_topology": display_topology,
        "nodes": nodes,
        "num_nodes": len(nodes),
        "isl": isl,
        "osl": int(suite_cfg["osl"]),
        "concurrency": concurrency,
        "concurrency_x": "x".join(str(value) for value in concurrency),
        "random_range_ratio": str(benchmark_cfg.get("random_range_ratio", 0.8)),
        "request_rate": str(benchmark_cfg.get("request_rate", "inf")),
        "num_prompts_multiplier": int(benchmark_cfg.get("num_prompts_multiplier", 10)),
        "wait_server_timeout": int(benchmark_cfg.get("wait_server_timeout", 2500)),
        "wait_router_timeout": int(benchmark_cfg.get("wait_router_timeout", 300)),
        "runner": runner_cfg,
        "service": {
            "prefill": prefill_cfg,
            "decode": decode_cfg,
            "router": router_cfg,
        },
        "server_args": server_args,
        "env": {
            "common": role_env(defaults, backend_cfg, model_cfg, suite_cfg, "common"),
            "prefill": role_env(defaults, backend_cfg, model_cfg, suite_cfg, "prefill"),
            "decode": role_env(defaults, backend_cfg, model_cfg, suite_cfg, "decode"),
            "router": role_env(defaults, backend_cfg, model_cfg, suite_cfg, "router"),
        },
        "run_eval": bool(suite_cfg.get("run_eval", False)),
        "accuracy": {
            "task": str(accuracy_cfg.get("task", "gsm8k")),
            "fewshot": int(accuracy_cfg.get("fewshot", 3)),
            "limit": suite_cfg.get("eval_limit", accuracy_cfg.get("limit")),
        },
    }


def build_cells(
    cfg: dict[str, Any],
    *,
    suite: str,
    model_filter: str | None,
    override_image: str | None,
) -> list[dict[str, Any]]:
    cells = []
    for model_name, model_cfg in (cfg.get("models") or {}).items():
        if model_filter and model_name != model_filter:
            continue
        suites = model_cfg.get("suites", {})
        for suite_cfg in normalize_list(suites.get(suite)):
            cells.append(
                build_cell(
                    cfg=cfg,
                    model_name=str(model_name),
                    model_cfg=model_cfg,
                    suite_name=suite,
                    suite_cfg=suite_cfg,
                    override_image=override_image,
                )
            )
    return cells


def write_github_outputs(cells: list[dict[str, Any]]) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if not output:
        return
    matrix = {"include": cells}
    with Path(output).open("a", encoding="utf-8") as handle:
        handle.write(f"matrix_json={json.dumps(matrix, separators=(',', ':'))}\n")
        handle.write(f"has_matrix={'true' if cells else 'false'}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=".github/benchmark/models_atomesh.yaml")
    parser.add_argument("--suite", default=os.environ.get("SUITE", "smoke"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME") or None)
    parser.add_argument("--image", default=os.environ.get("ATOMESH_IMAGE") or None)
    parser.add_argument("--output", help="Optional output JSON path")
    parser.add_argument("--github-output", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    cells = build_cells(
        config,
        suite=args.suite,
        model_filter=args.model,
        override_image=args.image,
    )
    print(f"Generated {len(cells)} ATOMesh benchmark cell(s) for suite={args.suite}")
    for cell in cells:
        print(
            f"  {cell['id']}: {cell['model']} {cell['display_topology']} "
            f"nodes={','.join(cell['nodes'])} isl={cell['isl']} osl={cell['osl']} "
            f"conc={cell['concurrency']}"
        )

    if args.output:
        Path(args.output).write_text(
            json.dumps({"include": cells}, indent=2), encoding="utf-8"
        )
    if args.github_output:
        write_github_outputs(cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
