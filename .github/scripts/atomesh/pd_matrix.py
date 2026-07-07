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

ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


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


def parse_int_list(value: Any, field_name: str) -> list[int]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        raw_values = [item for item in re.split(r"[,\s]+", value.strip()) if item]
    else:
        raw_values = normalize_list(value)
    try:
        return [int(item) for item in raw_values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a list of integers") from exc


def parse_model_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    models = {
        item.strip()
        for item in value.split(",")
        if item.strip() and item.strip().lower() != "all"
    }
    return models or None


def model_path_env_key(model_name: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").upper()
    return f"ATOMESH_MODEL_PATH_{suffix}"


def resolve_model_path(model_name: str, model_cfg: dict[str, Any]) -> str:
    env_value = os.environ.get(model_path_env_key(model_name), "").strip()
    model_path = env_value or str(model_cfg["model_path"])
    return resolve_env_refs(model_path) if "${" in model_path else model_path


def resolve_env_refs(value: str) -> str:
    def expand_env(match: re.Match[str]) -> str:
        name = match.group(1)
        env_value = os.environ.get(name, "").strip()
        if not env_value:
            raise ValueError(f"Environment variable {name} is not set")
        return env_value

    return ENV_REF_RE.sub(expand_env, value)


def resolve_nodes(value: Any) -> list[str]:
    nodes: list[str] = []
    for item in normalize_list(value):
        text = str(item).strip()
        expanded = resolve_env_refs(text)
        nodes.extend(node.strip() for node in expanded.split(",") if node.strip())
    return nodes


def resolve_runner(runner_cfg: dict[str, Any]) -> dict[str, Any]:
    runner = copy.deepcopy(runner_cfg)
    for key in ("slurm_account", "slurm_partition", "slurm_submit_runner", "log_root"):
        value = runner.get(key)
        if isinstance(value, str):
            runner[key] = resolve_env_refs(value)
    return runner


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
    override_benchmark_concurrency: list[int] | None,
    override_eval_concurrency: list[int] | None,
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

    prefill_cfg = deep_merge(
        backend_cfg.get("service", {}).get("prefill", {}),
        model_cfg.get("service", {}).get("prefill", {}),
        suite_cfg.get("prefill", {}),
    )
    decode_cfg = deep_merge(
        backend_cfg.get("service", {}).get("decode", {}),
        model_cfg.get("service", {}).get("decode", {}),
        suite_cfg.get("decode", {}),
    )
    router_cfg = deep_merge(
        backend_cfg.get("service", {}).get("router", {}),
        model_cfg.get("service", {}).get("router", {}),
        suite_cfg.get("router", {}),
    )
    pd_worker_layout = str(suite_cfg.get("pd_worker_layout", "multi_node"))
    single_node_pd = pd_worker_layout == "single_node"
    prefill_single_node_pd = pd_worker_layout == "prefill_single_node"

    nodes = resolve_nodes(suite_cfg.get("nodes"))
    if single_node_pd:
        if not nodes:
            raise ValueError(
                f"{suite_cfg.get('name', model_name)} needs at least one node"
            )
        nodes = nodes[:1]
    elif prefill_single_node_pd:
        required_nodes = 1 + int(decode_cfg.get("workers", 1))
        if len(nodes) < required_nodes:
            raise ValueError(
                f"{suite_cfg.get('name', model_name)} needs at least "
                f"{required_nodes} node(s)"
            )
        nodes = nodes[:required_nodes]
    elif len(nodes) < 2:
        raise ValueError(
            f"{suite_cfg.get('name', model_name)} needs at least two nodes"
        )

    server_args = deep_merge(
        model_cfg.get("server", {}).get("common_args", {}),
        suite_cfg.get("server", {}).get("common_args", {}),
    )
    runner_cfg = resolve_runner(
        deep_merge(defaults.get("runner", {}), suite_cfg.get("runner", {}))
    )
    benchmark_cfg = deep_merge(
        defaults.get("benchmark", {}),
        suite_cfg.get("benchmark", {}),
    )
    accuracy_cfg = deep_merge(
        model_cfg.get("accuracy", {}), suite_cfg.get("accuracy", {})
    )

    concurrency = override_benchmark_concurrency or parse_int_list(
        suite_cfg.get("concurrency"), "concurrency"
    )
    isl = [int(value) for value in normalize_list(suite_cfg.get("isl"))]
    if not concurrency or not isl:
        raise ValueError(
            f"{suite_cfg.get('name', model_name)} must define isl and concurrency"
        )
    eval_concurrency = (
        override_eval_concurrency
        or suite_cfg.get("eval_concurrency")
        or accuracy_cfg.get("eval_concurrency")
        or concurrency
    )
    accuracy_concurrency = parse_int_list(eval_concurrency, "eval_concurrency")

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
        "model_path": resolve_model_path(model_name, model_cfg),
        "precision": str(model_cfg.get("precision", "")),
        "topology": topology,
        "display_topology": display_topology,
        "pd_worker_layout": pd_worker_layout,
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
            "concurrency": accuracy_concurrency,
            "model_type": str(accuracy_cfg.get("model_type", "local-completions")),
            "endpoint": str(accuracy_cfg.get("endpoint", "completions")),
            "batch_size": accuracy_cfg.get("batch_size"),
            "max_gen_toks": accuracy_cfg.get("max_gen_toks"),
            "apply_chat_template": bool(accuracy_cfg.get("apply_chat_template", False)),
            "fewshot_as_multiturn": bool(
                accuracy_cfg.get("fewshot_as_multiturn", False)
            ),
        },
    }


def build_cells(
    cfg: dict[str, Any],
    *,
    suite: str,
    model_filter: set[str] | None,
    override_image: str | None,
    override_benchmark_concurrency: list[int] | None,
    override_eval_concurrency: list[int] | None,
) -> list[dict[str, Any]]:
    cells = []
    for model_name, model_cfg in (cfg.get("models") or {}).items():
        if model_filter and model_name not in model_filter:
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
                    override_benchmark_concurrency=override_benchmark_concurrency,
                    override_eval_concurrency=override_eval_concurrency,
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
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME") or None,
        help="Optional model filter: one model name, comma-separated model names, or all",
    )
    parser.add_argument("--image", default=os.environ.get("ATOMESH_IMAGE") or None)
    parser.add_argument(
        "--benchmark-concurrency",
        default=os.environ.get("ATOMESH_BENCHMARK_CONCURRENCY") or None,
        help="Optional comma-separated benchmark concurrency override",
    )
    parser.add_argument(
        "--eval-concurrency",
        default=os.environ.get("ATOMESH_EVAL_CONCURRENCY") or None,
        help="Optional comma-separated lm_eval concurrency override",
    )
    parser.add_argument("--output", help="Optional output JSON path")
    parser.add_argument("--github-output", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    benchmark_concurrency = parse_int_list(
        args.benchmark_concurrency, "benchmark_concurrency"
    )
    eval_concurrency = parse_int_list(args.eval_concurrency, "eval_concurrency")
    cells = build_cells(
        config,
        suite=args.suite,
        model_filter=parse_model_filter(args.model),
        override_image=args.image,
        override_benchmark_concurrency=benchmark_concurrency or None,
        override_eval_concurrency=eval_concurrency or None,
    )
    print(f"Generated {len(cells)} ATOMesh benchmark cell(s) for suite={args.suite}")
    for cell in cells:
        print(
            f"  {cell['id']}: {cell['model']} {cell['display_topology']} "
            f"nodes={','.join(cell['nodes'])} isl={cell['isl']} osl={cell['osl']} "
            f"conc={cell['concurrency']} eval_conc={cell['accuracy']['concurrency']}"
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
