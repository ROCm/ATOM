#!/usr/bin/env python3
"""Build scheduled or manually selected agentic benchmark configurations."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_benchmark_matrix import _emit
from catalog import build_cell_configs, load_variants

CATALOG = ".github/benchmark/models_agentic.json"
NIGHTLY_CATALOG = ".github/benchmark/models_agentic_nightly.json"
PROFILE_CATALOGS = {"test": CATALOG, "nightly": NIGHTLY_CATALOG}


def build_configs(path=None, inputs=None):
    """Apply manual overrides while retaining catalog concurrency bands."""
    inputs = inputs or {}
    profile = inputs.get("profile") or "test"
    if profile not in PROFILE_CATALOGS:
        raise ValueError(f"Unknown agentic profile: {profile}")
    path = path or PROFILE_CATALOGS[profile]
    known = {variant["prefix"] for variant in load_variants(path)}
    selected = {
        name.strip() for name in inputs.get("models", "").split(",") if name.strip()
    }
    if selected - known:
        raise ValueError(f"Unknown agentic models: {sorted(selected - known)}")

    param_lists = None
    values = None
    if inputs.get("concurrency", "").strip():
        values = [int(value.strip()) for value in inputs["concurrency"].split(",")]
        if (
            not all(value > 0 for value in values)
            or len(set(values)) != len(values)
            or len(values) > 256
        ):
            raise ValueError("Concurrency must contain 1–256 unique positive integers")
        # ISL/OSL/ratio are naming placeholders for the shared template. Agentic
        # metrics use the actual trace tokens and store these dimensions as null.
        if profile == "test":
            param_lists = ";".join(f"0,0,{value},1" for value in values)

    configs = build_cell_configs(
        path, param_lists=param_lists, model_filter=selected or None
    )
    if profile == "nightly" and values is not None:
        # Keep the catalog's per-point capture recipe. A blind scenario override
        # would duplicate c32 in both sparse and dense capture variants.
        available = {c for config in configs for c in json.loads(config["concurrency"])}
        if set(values) - available:
            raise ValueError("Nightly concurrency must be a subset of its catalog grid")
        for config in configs:
            config["concurrency"] = json.dumps(
                [c for c in json.loads(config["concurrency"]) if c in values]
            )
        configs = [config for config in configs if json.loads(config["concurrency"])]
    if not configs or len(configs) > 256:
        raise ValueError("The agentic catalog must produce 1–256 matrix configurations")
    for config in configs:
        if config["bench_kind"] != "aiperf_agentic":
            raise ValueError(
                "The agentic catalog must only contain aiperf_agentic cells"
            )
        env = dict(
            line.split("=", 1) for line in config["env_vars"].splitlines() if line
        )
        duration = inputs.get("duration_seconds")
        if duration is None or duration == "":
            duration = env.get("AIPERF_BENCHMARK_DURATION", "900")
        seconds = int(duration)
        if isinstance(duration, bool) or float(duration) != seconds:
            raise ValueError("Duration must be a whole number of seconds")
        # Replay + the runner's 90-minute warmup/drain budget stays below the
        # reusable workflow's 180-minute benchmark step timeout.
        if not 900 <= seconds <= 3600:
            raise ValueError("Duration must be between 900 and 3600 seconds")
        if len(json.loads(config["concurrency"])) > 256:
            raise ValueError("A concurrency matrix cannot exceed 256 cells")
        env["AIPERF_BENCHMARK_DURATION"] = str(seconds)
        config["env_vars"] = "\n".join(f"{key}={value}" for key, value in env.items())
        config["image"] = inputs.get("image") or "rocm/atom-dev:latest"
    return configs


def write_run_config(configs, inputs, event, output_dir):
    """Save the resolved matrix, dispatch inputs and an Actions run summary."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dispatch = {**inputs, "atom_commit": commit}
    record = {
        "event": event,
        "actor": os.environ.get("GITHUB_ACTOR", ""),
        "triggering_actor": os.environ.get("GITHUB_TRIGGERING_ACTOR", ""),
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "workflow_sha": os.environ.get("GITHUB_SHA", ""),
        "checkout_sha": commit,
        "inputs": inputs,
        "configs": configs,
    }
    (output / "run-config.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "dispatch-inputs.json").write_text(json.dumps(dispatch, indent=2) + "\n")
    command = (
        shlex.join(
            [
                "gh",
                "workflow",
                "run",
                "atom-agentic-benchmark.yaml",
                "--repo",
                os.environ.get("GITHUB_REPOSITORY", "ROCm/ATOM"),
                "--ref",
                os.environ.get("GITHUB_REF_NAME", "main"),
                "--json",
            ]
        )
        + " < dispatch-inputs.json"
    )
    count = sum(len(json.loads(config["concurrency"])) for config in configs)
    mode = "Preview only; no GPU jobs" if inputs.get("dry_run") else "GPU benchmark"
    lines = [
        "## Agentic run configuration",
        "",
        f"**Mode:** {mode} · **Points:** {count} · **Trigger:** {event}",
        "",
        f"**ATOM checkout:** `{commit}`",
        "",
        (
            "Requested configuration is shown below. Runtime image/model/software identities "
            "are recorded in each point's bundle."
        ),
        "",
    ]
    for config in configs:
        effective = {
            "model": config["model_path"],
            "runner": inputs.get("runner") or config["runner"],
            "image": config["image"],
            "concurrency": json.loads(config["concurrency"]),
            "server_args": config["server_args"],
            "extra_args": inputs.get("extra_args") or "",
            "env_vars": dict(
                line.split("=", 1) for line in config["env_vars"].splitlines()
            ),
            "aiter_ref": inputs.get("aiter_commit") or "image version",
            "enable_profiler": inputs.get("enable_profiler", False),
            "enable_rtl": inputs.get("enable_rtl", False),
        }
        lines.extend(
            [
                f"### {escape(config['display'])}",
                "",
                "**Concurrency:** "
                + ", ".join(str(value) for value in effective["concurrency"])
                + " · **Seconds per point:** "
                + effective["env_vars"]["AIPERF_BENCHMARK_DURATION"],
                "",
                "<details><summary>Server, environment and execution options</summary>",
                "",
                "<pre>" + escape(json.dumps(effective, indent=2)) + "</pre>",
                "</details>",
                "",
            ]
        )
    lines.extend(
        [
            "### Repeat this configuration",
            "",
            (
                "Download and extract `atom-agentic-run-config-<attempt>` from this run's "
                "artifacts, then run:"
            ),
            "",
            "```sh",
            command,
            "```",
            "",
            (
                "The dispatch file pins the ATOM checkout; the workflow ref must still exist. "
                "Set an immutable image and AITER ref for version comparisons. "
                "A preview keeps `dry_run: true`; change it to `false` to execute."
            ),
            "",
            (
                "Each GPU job links its summary, full bundle and available failure diagnostics. "
                "Artifacts expire after 15 days (failure diagnostics: 14 days); "
                "download them for long-term storage or import into AgenticViewer."
            ),
            "",
        ]
    )
    summary = "\n".join(lines)
    (output / "README.md").write_text(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(summary)


def main():
    try:
        event = os.environ.get("EVENT_NAME", "schedule")
        if event not in ("schedule", "workflow_dispatch"):
            raise ValueError(f"Unsupported agentic benchmark event: {event}")
        inputs = (
            json.loads(os.environ.get("INPUTS_JSON") or "{}")
            if event == "workflow_dispatch"
            else {"profile": "nightly"}
        )
        configs = build_configs(inputs=inputs)
        if os.environ.get("AGENTIC_RUN_CONFIG_DIR"):
            write_run_config(
                configs, inputs, event, os.environ["AGENTIC_RUN_CONFIG_DIR"]
            )
        _emit(configs)
        count = sum(len(json.loads(config["concurrency"])) for config in configs)
        print(f"Event={event}: {count} agentic cells", file=sys.stderr)
        return 0
    except (ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
