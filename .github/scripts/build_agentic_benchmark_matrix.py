#!/usr/bin/env python3
"""Build scheduled or manually selected agentic benchmark configurations."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_benchmark_matrix import _emit
from catalog import build_args, build_env_vars

CATALOG = ".github/benchmark/models_agentic.json"
NIGHTLY_CATALOG = ".github/benchmark/models_agentic_nightly.json"
PROFILE_CATALOGS = {"test": CATALOG, "nightly": NIGHTLY_CATALOG}


def resolve_run_image(image):
    """Resolve once on CPU; all cells and replay inputs use the immutable digest."""
    if "@" in image:
        if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", image):
            raise ValueError("Image digest must be a sha256 reference")
        return {"requested": image, "pinned": image, "display": image}
    if image == "rocm/atom-dev:latest":
        from resolve_atom_image import resolve_image

        resolved = resolve_image("rocm/atom-dev", "latest", None, "native")
        digest = resolved["reference_digest"]
        display = resolved["resolved_image"]
    else:
        # buildx resolves public OCI registries without pulling GPU image layers.
        output = subprocess.check_output(
            ["docker", "buildx", "imagetools", "inspect", image],
            text=True,
            timeout=90,
        )
        match = re.search(r"^Digest:\s+(sha256:[0-9a-f]{64})\s*$", output, re.MULTILINE)
        if not match:
            raise ValueError("Registry did not return an image digest")
        digest, display = match[1], image
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(digest)):
        raise ValueError("Registry returned an invalid image digest")
    return {"requested": image, "pinned": f"{image}@{digest}", "display": display}


def _check_fields(record, allowed, context):
    if not isinstance(record, dict):
        raise TypeError(f"{context} must be an object")
    unknown = record.keys() - allowed
    if unknown:
        raise ValueError(f"Unsupported {context} fields: {sorted(unknown)}")


def _concurrency(values):
    if (
        not isinstance(values, list)
        or not 1 <= len(values) <= 256
        or any(type(value) is not int or not 1 <= value <= 256 for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("Concurrency must contain unique integers between 1 and 256")
    return sorted(values)


def _load_configs(path):
    """Load agentic variants directly, without random-workload dimensions."""
    catalog = json.loads(Path(path).read_text())
    _check_fields(catalog, {"models", "reference"}, "catalog")
    configs = []
    identities = set()
    for model in catalog["models"]:
        _check_fields(
            model,
            {"display", "path", "prefix", "runner", "env_vars", "config", "variants"},
            "model",
        )
        _check_fields(
            model.get("config", {}),
            {"tp", "kv_cache_dtype", "trust_remote_code", "extra_args"},
            "server config",
        )
        for variant in model["variants"]:
            _check_fields(
                variant,
                {"label", "suffix", "extra_args", "env_vars", "concurrency"},
                "variant",
            )
            identity = model["prefix"] + variant["suffix"]
            if identity in identities:
                raise ValueError(f"Duplicate agentic artifact prefix: {identity}")
            identities.add(identity)
            configs.append(
                {
                    "display": f"{model['display']} {variant['label']}",
                    "prefix": model["prefix"],
                    "suffix": variant["suffix"],
                    "model_path": model["path"],
                    "server_args": build_args(model.get("config", {}), variant),
                    "bench_kind": "aiperf_agentic",
                    "env_vars": build_env_vars(model, variant),
                    "runner": model["runner"],
                    "concurrency": json.dumps(_concurrency(variant["concurrency"])),
                }
            )
    return configs


def build_configs(path=None, inputs=None):
    """Apply test overrides or select a subset of the nightly concurrency grid."""
    inputs = inputs or {}
    from prepare_benchmark_aiter_wheel import validate_request

    validate_request(inputs.get("aiter_wheel", ""), inputs.get("aiter_commit", ""))
    profile = inputs.get("profile") or "test"
    if profile not in PROFILE_CATALOGS:
        raise ValueError(f"Unknown agentic profile: {profile}")
    path = path or PROFILE_CATALOGS[profile]
    configs = _load_configs(path)
    known = {config["prefix"] for config in configs}
    selected = {
        name.strip() for name in inputs.get("models", "").split(",") if name.strip()
    }
    if selected - known:
        raise ValueError(f"Unknown agentic models: {sorted(selected - known)}")
    configs = [c for c in configs if not selected or c["prefix"] in selected]

    if inputs.get("concurrency", "").strip():
        values = _concurrency(
            [int(value.strip()) for value in inputs["concurrency"].split(",")]
        )
        if profile == "nightly":
            available = {
                c for config in configs for c in json.loads(config["concurrency"])
            }
            if set(values) - available:
                raise ValueError(
                    "Nightly concurrency must be a subset of its catalog grid"
                )
        for config in configs:
            config["concurrency"] = json.dumps(
                [c for c in json.loads(config["concurrency"]) if c in values]
                if profile == "nightly"
                else values
            )
        configs = [config for config in configs if json.loads(config["concurrency"])]
    if not configs or len(configs) > 256:
        raise ValueError("The agentic catalog must produce 1–256 matrix configurations")
    for config in configs:
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
        env["AIPERF_BENCHMARK_DURATION"] = str(seconds)
        config["env_vars"] = "\n".join(f"{key}={value}" for key, value in env.items())
        config["image"] = inputs.get("image") or "rocm/atom-dev:latest"
    return configs


def write_run_config(configs, inputs, event, output_dir, image_resolution=None):
    """Save the resolved matrix, dispatch inputs and an Actions run summary."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    # gh workflow run --json accepts string-valued inputs, including booleans.
    dispatch = {
        key: str(value).lower() if isinstance(value, bool) else str(value)
        for key, value in {**inputs, "atom_commit": commit}.items()
    }
    if image_resolution:
        dispatch["image"] = image_resolution["pinned"]
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
        "image_resolution": image_resolution,
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
            "aiter_ref": inputs.get("aiter_commit") or None,
            "aiter_wheel": inputs.get("aiter_wheel") or None,
            "aiter_source": (
                "override"
                if inputs.get("aiter_commit") or inputs.get("aiter_wheel")
                else "image version"
            ),
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
                "The image and downloaded wheel are pinned too; an AITER source ref must use a SHA for exact replays. "
                "A preview keeps `dry_run` set to the string `true`; change it to `false` to execute."
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
        resolution = resolve_run_image(configs[0]["image"])
        for config in configs:
            config["image"] = resolution["pinned"]
            config["image_display"] = resolution["display"]
        if os.environ.get("AGENTIC_RUN_CONFIG_DIR"):
            write_run_config(
                configs, inputs, event, os.environ["AGENTIC_RUN_CONFIG_DIR"], resolution
            )
        _emit(configs)
        count = sum(len(json.loads(config["concurrency"])) for config in configs)
        print(f"Event={event}: {count} agentic cells", file=sys.stderr)
        return 0
    except (
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
