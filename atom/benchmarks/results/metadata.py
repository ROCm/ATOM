# SPDX-License-Identifier: MIT
"""Capture execution identities; unknown metadata remains explicitly unknown."""

import ctypes
import importlib.metadata
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from . import SCHEMA_VERSION
from .io import read_json

PERFORMANCE_KEYS = {
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "GPU_MAX_HW_QUEUES",
    "MORI_SHMEM_HEAP_SIZE",
    "AITER_AR_1STAGE_MAX_KB",
    "AITER_BF16_FP8_MOE_BOUND",
    "AITER_LOG_LEVEL",
    "AITER_QUICK_REDUCE_QUANTIZATION",
    "AITER_USE_FLYDSL_MOE_SORTING",
    "ATOM_FORCE_ATTN_TRITON",
    "ATOM_MOE_GU_ITLV",
    "ATOM_NUMA_BIND",
    "ATOM_DSV41_BENCHMARK_SYNTHETIC",
    "ENABLE_TORCH_PROFILER",
    "ENABLE_RTL_PROFILER",
    "ATOM_AGENTIC_PROFILE_SECONDS",
    "AIPERF_UNSAFE_OVERRIDE",
    "AIPERF_SCENARIO",
    "AIPERF_PUBLIC_DATASET",
    "AIPERF_MAX_CONTEXT_LENGTH",
    "AIPERF_NUM_DATASET_ENTRIES",
    "AIPERF_BENCHMARK_DURATION",
    "AIPERF_WARMUP_REQUESTS_PER_LANE",
    "AIPERF_TRACE_IDLE_GAP_CAP_SECONDS",
    "AIPERF_AGENTIC_WARMUP_GRACE_PERIOD",
    "AIPERF_TRAJECTORY_START_MIN_RATIO",
    "AIPERF_TRAJECTORY_START_MAX_RATIO",
    "AIPERF_FAILED_REQUEST_THRESHOLD",
    "AIPERF_SLICE_DURATION",
    "AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT",
    "AIPERF_HTTP_TCP_USER_TIMEOUT",
    "AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES",
    "AIPERF_DATASET_CONFIGURATION_TIMEOUT",
    "AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT",
    "AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID",
    "AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID",
    "AIPERF_MODEL",
}


def git_sha(path):
    # Container bind mounts can be owned by the host UID. Trust only the
    # inspected checkout for this read, without changing global Git config.
    path = Path(path).resolve()
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={path}",
                "-C",
                str(path),
                "rev-parse",
                "HEAD",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def capture_hf_dataset(cache_dir, export_path):
    """Hash this replay's isolated HF cache, including the actual Arrow inputs.

    AIPerf deliberately omits inputs.json for Weka traces. The caller gives HF
    a fresh cache and disables AIPerf's mmap cache so these are the loaded data,
    not an unrelated cached revision. Retain the ledger, not a second dataset.
    """
    from .io import fingerprint, sha256

    export = read_json(export_path)
    provenance = export.get("metadata", {}).get("dataset", {})
    if not provenance.get("hf_dataset_name"):
        raise ValueError("AIPerf export is missing the HF dataset identity")
    cache = Path(cache_dir)
    infos = list(cache.rglob("dataset_info.json"))
    if len(infos) != 1:
        raise ValueError("Expected exactly one dataset in the isolated HF cache")
    directory = infos[0].parent
    arrows = sorted(directory.glob("*.arrow"))
    if not arrows:
        raise ValueError("The replay's HF cache contains no Arrow input files")
    files = [
        {"path": path.name, "size": path.stat().st_size, "sha256": sha256(path)}
        for path in [infos[0], *arrows]
    ]
    return {
        "source": "isolated_hf_cache",
        "dataset_provenance": provenance,
        "files": files,
        "dataset_content_sha256": fingerprint(files),
    }


def performance_env():
    return {k: v for k, v in os.environ.items() if k in PERFORMANCE_KEYS}


def installed_package(package):
    """Inspect the executing interpreter's installation, never the desired pin."""
    result = {"version": None, "sha": None, "sha_source": None, "wheel_sha256": None}
    try:
        distribution = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        return result
    result["version"] = distribution.version
    try:
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    except ValueError:
        direct = {}
    commit = direct.get("vcs_info", {}).get("commit_id")
    if commit:
        result.update(sha=commit, sha_source="direct_url_vcs")
    url = direct.get("url", "")
    if not commit and url.startswith("file://"):
        from urllib.parse import unquote, urlparse

        source = Path(unquote(urlparse(url).path))
        # A local wheel is a file, not an editable checkout. Never attribute
        # the wheel to the Git repository containing its download directory.
        if source.is_dir():
            commit = git_sha(source)
            if commit:
                result.update(sha=commit, sha_source="local_git")
    if not commit and re.sub(r"[-_.]+", "-", package).lower() == "amd-aiter":
        # AITER wheels embed a Git abbreviation, e.g. 0.1.1.dev1+ga75ba53de.
        # Retain it as an abbreviation; do not invent a full commit or use
        # an unrelated checkout. Dirty/unknown version suffixes stay unknown.
        match = re.search(r"\+g([0-9a-f]{7,40})$", distribution.version, re.IGNORECASE)
        if match:
            result.update(sha=match[1].lower(), sha_source="package_version")
    if url.split("?", 1)[0].lower().endswith(".whl"):
        archive = direct.get("archive_info", {})
        digest = archive.get("hashes", {}).get("sha256")
        if not digest and archive.get("hash", "").startswith("sha256="):
            digest = archive["hash"].split("=", 1)[1]
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            result["wheel_sha256"] = digest.lower()
    return result


def hip_pci_ids():
    """Resolve HIP logical ordinals (including visibility masks) to PCI IDs."""
    try:
        hip = ctypes.CDLL("libamdhip64.so")
        count = ctypes.c_int()
        if hip.hipGetDeviceCount(ctypes.byref(count)) != 0:
            return []
        result = []
        for index in range(count.value):
            bus = ctypes.create_string_buffer(32)
            if hip.hipDeviceGetPCIBusId(bus, len(bus), index) != 0:
                return []
            result.append(bus.value.decode().lower())
        return result
    except (OSError, AttributeError):
        return []


def option(argv, names, default=None):
    result = default
    for i, arg in enumerate(argv):
        normalized = arg.split("=", 1)[0].replace("_", "-")
        if normalized in names and "=" not in arg and i + 1 < len(argv):
            result = argv[i + 1]
        elif normalized in names and "=" in arg:
            result = arg.split("=", 1)[1]
    return result


def synthetic_settings(argv, environment):
    """Derive forced acceptance from executed flags, retaining legacy declarations."""
    settings = {}
    for key in ("length", "rate"):
        value = option(argv, (f"--spec-decode-acceptance-{key}",))
        if value is not None:
            settings[f"acceptance_{key}"] = float(value)
    declared = environment.get("ATOM_DSV41_BENCHMARK_SYNTHETIC")
    forced = bool(settings)
    # A declaration can describe older/custom synthetic mechanisms. Preserve it
    # conservatively, but it must never hide forced acceptance in actual argv.
    result = {"synthetic": forced or declared in ("1", "true"), **settings}
    if declared is not None:
        result["synthetic_declared"] = declared in ("1", "true")
        result["synthetic_declaration_mismatch"] = (
            result["synthetic_declared"] != forced
        )
    return result


def devices(sysfs="/sys/class/drm"):
    result = []
    for card in sorted(Path(sysfs).glob("card[0-9]*"), key=lambda p: p.name):
        if not re.fullmatch(r"card\d+", card.name):
            continue
        device = card / "device"
        try:
            if (device / "vendor").read_text().strip() != "0x1002":
                continue
            unique_id = (
                (device / "unique_id").read_text().strip()
                if (device / "unique_id").exists()
                else None
            )
            result.append(
                {
                    "pci_bus_id": device.resolve().name,
                    "uuid": unique_id,
                    "sysfs_path": str(device.resolve()),
                }
            )
        except OSError:
            continue
    return result


def select_devices(all_devices, gpu_count, env, logical_pci_ids=None):
    """Only an explicit PCI selection is safe across HIP/DRM index reorderings."""
    selected = env.get("ATOM_BENCHMARK_GPU_PCI_IDS", "").split(",")
    selected = [x.strip() for x in selected if x.strip()]
    if selected:
        by_pci = {d["pci_bus_id"]: d for d in all_devices}
        if (
            len(selected) != gpu_count
            or len(set(selected)) != gpu_count
            or any(x not in by_pci for x in selected)
        ):
            raise ValueError(
                "ATOM_BENCHMARK_GPU_PCI_IDS must name each allocated GPU exactly once"
            )
        return [by_pci[x] for x in selected], "explicit_pci"
    if logical_pci_ids and len(logical_pci_ids) >= gpu_count:
        by_pci = {d["pci_bus_id"].lower(): d for d in all_devices}
        selected = logical_pci_ids[:gpu_count]
        if len(set(selected)) == gpu_count and all(x in by_pci for x in selected):
            return [
                dict(by_pci[x], logical_ordinal=i) for i, x in enumerate(selected)
            ], "hip_logical_ordinals"
    # All cards used: no ordinal ambiguity. A subset needs explicit identities.
    if len(all_devices) == gpu_count and not any(
        env.get(k)
        for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    ):
        return all_devices, "all_detected_devices"
    return [], "unresolved_subset"


def capture_config(model, server_argv, kind, concurrency, launch=None):
    argv = launch.get("argv", server_argv) if launch else server_argv
    tp = int(option(argv, ("-tp", "--tensor-parallel-size"), "1"))
    pp = int(option(argv, ("-pp", "--pipeline-parallel-size"), "1"))
    dp = int(option(argv, ("-dp", "--data-parallel-size"), "1"))
    dp_attention = any(x.replace("_", "-") == "--enable-dp-attention" for x in argv)
    gpu_count = tp * pp * (1 if dp_attention else dp)
    env = launch.get("environment", performance_env()) if launch else performance_env()
    all_devices = devices()
    allocated, selection_source = select_devices(
        all_devices, gpu_count, os.environ, hip_pci_ids()
    )
    gpu_name = os.environ.get("GPU_NAME", "")
    hw = os.environ.get("BENCHMARK_HARDWARE")
    if not hw:
        hw = next(
            (
                key
                for key in ("mi355x", "mi350x", "mi325x", "mi300x")
                if key in gpu_name.lower().replace(" ", "")
            ),
            None,
        )
    harness = (
        installed_package("aiperf")
        if kind == "agentic"
        else {"version": "atom", "sha": git_sha(Path.cwd())}
    )
    model_revision = os.environ.get("BENCHMARK_MODEL_REVISION")
    marker = Path(model) / ".hf-revision"
    if marker.is_file():
        model_revision = marker.read_text().strip()
    precision = os.environ.get("BENCHMARK_PRECISION")
    # Precision is intentionally not inferred from KV dtype or a display label.
    aiter = installed_package("amd-aiter")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "repository": os.environ.get("GITHUB_REPOSITORY", "ROCm/ATOM"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
            "job": os.environ.get("GITHUB_JOB"),
            "cell_id": os.environ.get("RESULT_FILENAME"),
            "workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF"),
            "workflow_sha": os.environ.get("GITHUB_SHA"),
        },
        "software": {
            "atom_sha": git_sha(Path.cwd()),
            "aiter_sha": aiter["sha"],
            "aiter_sha_source": aiter["sha_source"],
            "aiter_version": aiter["version"],
            "aiter_wheel_sha256": aiter["wheel_sha256"],
            "image": os.environ.get("DOCKER_IMAGE"),
            "image_digest": os.environ.get("ATOM_IMAGE_DIGEST"),
            "harness_version": os.environ.get("ATOM_HARNESS_VERSION")
            or harness["version"],
            "harness_sha": os.environ.get("ATOM_HARNESS_SHA") or harness["sha"],
            "rocm_version": os.environ.get("ROCM_VERSION"),
        },
        "model": {
            "name": model,
            "key": os.environ.get("BENCHMARK_MODEL_KEY"),
            "revision": model_revision,
            "precision": precision,
        },
        "hardware": {
            "key": hw,
            "name": gpu_name,
            "gpu_count": gpu_count,
            "devices": allocated,
            "runner_gpu_count": len(all_devices),
            "selection_source": selection_source,
        },
        "parallelism": {
            "tp": tp,
            "pp": pp,
            "dp": dp,
            "ep": int(option(argv, ("--expert-parallel-size",), "1")),
            "dp_attention": dp_attention,
        },
        "workload": {
            "kind": kind,
            "concurrency": concurrency,
            "isl": int(os.environ.get("ISL", "0")) if kind == "random" else None,
            "osl": int(os.environ.get("OSL", "0")) if kind == "random" else None,
            "range_ratio": (
                os.environ.get("RANDOM_RANGE_RATIO") if kind == "random" else None
            ),
            "dataset": (
                os.environ.get("AIPERF_PUBLIC_DATASET")
                if kind == "agentic"
                else "random"
            ),
            "dataset_revision": os.environ.get("AIPERF_DATASET_REVISION"),
            "duration_s": os.environ.get("AIPERF_BENCHMARK_DURATION"),
            "warmup_requests_per_lane": os.environ.get(
                "AIPERF_WARMUP_REQUESTS_PER_LANE"
            ),
            "seed": 42 if kind == "agentic" else 0,
        },
        "recipe": {
            "server_argv": argv,
            "environment": env,
            "spec_method": option(argv, ("--method",), "none"),
            "spec_tokens": option(argv, ("--num-speculative-tokens",)),
            "kv_cache_dtype": option(argv, ("--kv-cache-dtype",)),
            "benchmark_extra_argv": shlex.split(os.environ.get("BENCH_EXTRA_ARGS", "")),
        },
        "timing": {
            "timestamp_unit": "unix_ns_decimal_string",
            "duration_unit": "s",
            "alignment": "client and collector share the host clock",
        },
        "validity": {
            **synthetic_settings(argv, env),
            "instrumented": any(
                env.get(key) == "1"
                for key in ("ENABLE_TORCH_PROFILER", "ENABLE_RTL_PROFILER")
            ),
            "unsafe_override": os.environ.get("AIPERF_UNSAFE_OVERRIDE") in ("1", "true")
            or (
                kind == "agentic"
                and int(os.environ.get("AIPERF_BENCHMARK_DURATION", "3600")) < 900
            ),
        },
    }


def resolve_client_config(config, client_launch):
    """Apply the executed client arguments, after all shell overrides."""
    if client_launch and Path(client_launch).is_file():
        launch = read_json(client_launch)
        config["recipe"]["client_argv"] = launch["argv"]
        config["recipe"]["client_environment"] = launch.get("environment", {})
    client = config["recipe"].get("client_argv", [])
    if config["workload"]["kind"] == "random":
        for key, flag, cast in (
            ("isl", "--random-input-len", int),
            ("osl", "--random-output-len", int),
            ("concurrency", "--max-concurrency", int),
            ("seed", "--seed", int),
            ("num_prompts", "--num-prompts", int),
            ("warmup_requests", "--num-warmups", int),
            ("range_ratio", "--random-range-ratio", float),
        ):
            value = option(client, (flag,))
            if value is not None:
                config["workload"][key] = cast(value)
        config["workload"]["request_rate"] = option(client, ("--request-rate",), "inf")
        config["workload"]["ignore_eos"] = "--ignore-eos" in client
        for total, ratio in (
            ("warmup_requests", "warmup_requests_per_lane"),
            ("num_prompts", "requests_per_lane"),
        ):
            if total in config["workload"]:
                config["workload"][ratio] = (
                    config["workload"][total] / config["workload"]["concurrency"]
                )
