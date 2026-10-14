"""Fail-closed provenance and host checks for the single 81198 prefix READ case.

Experiment hooks close observation after the existing warmup drain. Local
readiness verifies the reviewed bundle, not runtime resources or GPU evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

CASE = "prefix-read-fixed"
CASE_NAME = "kimi-k3-vllm-c48-prefix-read-fixed"
ENGINE = "81198e97ba7eee2a22540caaa756b7fdddcb4d93"
IMAGE = f"vllm/vllm-openai-rocm:nightly-{ENGINE}@sha256:a401e4f46872dcde77e07ae3d7fa2d3a71b7f2773714316116532e2b39deb179"
BASE = "11093795802a92494aeda37835789fdf3b87be5f8d01b3e0305d18959e189f8d"
AFTER = "c8713a6590cb1a023597ae43d19e16b4bb8a98ebc2589fc40621e738231be8bb"
PATCH = "75e31a74a2afc80bd7828d033d8047837dc6839eadbef71e384979b8b5f1d0da"
RELATIVE = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/" + RELATIVE
PAYLOAD = ".github/benchmark/prefix-read-fixed"
LOCAL_READY = "LOCAL_CHECKS_PASSED_RUNTIME_GATED"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_ready() -> dict:
    manifest = verify_bundle(Path(__file__).resolve().parents[3])
    if manifest.get("readiness") != LOCAL_READY or manifest.get("blockers") != []:
        raise RuntimeError("BLOCKED: local checks and review are incomplete")
    if not manifest.get("local_review", {}).get("passed"):
        raise RuntimeError("BLOCKED: missing local review evidence")
    return {
        "status": LOCAL_READY,
        "gpu_verified": False,
        "runtime_gates_required": True,
        "dispatch_authorization": "separate shared reservation; at most one workflow",
    }


def readonly_file(path: Path, mountinfo: str) -> bool:
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) > 6 and fields[4].replace("\\040", " ") == str(path):
            return "ro" in fields[5].split(",")
    return False


def verify_overlay(vllm_root: Path, mountinfo: str | None = None) -> dict:
    path = vllm_root / RELATIVE
    actual = digest(path)
    if actual != AFTER:
        raise RuntimeError(f"prefix READ candidate hash mismatch: {actual}")
    if mountinfo is None:
        mountinfo = Path("/proc/self/mountinfo").read_text()
    if not readonly_file(path, mountinfo):
        raise RuntimeError(f"prefix READ product is not a readonly file mount: {path}")
    return {
        "name": CASE,
        "engine": ENGINE,
        "base_sha256": BASE,
        "after_sha256": AFTER,
        "observed_sha256": actual,
        "destination": str(path),
        "readonly_mount_observed": True,
        "runtime_uid": os.geteuid(),
        "scope": "product attention page selector only",
    }


def verify_bundle(repo: Path) -> dict:
    root = repo / PAYLOAD
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["image"] != IMAGE or manifest["product"]["base_sha256"] != BASE:
        raise RuntimeError("unexpected base image/product identity")
    for relative, expected in manifest["files"].items():
        if digest(repo / relative) != expected:
            raise RuntimeError(f"manifest hash mismatch: {relative}")
    if (
        digest(root / "moriio_connector.py") != AFTER
        or digest(root / "fix-81198.patch") != PATCH
    ):
        raise RuntimeError("frozen product payload mismatch")
    reference = json.loads(
        (repo / ".github/benchmark/c48-model-gap/V0.json").read_text()
    )
    config = json.loads((root / "config.json").read_text())
    if config != dict(reference, name=CASE_NAME):
        raise RuntimeError("serving recipe differs from V0 beyond case name")
    return manifest


def verify_runtime_overlays(vllm_root: Path) -> list[dict]:
    repo = Path(__file__).resolve().parents[3]
    rows = json.loads((repo / PAYLOAD / "runtime-overlays.json").read_text())
    mounts = Path("/proc/self/mountinfo").read_text()
    result = []
    for row in rows:
        if not row["source"].startswith("vllm/"):
            continue
        path = vllm_root / row["source"].removeprefix("vllm/")
        if digest(path) != row["after_sha256"] or not readonly_file(path, mounts):
            raise RuntimeError(f"runtime overlay hash/readonly check failed: {path}")
        result.append(dict(row, observed_sha256=digest(path), readonly=True))
    return result


def check_capacity(
    paths: list[Path], minimum_bytes: int, minimum_inodes: int
) -> list[dict]:
    records = []
    for path in paths:
        stats = os.statvfs(path)  # Missing/unknown storage fails closed.
        row = {
            "path": str(path),
            "available_bytes": stats.f_bavail * stats.f_frsize,
            "available_inodes": stats.f_favail,
        }
        if (
            row["available_bytes"] < minimum_bytes
            or row["available_inodes"] < minimum_inodes
        ):
            raise RuntimeError(f"insufficient storage: {row}")
        records.append(row)
    return records


def check_model(root: Path) -> dict:
    config = root / "config.json"
    index = root / "model.safetensors.index.json"
    mapping = json.loads(index.read_text())["weight_map"]
    if not mapping:
        raise RuntimeError(f"empty model shard index: {root}")
    shards = []
    for name in sorted(set(mapping.values())):
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError(f"unsafe model shard name: {name}")
        if (
            not path.is_file()
            or not os.access(path, os.R_OK)
            or path.stat().st_size <= 0
        ):
            raise RuntimeError(f"missing/unreadable/empty model shard: {path}")
        shards.append({"path": name, "bytes": path.stat().st_size})
    return {
        "path": str(root),
        "config_sha256": digest(config),
        "index_sha256": digest(index),
        "shards": shards,
        "weight_contents_hashed": False,
    }


def cgroup_headroom(proc_root=Path("/proc/self"), mountinfo=None) -> list[dict]:
    groups = [
        line.split(":", 2) for line in (proc_root / "cgroup").read_text().splitlines()
    ]
    mounts = (proc_root / "mountinfo").read_text() if mountinfo is None else mountinfo
    records = []
    for line in mounts.splitlines():
        left, right = line.split(" - ", 1)
        fields, fs = left.split(), right.split()
        if fs[0] == "cgroup2":
            matches = [g[2] for g in groups if g[0] == "0" and g[1] == ""]
            limit_name, usage_name = "memory.max", "memory.current"
        elif fs[0] == "cgroup" and "memory" in fs[2].split(","):
            matches = [g[2] for g in groups if "memory" in g[1].split(",")]
            limit_name, usage_name = "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        if len(matches) != 1:
            raise RuntimeError("cannot resolve memory cgroup")
        root, mount = Path(fields[3]), Path(fields[4])
        group = Path(matches[0])
        try:
            relative = group.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("memory cgroup outside visible mount") from exc
        current = mount / relative
        while True:
            # Unified cgroup root has no memory controller budget files; its
            # finite ceiling is host MemAvailable, already checked separately.
            if (
                fs[0] == "cgroup2"
                and current == mount
                and root == Path("/")
                and not (current / limit_name).exists()
            ):
                records.append(
                    {
                        "path": str(current),
                        "limit": None,
                        "usage": 0,
                        "headroom": None,
                        "basis": "root cgroup; host MemAvailable",
                    }
                )
                break
            raw = (current / limit_name).read_text().strip()
            usage = int((current / usage_name).read_text())
            limit = None if raw == "max" else int(raw)
            records.append(
                {
                    "path": str(current),
                    "limit": limit,
                    "usage": usage,
                    "headroom": None if limit is None else max(0, limit - usage),
                }
            )
            if current == mount:
                break
            current = current.parent
    if not records:
        raise RuntimeError("memory cgroup limit unavailable")
    return records


def check_memory(meminfo: str, rank: int, cgroups=None) -> dict:
    values = {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in meminfo.splitlines()
    }
    amount, unit = values["MemAvailable"].split()
    if unit != "kB":
        raise RuntimeError("unknown MemAvailable unit")
    available = int(amount) * 1024
    if cgroups is not None:
        if not cgroups:
            raise RuntimeError("missing memory cgroup evidence")
        available = min(
            [available]
            + [row["headroom"] for row in cgroups if row["headroom"] is not None]
        )
    # 1799 decimal GB shared by P's eight ranks, plus explicit 64 GiB reserve.
    required = (1799000000000 if rank == 0 else 0) + 64 * 1024**3
    if available < required:
        raise RuntimeError(
            f"CPU memory budget insufficient: available={available}, required={required}"
        )
    return {
        "available_bytes": available,
        "required_bytes": required,
        "offload_bytes": 1799000000000 if rank == 0 else 0,
        "cgroup_limits": cgroups,
    }


def host_preflight(model: Path, run_dir: Path, rank: int, image: str) -> dict:
    if image != IMAGE or rank not in (0, 1):
        raise RuntimeError("wrong image or node rank")
    docker_root = subprocess.check_output(
        ["docker", "info", "--format", "{{.DockerRootDir}}"], text=True, timeout=30
    ).strip()
    if not docker_root or not Path(docker_root).is_absolute():
        raise RuntimeError("unknown Docker storage root")
    inspected = subprocess.run(
        ["docker", "image", "inspect", image],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if inspected.returncode:
        if "No such image" not in inspected.stderr:
            raise RuntimeError("image cache state unknown: " + inspected.stderr)
        cached = False
    else:
        entry = json.loads(inspected.stdout)[0]
        cached = any(
            item.endswith("@" + IMAGE.split("@", 1)[1])
            for item in entry.get("RepoDigests", [])
        )
        if not cached:
            raise RuntimeError("cached image digest mismatch")
    minimum = (40 if cached else 200) * 1024**3
    storage = check_capacity(
        [Path(docker_root), Path("/var/lib/containerd"), Path("/tmp"), run_dir],
        minimum,
        100000,
    )
    memory = check_memory(Path("/proc/meminfo").read_text(), rank, cgroup_headroom())
    for device in ("/dev/kfd", "/dev/dri", "/dev/infiniband"):
        if not Path(device).exists() or not os.access(device, os.R_OK):
            raise RuntimeError(f"missing/inaccessible device: {device}")
    rdma = list(Path("/sys/class/infiniband").iterdir())
    if len(rdma) < 8:
        raise RuntimeError("expected at least eight RDMA devices")
    models = [
        check_model(model),
        check_model(model.parent.parent / "Inferact/Kimi-K3-DSpark"),
    ]
    return {
        "status": "HOST_PREFLIGHT_ONLY",
        "image_cached": cached,
        "minimum_storage_bytes": minimum,
        "rank": rank,
        "image": image,
        "storage": storage,
        "cpu_memory": memory,
        "models": models,
        "policy": "readonly; no shared storage pruning; model contents not hashed",
    }


def verify_image(image: str) -> dict:
    if image != IMAGE:
        raise RuntimeError("unexpected serving image")
    inspected = json.loads(
        subprocess.check_output(
            ["docker", "image", "inspect", image], text=True, timeout=30
        )
    )[0]
    expected_digest = IMAGE.split("@", 1)[1]
    if not any(
        item.endswith("@" + expected_digest)
        for item in inspected.get("RepoDigests", [])
    ):
        raise RuntimeError("image digest is not present in Docker RepoDigests")
    repo = Path(__file__).resolve().parents[3]
    overlays = json.loads((repo / PAYLOAD / "runtime-overlays.json").read_text())
    expected_sources = {TARGET: BASE}
    expected_sources.update(
        {
            row["destination"]: row["base_sha256"]
            for row in overlays
            if "destination" in row
        }
    )
    script = (
        "import hashlib,pathlib,json; "
        f"paths={list(expected_sources)!r}; "
        "print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths}))"
    )
    actual = subprocess.check_output(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--user",
            f"{os.geteuid()}:{os.getegid()}",
            "--entrypoint",
            "python3",
            image,
            "-I",
            "-c",
            script,
        ],
        text=True,
        timeout=120,
    ).strip()
    observed_sources = json.loads(actual)
    if observed_sources != expected_sources:
        raise RuntimeError(f"pristine image source base mismatch: {observed_sources}")
    return {
        "image": image,
        "image_id": inspected["Id"],
        "repo_digests": inspected["RepoDigests"],
        "base_sha256_observed": observed_sources,
        "probe": "readonly CPU-only no-network no-device pristine image inspection",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["require-ready", "verify-bundle", "host-preflight", "verify-image"],
    )
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--rank", type=int, choices=[0, 1])
    parser.add_argument("--image")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.command == "require-ready":
        result = require_ready()
    elif options.command == "verify-bundle":
        result = verify_bundle(options.repo)
    elif options.command == "verify-image":
        result = verify_image(options.image)
    else:
        if any(
            value is None
            for value in (options.model, options.run_dir, options.rank, options.image)
        ):
            parser.error("host-preflight requires model, run-dir, rank and image")
        result = host_preflight(
            options.model, options.run_dir, options.rank, options.image
        )
    encoded = json.dumps(result, indent=2) + "\n"
    if options.output:
        options.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
