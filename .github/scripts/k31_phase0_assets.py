"""No-device inspection only: refuse mismatched runtime or cached model inputs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

from k31_phase0_inputs import checked_files, require


def identity(path, value=None):
    value = path.stat() if value is None else value
    require(stat.S_ISREG(value.st_mode), f"not a regular file: {path}")
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def measured_hash(path, seconds=90):
    before, started = identity(path), time.monotonic()
    value, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        require(
            identity(path, os.fstat(stream.fileno())) == before,
            "opened file identity differs",
        )
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            require(time.monotonic() - started < seconds, "checksum deadline")
            value.update(block)
            count += len(block)
        require(
            identity(path, os.fstat(stream.fileno())) == before, "opened file changed"
        )
    require(
        identity(path) == before and count == before["bytes"],
        "file changed during checksum",
    )
    return {"sha256": value.hexdigest(), "bytes_read": count, "identity": before}


def check_runtime(pins, spec, result):
    rows = result.setdefault("files", {})
    paths = {item["path"]: item["sha256"] for item in pins["package_files"]}
    paths.update(spec["native_files"])
    paths[str(Path(sys.executable).resolve())] = pins["python_sha256"]
    paths["/opt/venv/lib/python3.12/site-packages/lm_eval/tasks/gsm8k/gsm8k.yaml"] = (
        pins["task_sha256"]
    )
    paths.update(
        {
            f"/opt/venv/lib/python3.12/site-packages/{name}": value
            for name, value in pins["client_source_files"].items()
        }
    )
    for name, wanted in paths.items():
        rows[name] = measured_hash(Path(name))
        require(rows[name]["sha256"] == wanted, f"runtime hash differs: {name}")
    expected = {**pins["package_versions"], "flydsl": "0.5.0"}
    versions = {name: importlib.metadata.version(name) for name in expected}
    result["versions"] = versions
    require(versions == expected, "runtime package versions differ")
    command = ["git", "-c", "safe.directory=/app/aiter-test", "-C", "/app/aiter-test"]
    head = subprocess.check_output(
        command + ["rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    require(head == pins["aiter"], "AITER source ref differs")
    subprocess.run(
        command + ["diff", "--exit-code", "--quiet", "HEAD"], check=True, timeout=10
    )
    result.update(
        aiter=head, scope="pinned metadata/client files and four native modules only"
    )


def check_model(view, cache, result=None):
    result = {} if result is None else result
    root = cache / view["repo"]
    require(
        root.resolve().is_relative_to(cache.resolve()),
        "model root escapes readonly cache",
    )
    require(
        (root / ".hf-revision").read_text().strip()
        == view["expected_component_revision"],
        "model revision differs",
    )
    require(
        (root / ".atom_download_complete").is_file(), "model completion marker missing"
    )
    rows = result.setdefault("files", [])
    result["repo"] = view["repo"]
    for item in view["files"]:
        path = root / item["name"]
        require(
            path.resolve().is_relative_to(cache.resolve()),
            "model backing escapes readonly cache",
        )
        row = {
            "name": item["name"],
            "path": str(path),
            "resolved": str(path.resolve()),
            "identity": identity(path),
        }
        rows.append(row)
        if item.get("actual_sha256"):
            row["measured"] = measured_hash(path)
            require(
                row["measured"]["sha256"] == item["actual_sha256"],
                "model nonweight hash differs",
            )
        else:
            require(
                row["identity"]["bytes"] == item["source"]["bytes"],
                "weight size differs",
            )
            metadata = (
                root / ".cache/huggingface/download" / (item["name"] + ".metadata")
            )
            require(
                metadata.resolve().is_relative_to(cache.resolve()),
                "metadata escapes readonly cache",
            )
            meta_id = identity(metadata)
            require(meta_id["bytes"] <= 4096, "oversize acquisition metadata")
            raw = metadata.read_bytes()
            require(identity(metadata) == meta_id, "acquisition metadata changed")
            lines = raw.decode().splitlines()
            require(
                len(lines) == 3 and float(lines[2]) > 0, "invalid acquisition metadata"
            )
            require(
                lines[0]
                in (
                    view["original_local_revision"],
                    view["expected_component_revision"],
                ),
                "acquisition revision differs",
            )
            wanted = item["expected_lfs_sha256"]
            require(lines[1].strip('"') == wanted, "acquisition LFS identity differs")
            row.update(
                expected_lfs_sha256=wanted,
                acquisition_metadata=lines,
                metadata_sha256=hashlib.sha256(raw).hexdigest(),
                measured_payload_sha256=None,
            )
            if view["repo"].startswith("Inferact/"):
                row["measured"] = measured_hash(path)
                require(
                    row["measured"]["sha256"] == wanted, "draft payload hash differs"
                )
                row["measured_payload_sha256"] = row["measured"]["sha256"]
            require(
                identity(path) == row["identity"],
                "weight identity changed during inspection",
            )
    result["payload_bytes_read"] = sum(
        r.get("measured", {}).get("bytes_read", 0)
        for r in rows
        if "expected_lfs_sha256" in r
    )
    return result


def inspect(seed, spec_path, image_config, output):
    report = {
        "status": "failed",
        "model_execution_authorized": False,
        "started_epoch": time.time(),
        "image_configuration": image_config,
    }
    try:
        spec = json.loads(spec_path.read_text())
        checked_files(seed, spec)
        require(image_config == spec["image"]["config"], "OCI configuration differs")
        require(
            not Path("/dev/kfd").exists() and not Path("/dev/dri").exists(),
            "GPU devices exposed to Phase0",
        )
        for path in (seed, Path("/models")):
            require(
                os.statvfs(path).f_flag & os.ST_RDONLY, f"input is writable: {path}"
            )
        pins = json.loads((seed / "C/payload/pins.json").read_text())
        report.update(
            image_manifest=spec["image"]["manifest"],
            legacy_recorded_image_label=pins["image"],
        )
        report["runtime"] = {}
        check_runtime(pins, spec, report["runtime"])
        composition = json.loads(
            (seed / "C/payload/composition-manifest.json").read_text()
        )
        report["models"] = []
        for view in composition["views"]:
            result = {}
            report["models"].append(result)
            check_model(view, Path("/models"), result)
        report["status"] = (
            "passed scoped Phase0 inspection; model execution remains unauthorized"
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_epoch"] = time.time()
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--image-config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.seed, args.spec, args.image_config, args.output)
