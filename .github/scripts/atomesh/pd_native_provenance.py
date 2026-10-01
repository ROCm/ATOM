"""Fail closed on incompatible native-build dependencies or image extensions."""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
from importlib import metadata
from pathlib import Path

import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def requirements(path):
    for line in path.read_text().splitlines():
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            yield from requirements(path.parent / line[3:])
        else:
            yield Requirement(line)


def dependency_errors(reqs, versions):
    errors = []
    for req in reqs:
        if req.marker and not req.marker.evaluate({"extra": ""}):
            continue
        version = versions.get(canonicalize_name(req.name))
        if version is None or not req.specifier.contains(version, prereleases=True):
            errors.append(f"{req}: installed={version}")
    return errors


def active_versions():
    names = {
        canonicalize_name(dist.metadata["Name"]) for dist in metadata.distributions()
    }
    return {name: metadata.version(name) for name in sorted(names)}


def source_requirements(source):
    project = tomllib.loads((source / "pyproject.toml").read_text())
    reqs = list(requirements(source / "requirements/rocm.txt"))
    reqs.extend(Requirement(r) for r in project["build-system"]["requires"])
    reqs.extend(map(Requirement, ["triton>=3.8,<3.9", "amd-aiter>=0.1.23"]))
    return reqs


def active_dependency_errors(root="vllm", *, roots=None):
    pending = (
        [(root, frozenset())]
        if roots is None
        else [
            (req.name, frozenset(req.extras))
            for req in roots
            if not req.marker or req.marker.evaluate({"extra": ""})
        ]
    )
    seen, errors = set(), []
    while pending:
        name, extras = pending.pop()
        key = (canonicalize_name(name), extras)
        if key in seen:
            continue
        seen.add(key)
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            errors.append(f"Missing active dependency: {name}")
            continue
        for value in dist.requires or ():
            req = Requirement(value)
            if req.marker and not any(
                req.marker.evaluate({"extra": extra}) for extra in {"", *extras}
            ):
                continue
            try:
                version = metadata.version(req.name)
            except metadata.PackageNotFoundError:
                errors.append(f"{name} requires {req}: not installed")
                continue
            if not req.specifier.contains(version, prereleases=True):
                errors.append(f"{name} requires {req}: active={version}")
            pending.append((req.name, frozenset(req.extras)))
    return errors


def source_direct_url(dist, source):
    direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct_url.get("url") != source.resolve().as_uri():
        raise RuntimeError(
            f"vllm direct_url does not match native source: {direct_url}"
        )
    if direct_url.get("dir_info", {}).get("editable", False):
        raise RuntimeError(
            "Expected a native wheel installation, not an editable overlay"
        )
    return direct_url


def native_paths(prefix):
    paths = {}
    for name in ("vllm", "vllm._C"):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to(prefix.resolve()):
            raise RuntimeError(f"{name} loaded outside native build: {path}")
        paths[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "manifest"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", type=Path)
    args = parser.parse_args()
    versions = active_versions()
    report = {
        "packages": versions,
        "source_sha": subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    # Validate the complete source/build-rooted active closure both before and
    # after installation. Python sees inherited image packages that uv's resolver
    # does not; --no-deps is safe only behind this fail-closed compatibility gate.
    reqs = source_requirements(args.source)
    errors = dependency_errors(reqs, versions)
    errors.extend(active_dependency_errors(roots=reqs))
    report["source_dependency_errors"] = errors
    if errors:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError(f"Source dependency closure failed: {errors}")
    if args.mode == "preflight":
        torch = importlib.import_module("torch")
        if not torch.version.hip:
            errors.append("torch is not a ROCm build")
        report.update(
            errors=errors, hip=torch.version.hip, torch_git=torch.version.git_version
        )
        version_file = Path("/app/versions.txt")
        report["image_versions"] = (
            version_file.read_text() if version_file.exists() else None
        )
        for name in ("aiter", "mori", "triton", "transformers"):
            try:
                importlib.import_module(name)
            except (ImportError, OSError, RuntimeError) as exc:
                errors.append(f"{name} import failed: {exc!r}")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if errors:
            raise SystemExit(
                "Incompatible image; rebuild pinned dependencies, do not "
                "blanket-upgrade: " + "; ".join(errors)
            )
        constraints = args.output.with_suffix(".constraints.txt")
        constraints.write_text(
            "".join(
                f"{name}=={version}\n"
                for name, version in sorted(versions.items())
                if name != "vllm"
            )
        )
    else:
        if args.prefix is None:
            parser.error("manifest requires --prefix")
        report["modules"] = native_paths(args.prefix)
        dist = metadata.distribution("vllm")
        report["direct_url"] = source_direct_url(dist, args.source)
        report["dependency_errors"] = active_dependency_errors()
        if report["dependency_errors"]:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            raise RuntimeError(
                f"Native dependency closure failed: {report['dependency_errors']}"
            )
        report["image"] = os.environ.get("DOCKER_IMAGE")
        # Bind the native recipe to reviewed v0.1.23 dispatch and K3 tuning,
        # without imposing obsolete image-specific quantization implementations.
        aiter_root = Path(importlib.import_module("aiter").__file__).resolve().parent
        expected = {
            "fused_moe.py": "a1d6fe167f2eb70c21218a28a8eb2e2ab125bf8988e24f76880ba9218345e877",
            "configs/model_configs/kimik3_a4w4_tuned_fmoe.csv": "b88cc8aba72e2d37d05dbfd7b42bb34891f4ec84972780861c9a043d7df5ad7a",
        }
        actual = {
            name: hashlib.sha256((aiter_root / name).read_bytes()).hexdigest()
            for name in expected
        }
        report["aiter"] = {
            "root": str(aiter_root),
            "source_sha": "50da036acdedec2dd596f93188d6c615e2561672",
            "hashes": actual,
            "expected_hashes": expected,
        }
        ops = importlib.import_module("vllm._aiter_ops").rocm_aiter_ops
        report["aiter"]["activation"] = ops.get_fused_moe_situv2_activation()
        report["aiter"]["gate_up_interleaved"] = (
            ops.is_fused_moe_situv2_gate_up_interleaved()
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if actual != expected or report["aiter"]["activation"] != "a4w4":
            raise RuntimeError(
                "Native AITER source/activation differs from reviewed recipe"
            )
        if report["aiter"]["gate_up_interleaved"]:
            raise RuntimeError("K3 A4W4 requires separated gate/up layout")
        report["requirements_sha256"] = {
            str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.source / "requirements").rglob("*.txt")
        }
        importlib.import_module(
            "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector"
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
