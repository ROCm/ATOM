# SPDX-License-Identifier: MIT
"""Versioned cell bundles: identity, integrity, packaging and offline rebuild."""

import copy
import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path

from . import AGGREGATION_VERSION, SCHEMA_VERSION
from .io import (
    artifact_path,
    fingerprint,
    iter_jsonl,
    read_json,
    sha256,
    write_json,
    write_view,
)

SUMMARY_FILES = {
    "config.json",
    "summary.json",
    "validation.json",
    "views/overview.json",
    "exports/inferencex-v3.json",
}


def recipe_fingerprint(config, version=AGGREGATION_VERSION):
    if version not in ("1.0.0", "1.0.1", "1.0.2"):
        raise ValueError(f"Unsupported aggregation version: {version}")
    # Physical card IDs locate evidence but do not define a curve.
    # Software versions, graph settings and workload settings do define one.
    recipe = copy.deepcopy(config["recipe"])
    recipe.pop("client_argv", None)  # contains per-cell conc and output paths
    if version == "1.0.0":
        recipe.pop("client_environment", None)
    # Client settings include replay policy and timeouts absent from workload.
    # Both environments contain only the captured performance-key allowlist.
    for key in ("environment", "client_environment"):
        for variable in (
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        ):
            recipe.get(key, {}).pop(variable, None)
    return fingerprint(
        {
            "model": config["model"],
            "parallelism": config["parallelism"],
            "hardware": {
                k: config["hardware"].get(k) for k in ("key", "name", "gpu_count")
            },
            "software": config["software"],
            "recipe": recipe,
            "validity": config["validity"],
        }
    )


def config_errors(config):
    errors = []
    if not isinstance(config, dict):
        return ["config must be an object"]
    if config.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    for section in (
        "source",
        "software",
        "model",
        "hardware",
        "parallelism",
        "workload",
        "recipe",
        "validity",
    ):
        if not isinstance(config.get(section), dict):
            errors.append(f"missing object: {section}")
    if errors:
        return errors
    for section, key in (
        ("hardware", "gpu_count"),
        ("workload", "concurrency"),
        ("source", "attempt"),
    ):
        n = config[section].get(key)
        if type(n) is not int or n <= 0:
            errors.append(f"{section}.{key} must be a positive integer")
    if config["workload"].get("kind") not in ("random", "agentic"):
        errors.append("workload.kind must be random or agentic")
    for key in ("repository", "run_id"):
        if not isinstance(config["source"].get(key), str) or not config["source"][key]:
            errors.append(f"source.{key} must be a nonempty string")
    for key in ("tp", "pp", "dp", "ep"):
        value = config["parallelism"].get(key)
        if type(value) is not int or value <= 0:
            errors.append(f"parallelism.{key} must be a positive integer")
    cards = config["hardware"].get("devices", [])
    if not isinstance(cards, list) or any(not isinstance(card, dict) for card in cards):
        errors.append("hardware.devices must be an array of objects")
    for key in ("server_argv", "client_argv"):
        argv = config["recipe"].get(key, [])
        if not isinstance(argv, list) or any(not isinstance(arg, str) for arg in argv):
            errors.append(f"recipe.{key} must be an array of strings")
    return errors


def completeness_errors(config):
    errors = []
    for section, keys in {
        "model": ("name", "precision", "revision"),
        "software": ("atom_sha", "aiter_sha", "image_digest", "harness_version"),
    }.items():
        for key in keys:
            if not config[section].get(key):
                errors.append(f"missing {section}.{key}")
    cards = config["hardware"].get("devices", [])
    identities = [card.get("pci_bus_id") for card in cards]
    if (
        len(cards) != config["hardware"]["gpu_count"]
        or not all(identities)
        or len(set(identities)) != len(cards)
    ):
        errors.append("allocated GPU identities incomplete")
    if any(not card.get("uuid") for card in cards):
        errors.append("allocated GPU UUIDs missing")
    if not config["recipe"].get("server_argv") or not config["recipe"].get(
        "client_argv"
    ):
        errors.append("executed server/client argv missing")
    if config["workload"]["kind"] == "agentic":
        for key in ("dataset",):
            if not config["workload"].get(key):
                errors.append(f"missing workload.{key}")
        if not config["workload"].get("dataset_revision") and not config[
            "workload"
        ].get("dataset_content_sha256"):
            errors.append("missing workload.dataset_revision or dataset_content_sha256")
        if not config["software"].get("harness_sha"):
            errors.append("missing software.harness_sha")
    else:
        for key in ("isl", "osl"):
            if not config["workload"].get(key) or config["workload"][key] < 0:
                errors.append(f"missing positive workload.{key}")
    return errors


def _copy_file(src, dst):
    src, dst = Path(src), Path(dst)
    if not src.is_file() or src.is_symlink():
        raise ValueError(f"Not a regular source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix == ".gz" and src.suffix != ".gz":
        with open(src, "rb") as f, gzip.open(dst, "wb") as out:
            shutil.copyfileobj(f, out)
    else:
        shutil.copyfile(src, dst)


def build_bundle(
    config,
    records_path,
    output,
    *,
    record_format="atom",
    raw_dir=None,
    telemetry_dir=None,
    logs=None,
    exit_code=0,
    require_full=False,
    client_launch=None,
    harness_summary=None,
):
    from .aggregate import (
        COMPATIBILITY_COMMIT,
        aggregate,
        inferencex_export,
        metadata_errors,
    )
    from .metadata import resolve_client_config, synthetic_settings
    from .records import from_aiperf
    from .telemetry import attach_telemetry

    config = copy.deepcopy(config)
    actual = synthetic_settings(
        config["recipe"].get("server_argv", []), config["recipe"].get("environment", {})
    )
    # Old bundles may lack the original declaration or use another synthetic
    # mechanism. Rebuild must not erase their existing conservative marker.
    actual["synthetic"] |= bool(config["validity"].get("synthetic"))
    config["validity"].update(actual)
    resolve_client_config(config, client_launch)
    problems = config_errors(config)
    if problems:
        raise ValueError("; ".join(problems))
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite measurement: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=output.parent))
    try:
        missing = []
        measurement_errors = []
        if records_path and Path(records_path).exists():
            rows = iter_jsonl(records_path)
            if record_format == "aiperf":
                rows = (from_aiperf(row, i) for i, row in enumerate(rows))
        else:
            rows = iter(())
            missing.append("raw request records missing")

        def diagnosed_records():
            try:
                yield from rows
            except (OSError, ValueError, TypeError) as exc:
                measurement_errors.append(f"request parsing failed: {exc}")

        summary = aggregate(diagnosed_records(), config, temporary)
        harness = (
            read_json(harness_summary)
            if harness_summary and Path(harness_summary).is_file()
            else None
        )
        if harness:
            write_json(temporary / "raw" / "harness-summary.json", harness)
        if records_path and Path(records_path).is_file():
            name = (
                "profile_export.jsonl.gz"
                if record_format == "aiperf"
                else "requests.jsonl.gz"
            )
            _copy_file(records_path, temporary / "raw" / name)
        if raw_dir:
            raw_dir = Path(raw_dir)
            # Whitelist files; copying a dataset/cache/model directory is never necessary.
            for name in (
                "profile_export_aiperf.json",
                "server_metrics_export.json",
                "server_metrics_export.csv",
                "inputs.json",
                "dataset-identity.json",
                "profile_export_raw.jsonl",
                "profile_export_aiperf_timeslices.json",
                "aiperf.log",
            ):
                if (raw_dir / name).is_file():
                    _copy_file(
                        raw_dir / name, temporary / "raw" / "aiperf" / (name + ".gz")
                    )
        for label, path in (logs or {}).items():
            if label not in ("client", "server", "collector"):
                raise ValueError(f"Unknown log label: {label}")
            if Path(path).is_file():
                _copy_file(path, temporary / "logs" / f"{label}.log.gz")
        if telemetry_dir:
            for name in ("gpu_samples.jsonl", "server_samples.jsonl"):
                src = Path(telemetry_dir) / name
                if src.is_file():
                    _copy_file(src, temporary / "telemetry" / (name + ".gz"))
            attach_telemetry(summary, config, Path(telemetry_dir), temporary)
        has_server = bool(list((temporary / "raw").rglob("server_metrics_export*")))
        summary["capabilities"]["server_metrics_raw"] = (
            has_server or (temporary / "telemetry/server_samples.jsonl.gz").is_file()
        )
        if raw_dir and (Path(raw_dir) / "profile_export_aiperf.json").is_file():
            aiperf = read_json(Path(raw_dir) / "profile_export_aiperf.json")
            run_meta = aiperf.get("metadata", aiperf)
            # Keep the producer's verdict; an unsafe run may still have complete evidence.
            for field in ("submission_valid", "submission_invalid_reasons"):
                if field in run_meta:
                    config["validity"][field] = run_meta[field]
            config["workload"]["dataset_provenance"] = run_meta.get("dataset")
            inputs = Path(raw_dir) / "inputs.json"
            if inputs.is_file():
                config["workload"]["dataset_content_sha256"] = sha256(inputs)
            identity_path = Path(raw_dir) / "dataset-identity.json"
            if identity_path.is_file():
                identity = read_json(identity_path)
                if identity.get("dataset_provenance") != run_meta.get("dataset"):
                    raise ValueError(
                        "Dataset identity does not match the AIPerf export"
                    )
                files = identity.get("files", [])
                digest = identity.get("dataset_content_sha256")
                if not files or fingerprint(files) != digest:
                    raise ValueError("Invalid dataset content ledger")
                config["workload"]["dataset_content_sha256"] = digest
            if type(config["validity"].get("submission_valid")) is not bool:
                measurement_errors.append(
                    "AIPerf submission validity missing or malformed"
                )
            # Compare retained records against the harness's profiling counter.
            count = aiperf.get("request_count", {})
            expected = count.get("avg") if isinstance(count, dict) else count
            if expected is not None and expected != summary["accounting"]["counts"].get(
                "profiled_success", 0
            ):
                measurement_errors.append(
                    "AIPerf successful request count disagrees with retained records"
                )
        write_json(temporary / "config.json", config)
        counts = summary["accounting"]["counts"]
        if not counts.get("profiled_success"):
            missing.append("no eligible successful profiling requests")
        if counts.get("invalid"):
            missing.append("invalid request records")
        if summary["accounting"]["phases"].get("unknown"):
            missing.append("request phase missing")
        if not summary["capabilities"]["full_response_interactivity"]:
            missing.append("full-response timing unavailable")
        if config["workload"]["kind"] == "agentic":
            if not summary["capabilities"]["session_timeline"]:
                missing.append("session identity missing")
            if not has_server:
                missing.append("AIPerf server metrics export missing")
            if (
                not raw_dir
                or not (Path(raw_dir) / "profile_export_aiperf.json").is_file()
            ):
                missing.append("AIPerf aggregate export missing")
        if exit_code:
            missing.append(f"benchmark exit code {exit_code}")
        missing[:0] = completeness_errors(config) + measurement_errors
        # Recipe identity omits concurrency, provenance and phase timing, but includes
        # model/runtime/graph settings and workload definition to prevent mixed curves.
        workload_scope = {
            k: v
            for k, v in config["workload"].items()
            if k not in ("concurrency", "num_prompts", "warmup_requests")
        }
        recipe_id = recipe_fingerprint(config)
        point_id = fingerprint(
            {
                "source": config["source"],
                "recipe": recipe_id,
                "config": config,
                "requests": summary["requests_sha256"],
            }
        )
        summary.update(
            point_id=point_id,
            recipe_fingerprint=recipe_id,
            curve_group_id=fingerprint(
                {"recipe": recipe_id, "workload": workload_scope}
            ),
        )
        valid = not config["validity"].get("unsafe_override")
        valid = valid and config["validity"].get("submission_valid") is not False
        validation = {
            "complete": not missing,
            "missing": missing,
            "exit_code": exit_code,
            "measurement_valid": valid
            and not missing
            and not measurement_errors
            and not counts.get("invalid")
            and exit_code == 0
            and bool(counts.get("profiled_success")),
            "validity": config["validity"],
            "requested_full": require_full,
            "export_errors": {"inferencex": metadata_errors(config)},
        }
        validation["submission_eligible"] = (
            validation["measurement_valid"]
            and not config["validity"].get("synthetic")
            and not config["validity"].get("instrumented")
        )
        write_json(temporary / "validation.json", validation)
        write_json(temporary / "summary.json", summary)
        write_view(
            temporary / "views" / "overview.json",
            {"config": config, "summary": summary, "validation": validation},
            summary,
        )
        export = inferencex_export(config, summary, recipe_id)
        if export is not None:
            export["benchmark_outcome"] = {
                "status": "success" if validation["measurement_valid"] else "failed"
            }
            export["submission_valid"] = validation["submission_eligible"]
            write_json(temporary / "exports" / "inferencex-v3.json", export)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "aggregation_version": AGGREGATION_VERSION,
            "source": config["source"],
            "point_id": point_id,
            "recipe_fingerprint": recipe_id,
            "compatibility_commit": COMPATIBILITY_COMMIT,
            "curve_group_id": summary["curve_group_id"],
            "config_id": fingerprint(config),
            "status": (
                "failed"
                if exit_code or not counts.get("profiled_success")
                else "complete" if not missing else "partial"
            ),
            "capabilities": summary["capabilities"],
            "files": {},
        }
        manifest["availability"] = {
            key: {"status": "available" if available else "not_collected"}
            for key, available in summary["capabilities"].items()
        }
        if not summary["capabilities"]["power"]:
            manifest["availability"]["power"] = {
                "status": "failed" if "power" in summary else "not_collected",
                "reasons": summary.get("power", {}).get(
                    "reasons", ["no GPU telemetry"]
                ),
            }
        if config["workload"]["kind"] == "random":
            manifest["availability"]["session_timeline"] = {
                "status": "unsupported",
                "reasons": ["random requests have no sessions"],
            }
        cell = config["source"].get("cell_id")
        if cell and config["source"]["run_id"] != "local":
            suffix = f"{config['source']['attempt']}-{cell}"
            manifest["artifacts"] = {
                "summary": f"atom-benchmark-summary-v1-{suffix}",
                "full": f"atom-benchmark-bundle-v1-{suffix}",
            }
        for path in sorted(temporary.rglob("*")):
            path.chmod(0o755 if path.is_dir() else 0o644)
            if path.is_file():
                manifest["files"][path.relative_to(temporary).as_posix()] = {
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
        write_json(temporary / "manifest.json", manifest)
        temporary.chmod(0o755)  # Container output must be readable by the CI host.
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_bundle(root, summary_only=False):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported bundle schema")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(  # noqa: TRY004 - invalid bundle format
            "Missing file manifest"
        )
    required = {
        "config.json",
        "summary.json",
        "validation.json",
        "requests/requests.jsonl.gz",
        "views/overview.json",
    }
    if not required <= files.keys():
        raise ValueError(f"Missing required files: {sorted(required - files.keys())}")
    for name, expected in files.items():
        path = artifact_path(root, name)
        if summary_only and name not in SUMMARY_FILES:
            continue
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or sha256(path) != expected["sha256"]
        ):
            raise ValueError(f"Checksum/size mismatch: {name}")
    config = read_json(root / "config.json")
    problems = config_errors(config)
    if problems:
        raise ValueError("; ".join(problems))
    summary = read_json(root / "summary.json")
    if summary.get("point_id") != manifest.get("point_id"):
        raise ValueError("Point identity mismatch")
    recipe_id = recipe_fingerprint(config, manifest.get("aggregation_version"))
    expected_point = fingerprint(
        {
            "source": config["source"],
            "recipe": recipe_id,
            "config": config,
            "requests": summary["requests_sha256"],
        }
    )
    if expected_point != manifest.get("point_id") or fingerprint(
        config
    ) != manifest.get("config_id"):
        raise ValueError("Point/config fingerprint mismatch")
    if (
        config["source"] != manifest["source"]
        or recipe_id != manifest["recipe_fingerprint"]
    ):
        raise ValueError("Source/recipe identity mismatch")
    validation = read_json(root / "validation.json")
    if (manifest["status"] == "complete") != (validation["complete"] is True):
        raise ValueError("Manifest status contradicts completeness report")
    if validation["complete"] and (
        completeness_errors(config) or validation["missing"]
    ):
        raise ValueError("Incomplete metadata cannot be marked complete")
    if not summary_only:
        import hashlib

        digest = hashlib.sha256()
        count = 0
        for row in iter_jsonl(root / "requests/requests.jsonl.gz"):
            count += 1
            digest.update(
                (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
            )
        if (
            count != summary["accounting"]["counts"].get("total", 0)
            or digest.hexdigest() != summary["requests_sha256"]
        ):
            raise ValueError("Request count/source hash mismatch")
    return manifest


def summary_package(root, output):
    root, output = Path(root), Path(output)
    manifest = verify_bundle(root, summary_only=True)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for name in SUMMARY_FILES & manifest["files"].keys():
        _copy_file(root / name, output / name)
    # Keep the complete manifest so raw/detail hashes remain discoverable.
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "package.json",
        {"kind": "summary", "files": sorted(SUMMARY_FILES & manifest["files"].keys())},
    )


def run_index(root, output, summary_only=False, plan=None):
    entries = []
    for path in sorted(Path(root).rglob("manifest.json")):
        try:
            manifest = verify_bundle(path.parent, summary_only=summary_only)
            entries.append(
                {
                    "path": path.parent.relative_to(root).as_posix(),
                    **{
                        k: manifest[k]
                        for k in ("source", "point_id", "status", "capabilities")
                    },
                }
            )
        except (ValueError, OSError, KeyError) as exc:
            entries.append(
                {
                    "path": path.parent.relative_to(root).as_posix(),
                    "status": "corrupt",
                    "error": str(exc),
                }
            )
    planned = read_json(plan)["cells"] if plan else []
    found = {entry.get("source", {}).get("cell_id") for entry in entries}
    missing = [cell for cell in planned if cell["cell_id"] not in found]
    write_json(
        output,
        {
            "schema_version": SCHEMA_VERSION,
            "cells": entries,
            "missing_cells": missing,
            "planned_count": len(planned),
            "uploaded_count": len(entries),
        },
    )
    return entries


def rebuild_bundle(root, output):
    """Recompute using retained originals; no model, GPU, network or database."""
    root = Path(root)
    verify_bundle(root)
    config = read_json(root / "config.json")
    validation = read_json(root / "validation.json")
    with tempfile.TemporaryDirectory(prefix="atom-rebuild-") as staging:
        evidence = Path(staging)
        for path in root.rglob("*.gz"):
            relative = path.relative_to(root)
            if relative.parts[0] not in ("raw", "telemetry", "logs"):
                continue
            target = evidence / relative.with_suffix("")
            target.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "rb") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        aiperf = (evidence / "raw/profile_export.jsonl").is_file()
        records = evidence / (
            "raw/profile_export.jsonl" if aiperf else "raw/requests.jsonl"
        )
        return build_bundle(
            config,
            records if records.exists() else None,
            output,
            record_format="aiperf" if aiperf else "atom",
            raw_dir=evidence / "raw/aiperf",
            telemetry_dir=evidence / "telemetry",
            logs={
                k: evidence / f"logs/{k}.log" for k in ("client", "server", "collector")
            },
            harness_summary=root / "raw/harness-summary.json",
            exit_code=validation["exit_code"],
            require_full=validation["requested_full"],
        )
