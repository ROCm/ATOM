#!/usr/bin/env python3
"""Convert AIPerf profile_export.jsonl artifacts into Perfetto/Chrome traces.

Used by benchmark workflows after a client run so CI can upload a dedicated
artifact and print download links. Stdlib only: summarize jobs run on
ubuntu-latest without extra Python deps.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

RECORDS_FILENAME = "profile_export.jsonl"
AIPERF_SUMMARY_FILENAME = "profile_export_aiperf.json"
CONVERTER_PATH = Path(__file__).resolve().with_name("aiperf_to_chrome_trace.py")
DIR_CONCURRENCY_RE = re.compile(r"-c(\d+)$")


def load_converter(path: Path | None = None) -> Any:
    converter = path or CONVERTER_PATH
    if not converter.is_file():
        raise FileNotFoundError(f"AIPerf chrome-trace converter not found: {converter}")
    spec = importlib.util.spec_from_file_location("aiperf_to_chrome_trace", converter)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load converter from {converter}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 1:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 1:
        return int(value)
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed >= 1 else None
    return None


def concurrency_from_aiperf_summary(path: Path) -> int | None:
    """Prefer the configured profiling concurrency from AIPerf's JSON summary."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    nested = []
    for key in ("input_config", "config", "metadata"):
        value = data.get(key)
        if isinstance(value, dict):
            nested.append(value)
    for blob in (data, *nested):
        for key in ("concurrency", "max_concurrency", "configured_concurrency"):
            parsed = _positive_int(blob.get(key))
            if parsed is not None:
                return parsed
    return None


def infer_concurrency(jsonl: Path) -> int | None:
    summary = jsonl.with_name(AIPERF_SUMMARY_FILENAME)
    if summary.is_file():
        parsed = concurrency_from_aiperf_summary(summary)
        if parsed is not None:
            return parsed
    match = DIR_CONCURRENCY_RE.search(jsonl.parent.name)
    if match:
        return int(match.group(1))
    return None


def trace_stem(jsonl: Path, concurrency: int | None) -> str:
    if concurrency is None:
        return f"{jsonl.stem}.trace.json"
    return f"{jsonl.stem}.c{concurrency}.slots.trace.json"


def unique_trace_name(jsonl: Path, concurrency: int | None) -> str:
    """Prefix the artifact directory so merged CI uploads do not collide."""
    return f"{jsonl.parent.name}.{trace_stem(jsonl, concurrency)}"


def iter_profile_exports(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob(RECORDS_FILENAME) if path.is_file())


def write_gzip(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as reader, gzip.open(dst, "wb", compresslevel=6) as writer:
        shutil.copyfileobj(reader, writer)


def find_existing_trace(jsonl: Path, concurrency: int | None) -> Path | None:
    stem = jsonl.with_name(trace_stem(jsonl, concurrency))
    for candidate in (Path(str(stem) + ".gz"), stem):
        if candidate.is_file():
            return candidate
    return None


def _result_payload(
    jsonl: Path,
    destination: Path,
    concurrency: int | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    return {
        "input": str(jsonl),
        "output": str(destination),
        "concurrency": concurrency,
        "record_count": metadata.get("record_count"),
        "session_count": metadata.get("session_count"),
        "slot_count": metadata.get("slot_count"),
        "tree_count": metadata.get("tree_count"),
        "phase": metadata.get("phase"),
        "bytes": destination.stat().st_size,
    }


def _gzip_destination(path: Path) -> Path:
    return path if str(path).endswith(".gz") else Path(str(path) + ".gz")


def convert_export(
    jsonl: Path,
    *,
    converter: Any,
    output_dir: Path | None,
    gzip_output: bool,
) -> dict[str, Any]:
    concurrency = infer_concurrency(jsonl)
    if output_dir is not None:
        destination = output_dir / unique_trace_name(jsonl, concurrency)
    else:
        destination = jsonl.with_name(trace_stem(jsonl, concurrency))
    destination.parent.mkdir(parents=True, exist_ok=True)

    existing = find_existing_trace(jsonl, concurrency)
    if existing is not None:
        if output_dir is None:
            return _result_payload(jsonl, existing, concurrency)
        if gzip_output and existing.suffix != ".gz":
            destination = _gzip_destination(destination)
            write_gzip(existing, destination)
        else:
            if gzip_output:
                destination = _gzip_destination(destination)
            shutil.copy2(existing, destination)
        return _result_payload(jsonl, destination, concurrency)

    records = converter.load_records(jsonl)
    trace = converter.build_trace(
        records,
        phase=None,
        concurrency=concurrency,
        inspect_all_slots=concurrency is not None,
    )
    destination.write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
    if gzip_output:
        gzip_path = _gzip_destination(destination)
        write_gzip(destination, gzip_path)
        destination.unlink()
        destination = gzip_path
    return _result_payload(jsonl, destination, concurrency, trace.get("metadata", {}))


def write_index(
    results: list[dict[str, Any]],
    index_path: Path,
    *,
    artifact_url: str | None,
    run_url: str | None,
) -> None:
    lines = [
        "### AIPerf Chrome Traces",
        "",
        (
            "Download the `*.trace.json.gz` files from this run, then open them in "
            "[Perfetto UI](https://ui.perfetto.dev) (`Open trace file`) or "
            "`chrome://tracing` (gunzip first for Chrome)."
        ),
        "",
    ]
    if artifact_url:
        lines.append(f"- Artifact download: {artifact_url}")
    if run_url:
        lines.append(f"- Workflow run (all artifacts): {run_url}")
    if artifact_url or run_url:
        lines.append("")
    if not results:
        lines.append(
            "No `profile_export.jsonl` files were found; no traces were generated."
        )
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines.extend(
        [
            "| Trace | Concurrency | Requests | Trees | Size (bytes) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in results:
        lines.append(
            "| `{name}` | {conc} | {records} | {trees} | {size} |".format(
                name=Path(item["output"]).name,
                conc=item["concurrency"] if item["concurrency"] is not None else "--",
                records=(
                    item["record_count"] if item["record_count"] is not None else "--"
                ),
                trees=item["tree_count"] if item["tree_count"] is not None else "--",
                size=item["bytes"],
            )
        )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_dir", type=Path, help="Directory containing AIPerf artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write traces here (default: next to each profile_export.jsonl)",
    )
    parser.add_argument(
        "--index",
        type=Path,
        help="Markdown index path (default: <output-dir or result-dir>/TRACE_INDEX.md)",
    )
    parser.add_argument(
        "--converter", type=Path, help="Override aiperf_to_chrome_trace.py path"
    )
    parser.add_argument(
        "--run-url", default=None, help="Workflow run URL for the index"
    )
    parser.add_argument(
        "--artifact-url", default=None, help="Artifact download URL for the index"
    )
    parser.add_argument(
        "--no-gzip", action="store_true", help="Leave traces uncompressed"
    )
    args = parser.parse_args()

    root = args.result_dir
    if not root.is_dir():
        print(f"skip chrome traces: result directory not found: {root}")
        return 0

    converter = load_converter(args.converter)
    exports = iter_profile_exports(root)
    results: list[dict[str, Any]] = []
    failures = 0
    for jsonl in exports:
        try:
            results.append(
                convert_export(
                    jsonl,
                    converter=converter,
                    output_dir=args.output_dir,
                    gzip_output=not args.no_gzip,
                )
            )
            print(f"wrote {results[-1]['output']}")
        except (OSError, TypeError, ValueError) as exc:
            failures += 1
            print(f"WARNING: failed to convert {jsonl}: {exc}", file=sys.stderr)

    index_path = args.index
    if index_path is None:
        base = args.output_dir if args.output_dir is not None else root
        index_path = base / "TRACE_INDEX.md"
    write_index(
        results,
        index_path,
        artifact_url=args.artifact_url,
        run_url=args.run_url,
    )
    print(
        f"converted {len(results)}/{len(exports)} AIPerf export(s); index={index_path}"
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
