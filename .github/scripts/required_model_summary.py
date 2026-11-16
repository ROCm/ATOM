#!/usr/bin/env python3
"""Write required model test lists to GitHub Actions summaries."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any


LEVEL_BY_EVENT = {
    "push": "main",
    "schedule": "nightly",
    "workflow_dispatch": "nightly",
}

ALLOWED_LEVELS = {
    "pr": {"pr"},
    "main": {"pr", "main"},
    "nightly": {"pr", "main", "nightly"},
}

SUMMARY_KEYS = (
    "display_name",
    "model_name",
    "model_path",
    "runner",
    "test_level",
    "accuracy_test_threshold",
    "accuracy_threshold",
)


def event_level(event_name: str) -> str:
    return LEVEL_BY_EVENT.get(event_name, "pr")


def load_catalog_models(
    catalog_path: Path,
    event_name: str,
) -> tuple[list[dict[str, Any]], str, int]:
    models = json.loads(catalog_path.read_text(encoding="utf-8"))
    level = event_level(event_name)
    allowed = ALLOWED_LEVELS[level]
    filtered = [m for m in models if m.get("test_level", "nightly") in allowed]
    return filtered, level, len(models)


def load_workflow_matrix(workflow_path: Path, job_name: str) -> list[dict[str, Any]]:
    text = workflow_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return parse_workflow_matrix_text(text, job_name)

    workflow = yaml.safe_load(text)
    return workflow["jobs"][job_name]["strategy"]["matrix"].get("include", [])


def parse_workflow_matrix_text(text: str, job_name: str) -> list[dict[str, Any]]:
    """Parse the simple matrix.include shape used by the accuracy workflows."""

    lines = text.splitlines()
    job_start = None
    job_indent = None
    job_pattern = re.compile(rf"^(\s*){re.escape(job_name)}:\s*$")

    for index, line in enumerate(lines):
        match = job_pattern.match(line)
        if match:
            job_start = index + 1
            job_indent = len(match.group(1))
            break

    if job_start is None or job_indent is None:
        raise ValueError(f"Could not find job {job_name!r}")

    job_lines: list[tuple[int, str]] = []
    for line in lines[job_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            job_lines.append((_indent(line), line))
            continue
        indent = _indent(line)
        if indent <= job_indent:
            break
        job_lines.append((indent, line))

    include_index = None
    include_indent = None
    for index, (indent, line) in enumerate(job_lines):
        if line.strip() == "include:":
            include_index = index + 1
            include_indent = indent
            break

    if include_index is None or include_indent is None:
        raise ValueError(f"Could not find matrix.include for job {job_name!r}")

    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    item_indent: int | None = None
    block_scalar_indent: int | None = None

    for indent, line in job_lines[include_index:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if indent <= include_indent:
            break

        if block_scalar_indent is not None and indent > block_scalar_indent:
            continue
        block_scalar_indent = None

        if stripped.startswith("- "):
            if current is not None:
                rows.append(current)
            current = {}
            item_indent = indent
            _parse_key_value_into(current, stripped[2:])
            if _is_block_scalar(stripped[2:]):
                block_scalar_indent = indent
            continue

        if current is None or item_indent is None or indent <= item_indent:
            continue

        parsed = _parse_key_value_into(current, stripped)
        if parsed and _is_block_scalar(stripped):
            block_scalar_indent = indent

    if current is not None:
        rows.append(current)

    if not rows:
        raise ValueError(f"No matrix.include entries found for job {job_name!r}")

    return rows


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parse_key_value_into(row: dict[str, Any], text: str) -> bool:
    match = re.match(r"([A-Za-z_][\w-]*):(?:\s+(.*)|\s*)$", text)
    if not match:
        return False
    key, raw_value = match.groups()
    if key in SUMMARY_KEYS:
        row[key] = _parse_scalar(raw_value or "")
    return True


def _is_block_scalar(text: str) -> bool:
    return re.match(r"[A-Za-z_][\w-]*:\s*(\||>|[|>][-+]?)\s*$", text) is not None


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value[0:1] in {"'", '"'} and value[-1:] == value[0]:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
    if value in {"true", "false"}:
        return value == "true"
    if value == "null":
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def write_github_output(key: str, value: Any) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as output:
        output.write(f"{key}={json.dumps(value, separators=(',', ':'))}\n")


def append_step_summary(markdown: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write(markdown)
        if not markdown.endswith("\n"):
            summary.write("\n")


def markdown_table(
    title: str,
    rows: list[dict[str, Any]],
    *,
    source: str,
    event_name: str | None = None,
    level: str | None = None,
    total_count: int | None = None,
) -> str:
    selected_count = len(rows)
    total = total_count if total_count is not None else selected_count
    lines = [f"## {title}", ""]
    if event_name:
        lines.append(f"- Event: `{_md(event_name)}`")
    if level:
        lines.append(f"- Model level: `{_md(level)}`")
    lines.append(f"- Source: `{_md(source)}`")
    lines.append(f"- Selected models: `{selected_count}/{total}`")
    lines.extend(
        [
            "",
            "| # | Model | Level | Runner | Threshold |",
            "|---:|---|---|---|---:|",
        ]
    )
    for index, row in enumerate(rows, 1):
        model = row.get("display_name") or row.get("model_name") or row.get("model_path") or "unknown"
        level_value = row.get("test_level", "required")
        runner = row.get("runner", "")
        threshold = row.get("accuracy_test_threshold", row.get("accuracy_threshold", ""))
        lines.append(
            f"| {index} | {_md(model)} | {_md(level_value)} | {_md(runner)} | {_md(threshold)} |"
        )
    lines.append("")
    return "\n".join(lines)


def print_console_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"{title}: {len(rows)} models")
    print(f"{'Model':<48} {'Level':<10} {'Runner'}")
    print("-" * 90)
    for row in rows:
        model = row.get("display_name") or row.get("model_name") or row.get("model_path") or "unknown"
        level_value = row.get("test_level", "required")
        runner = row.get("runner", "")
        print(f"{model:<48} {level_value:<10} {runner}")


def _md(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog")
    catalog.add_argument("--title", required=True)
    catalog.add_argument("--catalog", type=Path, required=True)
    catalog.add_argument("--event-name", required=True)
    catalog.add_argument("--github-output-key")

    workflow = subparsers.add_parser("workflow-matrix")
    workflow.add_argument("--title", required=True)
    workflow.add_argument("--workflow", type=Path, required=True)
    workflow.add_argument("--job", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "catalog":
        rows, level, total = load_catalog_models(args.catalog, args.event_name)
        if args.github_output_key:
            write_github_output(args.github_output_key, rows)
        markdown = markdown_table(
            args.title,
            rows,
            source=str(args.catalog),
            event_name=args.event_name,
            level=level,
            total_count=total,
        )
    else:
        rows = load_workflow_matrix(args.workflow, args.job)
        markdown = markdown_table(args.title, rows, source=f"{args.workflow}:{args.job}")

    print_console_table(args.title, rows)
    append_step_summary(markdown)


if __name__ == "__main__":
    main()
