#!/usr/bin/env python3
"""Choose between the full and source-overlay ATOM image builds.

The incremental image is intentionally conservative.  It is only safe when
the published base image was built with the same native dependency stack and
the commits between the base image and the requested target only change files
that are consumed directly from the editable ``/app/ATOM`` checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

LABEL_SOURCE = "org.opencontainers.image.source"
LABEL_REVISION = "org.opencontainers.image.revision"
LABEL_FOUNDATION = "io.rocm.atom.foundation-image"
LABEL_AITER_SOURCE = "io.rocm.atom.aiter.source"
LABEL_AITER_REVISION = "io.rocm.atom.aiter.revision"
LABEL_RCCL_SOURCE = "io.rocm.atom.rccl.source"
LABEL_RCCL_REVISION = "io.rocm.atom.rccl.revision"

SAFE_PREFIXES = ("docs/", "examples/", "tests/", "tools/")
SAFE_ROOT_FILES = {"LICENSE", "README.md"}
SAFE_ATOM_SUFFIXES = {".json", ".md", ".py", ".pyi", ".txt", ".yaml", ".yml"}
UNSAFE_ATOM_PREFIXES = ("atom/entrypoints/atomesh/", "atom/mesh/")


@dataclass(frozen=True)
class BuildDecision:
    mode: str
    reason: str


def is_incremental_safe(path: str) -> bool:
    """Return whether *path* can be refreshed without rebuilding native code."""

    if path in SAFE_ROOT_FILES or path.startswith(SAFE_PREFIXES):
        return True
    if not path.startswith("atom/") or path.startswith(UNSAFE_ATOM_PREFIXES):
        return False
    return Path(path).suffix in SAFE_ATOM_SUFFIXES


def choose_build_mode(
    *,
    requested_mode: str,
    event_name: str,
    base_labels: Mapping[str, str],
    expected_labels: Mapping[str, str],
    base_is_ancestor: bool,
    changed_files: Iterable[str],
) -> BuildDecision:
    """Select the safe build mode and explain the decision."""

    if event_name == "schedule":
        return BuildDecision("full", "scheduled releases always refresh the full stack")
    if requested_mode == "full":
        return BuildDecision("full", "full build explicitly requested")

    missing = sorted(key for key in expected_labels if not base_labels.get(key))
    if missing:
        reason = "incremental base is missing metadata: " + ", ".join(missing)
        return _fallback_or_fail(requested_mode, reason)

    base_commit = str(base_labels.get(LABEL_REVISION, ""))
    if not re.fullmatch(r"[0-9a-f]{40}", base_commit):
        return _fallback_or_fail(
            requested_mode,
            "incremental base does not identify an immutable ATOM commit",
        )

    mismatched = sorted(
        key
        for key, expected in expected_labels.items()
        if base_labels.get(key) != expected
    )
    if mismatched:
        reason = "incremental base does not match requested stack: " + ", ".join(
            mismatched
        )
        return _fallback_or_fail(requested_mode, reason)

    if not base_is_ancestor:
        return _fallback_or_fail(
            requested_mode,
            "incremental base commit is not an ancestor of the requested commit",
        )

    unsafe = sorted(path for path in changed_files if not is_incremental_safe(path))
    if unsafe:
        preview = ", ".join(unsafe[:8])
        if len(unsafe) > 8:
            preview += f", and {len(unsafe) - 8} more"
        return _fallback_or_fail(
            requested_mode,
            f"changes require a full build: {preview}",
        )

    return BuildDecision(
        "incremental",
        "base metadata matches and all changed files are source-overlay safe",
    )


def _fallback_or_fail(requested_mode: str, reason: str) -> BuildDecision:
    if requested_mode == "incremental":
        raise ValueError(f"forced incremental build is unsafe: {reason}")
    return BuildDecision("full", reason)


def _git(
    repo_root: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=check,
        text=True,
        capture_output=True,
    )


def inspect_commit_range(
    repo_root: Path, base_commit: str, target_commit: str
) -> tuple[bool, list[str]]:
    """Return ancestry and changed paths for commits available in *repo_root*."""

    for commit in (base_commit, target_commit):
        if _git(
            repo_root, "cat-file", "-e", f"{commit}^{{commit}}", check=False
        ).returncode:
            return False, []

    ancestor = (
        _git(
            repo_root,
            "merge-base",
            "--is-ancestor",
            base_commit,
            target_commit,
            check=False,
        ).returncode
        == 0
    )
    if not ancestor:
        return False, []

    result = _git(
        repo_root,
        "diff",
        "--name-only",
        "--no-renames",
        base_commit,
        target_commit,
        "--",
    )
    return True, [line for line in result.stdout.splitlines() if line]


def _write_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"{name}={value}\n")
    print(f"{name}={value}")


def _write_summary(decision: BuildDecision, changed_files: list[str]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write("## Native image build mode\n\n")
        summary.write(f"- Mode: `{decision.mode}`\n")
        summary.write(f"- Reason: {decision.reason}\n")
        summary.write(f"- Compared files: {len(changed_files)}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requested-mode", choices=("auto", "full", "incremental"))
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--base-labels", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--atom-repo", required=True)
    parser.add_argument("--atom-commit", required=True)
    parser.add_argument("--foundation-image", required=True)
    parser.add_argument("--aiter-repo", required=True)
    parser.add_argument("--aiter-commit", required=True)
    parser.add_argument("--rccl-repo", required=True)
    parser.add_argument("--rccl-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested_mode = args.requested_mode or "full"
    base_labels = json.loads(args.base_labels.read_text(encoding="utf-8"))
    if not isinstance(base_labels, dict):
        base_labels = {}

    base_commit = str(base_labels.get(LABEL_REVISION, ""))
    base_is_ancestor = False
    changed_files: list[str] = []
    if base_commit:
        base_is_ancestor, changed_files = inspect_commit_range(
            args.repo_root, base_commit, args.atom_commit
        )

    expected_labels = {
        LABEL_SOURCE: args.atom_repo,
        LABEL_FOUNDATION: args.foundation_image,
        LABEL_AITER_SOURCE: args.aiter_repo,
        LABEL_AITER_REVISION: args.aiter_commit,
        LABEL_RCCL_SOURCE: args.rccl_repo,
        LABEL_RCCL_REVISION: args.rccl_commit,
    }

    try:
        decision = choose_build_mode(
            requested_mode=requested_mode,
            event_name=args.event_name,
            base_labels=base_labels,
            expected_labels=expected_labels,
            base_is_ancestor=base_is_ancestor,
            changed_files=changed_files,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    _write_output("mode", decision.mode)
    _write_output("reason", decision.reason)
    _write_output("base_commit", base_commit)
    _write_summary(decision, changed_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
