#!/usr/bin/env python3
"""Backfill published ATOMesh agentic points with both interactivity definitions.

``process_result.py`` computes them for every *future* run. Points already on
gh-pages predate that: the oldest still carry ``1000 / median_tpot_ms`` on the
main axis, and none carry the plain ``1/p90(ITL)`` number the dashboard now
offers as a second axis. This script recomputes both from the per-request records
stored in each run's ``atomesh-model-benchmark-*`` artifact and rewrites
``benchmark-dashboard/data.js`` in place.

    # one-time: pull the artifacts, then rewrite
    ./backfill_interactivity.py --data-js data.js --artifacts /tmp/bf --download --dry-run
    ./backfill_interactivity.py --data-js data.js --artifacts /tmp/bf --out data.js.new

Each dashboard point lives url-encoded inside a bench's ``extra`` string as
``perf_point=<json>``. The rewrite is deliberately textual -- only the matched
``perf_point`` tokens are spliced, so a 22 MB file produces a diff that touches
nothing but the points that actually changed.

Points whose artifact has expired or whose records are missing keep their old
value and are reported, not silently left looking current.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any

from interactivity import (
    DEFAULT_PERCENTILE,
    METHOD_MEDIAN_TPOT,
    METHOD_P90_E2E,
    agentic_interactivity,
    locate_records,
)
from process_result import AGENTIC_BENCHMARK_KIND, RESULT_RE, round_or_none

# safe="" mirrors perf_point_extra() in process_result.py, so an untouched point
# re-encodes to the exact bytes already in the file.
POINT_RE = re.compile(r"perf_point=([A-Za-z0-9_.\-~%]+)")
ARTIFACT_PATTERN = "atomesh-model-benchmark-*"
DEFAULT_REPO = "ROCm/ATOM"


def encode_point(point: dict[str, Any]) -> str:
    return urllib.parse.quote(
        json.dumps(point, separators=(",", ":"), sort_keys=True), safe=""
    )


def run_id_from_url(run_url: str) -> str | None:
    """``.../actions/runs/31989815489`` -> ``31989815489``."""
    match = re.search(r"/actions/runs/(\d+)", run_url or "")
    return match.group(1) if match else None


def download_artifacts(gh_run: str, dest: Path, repo: str) -> bool:
    if dest.is_dir() and any(dest.iterdir()):
        print(f"  [skip download] {gh_run}: {dest} already populated")
        return True
    if not shutil.which("gh"):
        sys.exit("ERROR: --download needs the `gh` CLI on PATH")
    dest.mkdir(parents=True, exist_ok=True)
    command = [
        "gh",
        "run",
        "download",
        gh_run,
        "-R",
        repo,
        "-p",
        ARTIFACT_PATTERN,
        "-D",
        str(dest),
    ]
    print(f"  [download] {' '.join(command)}")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip().splitlines()
        print(f"  [download failed] {gh_run}: {message[-1] if message else '?'}")
        return False
    return True


def find_result_file(root: Path, result_stem: str) -> list[Path]:
    """All ``<result_stem>.json`` files under a run's downloaded artifacts.

    More than one means two matrix entries produced the same result name; the
    caller treats that as ambiguous rather than guessing.
    """
    if not root.is_dir():
        return []
    return sorted(root.rglob(f"{result_stem}.json"))


def recompute(
    point: dict[str, Any], run_root: Path, percentile: float
) -> tuple[dict[str, Any] | None, str]:
    """Return (result, reason). result is None when the point cannot be redone."""
    result_stem = point.get("run_id")
    if not result_stem:
        return None, "point has no run_id"
    matches = find_result_file(run_root, str(result_stem))
    if not matches:
        return None, f"no {result_stem}.json under {run_root.name}"

    computed = []
    for result_path in matches:
        # The AIPerf artifact directory is named after the same model/topology/
        # concurrency triple the result filename encodes, so read them back off
        # the file rather than off the point, which stores neither raw topology
        # nor the model spelling used on disk.
        parsed = RESULT_RE.match(result_path.name)
        records = locate_records(
            result_path,
            model=parsed.group("model") if parsed else point.get("model"),
            topology=parsed.group("topology") if parsed else None,
            concurrency=(
                int(parsed.group("conc")) if parsed else point.get("concurrency")
            ),
        )
        if records is None:
            continue
        try:
            computed.append(agentic_interactivity(records, percentile))
        except (OSError, ValueError) as exc:
            return None, f"{records}: {exc}"
    if not computed:
        return None, f"no profile_export.jsonl beside {matches[0].name}"
    # Both metrics have to agree, not just the primary one: a disagreement in
    # either means the two result files describe different runs.
    values = {
        (round(entry["value"], 6), round(entry["itl_value"], 6)) for entry in computed
    }
    if len(values) > 1:
        return None, (
            f"{len(matches)} result files named {result_stem}.json disagree "
            f"({sorted(values)}) -- resolve by hand"
        )
    return computed[0], "ok"


def backfill(
    raw: str,
    artifacts: Path,
    percentile: float,
    download: bool,
    repo: str,
) -> tuple[str, list[dict[str, Any]], list[tuple[str, str]]]:
    downloaded: dict[str, bool] = {}
    changed: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    pieces: list[str] = []
    cursor = 0

    for match in POINT_RE.finditer(raw):
        point = json.loads(urllib.parse.unquote(match.group(1)))
        if (
            point.get("source") != "ATOMesh"
            or point.get("benchmark_kind") != AGENTIC_BENCHMARK_KIND
        ):
            continue
        label = f"{point.get('run_id')}"
        gh_run = run_id_from_url(str(point.get("run_url") or ""))

        result: dict[str, Any] | None = None
        if not gh_run:
            reason = "point has no parseable run_url"
        else:
            run_root = artifacts / gh_run
            if download and gh_run not in downloaded:
                downloaded[gh_run] = download_artifacts(gh_run, run_root, repo)
            if download and not downloaded[gh_run]:
                reason = f"artifact download failed for run {gh_run}"
            else:
                result, reason = recompute(point, run_root, percentile)

        updated = dict(point)
        if result is None:
            # Stamp the legacy definition instead of leaving the field absent. The
            # dashboard keeps any agentic point that is not p90 e2e out of the
            # Pareto frontier, so "measured the old way" must not be
            # indistinguishable from "field missing because something upstream
            # broke" -- those want opposite responses from whoever reads it next.
            #
            # setdefault, not assignment: on a second pass the artifacts behind an
            # earlier successful backfill may have expired, and overwriting would
            # relabel a correctly computed p90 point as legacy -- ejecting it from
            # the frontier over nothing worse than an aged-out artifact.
            skipped.append((label, reason))
            updated.setdefault("interactivity_method", METHOD_MEDIAN_TPOT)
        else:
            updated["interactivity"] = round_or_none(result["value"])
            updated["interactivity_method"] = METHOD_P90_E2E
            updated["interactivity_n_requests"] = result["n_requests"]
            updated["interactivity_p90_itl"] = round_or_none(result["itl_value"])

        # Re-running against an already-backfilled file must be a no-op -- which
        # includes the report: a point recomputed to the value it already had is
        # not a change, and listing it would make a second pass look like it did
        # work and hide any point that genuinely did move.
        if updated == point:
            continue
        if result is not None:
            changed.append(
                {
                    "gh_run": gh_run,
                    "run_id": label,
                    "model": point.get("model"),
                    "config": point.get("config_label"),
                    "concurrency": point.get("concurrency"),
                    "old": point.get("interactivity"),
                    "new": updated["interactivity"],
                    "p90_itl": updated["interactivity_p90_itl"],
                    "n_requests": result["n_requests"],
                }
            )
        pieces.append(raw[cursor : match.start(1)])
        pieces.append(encode_point(updated))
        cursor = match.end(1)

    pieces.append(raw[cursor:])
    return "".join(pieces), changed, skipped


def report(changed: list[dict[str, Any]], skipped: list[tuple[str, str]]) -> None:
    if changed:
        print(
            f"\n{'run':<12} {'model':<16} {'config':<44} {'conc':>5} "
            f"{'old':>10} {'new':>10} {'change':>9} {'p90_itl':>10} {'n':>7}"
        )
        for row in sorted(changed, key=lambda r: (r["gh_run"], str(r["run_id"]))):
            old, new = row["old"], row["new"]
            delta = (
                f"{(new - old) / old * 100:+.1f}%"
                if isinstance(old, (int, float)) and old
                else "--"
            )
            print(
                f"{row['gh_run']:<12} {str(row['model'])[:16]:<16} "
                f"{str(row['config'])[:44]:<44} {row['concurrency']!s:>5} "
                f"{old!s:>10} {new!s:>10} {delta:>9} {row['p90_itl']!s:>10} "
                f"{row['n_requests']:>7}"
            )
    if skipped:
        print(
            f"\nskipped {len(skipped)} point(s), old value kept "
            f"({METHOD_MEDIAN_TPOT}):"
        )
        for label, reason in skipped:
            print(f"  {label}: {reason}")
    print(f"\nrewritten {len(changed)} point(s), skipped {len(skipped)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-js",
        required=True,
        type=Path,
        help="gh-pages benchmark-dashboard/data.js",
    )
    parser.add_argument(
        "--artifacts",
        required=True,
        type=Path,
        help="directory holding <gh_run_id>/ artifact trees",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=f"fetch missing runs with `gh run download -p {ARTIFACT_PATTERN}`",
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help="owner/name for --download"
    )
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument("--out", type=Path, help="output path (default: <data-js>.new)")
    parser.add_argument(
        "--dry-run", action="store_true", help="report only, write nothing"
    )
    args = parser.parse_args()

    raw = args.data_js.read_text(encoding="utf-8")
    rewritten, changed, skipped = backfill(
        raw, args.artifacts, args.percentile, args.download, args.repo
    )
    report(changed, skipped)

    if args.dry_run:
        print("dry run: no file written")
        return
    if not changed:
        print("nothing changed: no file written")
        return
    out = args.out or args.data_js.with_suffix(args.data_js.suffix + ".new")
    out.write_text(rewritten, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
