"""Stage only this matrix cell's current Slurm reports for Actions artifacts."""

import argparse
import json
import os
import re
import shutil
from pathlib import Path


def stage_reports(results: Path, cell: str, output: Path) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cell):
        raise ValueError("Invalid matrix cell ID")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"cell": cell, "slurm_job_id": None, "reports": [], "notes": []}
    job_file = results / f"{cell}.slurm-job-id"
    if not job_file.is_file():
        manifest["notes"].append(
            "No Slurm job ID was recorded; no previous reports were selected."
        )
    else:
        job = job_file.read_text().strip()
        if not re.fullmatch(r"[0-9]+", job):
            raise ValueError("Invalid Slurm job ID")
        manifest["slurm_job_id"] = job
        current = results / cell / f"slurm_job-{job}"
        for folder in sorted(current.glob("benchmark_results/aiperf-*/metrics")):
            if not folder.resolve().is_relative_to(current.resolve()):
                raise ValueError("Metrics directory escapes the current Slurm run")
            name = folder.parent.name
            destination = output / name
            destination.mkdir(exist_ok=True)
            for source in folder.iterdir():
                if (
                    source.is_file()
                    and not source.is_symlink()
                    and source.suffix in {".html", ".json", ".log", ".yml"}
                ):
                    shutil.copy2(source, destination / source.name)
            status_file = destination / "status.json"
            try:
                status = (
                    json.loads(status_file.read_text()) if status_file.is_file() else {}
                )
            except (OSError, ValueError):
                status = {
                    "status": "unavailable",
                    "errors": [
                        "Report status is unreadable; the job may have been interrupted."
                    ],
                }
            has_html = (destination / "report.html").is_file()
            manifest["reports"].append(
                {
                    "name": name,
                    "html": f"{name}/report.html" if has_html else None,
                    "status": (
                        status.get("status", "unknown") if has_html else "unavailable"
                    ),
                    "benchmark_exit_code": status.get("benchmark_exit_code"),
                    "errors": status.get("errors", []),
                    "publication_errors": status.get("publication_errors", []),
                }
            )
        if not manifest["reports"]:
            manifest["notes"].append(
                "This Slurm run produced no metrics report; check its service and benchmark logs."
            )
    (output / "reports.json").write_text(json.dumps(manifest, indent=2))
    (output / "README.txt").write_text(
        "Download and extract this artifact, then open a report.html file in your browser.\n"
        "Reports contain embedded data and work offline. No server is required.\n"
        "reports.json lists report status and any collection errors.\n"
        + "\n".join(manifest["notes"])
        + "\n"
    )
    return manifest


def write_summary(manifest: dict, artifact_url: str, summary: Path) -> None:
    lines = ["### Agentic PD latency reports", ""]
    if artifact_url:
        lines += [f"[Download HTML reports and data]({artifact_url})", ""]
    lines += [
        "Extract the artifact and open `report.html` locally. The interactive report works offline.",
        "",
    ]
    if manifest["slurm_job_id"]:
        lines += [f"Slurm job: `{manifest['slurm_job_id']}`", ""]
    for report in manifest["reports"]:
        name = report["name"].replace("`", "").replace("\n", " ")
        lines.append(f"- `{name}` — **{report['status']}**")
    lines += ["", *manifest["notes"], ""]
    with summary.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("atomesh-results"))
    parser.add_argument("--cell", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--artifact-url", default=os.environ.get("METRICS_ARTIFACT_URL", "")
    )
    args = parser.parse_args()
    if args.summary_only:
        manifest = json.loads((args.output / "reports.json").read_text())
        write_summary(
            manifest, args.artifact_url, Path(os.environ["GITHUB_STEP_SUMMARY"])
        )
    else:
        manifest = stage_reports(args.results, args.cell, args.output)
        print(json.dumps(manifest, indent=2))
    if (
        any(report["status"] != "complete" for report in manifest["reports"])
        or manifest["notes"]
    ):
        print(
            "::warning::Metrics reports are incomplete; see the latency report artifact and Slurm logs."
        )


if __name__ == "__main__":
    main()
