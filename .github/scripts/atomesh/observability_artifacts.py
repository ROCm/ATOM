"""Stage self-contained HTML for one benchmark case and Slurm execution."""

import argparse
import html
import json
import shutil
from pathlib import Path


def stage_reports(source, destination, case_id, slurm_job_id):
    # pd_submit can copy a shared log root containing older runs and other
    # matrix cases. Require both identities; never pick a stale report when
    # this task failed before a Slurm job was submitted.
    if not slurm_job_id:
        return 0
    if not slurm_job_id.isdigit():
        raise ValueError("Slurm job ID must be numeric")
    links = []
    phases = set()
    for report in sorted(source.glob("**/observability/*/index.html")):
        metadata = json.loads(report.with_name("run.json").read_text())
        if metadata["case"] != case_id or str(metadata["run_id"]) != slurm_job_id:
            continue
        phase = metadata["phase"]
        if phase not in ("combined", "benchmark", "eval"):
            raise ValueError("Invalid report destination")
        if phase in phases:
            raise ValueError(f"Duplicate report for {case_id}, {slurm_job_id}, {phase}")
        phases.add(phase)
        target = destination / phase
        target.mkdir(parents=True, exist_ok=True)
        # upload-artifact compresses these files. After extraction each report
        # works under file://, with no fetch, server or Pages deployment.
        shutil.copyfile(report, target / "index.html")
        links.append(f'<li><a href="{phase}/index.html">{phase}</a></li>')
    if links:
        (destination / "index.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>ATOM CI Reports</title>'
            f"<h1>ATOM CI Observability</h1><p>{html.escape(case_id)} · Slurm job {slurm_job_id}</p><ul>"
            + "\n".join(links)
            + "</ul><p>Open a report above in your local browser. No server is required.</p>"
            "<p>For individual streaming requests, select an events/*.jsonl file "
            "from this case's full model benchmark artifact.</p></html>"
        )
    return len(links)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--slurm-job-id-file", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    job_id = (
        args.slurm_job_id_file.read_text().strip()
        if args.slurm_job_id_file.is_file()
        else None
    )
    count = stage_reports(args.source, args.destination, args.case_id, job_id)
    if args.github_output:
        with args.github_output.open("a") as handle:
            handle.write(
                f"has_reports={'true' if count else 'false'}\nreport_count={count}\n"
            )
    print(f"Staged {count} observability report(s) for {args.case_id}")


if __name__ == "__main__":
    main()
