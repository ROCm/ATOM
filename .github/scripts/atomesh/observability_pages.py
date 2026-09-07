"""Stage only HTML reports for the existing opt-in GitHub Pages workflow."""

import argparse
import gzip
import html
import json
import re
from pathlib import Path
from urllib.parse import quote

LOADER = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>ATOM CI Observability</title><body><p id="status">Loading observability report…</p>
<script>
(async()=>{try{
 const response=await fetch('report.html.gz');
 if(!response.ok)throw Error('HTTP '+response.status);
 const stream=new Blob([await response.arrayBuffer()]).stream().pipeThrough(new DecompressionStream('gzip'));
 const page=await new Response(stream).text();
 document.open();document.write(page);document.close();
}catch(error){document.getElementById('status').textContent='Unable to load report: '+error.message;}})();
</script></body></html>
"""


def stage_reports(source, destination, run_id, attempt):
    if not run_id.isdigit() or not attempt.isdigit():
        raise ValueError("Run ID and attempt must be numeric")
    run_root = Path("observability") / run_id / attempt
    output = destination / run_root
    links = []
    for report in sorted(source.glob("**/observability/*/index.html")):
        metadata = json.loads(report.with_name("run.json").read_text())
        case = re.sub(r"[^A-Za-z0-9_.-]", "-", metadata["case"])
        phase = metadata["phase"]
        if case in ("", ".", "..") or phase not in ("combined", "benchmark", "eval"):
            raise ValueError("Invalid report destination")
        target = output / case / phase
        target.mkdir(parents=True, exist_ok=True)
        # 100ms time series produce large HTML. Pages serves a small loader
        # and a compressed report; raw events and VM data stay in CI artifacts.
        (target / "report.html.gz").write_bytes(
            gzip.compress(report.read_bytes(), mtime=0)
        )
        (target / "index.html").write_text(LOADER)
        links.append(
            f'<li><a href="{quote(case)}/{phase}/">{html.escape(case)} · {phase}</a></li>'
        )
    if links:
        (output / "index.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>ATOM CI Reports</title>'
            f"<h1>ATOM CI Observability · Run {run_id}, attempt {attempt}</h1><ul>"
            + "\n".join(links)
            + "</ul><p>Detailed event files are available in the CI artifacts.</p></html>"
        )
    return run_root, len(links)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    path, count = stage_reports(
        args.source, args.destination, args.run_id, args.attempt
    )
    if args.github_output:
        with args.github_output.open("a") as handle:
            handle.write(
                f"has_reports={'true' if count else 'false'}\nreport_path={path.as_posix()}/\n"
            )
    print(f"Staged {count} observability report(s) under {path}")


if __name__ == "__main__":
    main()
