import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUMMARIZE = REPO / ".github" / "scripts" / "summarize.py"


def _write(path, model, conc, tput, tpot, ttft):
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{model}-8192-1024-{conc}-0.8.json").write_text(
        json.dumps(
            {
                "date": "20260101-000000",
                "benchmark_model_name": model,
                "model_id": f"org/{model}",
                "random_input_len": 8192,
                "random_output_len": 1024,
                "max_concurrency": conc,
                "duration": 1.0,
                "total_input_tokens": 1.0,
                "total_output_tokens": 1.0,
                "request_throughput": 1.0,
                "output_throughput": tput,
                "total_token_throughput": tput * 9,
                "mean_ttft_ms": ttft,
                "median_ttft_ms": ttft,
                "std_ttft_ms": 0.0,
                "p99_ttft_ms": ttft,
                "mean_tpot_ms": tpot,
                "median_tpot_ms": tpot,
                "std_tpot_ms": 0.0,
                "p99_tpot_ms": tpot,
                "mean_itl_ms": 0.0,
                "median_itl_ms": 0.0,
                "std_itl_ms": 0.0,
                "p99_itl_ms": 0.0,
                "mean_e2el_ms": 0.0,
                "median_e2el_ms": 0.0,
                "std_e2el_ms": 0.0,
                "p99_e2el_ms": 0.0,
            }
        )
    )


def _baseline_result(tmp, name, model, conc, tput, tpot, ttft):
    """A baseline result, nested one directory deep.

    ``main()`` loads the baseline with ``recursive=True`` because
    ``gh run download`` nests each artifact in its own subfolder (see
    summarize.py's ``load_results`` docstring and
    ``atom-benchmark.yaml:246``), so the fixture nests here to match.
    """
    _write(tmp / name, model, conc, tput, tpot, ttft)


def _current_result(tmp, model, conc, tput, tpot, ttft):
    """A current-run result, flat in the result dir.

    ``main()`` loads the current run with ``load_results(args.result_dir)``
    (no ``recursive=True``) because the workflow invokes
    ``summarize.py .`` directly on the job's own working directory
    (``atom-benchmark.yaml:273``), which is not nested like a downloaded
    artifact.
    """
    _write(tmp, model, conc, tput, tpot, ttft)


def _run(tmp_path, cur, base):
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SUMMARIZE),
            str(cur),
            "--baseline-dir",
            str(base),
            "--output-json",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    return proc, json.loads(out.read_text())


def test_report_carries_a_linkage_block(tmp_path):
    base = tmp_path / "base"
    cur = tmp_path / "cur"
    for conc in (32, 128):
        _baseline_result(base, "base", "M", conc, 1000.0, 10.0, 100.0)
        _current_result(cur, "M", conc, 1000.0, 10.0, 100.0)
    _proc, report = _run(tmp_path, cur, base)
    assert "linkage" in report
    assert report["linkage"]["count"] == 0


def test_linkage_trips_where_the_family_moves_together(tmp_path):
    base = tmp_path / "base"
    cur = tmp_path / "cur"
    for conc in (32, 128):
        _baseline_result(base, "base", "M", conc, 1000.0, 10.0, 100.0)
        _current_result(cur, "M", conc, 920.0, 11.0, 112.0)  # -8% tput, +10% tpot
    _proc, report = _run(tmp_path, cur, base)
    assert report["linkage"]["count"] == 1
    entry = report["linkage"]["triggered"][0]
    assert entry["model"] == "M"
    assert entry["mirror_via"] == "TPOT"


def test_shipped_fields_are_untouched(tmp_path):
    """Three consumers read these; adding linkage must not disturb them."""
    base = tmp_path / "base"
    cur = tmp_path / "cur"
    for conc in (32, 128):
        _baseline_result(base, "base", "M", conc, 1000.0, 10.0, 100.0)
        _current_result(cur, "M", conc, 920.0, 11.0, 112.0)
    proc, report = _run(tmp_path, cur, base)
    assert set(report) >= {
        "timestamp",
        "regression_count",
        "regressions",
        "all_results",
    }
    assert report["regression_count"] == len(report["regressions"])
    assert proc.returncode == 2  # regressions found, exit code unchanged


def test_workflow_reads_linkage_count_without_gating_on_it():
    """The shadow number must reach the summary, and must not move the gate."""
    import yaml

    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "atom-benchmark.yaml").read_text()
    )
    steps = workflow["jobs"]["summarize-benchmark-result"]["steps"]
    step = next(s for s in steps if s.get("id") == "check-regression")
    assert "linkage_count" in step["run"]
    # The gate is still the shipped criterion.
    assert "has_regression=true" in step["run"]
    # The issue-creating step lives in a downstream job (profiler-analysis),
    # not in summarize-benchmark-result itself -- search all jobs for it.
    issue_step = next(
        s
        for job in workflow["jobs"].values()
        for s in job.get("steps", [])
        if isinstance(s.get("name"), str) and "ssue" in s["name"]
    )
    assert "linkage" not in str(issue_step.get("if", ""))
