# SPDX-License-Identifier: MIT
"""Tests for the PR performance check.

Covers both halves of the feature, mirroring tests/test_benchmark_catalog.py's
convention of one test module per CI feature rather than one per source file:

- perf_judge.py    -- what counts as a regression, and what must never read as
                      a pass
- perf_check_half.sh -- the pairing mechanics: checkout, launch, benchmark,
                      stop, archive, in that order

Everything runs on CPU. `docker` and the container's atom_test.sh are replaced
by shims on PATH, so argument passing and ordering are exercised for real
without a GPU.

The cases that matter most are the ones where a naive implementation quietly
returns a pass:

- an entry that regressed but lost a concurrency level
- a matrix where most jobs died
- a shape the criterion cannot summarise
- low-concurrency noise (must not trip the verdict, must still be visible)
- a half that produced nothing (must not be filled in with an invented result)
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / ".github" / "scripts"
HALF_SH = SCRIPTS / "perf_check_half.sh"

sys.path.insert(0, str(SCRIPTS))

import perf_judge as pj

JUDGING = [32, 256]
REFERENCE = []
# The drift criterion is the nightly monitor's, ported with its constants
# intact (DRIFT_MIN_CONC=64, DRIFT_FAM_MIN_CONFIGS=3). It reads dashboard
# history, not this run's levels, so its fixtures are its own -- reusing
# JUDGING here coupled two things that only looked alike.
DRIFT_LEVELS = [64, 128, 256]
ALL_CONCS = REFERENCE + JUDGING

BASE_TPUT = {
    "DeepSeek-V4-Pro": 17354.8,
    "DeepSeek-V4-Pro-mtp3": 19743.0,
    "Kimi-K3": 15120.4,
    "MiniMax-M3-MXFP8": 16402.1,
    "GLM-5.2-FP8": 11718.1,
}


def _result(model, conc, tput, tpot, ttft=420.0):
    return {
        "benchmark_backend": "ATOM",
        "benchmark_model_name": model,
        "random_input_len": 8192,
        "random_output_len": 1024,
        "max_concurrency": conc,
        "output_throughput": tput / 2,
        "total_token_throughput": tput,
        "mean_ttft_ms": ttft,
        "mean_tpot_ms": tpot,
    }


def write_pair(tmp_path, spec, concs=ALL_CONCS, drop=()):
    """Materialise a base/head pair.

    ``spec`` maps model -> {"tput": pct, "tpot": pct, "per_conc": {conc: pct}}.
    ``drop`` lists (model, conc) whose head half is missing, standing in for a
    job that died.
    """
    base_dir, head_dir = tmp_path / "base", tmp_path / "head"
    base_dir.mkdir(parents=True, exist_ok=True)
    head_dir.mkdir(parents=True, exist_ok=True)

    for model, entry in spec.items():
        for conc in concs:
            # Low concurrency genuinely yields lower absolute throughput; the
            # judge works on ratios, but keeping it realistic guards against an
            # accidental dependence on magnitude.
            base_value = BASE_TPUT[model] * (0.6 if conc < 64 else 1.0)
            delta = entry.get("per_conc", {}).get(conc, entry.get("tput", 0.0))
            tpot_delta = entry.get("tpot", 0.0)
            ttft_delta = entry.get("ttft", 0.0)

            name = f"{model}-8192-1024-{conc}-0.8.json"
            (base_dir / name).write_text(
                json.dumps(_result(model, conc, base_value, 30.0))
            )
            if (model, conc) in drop:
                continue
            (head_dir / name).write_text(
                json.dumps(
                    _result(
                        model,
                        conc,
                        base_value * (1 + delta / 100),
                        30.0 * (1 + tpot_delta / 100),
                        420.0 * (1 + ttft_delta / 100),
                    )
                )
            )
    return base_dir, head_dir


def run_judge(tmp_path, spec, expect=5, concs=ALL_CONCS, drop=()):
    base_dir, head_dir = write_pair(tmp_path, spec, concs=concs, drop=drop)
    pairs = pj.pair_results(
        pj.load_results(base_dir, recursive=True),
        pj.load_results(head_dir, recursive=True),
    )
    return pj.judge(pairs, {}, expected_entries=expect)


def _run_cli(tmp_path, base_dir, head_dir, history=None, expect=5):
    """Drive the judge through its command line, the way CI does."""
    out = tmp_path / "verdict.json"
    cmd = [
        sys.executable,
        str(SCRIPTS / "perf_judge.py"),
        "--base-dir",
        str(base_dir),
        "--head-dir",
        str(head_dir),
        "--expect-entries",
        str(expect),
        "--output-json",
        str(out),
    ]
    cmd += ["--history", str(history)] if history else ["--no-history"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


def flat(tput=0.0, tpot=0.0):
    return {m: {"tput": tput, "tpot": tpot} for m in BASE_TPUT}


# ---------------------------------------------------------------- verdicts ---
def test_no_change_is_clean(tmp_path):
    report = run_judge(tmp_path, flat(tput=-0.4, tpot=-0.1))
    assert report["verdict"] == "clean"
    assert report["n_triggered"] == 0
    assert report["n_judged"] == 5


def test_one_entry_regressing_is_scoped_to_that_entry(tmp_path):
    spec = flat(tput=-0.4, tpot=-0.1)
    spec["DeepSeek-V4-Pro-mtp3"] = {"tput": -6.0, "tpot": 5.1}
    report = run_judge(tmp_path, spec)
    assert report["verdict"] == "regression"
    assert report["n_triggered"] == 1
    assert report["scope"] == "single_entry"
    tripped = [f for f in report["families"] if f["status"] == "triggered"]
    assert tripped[0]["model"] == "DeepSeek-V4-Pro-mtp3"


def test_every_entry_regressing_is_scoped_as_shared_path(tmp_path):
    report = run_judge(tmp_path, flat(tput=-5.5, tpot=4.8))
    assert report["verdict"] == "regression"
    assert report["scope"] == "all_entries"


def test_drop_without_tpot_mirroring_does_not_trip(tmp_path):
    """Throughput down but TPOT flat is measurement wobble, not a slowdown."""
    spec = flat(tput=-0.4, tpot=-0.1)
    spec["Kimi-K3"] = {"tput": -4.0, "tpot": 0.5}
    report = run_judge(tmp_path, spec)
    assert report["verdict"] == "clean"
    kimi = next(f for f in report["families"] if f["model"] == "Kimi-K3")
    assert kimi["status"] == "clean"
    assert kimi["mirror"] is False


def test_queueing_regression_trips_via_ttft(tmp_path):
    """Throughput down while requests queue is a regression, even as TPOT falls.

    Measured on MI308 with max_num_seqs capped at 16: -29% throughput, TTFT
    +495%, TPOT -78%. Capping how many requests run at once makes each served
    request faster while the queue behind it grows, so a gate that demanded
    TPOT rise called a 29% capacity loss `unclear` and reported nothing.
    """
    spec = flat(tput=-0.4, tpot=-0.1)
    spec["Kimi-K3"] = {"tput": -29.3, "tpot": -77.8, "ttft": 494.8}
    report = run_judge(tmp_path, spec)
    assert report["verdict"] == "regression"
    kimi = next(f for f in report["families"] if f["model"] == "Kimi-K3")
    assert kimi["status"] == "triggered"
    assert kimi["mirror"] is True
    assert kimi["mirror_via"] == "TTFT"


def test_drop_with_neither_latency_moving_does_not_trip(tmp_path):
    """Widening the gate to TTFT must not turn it into no gate at all."""
    spec = flat(tput=-0.4, tpot=-0.1)
    spec["Kimi-K3"] = {"tput": -4.0, "tpot": 0.5, "ttft": 0.2}
    report = run_judge(tmp_path, spec)
    assert report["verdict"] == "clean"
    kimi = next(f for f in report["families"] if f["model"] == "Kimi-K3")
    assert kimi["mirror"] is False
    assert kimi["mirror_via"] is None


def test_missing_history_cannot_report_clean(tmp_path):
    """A run whose baseline check could not execute is not a green run.

    The baseline check is what catches a base measurement that is itself
    broken, where the delta is near zero and clean is exactly the wrong call.
    Degrading quietly would leave a report indistinguishable from one that was
    actually checked.
    """
    base_dir, head_dir = write_pair(tmp_path, flat(tput=-0.4, tpot=-0.1))
    missing = tmp_path / "nope.js"
    report = _run_cli(tmp_path, base_dir, head_dir, history=missing)
    assert report["history_status"] == "unavailable"
    assert report["verdict"] == "partial"
    assert report["history_downgraded"] is True


def test_history_that_matches_nothing_is_not_a_pass(tmp_path):
    """History that lines up with no measured configuration is not history.

    This is what a renamed model or a changed dashboard format produces, and
    from inside the judge it is indistinguishable from a healthy check unless
    the mismatch is stated.
    """
    base_dir, head_dir = write_pair(tmp_path, flat(tput=-0.4, tpot=-0.1))
    stale = tmp_path / "stale.js"
    # Enough runs for the history to be usable -- a single point yields no
    # spread and would read as "unavailable", which is a different failure.
    runs = [
        {
            "date": 1786149015906 + i * 86400000,
            "commit": {"id": chr(97 + i) * 40},
            "benches": [
                {
                    "name": f"ATOM::Renamed-Model 8192/1024 c={c} {metric}",
                    "value": value,
                    "unit": unit,
                    "extra": f"Run: actions/runs/{i}",
                }
                for c in (64, 128, 256)
                for metric, value, unit in (
                    ("Total Tput", 1000.0 + i, "tok/s"),
                    ("TPOT", 30.0, "ms"),
                )
            ],
        }
        for i in range(8)
    ]
    stale.write_text(
        "window.BENCHMARK_DATA = " + json.dumps({"entries": {"Benchmark": runs}})
    )
    report = _run_cli(tmp_path, base_dir, head_dir, history=stale)
    assert report["history_status"] == "unmatched"
    assert report["verdict"] == "partial"


def test_families_outside_drift_coverage_are_listed(tmp_path):
    """A family the trend criterion cannot reach is named, not skipped.

    Absent and clean look the same in a report, so a model that has just
    started running would sit outside coverage indefinitely with nobody
    noticing it never got checked.
    """
    now = 1786149015906
    day = 86400000
    series = {}
    # Enough history to be judged.
    for c in DRIFT_LEVELS:
        series[("M", "8192/1024", c)] = [
            {"date": now - i * day, "tput": 1000.0 - i, "tpot": 30.0 + i}
            for i in range(pj.DRIFT_MIN_RUNS + 2)
        ]
    # Two levels only, so the family cannot form.
    for c in (64, 128):
        series[("Thin", "8192/1024", c)] = [
            {"date": now - i * day, "tput": 1000.0, "tpot": 30.0}
            for i in range(pj.DRIFT_MIN_RUNS + 2)
        ]
    # Enough levels, but none has run often enough.
    for c in JUDGING:
        series[("New", "8192/1024", c)] = [
            {"date": now - i * day, "tput": 1000.0, "tpot": 30.0} for i in range(3)
        ]
    waiting = pj.main_drift(series, {"M", "Thin", "New"})["waiting"]
    named = {w[0]: w[2] for w in waiting}
    assert "Thin" in named and "levels have enough history" in named["Thin"]
    assert "New" in named and "runs yet" in named["New"]
    assert "M" not in named


def test_peak_does_not_forget_a_drop_that_aged_out(tmp_path):
    """A level that dropped long ago and never recovered still reads as down.

    Every window rolls forward. Once the good runs leave it the baseline falls
    with them, the shortfall shrinks, and a permanent regression becomes the
    new normal -- upstream closed a family as healthy while it sat 12.3% under
    its peak. The anchor is taken over all history for that reason.
    """
    old_high = [1000.0, 1010.0, 1005.0]
    settled = [900.0] * 12  # dropped 10%, long enough to fill any window
    values = old_high + settled
    anchor = pj._anchor(values)
    assert anchor == pytest.approx(1005.0, abs=1.0)

    points = [
        {"date": 1786149015906 + i * 86400000, "tput": v, "tpot": 30.0}
        for i, v in enumerate(values)
    ]
    # The level window sees only the settled runs...
    assert pj._level_window(points) == pytest.approx([900.0] * 3, abs=0.1)
    # ...but the anchor still remembers what main used to do.
    assert (900.0 / anchor - 1) * 100 < -9


# ------------------------------------------------- incomplete data is not a pass ---
def test_regressed_entry_missing_a_level_does_not_read_as_clean(tmp_path):
    """The failure this guards against: the one entry that regressed loses a
    concurrency level, becomes unjudgeable, and the run reports clean."""
    spec = flat(tput=-0.4, tpot=-0.1)
    spec["DeepSeek-V4-Pro-mtp3"] = {"tput": -6.0, "tpot": 5.1}
    report = run_judge(tmp_path, spec, drop=[("DeepSeek-V4-Pro-mtp3", 256)])

    assert report["verdict"] != "clean"
    assert report["verdict"] == "partial"
    assert report["n_insufficient"] == 1

    # ... and the drop it did measure must still be spelled out to the reader.
    body = pj.render(report, "")
    assert "could not be judged" in body
    assert "-6.0%" in body


def test_mostly_dead_matrix_does_not_read_as_clean(tmp_path):
    """Four of five entries produced nothing. The survivor being fine says
    nothing about the run, and expected_entries is what makes that visible."""
    spec = {"GLM-5.2-FP8": {"tput": -0.2, "tpot": 0.1}}
    report = run_judge(tmp_path, spec, expect=5)
    assert report["verdict"] == "partial"
    assert report["n_missing"] == 4


def test_nothing_judgeable_is_inconclusive(tmp_path):
    report = run_judge(tmp_path, flat(), concs=[4, 32], expect=5)
    assert report["verdict"] == "inconclusive"
    assert report["n_judged"] == 0
    assert "not** a pass" in pj.render(report, "")


def test_positive_finding_survives_missing_coverage(tmp_path):
    """A regression stands on its own; gaps elsewhere do not soften it."""
    spec = flat(tput=-0.4, tpot=-0.1)
    spec["DeepSeek-V4-Pro"] = {"tput": -6.0, "tpot": 5.1}
    report = run_judge(tmp_path, spec, drop=[("Kimi-K3", 128), ("Kimi-K3", 256)])
    assert report["verdict"] == "regression"


# ------------------------------------------------ judging vs reference levels ---
def test_c32_now_reaches_the_verdict(tmp_path):
    """c=32 is judged, not shown. Its own run-to-run spread is 2.2% for a
    tp>=4 MoE against 0.5-1.2% at c=256, and DOWN_EPS_PCT is -2.0% -- so this
    level can cross the "down" line without any code having changed. The
    matrix runs it anyway, because two jobs per entry is the budget."""
    spec = flat(tput=-0.4, tpot=-0.1)
    spec["Kimi-K3"] = {"per_conc": {32: -9.0, 256: -0.1}, "tpot": 0.1}
    report = run_judge(tmp_path, spec)

    kimi = next(f for f in report["families"] if f["model"] == "Kimi-K3")
    assert [m["conc"] for m in kimi["judging"]] == JUDGING
    assert kimi["reference"] == []
    # Averaged, not discarded: -9.0 and -0.1 reach the verdict as -4.55.
    assert kimi["median_tput_pct"] == pytest.approx(-4.55, abs=0.2)

    # No level is italicised any more, because none is excluded.
    body = pj.render(report, "")
    assert "-9.0%" in body
    assert "| *32* |" not in body


def test_reference_only_entry_is_insufficient(tmp_path):
    report = run_judge(tmp_path, flat(tput=-9.0, tpot=6.0), concs=[4, 32])
    assert all(f["status"] == "insufficient" for f in report["families"])


# ------------------------------------------------------------- shape flags ---
def test_shape_detection_is_off_below_three_levels(tmp_path):
    """The other thing the two-level matrix gave up. `_nonmonotonic` needs an
    interior level to compare against its neighbours, so with a pair there is
    nothing to detect -- a family whose levels disagree is averaged and
    reported as a number, with no `unclear` to flag that the average is not
    describing the data. Three levels used to catch exactly this."""
    spec = flat(tput=-0.4, tpot=-0.1)
    spec["GLM-5.2-FP8"] = {"per_conc": {32: -8.6, 256: -7.8}, "tpot": 5.0}
    report = run_judge(tmp_path, spec)

    glm = next(f for f in report["families"] if f["model"] == "GLM-5.2-FP8")
    assert glm["nonmonotonic"] is None
    assert report["verdict"] == "regression"


def test_uniform_drop_is_not_flagged_as_nonmonotonic(tmp_path):
    spec = flat(tput=-0.4, tpot=-0.1)
    spec["Kimi-K3"] = {"tput": -6.0, "tpot": 5.1}
    report = run_judge(tmp_path, spec)
    assert all(not f.get("nonmonotonic") for f in report["families"])


def test_one_judging_level_is_not_enough(tmp_path):
    """A pair is the floor. One level is a single measurement, and the spread
    between levels on identical code reached 2.58 points on mi355-gpu-41, so a
    lone number cannot carry a verdict."""
    report = run_judge(tmp_path, flat(tput=-6.0, tpot=5.0), concs=[256])
    assert all(f["status"] == "insufficient" for f in report["families"])
    assert report["verdict"] == "inconclusive"


def test_a_pair_is_a_mean_and_one_level_moves_it_by_half(tmp_path):
    """What the two-level matrix gave up. With three levels the median ignores
    one aberrant level; with two it carries half of it, so the effective
    resolution is coarser than FAMILY_MEDIAN_PCT reads."""
    spec = flat(tput=-0.4, tpot=-0.1)
    spec["Kimi-K3"] = {"per_conc": {32: -8.0, 256: 0.0}, "tpot": 4.0}
    report = run_judge(tmp_path, spec)
    kimi = next(f for f in report["families"] if f["model"] == "Kimi-K3")
    # -8.0 and 0.0 average to -4.0: half of the outlier reaches the verdict.
    assert kimi["median_tput_pct"] == pytest.approx(-4.0, abs=0.2)


# ------------------------------------------------------- history / sigma ---
def _data_js(points):
    runs = [
        {
            "date": i,
            "commit": {"id": f"c{i}"},
            "benches": [
                {
                    "name": "ATOM::M 8192/1024 c=256 Total Tput (tok/s)",
                    "unit": "tok/s",
                    "value": value,
                    "extra": f"Run: https://github.com/x/y/actions/runs/{run_id}",
                }
            ],
        }
        for i, (run_id, value) in enumerate(points)
    ]
    return "window.BENCHMARK_DATA = " + json.dumps({"entries": {"Benchmark": runs}})


def test_history_deduplicates_by_actions_run_id(tmp_path):
    """The dashboard republishes one run against several commits. Duplicates
    have zero spread, so leaving them in biases sigma toward 'very stable'."""
    src = tmp_path / "data.js"
    src.write_text(
        _data_js([("1", 100.0), ("1", 100.0), ("1", 100.0), ("2", 110.0), ("3", 90.0)])
    )
    # Three of five points collapse to one, leaving three observations -- below
    # the six required, so nothing is reported rather than a fabricated sigma.
    assert pj.load_history(str(src))[1] == {}


def test_history_failure_is_not_fatal(tmp_path):
    assert pj.load_history(str(tmp_path / "absent.js")) == ({}, {})
    (tmp_path / "junk.js").write_text("not json at all")
    assert pj.load_history(str(tmp_path / "junk.js")) == ({}, {})


def test_verdict_holds_without_history(tmp_path):
    """Family linkage needs no noise band; sigma must never gate a verdict."""
    spec = flat(tput=-0.4, tpot=-0.1)
    spec["Kimi-K3"] = {"tput": -6.0, "tpot": 5.1}
    assert run_judge(tmp_path, spec)["verdict"] == "regression"


def test_a_baseline_far_below_normal_outranks_everything(tmp_path):
    """A paired comparison is blind to main already being broken: base and head
    both down 30% gives a delta of zero and reads as clean. Checking the base
    measurement against main's recent level is the only thing here that can
    see it."""
    history = {
        ("M", "8192/1024", c): {"cv": 0.01, "median": 10000.0, "n": 8} for c in JUDGING
    }
    # Both halves sit 30% under what main normally does.
    base = [_result("M", c, 7000.0, 30.0) for c in JUDGING]
    head = [_result("M", c, 7000.0, 30.0) for c in JUDGING]

    report = pj.judge(pj.pair_results(base, head), history, expected_entries=1)
    assert report["verdict"] == "bad_baseline"
    assert report["n_bad_baseline"] == 1

    body = pj.render(report, "")
    assert "under main's recent level" in body
    assert "cannot be acted on" in body
    # Reported once for the model, not repeated per level.
    assert body.count("cannot be acted on") == 1


def test_a_naturally_jumpy_configuration_is_not_condemned(tmp_path):
    """A fixed percentage alone condemns a configuration for behaving the way
    it always has. The 8-run spread on the dashboard reaches 25% at P90, so the
    deviation has to clear the configuration's own noise as well."""
    history = {
        ("M", "8192/1024", c): {"cv": 0.20, "median": 10000.0, "n": 8} for c in JUDGING
    }
    base = [_result("M", c, 7400.0, 30.0) for c in JUDGING]  # -26%, but 1.3 sigma
    head = [_result("M", c, 7400.0, 30.0) for c in JUDGING]
    report = pj.judge(pj.pair_results(base, head), history, expected_entries=1)
    assert report["verdict"] == "clean"
    assert report["n_bad_baseline"] == 0


def test_one_level_alone_is_not_enough_to_condemn_the_baseline(tmp_path):
    """One level off is the shape a scheduling blip takes; several at once is
    the machine or the commit."""
    history = {
        ("M", "8192/1024", c): {"cv": 0.01, "median": 10000.0, "n": 8} for c in JUDGING
    }
    base = [
        _result("M", 64, 7000.0, 30.0),  # only this one is far down
        _result("M", 128, 9900.0, 30.0),
        _result("M", 256, 9950.0, 30.0),
    ]
    head = [
        _result("M", c, v, 30.0)
        for c, v in ((64, 7000.0), (128, 9900.0), (256, 9950.0))
    ]
    report = pj.judge(pj.pair_results(base, head), history, expected_entries=1)
    assert report["n_bad_baseline"] == 0


def test_a_healthy_baseline_does_not_trip_the_sanity_check(tmp_path):
    history = {
        ("M", "8192/1024", c): {"cv": 0.01, "median": 10000.0, "n": 8} for c in JUDGING
    }
    base = [_result("M", c, 9800.0, 30.0) for c in JUDGING]
    head = [_result("M", c, 9750.0, 30.0) for c in JUDGING]
    report = pj.judge(pj.pair_results(base, head), history, expected_entries=1)
    assert report["verdict"] == "clean"
    assert report["n_bad_baseline"] == 0


def test_bad_baseline_outranks_a_trip(tmp_path):
    """Even a clear regression is not reportable against a broken baseline."""
    history = {
        ("M", "8192/1024", c): {"cv": 0.01, "median": 10000.0, "n": 8} for c in JUDGING
    }
    base = [_result("M", c, 7000.0, 30.0) for c in JUDGING]
    head = [_result("M", c, 6300.0, 33.0) for c in JUDGING]  # -10% on top
    report = pj.judge(pj.pair_results(base, head), history, expected_entries=1)
    assert report["verdict"] == "bad_baseline"


def test_sanity_check_is_skipped_without_history(tmp_path):
    """No history, no check -- it must not invent a verdict from absence."""
    base = [_result("M", c, 7000.0, 30.0) for c in JUDGING]
    head = [_result("M", c, 7000.0, 30.0) for c in JUDGING]
    report = pj.judge(pj.pair_results(base, head), {}, expected_entries=1)
    assert report["verdict"] == "clean"
    assert report["n_bad_baseline"] == 0


def _series(model, concs, days, start, end, tpot_start=30.0, tpot_end=30.0):
    """A per-configuration series sliding linearly from start to end."""
    out = {}
    n = 8
    for c in concs:
        pts = []
        for i in range(n):
            frac = i / (n - 1)
            pts.append(
                {
                    "date": (days * 86400 * 1000 * i) // (n - 1),
                    "tput": start + (end - start) * frac,
                    "tpot": tpot_start + (tpot_end - tpot_start) * frac,
                }
            )
        out[(model, "8192/1024", c)] = pts
    return out


def test_main_side_drift_is_reported_apart_from_the_verdict(tmp_path):
    """A slide on main leaves the paired delta honest and the absolute level
    wrong. Reporting only "no change" would read as "fine"."""
    series = _series(
        "M", DRIFT_LEVELS, 14, 10000.0, 9000.0, 30.0, 33.0
    )  # -10%, TPOT +10%
    result = pj.main_drift(series, {"M"})
    drift = result["rows"]
    assert len(drift) == 1
    assert drift[0]["median_pct"] < pj.DRIFT_MEDIAN_TH
    assert drift[0]["n_down"] == len(DRIFT_LEVELS)

    base = [_result("M", c, 9000.0, 33.0) for c in JUDGING]
    head = [_result("M", c, 8995.0, 33.0) for c in JUDGING]
    report = pj.judge(pj.pair_results(base, head), {}, expected_entries=1, drift=result)

    # The PR itself is clean, and the drift does not change that.
    assert report["verdict"] == "clean"
    body = pj.render(report, "")
    assert "<details>" in body
    assert "Sliding on main" in body  # no history stats: falls back to a line


def test_drift_needs_tpot_to_mirror(tmp_path):
    """Throughput sliding with TPOT flat is measurement wobble, not a slowdown."""
    series = _series("M", DRIFT_LEVELS, 14, 10000.0, 9000.0, 30.0, 30.0)
    assert pj.main_drift(series, {"M"})["rows"] == []


def test_drift_is_scoped_to_the_models_measured(tmp_path):
    """Listing every drifting family in the repository would bury the one the
    reader came for."""
    series = _series("Other", DRIFT_LEVELS, 14, 10000.0, 9000.0, 30.0, 33.0)
    assert pj.main_drift(series, {"M"})["rows"] == []
    assert pj.main_drift(series, {"Other"})["rows"] != []


def test_drift_ignores_low_concurrency(tmp_path):
    series = _series("M", [4, 8, 32], 14, 10000.0, 9000.0, 30.0, 33.0)
    assert pj.main_drift(series, {"M"})["rows"] == []


# ------------------------------------------------------------- contract ---
def test_thresholds_report_the_measured_residual_bias(tmp_path):
    """The trip point is not the threshold: a measured, systematic bias favours
    head, so a real drop has to exceed both before it registers. Reporting the
    threshold alone would overstate what the check resolves."""
    report = run_judge(tmp_path, flat())
    bias = report["thresholds"]["measured_residual_bias_pct"]
    assert bias > 0
    assert bias == pj.MEASURED_RESIDUAL_BIAS_PCT


def test_judging_levels_start_at_the_documented_boundary():
    assert pj.JUDGE_MIN_CONC == 32
    assert pj.FAMILY_MIN_CONFIGS == 2
    # Asking for more levels down than a family can have means nothing ever
    # trips, and it fails silently: every run reads clean.
    assert pj.FAMILY_MIN_DOWN <= pj.FAMILY_MIN_CONFIGS
    assert pj.FAMILY_MIN_DOWN == pj.FAMILY_MIN_CONFIGS


def test_cli_never_signals_failure_through_the_exit_code(tmp_path, monkeypatch):
    """The check informs the reviewer; it does not gate the merge. Signalling
    through the exit code would turn it into a required check by accident."""
    spec = flat(tput=-8.0, tpot=6.0)
    base_dir, head_dir = write_pair(tmp_path, spec)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perf_judge.py",
            "--base-dir",
            str(base_dir),
            "--head-dir",
            str(head_dir),
            "--no-history",
            "--expect-entries",
            "5",
            "--output-json",
            str(tmp_path / "verdict.json"),
        ],
    )
    assert pj.main() == 0
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["verdict"] == "regression"


# ======================== perf_check_half.sh ========================

RESULT_FILENAME = "deepseek-v4-pro-mtp3-8192-1024-128-0.8"


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _two_commit_repo(ws, script_in_head_only=False):
    """A git repo with a base and a head commit, standing in for the PR.

    `script_in_head_only` reproduces the merge-base that predates this
    feature: the base commit has no .github/scripts at all, and the head
    commit adds the pairing script. Every other caller wants both commits to
    differ only in marker.txt, which is what the shim reads to tell which one
    is checked out.
    """
    _git(ws, "init", "-q")
    _git(ws, "config", "user.email", "t@example.com")
    _git(ws, "config", "user.name", "t")

    def commit(message):
        _git(ws, "add", "-A")
        _git(ws, "commit", "-qm", message)
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ws,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    (ws / "marker.txt").write_text("base\n")
    base_sha = commit("base")

    if script_in_head_only:
        scripts = ws / ".github" / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "perf_check_half.sh").write_text(HALF_SH.read_text())
        (scripts / "perf_check_half.sh").chmod(0o755)
    (ws / "marker.txt").write_text("head\n")
    head_sha = commit("head")

    return base_sha, head_sha


@pytest.fixture
def workspace(tmp_path):
    """A checked-out PR whose two commits differ only in marker.txt."""
    ws = tmp_path / "ws"
    ws.mkdir()
    return (ws, *_two_commit_repo(ws))


def _docker_shim(bindir, log, concs=(128,), model="DeepSeek-V4-Pro-mtp3"):
    """Write a `docker` shim that records calls and fabricates benchmark results.

    It reads marker.txt from the workspace, so the throughput it reports
    depends on which commit is currently checked out -- that is how the tests
    prove the checkout actually took effect before the benchmark ran.

    One shim, parameterised by concurrency level, rather than a second copy
    for the multi-level case: two shims drift, and the one that is not the
    fixture is the one nobody notices has stopped resembling atom_test.sh.
    """
    # A single level writes `${RESULT_FILENAME}.json` -- the catalog's
    # result_filename already carries isl/osl/conc, so the name is complete.
    # Several levels append their own, using RESULT_FILENAME as the prefix
    # perf_check_half.sh actually globs on.
    writes = "\n".join(textwrap.dedent(f"""\
            cat > "${{RESULT_FILENAME}}{'' if len(concs) == 1 else f'-8192-1024-{conc}-0.8'}.json" <<EOF
        {{"benchmark_backend":"ATOM",
         "benchmark_model_name":"{model}",
         "random_input_len":8192,"random_output_len":1024,
         "max_concurrency":{conc},
         "output_throughput":$((base_tput/2)),"total_token_throughput":$base_tput,
         "mean_ttft_ms":420.0,"mean_tpot_ms":$tpot}}
        EOF""") for conc in concs)

    (bindir / "docker").write_text(textwrap.dedent(f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            echo "ARGS: $*" >> "{log}"
            # `docker exec -i ... bash -l` receives the launch command on stdin.
            if [[ "$*" == *"-i"* ]]; then
              echo "STDIN: $(cat)" >> "{log}"
              exit 0
            fi
            case "$*" in
              *benchmark*)
                marker=$(cat marker.txt)
                if [ "$marker" = "head" ]; then base_tput=16000; tpot=32
                else base_tput=17000; tpot=30; fi
            @WRITES@
                ;;
              *stop*) echo "STOP" >> "{log}" ;;
            esac
            """).replace("@WRITES@", writes))
    (bindir / "docker").chmod(0o755)
    return bindir, log


@pytest.fixture
def fake_docker(tmp_path):
    """The single-level shim, which every pairing-mechanics test uses."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    return _docker_shim(bindir, tmp_path / "docker.log")


def run_half(ws, bindir, sha, half, script=None, **env):
    """Invoke one half the way the workflow does.

    `script` points at the copy to run. It defaults to the one in this
    checkout; the self-hosting test passes a staged copy outside the
    workspace, which is what the workflow does and the only reason the
    parameter exists.
    """
    result = subprocess.run(
        ["bash", str(script or HALF_SH), sha, half],
        cwd=ws,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "CONTAINER": "atom-perf-check",
            "MODEL_PATH": "deepseek-ai/DeepSeek-V4-Pro",
            "ARGS": "--kv_cache_dtype fp8 -tp 8 --method mtp",
            "RESULT_FILENAME": RESULT_FILENAME,
            "CONC": "128",
            "ISL": "8192",
            "OSL": "1024",
            "RANDOM_RANGE_RATIO": "0.8",
            **env,
        },
    )
    return result


# ------------------------------------------------------------ arguments ---
def test_rejects_an_unknown_half(workspace, fake_docker):
    ws, base_sha, _ = workspace
    bindir, _ = fake_docker
    result = run_half(ws, bindir, base_sha, "middle")
    assert result.returncode == 2
    assert "must be warmup|base|head|base2" in result.stderr


def test_requires_its_environment(workspace, fake_docker):
    ws, base_sha, _ = workspace
    bindir, _ = fake_docker
    result = subprocess.run(
        ["bash", str(HALF_SH), base_sha, "base"],
        cwd=ws,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"},
    )
    assert result.returncode != 0
    assert "CONTAINER" in result.stderr


# -------------------------------------------------------------- one half ---
def test_checks_out_the_requested_commit_before_benchmarking(workspace, fake_docker):
    ws, base_sha, _head_sha = workspace
    bindir, _ = fake_docker

    run_half(ws, bindir, base_sha, "base")
    # The shim reports throughput based on marker.txt, so this value can only
    # be produced if the checkout landed before the benchmark ran.
    payload = json.loads(
        (ws / "perf-pair/base" / f"{RESULT_FILENAME}.json").read_text()
    )
    assert payload["total_token_throughput"] == 17000


def test_passes_server_args_through_stdin_not_the_command_line(workspace, fake_docker):
    """Quoting in ARGS must be parsed once, by the container's bash. Inlining it
    into `bash -lc "..."` collapses single-quoted JSON values."""
    ws, base_sha, _ = workspace
    bindir, log = fake_docker
    run_half(ws, bindir, base_sha, "base")
    text = log.read_text()
    assert "STDIN: .github/scripts/atom_test.sh launch" in text
    assert "--method mtp" in text


def test_benchmark_parameters_reach_the_container(workspace, fake_docker):
    """atom_test.sh reads ISL/OSL/CONC/RANDOM_RANGE_RATIO from its own
    environment under `set -u`, so anything missing aborts the benchmark after
    the model has already loaded. CI injects them when the container starts,
    which hides the omission wherever the container was started that way -- it
    surfaced only against a locally started container, as
    "line 447: ISL: unbound variable".
    """
    ws, base_sha, _ = workspace
    bindir, log = fake_docker
    run_half(ws, bindir, base_sha, "base")

    bench_call = [
        ln
        for ln in log.read_text().splitlines()
        if ln.startswith("ARGS:") and "benchmark" in ln
    ]
    assert bench_call, "no benchmark invocation recorded"
    for var in ("ISL=8192", "OSL=1024", "CONC=128", "RANDOM_RANGE_RATIO=0.8"):
        assert var in bench_call[0], f"{var} not passed to the container"


def test_stops_the_server_after_measuring(workspace, fake_docker):
    ws, base_sha, _ = workspace
    bindir, log = fake_docker
    run_half(ws, bindir, base_sha, "base")
    lines = log.read_text().splitlines()
    bench = next(i for i, ln in enumerate(lines) if "benchmark" in ln)
    stop = next(i for i, ln in enumerate(lines) if ln == "STOP")
    assert stop > bench, "stop must follow the benchmark, not precede it"


def test_missing_results_warn_rather_than_fabricate(workspace, tmp_path):
    """A half that produced nothing must leave the directory empty. Inventing a
    result here would turn missing evidence into a passing comparison."""
    ws, base_sha, _ = workspace
    bindir = tmp_path / "silent"
    bindir.mkdir()
    (bindir / "docker").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bindir / "docker").chmod(0o755)

    result = run_half(ws, bindir, base_sha, "base")
    assert result.returncode == 0
    assert "::warning::" in result.stdout
    assert list((ws / "perf-pair/base").glob("*.json")) == []


# ------------------------------------------------------- both halves ---
def test_archiving_survives_the_second_checkout(workspace, fake_docker):
    """The ordering guard: `git clean` between halves would wipe the base
    results if they were not copied out first."""
    ws, base_sha, head_sha = workspace
    bindir, _ = fake_docker

    run_half(ws, bindir, base_sha, "base")
    run_half(ws, bindir, head_sha, "head")

    base_files = list((ws / "perf-pair/base").glob("*.json"))
    head_files = list((ws / "perf-pair/head").glob("*.json"))
    assert len(base_files) == 1, "base results were destroyed by the head checkout"
    assert len(head_files) == 1

    assert json.loads(base_files[0].read_text())["total_token_throughput"] == 17000
    assert json.loads(head_files[0].read_text())["total_token_throughput"] == 16000


def test_runs_when_the_target_commit_does_not_contain_the_script(tmp_path, fake_docker):
    """The merge-base predates this feature, so checking it out removes the
    script from the workspace. Invoking it from the workspace therefore works
    for the first half and fails with "No such file or directory" for the
    second -- the script deletes itself partway through the pairing.

    Guards the fix: the workflow stages the script outside the workspace and
    runs it from there. Reproduced from a real CI failure (exit 127 on the
    HEAD half after the BASE half had succeeded).
    """
    bindir, _ = fake_docker

    ws = tmp_path / "selfhost"
    ws.mkdir()
    base_sha, head_sha = _two_commit_repo(ws, script_in_head_only=True)

    # Staged outside the workspace, which is what the workflow does.
    staged = tmp_path / "staged.sh"
    staged.write_text(HALF_SH.read_text())
    staged.chmod(0o755)

    for sha, half in ((base_sha, "base"), (head_sha, "head")):
        result = run_half(ws, bindir, sha, half, script=staged, MODEL_PATH="m", ARGS="")
        assert result.returncode == 0, f"{half} half failed: {result.stderr[-400:]}"

    assert len(list((ws / "perf-pair/base").glob("*.json"))) == 1
    assert len(list((ws / "perf-pair/head").glob("*.json"))) == 1


def test_warmup_half_discards_its_results(workspace, fake_docker):
    """The warmup exists to pay for the caches the first measured half would
    otherwise populate. A warmup result reaching the judge would be
    indistinguishable from a measurement, so it must not be collected."""
    ws, base_sha, _ = workspace
    bindir, _log = fake_docker

    result = run_half(ws, bindir, base_sha, "warmup")
    assert result.returncode == 0
    assert "results discarded" in result.stdout
    assert list((ws / "perf-pair/warmup").glob("*.json")) == []
    # ... and it asks for a shorter run rather than a full one.
    assert "NUM_PROMPTS_OVERRIDE" in result.stdout


def test_repeated_base_reading_yields_a_drift_figure(tmp_path):
    """base is measured twice, before and after head. The distance between the
    two readings is what drifted while the pairing ran, and the baseline is
    their mean so that drift cancels to first order."""
    base = [_result("M", c, 10000.0, 30.0) for c in JUDGING]
    base2 = [_result("M", c, 10200.0, 30.0) for c in JUDGING]  # +2% over the run
    head = [_result("M", c, 10100.0, 30.0) for c in JUDGING]

    pairs = pj.pair_results(base, head, base2)
    assert len(pairs) == len(JUDGING)
    for p in pairs:
        assert p["drift_pct"] == pytest.approx(2.0, abs=0.01)
        # Naively head/base reads +1%; against the mean of the two it is flat,
        # which is what a commit that changed nothing should look like.
        assert p["tput_pct"] == pytest.approx(0.0, abs=0.02)


def test_drift_beyond_the_resolvable_threshold_is_not_trusted(tmp_path):
    """When the base commit does not measure the same twice, the comparison
    cannot answer the question, whatever the deltas look like."""
    base = [_result("M", c, 10000.0, 30.0) for c in JUDGING]
    base2 = [_result("M", c, 10800.0, 30.0) for c in JUDGING]  # +8%
    head = [_result("M", c, 9600.0, 33.0) for c in JUDGING]  # would look like -8%

    report = pj.judge(pj.pair_results(base, head, base2), {}, expected_entries=1)
    assert report["verdict"] == "untrustworthy"
    assert report["n_drifted"] == 1
    body = pj.render(report, "")
    assert "did not measure the same twice" in body
    assert "cannot be attributed to the change" in body


def test_absent_second_reading_falls_back_to_a_single_baseline(tmp_path):
    base = [_result("M", c, 10000.0, 30.0) for c in JUDGING]
    head = [_result("M", c, 9000.0, 33.0) for c in JUDGING]
    pairs = pj.pair_results(base, head)
    assert all(p["drift_pct"] is None for p in pairs)
    assert pairs[0]["tput_pct"] == pytest.approx(-10.0, abs=0.01)
    report = pj.judge(pairs, {}, expected_entries=1)
    assert report["verdict"] == "regression"


def test_pipeline_end_to_end_reaches_a_verdict(workspace, fake_docker):
    """Both halves, then the judge, on files the script actually produced --
    rather than on fixtures hand-shaped to match what it is assumed to write."""
    ws, base_sha, head_sha = workspace
    bindir, _ = fake_docker

    run_half(ws, bindir, base_sha, "base")
    run_half(ws, bindir, head_sha, "head")

    pairs = pj.pair_results(
        pj.load_results(ws / "perf-pair/base", recursive=True),
        pj.load_results(ws / "perf-pair/head", recursive=True),
    )
    assert len(pairs) == 1
    assert pairs[0]["model"] == "DeepSeek-V4-Pro-mtp3"
    assert pairs[0]["conc"] == 128
    assert pairs[0]["tput_pct"] == pytest.approx(-5.88, abs=0.05)

    # One concurrency level cannot satisfy the linkage criterion, and the judge
    # must say so rather than call a single -5.9% reading a regression.
    #
    # `inconclusive`, not `partial`: nothing at all could be judged here.
    # `partial` is the weaker claim -- some entries were judged and came back
    # clean, but the coverage behind that was incomplete.
    report = pj.judge(pairs, {}, expected_entries=1)
    assert report["verdict"] == "inconclusive"
    assert report["n_judged"] == 0
    assert report["n_insufficient"] == 1

    body = pj.render(report, "")
    assert "not** a pass" in body
    # The level that did report dropped ~6%; that must not vanish just because
    # the family could not be judged.
    assert "could not be judged" in body
    assert "-5.9%" in body


def test_full_matrix_of_halves_produces_a_regression(workspace, tmp_path):
    """Three judging levels from the script's own output, end to end."""
    ws, base_sha, head_sha = workspace
    bindir = tmp_path / "bin3"
    bindir.mkdir()
    _docker_shim(bindir, tmp_path / "docker3.log", concs=(64, 128, 256), model="M")

    for sha, half in ((base_sha, "base"), (head_sha, "head")):
        run_half(ws, bindir, sha, half, RESULT_FILENAME="run")

    pairs = pj.pair_results(
        pj.load_results(ws / "perf-pair/base", recursive=True),
        pj.load_results(ws / "perf-pair/head", recursive=True),
    )
    report = pj.judge(pairs, {}, expected_entries=1)
    assert len(pairs) == 3
    assert report["verdict"] == "regression"
    family = report["families"][0]
    assert family["n_down"] == 3
    assert family["mirror"] is True


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_pr_comment_script_is_valid_javascript():
    """The github-script block is JS embedded in YAML; nothing else checks it."""
    import yaml

    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "atom-perf-check.yaml").read_text()
    )
    steps = workflow["jobs"]["judge"]["steps"]
    script = next(
        s["with"]["script"]
        for s in steps
        if str(s.get("uses", "")).startswith("actions/github-script")
    )
    # Wrap in an async function: the body uses top-level await.
    subprocess.run(
        ["node", "--check", "-"],
        input=f"async function main() {{\n{script}\n}}",
        text=True,
        check=True,
        capture_output=True,
    )


DUMMY_ARGS = "--load_dummy=xavier --fake-eplb"


def test_dummy_weight_args_reach_every_phase(workspace, fake_docker):
    """Weight-loading flags must apply to warmup, base and head alike.

    A pairing where one phase loaded real weights and another loaded dummy
    ones would report the difference between two models as a code delta. The
    flags are appended inside this script for exactly that reason -- there is
    no per-phase place to set them, so there is no way to set them for one
    phase only. The cell's own server args have to survive alongside.
    """
    ws, base_sha, head_sha = workspace
    bindir, log = fake_docker
    for half, sha in (("warmup", base_sha), ("base", base_sha), ("head", head_sha)):
        run_half(ws, bindir, sha, half, DUMMY_WEIGHT_ARGS=DUMMY_ARGS)

    launches = [
        ln
        for ln in log.read_text().splitlines()
        if ln.startswith("STDIN:") and "launch" in ln
    ]
    assert len(launches) == 3, f"expected one launch per phase, got {launches}"
    for ln in launches:
        assert DUMMY_ARGS in ln, f"dummy flags missing from: {ln}"
        assert "--method mtp" in ln, f"cell server args dropped from: {ln}"


def test_dummy_weight_args_are_recorded_with_the_result(workspace, fake_docker):
    """SERVER_ARGS lands in the result JSON's metadata. If it carried only the
    cell's args, an artifact measured on dummy weights would be indis-
    tinguishable from one measured on the checkpoint."""
    ws, base_sha, _ = workspace
    bindir, log = fake_docker
    run_half(ws, bindir, base_sha, "base", DUMMY_WEIGHT_ARGS=DUMMY_ARGS)

    bench = [
        ln
        for ln in log.read_text().splitlines()
        if ln.startswith("ARGS:") and "benchmark" in ln
    ]
    assert bench, "no benchmark invocation recorded"
    assert DUMMY_ARGS in bench[0], "dummy flags not recorded in SERVER_ARGS"


def test_no_dummy_args_leaves_the_launch_line_untouched(workspace, fake_docker):
    """Clearing the variable has to restore a real-weight run exactly, not
    leave a bare `--load_dummy` or an empty value behind."""
    ws, base_sha, _ = workspace
    bindir, log = fake_docker
    run_half(ws, bindir, base_sha, "base", DUMMY_WEIGHT_ARGS="")

    text = log.read_text()
    assert "load_dummy" not in text and "fake-eplb" not in text
    launch = [
        ln for ln in text.splitlines() if ln.startswith("STDIN:") and "launch" in ln
    ]
    assert len(launch) == 1
    assert launch[0].rstrip().endswith("--kv_cache_dtype fp8 -tp 8 --method mtp")


def test_every_job_below_the_gate_reads_the_gate_decision():
    """The label gate only bites if each job checks its output.

    `check_heavy_ci_gate.sh` exits 0 whether or not the PR carries `ci:perf` --
    an unlabelled PR is not a failure, it is a PR that did not ask for a
    benchmark -- and writes the verdict to `should_run`. So `needs: gate`
    gates nothing on its own: `needs` blocks on failure, and the gate never
    fails. Without this test the workflow reads as gated and runs 25 GPU jobs
    on every push -- which is what it did until this test was added, with the
    label doing nothing at all.
    """
    import yaml

    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "atom-perf-check.yaml").read_text()
    )
    assert (
        workflow["jobs"]["gate"]["outputs"]["should_run"]
        == "${{ steps.gate.outputs.should_run }}"
    )
    for name, job in workflow["jobs"].items():
        if name == "gate":
            continue
        assert "gate" in job["needs"], f"{name} must depend on the gate"
        assert "needs.gate.outputs.should_run == 'true'" in job["if"], (
            f"{name} lists the gate in `needs` but never reads its decision, "
            "so it runs on unlabelled PRs"
        )


# --- matrix builder -------------------------------------------------------
# The matrix is built by a script rather than inline in the workflow so these
# can exist at all; `minimax-m3-mxfp8` (the catalog calls it `m3-mxfp8`) is the
# mistake that motivated it, and it cost a full GPU run to find.

MATRIX_PY = SCRIPTS / "build_perf_check_matrix.py"

MATRIX_ENV = {
    "PERF_MODELS": "deepseek-v4-pro,kimi-k3,m3-mxfp8,glm-5-2-fp8",
    "PERF_ISL": "8192",
    "PERF_OSL": "1024",
    "PERF_RATIO": "0.8",
    "PERF_JUDGING_CONCS": "64,128,256",
    "PERF_REFERENCE_CONCS": "4",
}


def _run_matrix(**overrides):
    env = {**os.environ, **MATRIX_ENV, **overrides}
    env.pop("GITHUB_OUTPUT", None)  # emit to stdout instead
    return subprocess.run(
        [sys.executable, str(MATRIX_PY)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_matrix_is_the_agreed_set():
    """Five entries across four levels, and the MTP3 pair alongside its base."""
    proc = _run_matrix()
    assert proc.returncode == 0, proc.stderr
    cells = json.loads(proc.stdout)
    assert len(cells) == 20
    by_entry = {}
    for c in cells:
        by_entry.setdefault(c["prefix"] + c["suffix"], []).append(c["conc"])
    assert set(by_entry) == {
        "deepseek-v4-pro",
        "deepseek-v4-pro-mtp3",
        "kimi-k3",
        "m3-mxfp8",
        "glm-5-2-fp8",
    }
    for entry, concs in by_entry.items():
        assert sorted(concs) == [4, 64, 128, 256], entry


def test_matrix_names_an_unknown_prefix():
    """A typo must say which name is wrong, not just that a count is short."""
    proc = _run_matrix(PERF_MODELS="deepseek-v4-pro,minimax-m3-mxfp8")
    assert proc.returncode == 1
    assert "minimax-m3-mxfp8" in proc.stderr
    assert "not a catalog prefix" in proc.stderr


def test_matrix_names_the_level_a_variant_cannot_reach():
    """A band that excludes a level is not a missing variant, and says so.

    Every DPA/TBO variant has `conc_min` at 64 or above, so adding one while a
    reference level below that is configured builds a short matrix. The message
    has to point at the band rather than at the entry, which is present.
    """
    proc = _run_matrix(PERF_MODELS="deepseek-v4-pro", PERF_REFERENCE_CONCS="4")
    assert proc.returncode == 0, proc.stderr  # base and MTP3 both reach c=4

    import build_perf_check_matrix as bpm

    keep = bpm.selected_variants({"deepseek-v4-pro"})
    assert keep == {("deepseek-v4-pro", ""), ("deepseek-v4-pro", "-mtp3")}
    assert bpm.selected_variants({"kimi-k3"}) == {("kimi-k3", "")}


def test_matrix_rejects_a_renamed_pair_variant():
    """Half a pair silently vanishing is the failure the pair exists to catch."""
    import build_perf_check_matrix as bpm

    problems = bpm.check_against_catalog(
        str(REPO / ".github" / "benchmark" / "models.json"),
        {"deepseek-v4-pro"},
        {("deepseek-v4-pro", ""), ("deepseek-v4-pro", "-no-such-variant")},
    )
    assert len(problems) == 1
    assert "-no-such-variant" in problems[0]
