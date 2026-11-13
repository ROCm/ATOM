# SPDX-License-Identifier: MIT
"""The shared speculative-acceptance gate, exercised as CI invokes it.

It is now the only thing standing between a broken draft head and a green
cell, on both the PR and the nightly path, so the cases below are the ones
that distinguish it from the overall-only check it replaced: a per-position
collapse that leaves the overall rate respectable, and a missing metric, which
has to fail rather than pass.

Both readers are run. The nightly host has jq and no python, the PR path runs
in a container with python and no jq, and a gate that answers differently
depending on which is installed would be worse than either.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[1] / ".github/scripts/check_mtp_acceptance.sh"


def _result(tmp_path, overall=0.47, per_pos=(0.78, 0.62, 0.43, 0.32, 0.21)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "result.json"
    meta = {}
    if overall is not None:
        meta["mtp_acceptance_overall"] = overall
    if per_pos is not None:
        meta["mtp_per_pos_acceptance"] = list(per_pos)
    path.write_text(json.dumps({"atom_ci_metadata": meta}))
    return path


def _run(result, overall_threshold, per_pos="", *, with_jq=True):
    env = dict(os.environ)
    # Pin the reader rather than hiding jq from PATH: stripping the entries
    # that hold it also strips the one that holds bash.
    env["MTP_GATE_READER"] = "jq" if with_jq else "python"
    return subprocess.run(
        [str(GATE), str(result), str(overall_threshold), per_pos],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


READERS = pytest.mark.parametrize(
    "with_jq",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                shutil.which("jq") is None, reason="jq reader needs jq"
            ),
        ),
        False,
    ],
    ids=["jq", "python"],
)


@READERS
def test_a_healthy_run_passes(tmp_path, with_jq):
    done = _run(_result(tmp_path), 0.35, "0.55,0.40,0.25,0.18,0.10", with_jq=with_jq)
    assert done.returncode == 0, done.stderr
    assert "PASSED" in done.stdout


@READERS
def test_a_per_position_collapse_fails_even_with_a_healthy_overall(tmp_path, with_jq):
    """The case the PR-side copy could not see.

    First position intact, the rest on the floor: the overall rate stays above
    its threshold, so an overall-only gate reports this cell green.
    """
    result = _result(tmp_path, overall=0.40, per_pos=(0.95, 0.02, 0.01, 0.0, 0.0))
    done = _run(result, 0.35, "0.55,0.40,0.25,0.18,0.10", with_jq=with_jq)
    assert done.returncode == 1
    assert "position 1 acceptance 0.02" in done.stdout
    # The overall line still reports a pass, which is the point.
    assert "MTP overall acceptance: 0.4" in done.stdout


@READERS
def test_a_low_overall_fails(tmp_path, with_jq):
    done = _run(_result(tmp_path, overall=0.008), 0.35, with_jq=with_jq)
    assert done.returncode == 1
    assert "overall acceptance 0.008" in done.stdout


@READERS
def test_a_missing_metric_fails_rather_than_passes(tmp_path, with_jq):
    """Absent means the spec-decode metrics were never captured."""
    done = _run(_result(tmp_path, overall=None), 0.35, with_jq=with_jq)
    assert done.returncode == 1
    assert "mtp_acceptance_overall missing" in done.stderr


@READERS
def test_a_missing_position_fails(tmp_path, with_jq):
    result = _result(tmp_path, per_pos=(0.78, 0.62))
    done = _run(result, 0.35, "0.55,0.40,0.25", with_jq=with_jq)
    assert done.returncode == 1
    assert "position 2 acceptance missing" in done.stdout


@READERS
def test_without_per_position_thresholds_only_the_overall_is_gated(tmp_path, with_jq):
    result = _result(tmp_path, overall=0.47, per_pos=(0.01, 0.01, 0.01))
    done = _run(result, 0.35, "", with_jq=with_jq)
    assert done.returncode == 0


def test_both_readers_agree_on_every_case(tmp_path):
    """A gate whose answer depends on which reader is installed is not a gate."""
    if shutil.which("jq") is None:
        pytest.skip("needs jq to compare the two readers")
    cases = [
        (0.47, (0.78, 0.62, 0.43), "0.55,0.40,0.25"),
        (0.40, (0.95, 0.02, 0.01), "0.55,0.40,0.25"),
        (0.008, (0.1, 0.1, 0.1), ""),
    ]
    for index, (overall, per_pos, thresholds) in enumerate(cases):
        result = _result(tmp_path / str(index), per_pos=per_pos, overall=overall)
        with_jq = _run(result, 0.35, thresholds, with_jq=True)
        with_py = _run(result, 0.35, thresholds, with_jq=False)
        assert with_jq.returncode == with_py.returncode, (overall, per_pos)


def test_a_missing_result_file_is_an_error_not_a_pass(tmp_path):
    done = _run(tmp_path / "absent.json", 0.35)
    assert done.returncode == 2
    assert "no results JSON" in done.stderr
