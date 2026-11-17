# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run atom-nightly-deadline.yaml's real Bash step against a fake `gh`:
the job must go red whenever runs could not be listed or stopped."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parent.parent
    / ".github/workflows/atom-nightly-deadline.yaml"
)
BENCHMARK = "atom-sglang-benchmark.yaml"
ACCURACY = "atom-sglang-accuracy-validation.yaml"
VLLM_BENCHMARK = "atom-vllm-benchmark.yaml"
VLLM_ACCURACY = "atom-vllm-accuracy-validation.yaml"

# State lives in $FAKE_GH: unfinished/<workflow> holds a run id per line,
# list_fails/<workflow> makes listing that workflow fail, and cancel_mode /
# force-cancel_mode say what a POST does: "stops" the run, "ignored" (accepted,
# run keeps going) or "fails" (the call errors). Every call goes to calls.
FAKE_GH = r"""#!/usr/bin/env bash
echo "$*" >> "${FAKE_GH}/calls"
if [ "$2" = "-X" ]; then
  path="$4"; action="${path##*/}"; id="${path%/*}"; id="${id##*/}"
  mode="$(cat "${FAKE_GH}/${action}_mode")"
  [ "${mode}" = "fails" ] && exit 1
  if [ "${mode}" = "stops" ]; then
    for f in "${FAKE_GH}"/unfinished/*; do sed -i "/^${id}$/d" "$f"; done
  fi
  exit 0
fi
workflow="${2#*/actions/workflows/}"; workflow="${workflow%%/*}"
[ -e "${FAKE_GH}/list_fails/${workflow}" ] && exit 1
cat "${FAKE_GH}/unfinished/${workflow}"
"""


def _script():
    job = yaml.safe_load(WORKFLOW.read_text())["jobs"]["stop-overrunning-nightly-runs"]
    step = next(s for s in job["steps"] if "run" in s)
    return job["env"]["WORKFLOWS"], step["run"]


@pytest.fixture
def deadline(tmp_path):
    state = tmp_path / "state"
    for sub in ("unfinished", "list_fails"):
        (state / sub).mkdir(parents=True)
    (state / "calls").touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(FAKE_GH)
    gh.chmod(0o755)
    (bin_dir / "sleep").symlink_to(shutil.which("true"))
    workflows, script = _script()

    def run(
        unfinished=None,
        list_fails=(),
        cancel="stops",
        force_cancel="stops",
    ):
        for workflow in workflows.split():
            ids = (unfinished or {}).get(workflow, [])
            (state / "unfinished" / workflow).write_text("".join(f"{i}\n" for i in ids))
        for workflow in list_fails:
            (state / "list_fails" / workflow).touch()
        (state / "cancel_mode").write_text(cancel)
        (state / "force-cancel_mode").write_text(force_cancel)
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_GH": str(state),
            "GH_REPO": "ROCm/ATOM",
            "WORKFLOWS": workflows,
            "GRACE_SECONDS": "0",
            "SETTLE_SECONDS": "0",
        }
        result = subprocess.run(
            ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
        )
        posts = [
            line.split()[3].rsplit("/", 2)[1:][::-1]
            for line in (state / "calls").read_text().splitlines()
            if line.startswith("api -X POST")
        ]
        return (
            result.returncode,
            [(action, int(run_id)) for action, run_id in posts],
            result.stdout,
        )

    return run


def test_nothing_unfinished_is_green(deadline):
    code, posts, _ = deadline()
    assert (code, posts) == (0, [])


def test_deadline_covers_sglang_and_vllm_benchmark_and_accuracy():
    assert _script()[0].split() == [BENCHMARK, ACCURACY, VLLM_BENCHMARK, VLLM_ACCURACY]


def test_runs_that_stop_on_cancel_are_not_force_cancelled(deadline):
    code, posts, _ = deadline({BENCHMARK: [1], ACCURACY: [2], VLLM_BENCHMARK: [3]})
    assert code == 0
    assert posts == [("cancel", 1), ("cancel", 2), ("cancel", 3)]


def test_runs_that_ignore_cancel_are_force_cancelled(deadline):
    code, posts, _ = deadline({BENCHMARK: [1]}, cancel="ignored")
    assert code == 0
    assert posts == [("cancel", 1), ("force-cancel", 1)]


def test_failed_cancel_is_covered_by_force_cancel(deadline):
    code, posts, out = deadline({BENCHMARK: [1]}, cancel="fails")
    assert code == 0
    assert posts == [("cancel", 1), ("force-cancel", 1)]
    assert "::warning::Cancelling run 1 failed" in out


@pytest.mark.parametrize("unlisted", [BENCHMARK, ACCURACY, VLLM_BENCHMARK])
def test_a_workflow_that_cannot_be_listed_is_red_but_the_others_are_still_stopped(
    deadline, unlisted
):
    runs = {BENCHMARK: 1, ACCURACY: 2, VLLM_BENCHMARK: 3}
    code, posts, out = deadline(
        {workflow: [run] for workflow, run in runs.items()}, list_fails=[unlisted]
    )
    assert code == 1
    assert posts == [("cancel", run) for w, run in runs.items() if w != unlisted]
    assert f"::error::Could not list the runs of {unlisted}." in out


def test_refused_force_cancel_is_red(deadline):
    code, posts, out = deadline(
        {BENCHMARK: [1]}, cancel="ignored", force_cancel="fails"
    )
    assert code == 1
    assert posts == [("cancel", 1), ("force-cancel", 1)]
    assert "::error::Force-cancelling run 1 failed." in out


def test_run_still_unfinished_after_force_cancel_is_red(deadline):
    code, _, out = deadline({BENCHMARK: [1]}, cancel="ignored", force_cancel="ignored")
    assert code == 1
    assert "::error::Still unfinished after force-cancel: 1" in out
