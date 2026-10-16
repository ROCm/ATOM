import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = REPO_ROOT / ".github" / "scripts" / "check_heavy_ci_gate.sh"

# `gh` is faked below, `jq` is not: the gate parses the event JSON with it. A
# GitHub runner ships jq, which is also the only place this script runs for
# real, so skipping here costs nothing there. Worth knowing that it costs
# everything on a box without jq -- these become silently unrun rather than red.
pytestmark = pytest.mark.skipif(
    shutil.which("jq") is None, reason="the CI gate script parses its event with jq"
)


def _run_gate(
    tmp_path,
    *,
    labels=(),
    review_decision="",
    review_query_fails=False,
    allowed="ci:full,ci:vllm",
    event_name="pull_request",
    action="synchronize",
    review_state="approved",
    run_attempt=1,
    current_pr=None,
    pr_query_fails=False,
    changed_files="atom/model_engine/scheduler.py",
    runs=(),
    jobs=None,
    runs_query_fails=False,
    jobs_query_fails=False,
):
    event_path = tmp_path / "event.json"
    output_path = tmp_path / "output.txt"
    summary_path = tmp_path / "summary.md"
    event_path.write_text(
        json.dumps(
            {
                "action": action,
                "review": {"state": review_state},
                "pull_request": {
                    "number": 2166,
                    "draft": False,
                    "base": {"ref": "main"},
                    "head": {
                        "sha": "current-sha",
                        "ref": "fix-ci",
                        "repo": {"full_name": "ROCm/ATOM"},
                    },
                    "labels": [{"name": label} for label in labels],
                },
            }
        ),
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${FAKE_API_LOG}"
if [[ " $* " == *"/pulls/"*"/files"* ]]; then
  printf '%s\\n' "${FAKE_CHANGED_FILES}"
elif [[ "${2:-}" == "repos/ROCm/ATOM/pulls/2166" ]]; then
  if [[ "${FAKE_PR_QUERY_FAIL:-0}" == "1" ]]; then
    exit 1
  fi
  printf '%s\\n' "${FAKE_CURRENT_PR}"
elif [[ " $* " == *"/actions/workflows/"* ]]; then
  [[ " $* " == *" --paginate "* && " $* " == *" --slurp "* ]]
  if [[ "${FAKE_RUNS_QUERY_FAIL}" == "1" ]]; then
    exit 1
  fi
  printf '%s\\n' "${FAKE_RUNS}"
elif [[ " $* " == *"/actions/runs/"*"/jobs?filter=all"* ]]; then
  [[ " $* " == *" --paginate "* && " $* " == *" --slurp "* ]]
  if [[ "${FAKE_JOBS_QUERY_FAIL}" == "1" ]]; then
    exit 1
  fi
  endpoint="${!#}"
  run_id="${endpoint#*/actions/runs/}"
  run_id="${run_id%%/*}"
  jq -c --arg id "$run_id" '.[$id] // [{"jobs": []}]' <<< "${FAKE_JOBS}"
elif [[ "${2:-}" == "graphql" ]]; then
  if [[ "${FAKE_REVIEW_QUERY_FAIL:-0}" == "1" ]]; then
    exit 1
  fi
  printf '%s\\n' "${FAKE_REVIEW_DECISION:-}"
elif [[ " $* " == *" /labels "* ]]; then
  exit 0
else
  echo "unexpected gh invocation: $*" >&2
  exit 2
fi
""",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "GITHUB_OUTPUT": str(output_path),
            "GITHUB_STEP_SUMMARY": str(summary_path),
            "GITHUB_EVENT_NAME": event_name,
            "GITHUB_EVENT_PATH": str(event_path),
            "GITHUB_RUN_ATTEMPT": str(run_attempt),
            "GITHUB_RUN_ID": "200",
            "GITHUB_REPOSITORY": "ROCm/ATOM",
            "CI_GATE_WORKFLOW": "atom-vllm-test.yaml",
            "CI_GATE_LABELS": allowed,
            "CI_GATE_PATHS_IGNORE": "docs/**",
            "FAKE_REVIEW_DECISION": review_decision,
            "FAKE_REVIEW_QUERY_FAIL": "1" if review_query_fails else "0",
            "FAKE_PR_QUERY_FAIL": "1" if pr_query_fails else "0",
            "FAKE_CURRENT_PR": json.dumps(
                current_pr
                if current_pr is not None
                else _current_pr(labels=[{"name": label} for label in labels])
            ),
            "FAKE_CHANGED_FILES": changed_files,
            "FAKE_RUNS": json.dumps(
                [{"workflow_runs": []}, {"workflow_runs": list(runs)}]
            ),
            "FAKE_JOBS": json.dumps(jobs or {}),
            "FAKE_RUNS_QUERY_FAIL": "1" if runs_query_fails else "0",
            "FAKE_JOBS_QUERY_FAIL": "1" if jobs_query_fails else "0",
            "FAKE_API_LOG": str(tmp_path / "api.log"),
        }
    )
    subprocess.run(["bash", str(GATE_SCRIPT)], check=True, env=env)
    return dict(
        line.split("=", 1)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    )


def test_matching_label_runs_without_review(tmp_path):
    result = _run_gate(tmp_path, labels=("ci:vllm",))

    assert result["should_run"] == "true"
    assert result["reason"] == "label-present"
    assert result["matched_label"] == "ci:vllm"


def test_unrelated_label_does_not_authorize_workflow(tmp_path):
    result = _run_gate(tmp_path, labels=("ci:atom",))

    assert result["should_run"] == "false"
    assert result["reason"] == "not-approved"


def test_current_approval_runs_without_label(tmp_path):
    result = _run_gate(tmp_path, review_decision="APPROVED")

    assert result["should_run"] == "true"
    assert result["reason"] == "current-approval"
    assert result["review_decision"] == "APPROVED"


def test_dismissed_or_superseded_review_does_not_run(tmp_path):
    result = _run_gate(tmp_path, review_decision="REVIEW_REQUIRED")

    assert result["should_run"] == "false"
    assert result["reason"] == "not-approved"
    assert result["approval_count"] == "0"


def test_changes_requested_does_not_run(tmp_path):
    result = _run_gate(tmp_path, review_decision="CHANGES_REQUESTED")

    assert result["should_run"] == "false"
    assert result["reason"] == "changes-requested"
    assert result["changes_requested_count"] == "1"


def test_review_query_failure_fails_closed(tmp_path):
    result = _run_gate(tmp_path, review_query_fails=True)

    assert result["should_run"] == "false"
    assert result["reason"] == "review-query-failed"


def _current_pr(**updates):
    return {
        "state": "open",
        "draft": False,
        "base": {"ref": "main"},
        "head": {"sha": "current-sha"},
        "labels": [],
        **updates,
    }


def test_queued_approval_sees_current_labels(tmp_path):
    result = _run_gate(
        tmp_path,
        labels=("bug",),
        event_name="pull_request_review",
        action="submitted",
        current_pr=_current_pr(labels=[{"name": "ci:vllm"}]),
    )
    assert result["should_run"] == "true"
    assert result["reason"] == "label-present"


def test_queued_approval_does_not_reuse_removed_label(tmp_path):
    result = _run_gate(
        tmp_path,
        labels=("ci:vllm",),
        event_name="pull_request_review",
        action="submitted",
        current_pr=_current_pr(),
    )
    assert result["should_run"] == "false"
    assert result["reason"] == "not-approved"


@pytest.mark.parametrize(
    "updates, reason",
    [
        ({"head": {"sha": "new-sha"}}, "stale-head-sha"),
        ({"state": "closed"}, "pull-request-closed"),
        ({"base": {"ref": "release"}}, "non-main-base-branch"),
        ({"draft": True}, "draft-pull-request"),
    ],
)
def test_queued_approval_revalidates_pr(tmp_path, updates, reason):
    result = _run_gate(
        tmp_path,
        review_decision="APPROVED",
        event_name="pull_request_review",
        action="submitted",
        current_pr=_current_pr(**updates),
    )
    assert result["should_run"] == "false"
    assert result["reason"] == reason


def test_approval_current_pr_query_fails_closed(tmp_path):
    result = _approve(tmp_path, pr_query_fails=True)
    assert result["should_run"] == "false"
    assert result["reason"] == "pr-query-failed"


def test_current_approval_releases_without_label(tmp_path):
    result = _approve(tmp_path)
    assert result["should_run"] == "true"
    assert result["reason"] == "current-approval"


@pytest.mark.parametrize("event_name", ["push", "schedule", "workflow_dispatch"])
def test_non_pr_events_are_unchanged(tmp_path, event_name):
    result = _run_gate(tmp_path, event_name=event_name)
    assert result["should_run"] == "true"
    assert result["reason"] == "non-pull-request-event"


@pytest.mark.parametrize("review_state", ["commented", "changes_requested"])
def test_non_approval_review_cannot_release_gate(tmp_path, review_state):
    result = _run_gate(
        tmp_path,
        event_name="pull_request_review",
        action="submitted",
        review_state=review_state,
        review_decision="APPROVED",
    )
    assert result["should_run"] == "false"
    assert result["reason"] == "review-not-approved-event"


def test_ignored_paths_remain_gated(tmp_path):
    result = _run_gate(
        tmp_path, review_decision="APPROVED", changed_files="docs/guide.rst"
    )
    assert result["should_run"] == "false"
    assert result["reason"] == "ignored-paths-only"


def _previous_run(**updates):
    return {
        "id": 100,
        "head_sha": "current-sha",
        "head_branch": "fix-ci",
        "head_repository": {"full_name": "ROCm/ATOM"},
        "event": "pull_request",
        "pull_requests": [{"number": 2166}],
        "conclusion": "success",
        **updates,
    }


def _jobs(signal="success"):
    return [
        {"jobs": [{"name": "Check Heavy CI Gate", "conclusion": "success"}]},
        {"jobs": [{"name": "Check Pre Checkin Signal", "conclusion": signal}]},
    ]


def _approve(tmp_path, **kwargs):
    return _run_gate(
        tmp_path,
        **{
            "event_name": "pull_request_review",
            "action": "submitted",
            "review_decision": "APPROVED",
            **kwargs,
        },
    )


def test_first_approval_releases_previously_skipped_heavy_jobs(tmp_path):
    result = _approve(tmp_path, runs=[_previous_run()], jobs={100: _jobs("skipped")})
    assert result["should_run"] == "true"
    assert result["reason"] == "current-approval"


@pytest.mark.parametrize("signal", ["success", "failure", "cancelled", None])
def test_approval_does_not_restart_existing_heavy_work(tmp_path, signal):
    result = _approve(tmp_path, runs=[_previous_run()], jobs={100: _jobs(signal)})
    assert result["should_run"] == "false"
    assert result["reason"] == "heavy-ci-already-started"
    assert result["existing_run_id"] == "100"


def test_partial_matrix_failure_keeps_successes_and_failures_in_original_run(tmp_path):
    jobs = _jobs() + [
        {
            "jobs": [
                {"name": "Accuracy (model A)", "conclusion": "success"},
                {"name": "Accuracy (model B)", "conclusion": "failure"},
            ]
        }
    ]
    result = _approve(
        tmp_path, runs=[_previous_run(conclusion="failure")], jobs={100: jobs}
    )
    assert result["should_run"] == "false"


def test_repeat_approval_reuses_previous_review_run(tmp_path):
    result = _approve(
        tmp_path,
        labels=("ci:vllm",),
        runs=[_previous_run(event="pull_request_review")],
        jobs={100: _jobs()},
    )
    assert result["reason"] == "heavy-ci-already-started"


def test_explicit_label_trigger_is_unchanged(tmp_path):
    result = _run_gate(
        tmp_path, action="labeled", labels=("ci:vllm",), runs_query_fails=True
    )
    assert result["should_run"] == "true"
    assert not (tmp_path / "api.log").exists()


@pytest.mark.parametrize(
    "updates",
    [
        {"head_sha": "old-sha"},
        {"head_branch": "different-branch"},
        {"pull_requests": [{"number": 999}]},
        {"event": "push"},
        {"event": "workflow_dispatch"},
        {"id": 200},
        {"id": 201},
        {"pull_requests": [], "head_repository": {"full_name": "someone/ATOM"}},
    ],
)
def test_unrelated_current_or_later_runs_do_not_suppress_approval(tmp_path, updates):
    run = _previous_run(**updates)
    result = _approve(tmp_path, runs=[run], jobs={run["id"]: _jobs()})
    assert result["should_run"] == "true"
    assert "/actions/runs/" not in (tmp_path / "api.log").read_text()


def test_run_without_pr_association_matches_head_repository_and_branch(tmp_path):
    result = _approve(
        tmp_path, runs=[_previous_run(pull_requests=[])], jobs={100: _jobs()}
    )
    assert result["reason"] == "heavy-ci-already-started"


def test_all_attempts_are_checked_not_just_latest_skipped_attempt(tmp_path):
    result = _approve(
        tmp_path, runs=[_previous_run()], jobs={100: _jobs() + _jobs("skipped")}
    )
    assert result["should_run"] == "false"


def test_missing_jobs_in_cancelled_run_do_not_suppress_first_approval(tmp_path):
    result = _approve(tmp_path, runs=[_previous_run(conclusion="cancelled")])
    assert result["should_run"] == "true"


@pytest.mark.parametrize("failed_query", ["runs_query_fails", "jobs_query_fails"])
def test_history_query_failure_fails_closed(tmp_path, failed_query):
    result = _approve(tmp_path, runs=[_previous_run()], **{failed_query: True})
    assert result["should_run"] == "false"
    assert result["reason"] == failed_query.replace("_", "-").replace("fails", "failed")


def test_unreadable_job_history_does_not_allow_duplicate_work(tmp_path):
    result = _approve(tmp_path, runs=[_previous_run()], jobs={100: [{"jobs": None}]})
    assert result["should_run"] == "false"
    assert result["reason"] == "jobs-query-failed"


def test_explicit_rerun_bypasses_automatic_deduplication(tmp_path):
    result = _approve(tmp_path, run_attempt=2, runs_query_fails=True)
    assert result["should_run"] == "true"
    assert "/actions/workflows/" not in (tmp_path / "api.log").read_text()


def test_queued_approval_revalidates_head_sha(tmp_path):
    result = _approve(tmp_path, current_pr=_current_pr(head={"sha": "new-sha"}))
    assert result["should_run"] == "false"
    assert result["reason"] == "stale-head-sha"


def test_workflow_wiring_preserves_existing_checks_and_separates_noop_reviews():
    for filename in ("atom-test.yaml", "atom-vllm-test.yaml", "atom-sglang-test.yaml"):
        config = yaml.safe_load(
            (REPO_ROOT / ".github/workflows" / filename).read_text()
        )
        events = config.get("on", config.get(True))
        assert "pull_request_review" in events
        assert "labeled" in events["pull_request"]["types"]
        concurrency = config["concurrency"]
        assert "github.event.review.state" not in concurrency["cancel-in-progress"]
        assert (
            "github.event_name != 'pull_request_review'"
            if filename != "atom-sglang-test.yaml"
            else "github.event_name == 'pull_request'"
        ) in concurrency["cancel-in-progress"]
        assert "github.event.review.state != 'approved'" in concurrency["group"]
        assert "format('-review-{0}', github.run_id)" in concurrency["group"]
        assert config["jobs"]["ci-gate"]["permissions"]["actions"] == "read"
        gate_env = next(
            step["env"]
            for step in config["jobs"]["ci-gate"]["steps"]
            if step.get("id") == "gate"
        )
        assert gate_env["CI_GATE_WORKFLOW"] == filename
        assert config["jobs"]["check-signal"]["name"] == "Check Pre Checkin Signal"
        if filename == "atom-test.yaml":
            smoke = config["jobs"]["offline-smoke-test"]
            assert smoke["if"] == "${{ github.event_name != 'pull_request_review' }}"
            assert "'Offline smoke not requested'" in smoke["name"]
            assert "'Offline inference smoke test'" in smoke["name"]
            assert config["jobs"]["atom-test"]["name"] == "Accuracy"
            assert {"push", "schedule", "workflow_dispatch"} <= events.keys()
