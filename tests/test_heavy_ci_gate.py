import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from atomesh_workflow_helpers import github_expressions

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = REPO_ROOT / ".github" / "scripts" / "check_heavy_ci_gate.sh"
# atom-test's name for the first job of a no-op label run (CI_GATE_NOOP_JOB).
NOOP_JOB = "Check Heavy CI Gate (no-op label)"

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
    live_labels=None,
    labels_query_fails=False,
    review_decision="",
    review_query_fails=False,
    allowed="ci:full,ci:vllm",
    event_name="pull_request",
    action="synchronize",
    added_label=None,
    review_state="approved",
    run_attempt=1,
    current_pr=None,
    pr_query_fails=False,
    changed_files="atom/model_engine/scheduler.py",
    runs=(),
    jobs=None,
    runs_query_fails=False,
    jobs_query_fails=False,
    pr_check_job="Offline inference smoke test",
    noop_job=NOOP_JOB,
    paths_ignore="docs/**",
    paths_include="",
):
    event_path = tmp_path / "event.json"
    output_path = tmp_path / "output.txt"
    summary_path = tmp_path / "summary.md"
    event_path.write_text(
        json.dumps(
            {
                "action": action,
                **({"label": {"name": added_label}} if added_label else {}),
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
elif [[ " $* " == *"/issues/"*"/labels"* ]]; then
  if [[ "${FAKE_LABELS_QUERY_FAIL:-0}" == "1" ]]; then
    exit 1
  fi
  printf '%s\\n' "${FAKE_LIVE_LABELS}"
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
            "CI_GATE_PR_CHECK_JOB": pr_check_job,
            "CI_GATE_NOOP_JOB": noop_job,
            "CI_GATE_LABELS": allowed,
            "CI_GATE_PATHS_IGNORE": paths_ignore,
            "CI_GATE_PATHS_INCLUDE": paths_include,
            "FAKE_REVIEW_DECISION": review_decision,
            "FAKE_REVIEW_QUERY_FAIL": "1" if review_query_fails else "0",
            "FAKE_PR_QUERY_FAIL": "1" if pr_query_fails else "0",
            "FAKE_CURRENT_PR": json.dumps(
                current_pr
                if current_pr is not None
                else _current_pr(labels=[{"name": label} for label in labels])
            ),
            "FAKE_LIVE_LABELS": "\n".join(
                labels if live_labels is None else live_labels
            ),
            "FAKE_LABELS_QUERY_FAIL": "1" if labels_query_fails else "0",
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
    assert result["decidable"] == "true"
    assert result["smoke_requested"] == "true"
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
    assert result["decidable"] == "false"
    assert result["smoke_requested"] == "false"


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
    assert result["decidable"] == "true"
    assert result["smoke_requested"] == "false"


def test_included_paths_can_release_gate(tmp_path):
    result = _run_gate(
        tmp_path,
        review_decision="APPROVED",
        changed_files="atom/mesh/router.py",
        paths_ignore="",
        paths_include="atom/mesh/**",
    )

    assert result["should_run"] == "true"
    assert result["reason"] == "current-approval"


def test_paths_outside_include_remain_gated(tmp_path):
    result = _run_gate(
        tmp_path,
        review_decision="APPROVED",
        changed_files="atom/plugin/vllm/model.py",
        paths_ignore="",
        paths_include="atom/mesh/**",
    )

    assert result["should_run"] == "false"
    assert result["reason"] == "paths-not-included"
    assert result["smoke_requested"] == "false"


@pytest.mark.parametrize(
    "changed_files, reason",
    [
        ("atom/model_engine/scheduler.py", "paths-not-included"),
        ("atom/plugin/sglang/model.py", "ignored-paths-only"),
    ],
)
def test_include_and_ignore_report_why_nothing_is_relevant(
    tmp_path, changed_files, reason
):
    result = _run_gate(
        tmp_path,
        review_decision="APPROVED",
        changed_files=changed_files,
        paths_ignore="atom/plugin/sglang/**",
        paths_include="atom/plugin/**",
    )

    assert result["should_run"] == "false"
    assert result["reason"] == reason


def _workflow_gate_env(filename):
    config = yaml.safe_load((REPO_ROOT / ".github/workflows" / filename).read_text())
    return next(
        step["env"]
        for step in config["jobs"]["ci-gate"]["steps"]
        if step.get("id") == "gate"
    )


@pytest.mark.parametrize(
    "filename, changed_file, labels, should_run",
    [
        # Approval alone does not run a plugin's CI for a core-only change.
        ("atom-vllm-test.yaml", "atom/model_engine/scheduler.py", (), "false"),
        ("atom-sglang-test.yaml", "atom/model_engine/scheduler.py", (), "false"),
        # Each plugin's own code runs only its own CI.
        ("atom-vllm-test.yaml", "atom/plugin/vllm/model.py", (), "true"),
        ("atom-sglang-test.yaml", "atom/plugin/vllm/model.py", (), "false"),
        ("atom-sglang-test.yaml", "atom/plugin/sglang/model.py", (), "true"),
        ("atom-vllm-test.yaml", "atom/plugin/sglang/model.py", (), "false"),
        # Code both plugins share runs both.
        ("atom-vllm-test.yaml", "atom/plugin/register.py", (), "true"),
        ("atom-sglang-test.yaml", "atom/plugin/register.py", (), "true"),
        ("atom-vllm-test.yaml", "docker/vllm_release.dockerfile", (), "true"),
        ("atom-sglang-test.yaml", "docker/vllm_release.dockerfile", (), "false"),
        # Scripts and files outside atom/plugin/ that the plugin CI runs.
        ("atom-vllm-test.yaml", ".github/scripts/atom_oot_test.sh", (), "true"),
        ("atom-vllm-test.yaml", ".github/scripts/check_mtp_acceptance.sh", (), "true"),
        ("atom-sglang-test.yaml", ".github/scripts/atom_sglang_test.sh", (), "true"),
        ("atom-sglang-test.yaml", ".github/workflows/docker-release.yaml", (), "true"),
        # The plugin's label still asks for its CI on any change.
        ("atom-vllm-test.yaml", "atom/model_engine/scheduler.py", ("ci:vllm",), "true"),
        (
            "atom-sglang-test.yaml",
            "atom/model_engine/scheduler.py",
            ("ci:sglang",),
            "true",
        ),
    ],
)
def test_plugin_ci_runs_for_plugin_paths_or_its_label(
    tmp_path, filename, changed_file, labels, should_run
):
    env = _workflow_gate_env(filename)
    result = _run_gate(
        tmp_path,
        review_decision="APPROVED",
        labels=labels,
        allowed=env["CI_GATE_LABELS"],
        changed_files=changed_file,
        paths_ignore=env["CI_GATE_PATHS_IGNORE"],
        paths_include=env["CI_GATE_PATHS_INCLUDE"],
    )

    assert result["should_run"] == should_run


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


@pytest.mark.parametrize("signal", ["success", None])
def test_approval_does_not_restart_existing_heavy_work(tmp_path, signal):
    result = _approve(tmp_path, runs=[_previous_run()], jobs={100: _jobs(signal)})
    assert result["should_run"] == "false"
    assert result["reason"] == "heavy-ci-already-started"
    assert result["existing_run_id"] == "100"


@pytest.mark.parametrize("signal", ["failure", "cancelled"])
def test_approval_runs_heavy_ci_when_pre_checkin_signal_never_passed(tmp_path, signal):
    result = _approve(tmp_path, runs=[_previous_run()], jobs={100: _jobs(signal)})
    assert result["should_run"] == "true"
    assert result["reason"] == "current-approval"


def test_failed_signal_attempt_does_not_hide_a_passed_one(tmp_path):
    result = _approve(
        tmp_path,
        runs=[_previous_run()],
        jobs={100: _jobs("failure") + _jobs("success")},
    )
    assert result["reason"] == "heavy-ci-already-started"


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


def _label(tmp_path, added_label, labels, **kwargs):
    return _run_gate(
        tmp_path,
        **{
            "action": "labeled",
            "added_label": added_label,
            "labels": labels,
            "allowed": "ci:full,ci:atom",
            **kwargs,
        },
    )


def test_label_that_opens_the_gate_runs_heavy_ci(tmp_path):
    result = _label(tmp_path, "ci:atom", ("ci:atom",), runs_query_fails=True)
    assert result["should_run"] == "true"
    assert result["reason"] == "label-present"
    assert result["smoke_requested"] == "true"
    assert not (tmp_path / "api.log").exists()


@pytest.mark.parametrize(
    "added_label, labels, reason",
    [
        ("LGTM", ("ci:atom", "LGTM"), "label-not-gating"),
        ("LGTM", ("LGTM",), "label-not-gating"),
        ("ci:vllm", ("ci:atom", "ci:vllm"), "label-not-gating"),
        ("ci:full", ("ci:atom", "ci:full"), "gate-already-open"),
    ],
)
def test_label_that_does_not_open_the_gate_is_a_noop(
    tmp_path, added_label, labels, reason
):
    result = _label(tmp_path, added_label, labels, review_decision="APPROVED")
    assert result["should_run"] == "false"
    assert result["reason"] == reason
    assert result["decidable"] == "true"
    assert result["smoke_requested"] == "false"
    assert not (tmp_path / "api.log").exists()


def _opened_gate(tmp_path, events, **kwargs):
    opened = []
    for index, (added_label, labels) in enumerate(events):
        run_dir = tmp_path / str(index)
        run_dir.mkdir(parents=True)
        result = _label(run_dir, added_label, labels, **kwargs)
        if result["should_run"] == "true":
            opened.append(added_label)
    return opened


def test_two_gate_labels_added_at_once_open_the_gate_exactly_once(tmp_path):
    # Both payloads carry both labels; only the event of the first gate label
    # in priority order (ci:full last) opens the gate.
    both = ("ci:full", "ci:atom")
    opened = _opened_gate(tmp_path, [("ci:atom", both), ("ci:full", both)])
    assert opened == ["ci:atom"]


def test_gate_labels_added_one_after_another(tmp_path):
    # ci:atom first: the later ci:full is a no-op.
    opened = _opened_gate(
        tmp_path / "atom-first",
        [("ci:atom", ("ci:atom",)), ("ci:full", ("ci:atom", "ci:full"))],
    )
    assert opened == ["ci:atom"]
    # ci:full first: the later ci:atom opens again -- the one accepted duplicate.
    opened = _opened_gate(
        tmp_path / "full-first",
        [("ci:full", ("ci:full",)), ("ci:atom", ("ci:full", "ci:atom"))],
    )
    assert opened == ["ci:full", "ci:atom"]


@pytest.mark.parametrize(
    "added_label, labels, live_labels, reason",
    [
        ("LGTM", ("ci:atom", "LGTM"), ("ci:atom", "LGTM"), "label-not-gating"),
        (
            "ci:full",
            ("ci:atom", "ci:full"),
            ("ci:atom", "ci:full"),
            "gate-already-open",
        ),
        # ci:atom was removed after ci:full was added beside it. The live labels
        # would open the gate, but the replayed payload keeps this run in its
        # -noop- group under "not requested". (The label query is made to fail
        # below, so a rerun that read the live labels would turn undecidable.)
        ("ci:full", ("ci:atom", "ci:full"), ("ci:full",), "gate-already-open"),
    ],
)
def test_rerun_of_noop_label_run_stays_a_noop(
    tmp_path, added_label, labels, live_labels, reason
):
    # Heavy CI started by a no-op run's rerun would run beside the PR group's.
    result = _label(
        tmp_path,
        added_label,
        labels,
        run_attempt=2,
        live_labels=live_labels,
        labels_query_fails=True,
    )
    assert result["should_run"] == "false"
    assert result["reason"] == reason
    assert result["decidable"] == "true"
    assert result["smoke_requested"] == "false"
    assert not (tmp_path / "api.log").exists()


@pytest.mark.parametrize(
    "added_label, labels, live_labels, kwargs, should_run, reason",
    [
        ("ci:atom", ("ci:atom",), ("ci:atom",), {}, "true", "label-present"),
        # ci:full opened the gate, then ci:atom was added: still gated.
        ("ci:full", ("ci:full",), ("ci:full", "ci:atom"), {}, "true", "label-present"),
        # The label was removed since: the PR's current state decides, as on
        # a synchronize rerun.
        ("ci:atom", ("ci:atom",), (), {}, "false", "not-approved"),
        (
            "ci:atom",
            ("ci:atom",),
            (),
            {"review_decision": "APPROVED"},
            "true",
            "current-approval",
        ),
        (
            "ci:atom",
            ("ci:atom",),
            (),
            {"review_decision": "APPROVED", "changed_files": "docs/guide.rst"},
            "false",
            "ignored-paths-only",
        ),
    ],
)
def test_rerun_of_gate_opening_label_run_is_judged_like_a_code_change(
    tmp_path, added_label, labels, live_labels, kwargs, should_run, reason
):
    result = _label(
        tmp_path, added_label, labels, run_attempt=2, live_labels=live_labels, **kwargs
    )
    assert result["should_run"] == should_run
    assert result["reason"] == reason
    assert result["decidable"] == "true"
    assert "/issues/2166/labels" in (tmp_path / "api.log").read_text()


def test_first_attempt_label_lookup_fails_closed(tmp_path):
    # The payload shows no labels, and the live labels cannot be read.
    result = _run_gate(tmp_path, labels=(), labels_query_fails=True)
    assert result["should_run"] == "false"
    assert result["reason"] == "labels-query-failed"
    assert result["decidable"] == "false"


def test_rerun_of_code_change_reads_live_labels(tmp_path):
    # The replayed payload still shows the labels from before ci:vllm was added.
    result = _run_gate(
        tmp_path, run_attempt=2, labels=("bug",), live_labels=("bug", "ci:vllm")
    )
    assert result["should_run"] == "true"
    assert result["reason"] == "label-present"
    assert "/issues/2166/labels" in (tmp_path / "api.log").read_text()


def test_rerun_label_query_failure_fails_closed(tmp_path):
    result = _label(
        tmp_path, "ci:atom", ("ci:atom",), run_attempt=2, labels_query_fails=True
    )
    assert result["should_run"] == "false"
    assert result["reason"] == "labels-query-failed"
    assert result["decidable"] == "false"


def test_gating_label_on_approved_pr_takes_over_work_it_may_have_cancelled(tmp_path):
    # The concurrency group cannot see the approval, so this run shares the
    # PR's group and may have cancelled the approval's heavy CI.
    result = _label(tmp_path, "ci:atom", ("ci:atom",), review_decision="APPROVED")
    assert result["should_run"] == "true"
    assert result["reason"] == "label-present"


def test_approval_does_not_repeat_heavy_ci_opened_by_label(tmp_path):
    result = _approve(
        tmp_path, labels=("ci:vllm",), runs=[_previous_run()], jobs={100: _jobs()}
    )
    assert result["should_run"] == "false"
    assert result["reason"] == "heavy-ci-already-started"


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


@pytest.mark.parametrize("labels", [(), ("ci:vllm",)])
def test_approval_that_replaced_pending_pr_run_takes_over_its_checks(tmp_path, labels):
    result = _approve(
        tmp_path,
        labels=labels,
        runs=[_previous_run(conclusion="cancelled")],
        jobs={100: [{"jobs": []}, {"jobs": []}]},
    )
    assert result["should_run"] == "true"
    assert result["pr_check_takeover"] == "true"
    assert result["replaced_run_id"] == "100"


def test_noop_label_run_does_not_hide_the_pr_run_an_approval_replaced(tmp_path):
    # A no-op label run (its own concurrency group) started between the pending
    # synchronize run and the approval that replaced it. It is not the PR
    # group's latest run, so the takeover stands.
    noop_jobs = [{"jobs": [{"name": NOOP_JOB, "conclusion": "success"}]}]
    runs = [
        _previous_run(conclusion="cancelled"),
        _previous_run(id=101, conclusion="success"),
    ]
    result = _approve(
        tmp_path, runs=runs, jobs={100: [], 101: noop_jobs}, changed_files="docs/a.md"
    )
    assert result["pr_check_takeover"] == "true"
    assert result["replaced_run_id"] == "100"
    assert result["smoke_requested"] == "true"


def test_queued_noop_label_run_without_jobs_does_not_hide_the_replaced_pr_run(
    tmp_path,
):
    # The same no-op run, still queued: its first job does not exist yet.
    runs = [
        _previous_run(conclusion="cancelled"),
        _previous_run(id=101, conclusion=None, status="queued"),
    ]
    result = _approve(
        tmp_path, runs=runs, jobs={100: [], 101: []}, changed_files="docs/a.md"
    )
    assert result["pr_check_takeover"] == "true"
    assert result["replaced_run_id"] == "100"


def test_pr_check_takeover_is_reported_even_when_approval_is_not_sufficient(tmp_path):
    result = _approve(
        tmp_path,
        review_decision="CHANGES_REQUESTED",
        runs=[_previous_run(conclusion="cancelled")],
    )
    assert result["should_run"] == "false"
    assert result["pr_check_takeover"] == "true"
    assert result["smoke_requested"] == "true"


def test_replaced_run_does_not_hide_heavy_ci_started_by_earlier_run(tmp_path):
    result = _approve(
        tmp_path,
        runs=[_previous_run(), _previous_run(id=150, conclusion="cancelled")],
        jobs={100: _jobs(), 150: []},
    )
    assert result["should_run"] == "false"
    assert result["reason"] == "heavy-ci-already-started"
    assert result["existing_run_id"] == "100"
    assert result["pr_check_takeover"] == "true"
    assert result["replaced_run_id"] == "150"


@pytest.mark.parametrize(
    "runs, jobs",
    [
        ([_previous_run(conclusion="cancelled")], {100: _jobs("skipped")}),
        ([_previous_run(conclusion="success")], {100: []}),
        ([_previous_run(conclusion=None)], {100: []}),
        (
            [
                _previous_run(conclusion="cancelled"),
                _previous_run(id=150, conclusion="success"),
            ],
            {100: [], 150: _jobs("skipped")},
        ),
        ([_previous_run(conclusion="cancelled", event="pull_request_review")], {}),
        ([_previous_run(conclusion="cancelled", id=201)], {}),
        ([_previous_run(conclusion="cancelled", head_sha="old-sha")], {}),
    ],
)
def test_pr_run_that_started_or_is_unrelated_is_not_treated_as_replaced(
    tmp_path, runs, jobs
):
    result = _approve(tmp_path, runs=runs, jobs=jobs)
    assert "pr_check_takeover" not in result


@pytest.mark.parametrize("failed_query", ["runs_query_fails", "jobs_query_fails"])
def test_replacement_check_failure_does_not_claim_replacement(tmp_path, failed_query):
    result = _approve(
        tmp_path,
        runs=[_previous_run(conclusion="cancelled")],
        **{failed_query: True},
    )
    assert result["should_run"] == "false"
    assert result["reason"] == failed_query.replace("_", "-").replace("fails", "failed")
    assert "pr_check_takeover" not in result


@pytest.mark.parametrize(
    "kwargs",
    [
        {"event_name": "pull_request", "action": "synchronize"},
        {"review_state": "commented"},
        {"current_pr": _current_pr(head={"sha": "new-sha"})},
    ],
)
def test_only_current_approval_runs_check_for_replaced_pr_run(tmp_path, kwargs):
    result = _approve(tmp_path, runs=[_previous_run(conclusion="cancelled")], **kwargs)
    assert "pr_check_takeover" not in result


def test_retargeted_pr_does_not_take_over_pr_checks(tmp_path):
    result = _approve(
        tmp_path,
        runs=[_previous_run(conclusion="cancelled")],
        current_pr=_current_pr(base={"ref": "release"}),
    )
    assert result["should_run"] == "false"
    assert result["reason"] == "non-main-base-branch"
    assert "pr_check_takeover" not in result


def test_workflow_without_pr_only_check_does_not_look_for_replaced_run(tmp_path):
    result = _approve(
        tmp_path, runs=[_previous_run(conclusion="cancelled")], pr_check_job=""
    )
    assert result["should_run"] == "true"
    assert "pr_check_takeover" not in result


def _smoke_jobs(conclusion):
    return [
        {"jobs": [{"name": "Offline inference smoke test", "conclusion": conclusion}]}
    ]


@pytest.mark.parametrize("conclusion", ["success", "failure", None])
def test_second_approval_does_not_repeat_takeover(tmp_path, conclusion):
    result = _approve(
        tmp_path,
        runs=[
            _previous_run(conclusion="cancelled"),
            _previous_run(id=150, event="pull_request_review", conclusion=None),
        ],
        jobs={100: [], 150: _jobs() + _smoke_jobs(conclusion)},
    )
    assert result["reason"] == "heavy-ci-already-started"
    assert "pr_check_takeover" not in result


def test_comment_review_between_does_not_hide_replacement(tmp_path):
    result = _approve(
        tmp_path,
        runs=[
            _previous_run(conclusion="cancelled"),
            _previous_run(id=150, event="pull_request_review"),
        ],
        jobs={100: [], 150: _smoke_jobs("skipped")},
    )
    assert result["should_run"] == "true"
    assert result["pr_check_takeover"] == "true"
    assert result["replaced_run_id"] == "100"


def test_approval_replaced_while_pending_is_taken_over_again(tmp_path):
    result = _approve(
        tmp_path,
        runs=[
            _previous_run(conclusion="cancelled"),
            _previous_run(id=150, event="pull_request_review", conclusion="cancelled"),
        ],
        jobs={100: [], 150: []},
    )
    assert result["should_run"] == "true"
    assert result["pr_check_takeover"] == "true"


def test_unreadable_later_review_run_does_not_claim_replacement(tmp_path):
    result = _approve(
        tmp_path,
        runs=[
            _previous_run(conclusion="cancelled"),
            _previous_run(id=150, event="pull_request_review"),
        ],
        jobs={100: [], 150: [{"jobs": None}]},
    )
    assert "pr_check_takeover" not in result


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
    result = _approve(
        tmp_path, run_attempt=2, runs=[_previous_run()], jobs={100: _jobs()}
    )
    assert result["should_run"] == "true"
    assert result["reason"] == "current-approval"
    # Read once, to tell a no-op label run from the PR group's when looking
    # for a replaced run; the deduplication check would read it again.
    assert (tmp_path / "api.log").read_text().count("/actions/runs/100/jobs") == 1


def test_explicit_rerun_of_replacing_approval_repeats_the_smoke_test(tmp_path):
    result = _approve(
        tmp_path, run_attempt=2, runs=[_previous_run(conclusion="cancelled")]
    )
    assert result["should_run"] == "true"
    assert result["pr_check_takeover"] == "true"


def test_queued_approval_revalidates_head_sha(tmp_path):
    result = _approve(tmp_path, current_pr=_current_pr(head={"sha": "new-sha"}))
    assert result["should_run"] == "false"
    assert result["reason"] == "stale-head-sha"


def _result_job_names(results):
    return {
        line.split("=", 1)[0].strip() for line in results.splitlines() if line.strip()
    }


def test_workflow_wiring_preserves_existing_checks_and_separates_noop_reviews():
    gate_expectations = {
        "atom-test.yaml": {
            "job": "atom-test-gate",
            "needs": [
                "ci-gate",
                "check-signal",
                "download_aiter_wheel",
                "load-test-models",
                "offline-smoke-test",
                "atom-test",
            ],
            "required": {
                "check-signal",
                "download_aiter_wheel",
                "load-test-models",
                "atom-test",
            },
            "optional": {"offline-smoke-test"},
        },
        "atom-vllm-test.yaml": {
            "job": "atom-vllm-test-gate",
            "needs": [
                "ci-gate",
                "check-signal",
                "download_aiter_wheel",
                "atom-vllm-oot",
            ],
            "required": {"check-signal", "download_aiter_wheel", "atom-vllm-oot"},
            "optional": set(),
        },
        "atom-sglang-test.yaml": {
            "job": "atom-sglang-test-gate",
            "needs": [
                "ci-gate",
                "check-signal",
                "download_aiter_wheel",
                "atom-sglang-accuracy",
            ],
            "required": {
                "check-signal",
                "download_aiter_wheel",
                "atom-sglang-accuracy",
            },
            "optional": set(),
        },
    }

    for filename in ("atom-test.yaml", "atom-vllm-test.yaml", "atom-sglang-test.yaml"):
        config = yaml.safe_load(
            (REPO_ROOT / ".github/workflows" / filename).read_text()
        )
        gate_expectation = gate_expectations[filename]
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
        assert (
            config["jobs"]["ci-gate"]["outputs"]["decidable"]
            == "${{ steps.gate.outputs.decidable }}"
        )
        gate_env = next(
            step["env"]
            for step in config["jobs"]["ci-gate"]["steps"]
            if step.get("id") == "gate"
        )
        assert gate_env["CI_GATE_WORKFLOW"] == filename
        assert config["jobs"]["check-signal"]["name"] == "Check Pre Checkin Signal"

        test_gate = config["jobs"][gate_expectation["job"]]
        assert test_gate["if"] == "always()"
        assert test_gate["needs"] == gate_expectation["needs"]
        evaluate_step = next(
            step
            for step in test_gate["steps"]
            if step.get("uses") == "./.github/actions/evaluate-test-gate"
        )
        evaluate_inputs = evaluate_step["with"]
        required_jobs = _result_job_names(evaluate_inputs.get("required-results", ""))
        optional_jobs = _result_job_names(evaluate_inputs.get("optional-results", ""))
        assert required_jobs == gate_expectation["required"]
        assert optional_jobs == gate_expectation["optional"]
        assert set(test_gate["needs"]) - {"ci-gate"} == required_jobs | optional_jobs

        if filename != "atom-test.yaml":
            assert "CI_GATE_PR_CHECK_JOB" not in gate_env
        if filename == "atom-test.yaml":
            assert gate_env["CI_GATE_PR_CHECK_JOB"] == "Offline inference smoke test"
            assert (
                config["jobs"]["ci-gate"]["outputs"]["smoke_requested"]
                == "${{ steps.gate.outputs.smoke_requested }}"
            )
            smoke = config["jobs"]["offline-smoke-test"]
            assert smoke["needs"] == ["ci-gate"]
            assert smoke["if"] == "${{ !cancelled() }}"
            assert (
                smoke["env"]["SMOKE_REQUESTED"]
                == "${{ needs.ci-gate.outputs.smoke_requested }}"
            )
            assert "needs.ci-gate.outputs.smoke_requested == 'true'" in smoke["runs-on"]
            assert "'ubuntu-latest'" in smoke["runs-on"]
            assert "'linux-atom-mi35x-1'" in smoke["runs-on"]
            assert "needs.ci-gate.outputs.smoke_requested == 'true'" in smoke["name"]
            assert all(
                "env.SMOKE_REQUESTED == 'true'" in step["if"] for step in smoke["steps"]
            )
            assert (
                config["jobs"]["ci-gate"]["outputs"]["pr_check_takeover"]
                == "${{ steps.gate.outputs.pr_check_takeover }}"
            )
            assert "'Offline smoke not requested'" in smoke["name"]
            assert "'Offline inference smoke test'" in smoke["name"]
            assert config["jobs"]["atom-test"]["name"] == "Accuracy"
            assert {"push", "schedule", "workflow_dispatch"} <= events.keys()
        else:
            assert "smoke_requested" not in config["jobs"]["ci-gate"]["outputs"]


GATED_WORKFLOWS = (
    "atom-test.yaml",
    "atom-vllm-test.yaml",
    "atom-sglang-test.yaml",
    "atomesh-benchmark.yaml",
    "atomesh-accuracy-validation.yaml",
)


def _evaluate(source, event_name, action="", added_label="", labels=(), **outputs):
    """A workflow value for one PR event; `outputs` are the ci-gate outputs."""
    return github_expressions(
        source,
        event_name,
        pull_request_number="2166",
        run_id="7",
        action=action,
        label_name=added_label or "",
        label_names=tuple(labels),
        needs_outputs=outputs,
    )


def _noop_group_suffix(group, *args):
    value = _evaluate(group, *args)
    return value[value.index("-noop-") :] if "-noop-" in value else ""


@pytest.mark.parametrize("filename", GATED_WORKFLOWS)
def test_only_gate_opening_label_events_share_the_cancelling_group(tmp_path, filename):
    config = yaml.safe_load((REPO_ROOT / ".github/workflows" / filename).read_text())
    group = config["concurrency"]["group"]
    gate_env = next(
        step["env"]
        for step in config["jobs"]["ci-gate"]["steps"]
        if step.get("id") == "gate"
    )
    allowed = gate_env["CI_GATE_LABELS"]
    gate_labels = [label.strip() for label in allowed.split(",")]
    assert len(gate_labels) == 2 and "ci:full" in gate_labels
    own_label = next(label for label in gate_labels if label != "ci:full")

    gate_job_name = next(
        job["name"]
        for job in config["jobs"].values()
        if "Gate not requested" in str(job.get("name", ""))
    )
    real_name = _evaluate(gate_job_name, "pull_request", should_run="true")
    assert not real_name.endswith("not requested")
    # Where the gate script looks for replaced PR runs, a no-op run's first job
    # is named CI_GATE_NOOP_JOB, which no run of the PR's group carries.
    noop_job = gate_env.get("CI_GATE_NOOP_JOB")
    assert (noop_job is None) == ("CI_GATE_PR_CHECK_JOB" not in gate_env)
    ci_gate_name = config["jobs"]["ci-gate"]["name"]

    cases = [
        ("synchronize", None, (own_label,)),
        ("closed", None, (own_label,)),
        ("labeled", "LGTM", (own_label, "LGTM")),
        ("labeled", "ci:other", ("ci:other",)),
        ("labeled", own_label, (own_label,)),
        ("labeled", "ci:full", ("ci:full",)),
        # Both gate labels on the PR: added at once, or one after the other.
        ("labeled", "ci:full", (own_label, "ci:full")),
        ("labeled", own_label, (own_label, "ci:full")),
        # A payload whose label list lacks the label it reports as added.
        ("labeled", own_label, ()),
        ("labeled", own_label, ("LGTM",)),
        ("labeled", "ci:full", ()),
        ("labeled", "ci:full", (own_label,)),
    ]
    for index, (action, added_label, labels) in enumerate(cases):
        event = ("pull_request", action, added_label, labels)
        suffix = _noop_group_suffix(group, *event)
        # Even with an undecidable gate (should_run false), only a no-op run
        # may hide the required gate check; a run that may have cancelled the
        # PR's previous run reports it.
        name = _evaluate(gate_job_name, *event, should_run="false")
        assert (name == real_name) == (suffix == ""), (action, labels)
        if noop_job:
            assert (_evaluate(ci_gate_name, *event) == noop_job) == (suffix != "")
        if action != "labeled":
            # Code changes and closing keep the PR group, so they still cancel
            # stale runs (closing or merging stops the PR's in-flight heavy CI).
            assert suffix == ""
            continue
        run_dir = tmp_path / str(index)
        run_dir.mkdir()
        result = _run_gate(
            run_dir,
            action=action,
            added_label=added_label,
            labels=labels,
            allowed=allowed,
        )
        # A label event may share the PR's cancelling group only if the gate
        # lets it run the heavy CI it might have cancelled there.
        assert (suffix == "") == (result["should_run"] == "true"), (action, labels)

    both = (own_label, "ci:full")
    opening = [
        label
        for label in both
        if _noop_group_suffix(group, "pull_request", "labeled", label, both) == ""
    ]
    assert opening == [own_label]

    for event_name in ("push", "schedule", "pull_request_review"):
        assert _noop_group_suffix(group, event_name) == ""
    # Reviews keep their naming: only a run that releases heavy CI or takes
    # over the PR checks reports the gate.
    review_name = _evaluate(gate_job_name, "pull_request_review", should_run="false")
    assert review_name.endswith("not requested")


def _run_evaluate_gate(tmp_path, **overrides):
    action = yaml.safe_load(
        (REPO_ROOT / ".github/actions/evaluate-test-gate/action.yml").read_text()
    )
    script = action["runs"]["steps"][0]["run"]
    env = os.environ.copy()
    env.update(
        {
            "GATE_NAME": "Example Gate",
            "SUITE_NAME": "Example tests",
            "SHOULD_RUN": "true",
            "DECIDABLE": "true",
            "REASON": "current-approval",
            "CI_GATE_RESULT": "success",
            "REQUIRED_RESULTS": "check-signal=success\naccuracy=success",
            "OPTIONAL_RESULTS": "",
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
            **overrides,
        }
    )
    return subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        encoding="utf-8",
        env=env,
    )


def test_evaluate_test_gate_rejects_undecidable_scope(tmp_path):
    result = _run_evaluate_gate(
        tmp_path,
        DECIDABLE="false",
        REASON="review-query-failed",
    )

    assert result.returncode != 0
    assert "scope was undecidable: review-query-failed" in result.stdout


def test_evaluate_test_gate_rejects_failed_ci_gate(tmp_path):
    result = _run_evaluate_gate(tmp_path, CI_GATE_RESULT="cancelled")

    assert result.returncode != 0
    assert "scope gate did not succeed: cancelled" in result.stdout


def test_evaluate_test_gate_allows_optional_skipped_when_required_passes(tmp_path):
    result = _run_evaluate_gate(
        tmp_path,
        OPTIONAL_RESULTS="offline-smoke-test=skipped",
    )

    assert result.returncode == 0, result.stderr


def test_evaluate_test_gate_rejects_required_skipped(tmp_path):
    result = _run_evaluate_gate(
        tmp_path,
        OPTIONAL_RESULTS="offline-smoke-test=skipped",
        REQUIRED_RESULTS="check-signal=success\naccuracy=skipped",
    )

    assert result.returncode != 0
    assert "accuracy=skipped" in result.stderr


def test_evaluate_test_gate_allows_not_required_doc_only(tmp_path):
    result = _run_evaluate_gate(
        tmp_path,
        SHOULD_RUN="false",
        REASON="ignored-paths-only",
        OPTIONAL_RESULTS="offline-smoke-test=skipped",
        REQUIRED_RESULTS="check-signal=skipped\naccuracy=skipped",
    )

    assert result.returncode == 0, result.stderr


def test_evaluate_test_gate_rejects_optional_failure_when_not_required(tmp_path):
    result = _run_evaluate_gate(
        tmp_path,
        SHOULD_RUN="false",
        REASON="ignored-paths-only",
        OPTIONAL_RESULTS="offline-smoke-test=failure",
    )

    assert result.returncode != 0
    assert "offline-smoke-test=failure" in result.stderr
