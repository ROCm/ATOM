# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""decide() of the heavy-CI gate, one test per rule (README.md there)."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".github" / "scripts"))

from ci_gate import decide as d
from ci_gate.suites import SUITES

SUITE = SUITES[0]
NOW = 10_000.0
HEAD = "h1"
APPROVED = {"review_decision": "APPROVED"}
NOT_RUN = (SUITE.not_run_job, "success")


def pr(**changes) -> d.PullRequest:
    base = d.PullRequest(
        number=1,
        is_open=True,
        draft=False,
        base_ref="main",
        head_sha=HEAD,
        head_repo="ROCm/ATOM",
        base_repo="ROCm/ATOM",
        labels=frozenset(),
        review_decision="REVIEW_REQUIRED",
        changed=changed("atom/x.py"),
        last_base_change_at=None,
        open_dependencies=(),
    )
    return replace(base, **changes)


def changed(*paths, complete=True) -> d.ChangedPaths:
    return d.ChangedPaths(paths, complete)


def run(
    *jobs,
    status="completed",
    created=100.0,
    old=False,
    run_id=5,
    attempts=None,
    event="pull_request",
):
    attempt = d.Attempt(1, "author", status, tuple(d.Job(n, c) for n, c in jobs))
    return d.Run(run_id, event, HEAD, created, attempts or (attempt,), old)


def state(decision: d.Decision) -> str | None:
    return decision.status[0] if decision.status else None


def test_closed_cancels_and_writes_nothing():
    decision = d.decide(pr(is_open=False), (), SUITE, NOW)
    assert decision.status is None and decision.action == d.ACT_CANCEL_ALL


def test_base_not_main_waits_and_plans_nothing():
    decision = d.decide(pr(base_ref="feat", **APPROVED), (), SUITE, NOW)
    assert decision.status == (d.PENDING, "Waiting: base is feat")
    assert decision.plan == d.PLAN_NONE


def test_open_dependency_holds_even_when_approved_and_labelled():
    labelled = pr(open_dependencies=(3410,), labels=frozenset({"ci:atom"}), **APPROVED)
    decision = d.decide(labelled, (), SUITE, NOW)
    assert decision.status[1].startswith("⛔ BLOCKED by unmerged #3410")
    assert decision.plan == d.PLAN_NONE and decision.action == d.ACT_HOLD


@pytest.mark.parametrize(
    "changes",
    [
        {"labels": frozenset({"ci:atom"})},
        {"base_ref": "feat"},  # the gate does not apply before its verdict
        {"open_dependencies": (7,)},
    ],
)
def test_draft_writes_nothing_and_runs_nothing(changes):
    decision = d.decide(pr(draft=True, **changes), (), SUITE, NOW)
    assert decision.status is None and decision.plan == d.PLAN_NONE


def test_paths_the_suite_ignores_are_success_even_with_a_label():
    docs = pr(
        changed=changed("docs/a.md", "x/README.md"), labels=frozenset({"ci:atom"})
    )
    decision = d.decide(docs, (), SUITE, NOW)
    assert decision.status == (d.SUCCESS, "Not required for this change")


def test_an_incomplete_file_list_rules_nothing_out():
    assert not d.needs_suite(changed("docs/a.md"), SUITE)
    assert d.needs_suite(changed("docs/a.md", complete=False), SUITE)


def test_include_list_suite_needs_an_included_path():
    mesh = next(s for s in SUITES if s.key == "atomesh-accuracy-validation")
    assert not d.needs_suite(changed("atom/model_engine/x.py"), mesh)
    assert d.needs_suite(changed("atom/mesh/router.rs"), mesh)


def test_ignore_wins_over_include():
    sglang = next(s for s in SUITES if s.key == "atom-sglang-test")
    assert d.needs_suite(changed("atom/plugin/sglang/x.py"), sglang)
    assert not d.needs_suite(changed("atom/plugin/vllm/x.py"), sglang)


@pytest.mark.parametrize(
    "event, speaks",
    [
        ("pull_request", True),
        ("pull_request_review", True),  # the old gate runs heavy CI on approval
        ("workflow_dispatch", False),  # may use other inputs (an aiter branch)
        ("schedule", False),
        ("push", False),
    ],
)
def test_only_runs_of_the_prs_own_code_speak_for_it(event, speaks):
    passed = run((SUITE.result_job, "success"), event=event)
    assert (state(d.decide(pr(), (passed,), SUITE, NOW)) == d.SUCCESS) is speaks


def test_tested_sha_is_success_without_authorization_and_plans_nothing():
    decision = d.decide(pr(), (run((SUITE.result_job, "success")),), SUITE, NOW)
    assert state(decision) == d.SUCCESS and decision.plan == d.PLAN_NONE


def test_any_valid_run_proves_the_sha_not_only_the_latest():
    passed = run((SUITE.result_job, "success"), run_id=1, created=100)
    later_not_run = run(NOT_RUN, run_id=2, created=200)
    assert state(d.decide(pr(), (passed, later_not_run), SUITE, NOW)) == d.SUCCESS


def test_run_created_before_the_last_base_change_proves_nothing():
    passed_before = run((SUITE.result_job, "success"), created=100)
    decision = d.decide(
        pr(last_base_change_at=150, **APPROVED), (passed_before,), SUITE, NOW
    )
    assert decision.status == (d.PENDING, d.PUSH_NEEDED)


def test_a_run_in_the_second_of_a_base_change_proves_nothing():
    same_second = run((SUITE.result_job, "success"), created=150)
    decision = d.decide(
        pr(last_base_change_at=150, **APPROVED), (same_second,), SUITE, NOW
    )
    assert decision.status == (d.PENDING, d.PUSH_NEEDED)


def test_a_base_that_never_changed_bounds_no_run():
    # The PR's first run can share the second the PR was opened in.
    first = run((SUITE.result_job, "success"), created=0)
    assert state(d.decide(pr(), (first,), SUITE, NOW)) == d.SUCCESS


SIGNAL = SUITE.old_signal_job


def test_an_old_run_reads_as_its_own_gate_read_it():
    opened_and_passed = run(
        (SIGNAL, "success"), (SUITE.old_gate_job, "success"), old=True
    )
    # The gate passes without heavy CI when it did not open: no evidence.
    not_opened = run((SIGNAL, "skipped"), (SUITE.old_gate_job, "success"), old=True)
    assert state(d.decide(pr(), (opened_and_passed,), SUITE, NOW)) == d.SUCCESS
    assert state(d.decide(pr(), (not_opened,), SUITE, NOW)) == d.PENDING


def test_an_old_run_passes_whatever_its_matrix_looked_like():
    # e.g. an empty model list: the gate's verdict is the evidence, not the jobs
    passed = run(
        (SIGNAL, "success"),
        ("Accuracy (skip)", "skipped"),
        (SUITE.old_gate_job, "success"),
        old=True,
    )
    assert state(d.decide(pr(), (passed,), SUITE, NOW)) == d.SUCCESS


def test_an_old_run_whose_gate_opened_and_failed_is_failure():
    # e.g. every accuracy job passed but the smoke test failed the old gate
    ran = run(
        (SIGNAL, "success"),
        ("Accuracy (a)", "success"),
        (SUITE.smoke_job, "failure"),
        (SUITE.old_gate_job, "failure"),
        old=True,
    )
    assert state(d.decide(pr(**APPROVED), (ran,), SUITE, NOW)) == d.FAILURE


def test_a_later_run_that_did_not_try_heavy_ci_does_not_hide_a_failure():
    failed = run((SUITE.result_job, "failure"), run_id=1, created=100)
    not_tried = run(NOT_RUN, run_id=2, created=200)
    decision = d.decide(pr(**APPROVED), (failed, not_tried), SUITE, NOW)
    assert decision.status == (d.FAILURE, "Failed (run 1)")


def test_a_run_still_going_may_add_evidence():
    failed = run((SUITE.result_job, "failure"), run_id=1, created=100)
    going = run(status="in_progress", run_id=2, created=200)
    decision = d.decide(pr(**APPROVED), (failed, going), SUITE, NOW)
    assert decision.status[1].startswith(d.RUNNING)


def test_old_definition_run_without_heavy_ci_is_rerun_by_hand_never_by_the_bot():
    not_run = run((SIGNAL, "skipped"), (SUITE.old_gate_job, "success"), old=True)
    decision = d.decide(pr(**APPROVED), (not_run,), SUITE, NOW)
    assert (
        decision.status == (d.PENDING, d.RERUN_OR_PUSH)
        and decision.action == d.ACT_NONE
    )


def test_a_cancelled_run_is_no_evidence_and_is_rerun():
    # Its aggregate job fails on its cancelled needs; that is not a failure.
    cancelled = run((SUITE.result_job, "failure"), attempts=None)
    cancelled = replace(
        cancelled, attempts=(replace(cancelled.latest, conclusion="cancelled"),)
    )
    decision = d.decide(pr(**APPROVED), (cancelled,), SUITE, NOW)
    assert (
        decision.status == (d.PENDING, "Re-running") and decision.action == d.ACT_RERUN
    )


@pytest.mark.parametrize(
    "key, path",
    [
        ("atom-test", ".github/workflows/atom-test.yaml"),
        ("atom-test", ".github/scripts/atom_test.sh"),
        ("atom-test", "scripts/wait_infer_drain.sh"),
        ("atom-test", "setup.py"),
        ("atom-vllm-test", "docker/vllm_release.dockerfile"),
        ("atom-vllm-test", "docker/patches/x.patch"),
    ],
)
def test_a_pass_by_the_prs_own_ci_definition_needs_an_approved_review(key, path):
    suite = next(s for s in SUITES if s.key == key)
    passed = run((suite.result_job, "success"))
    own_ci = changed("atom/plugin/vllm/x.py", path)
    label = frozenset(suite.labels)
    labelled = d.decide(pr(changed=own_ci, labels=label), (passed,), suite, NOW)
    approved = d.decide(pr(changed=own_ci, **APPROVED), (passed,), suite, NOW)
    assert labelled.status == (d.PENDING, d.CI_REVIEW_NEEDED)
    assert state(approved) == d.SUCCESS


@pytest.mark.parametrize(
    "path",
    [
        "atom/models/x.py",  # what the run tests, not how
        "docs/architecture_guide.rst",  # no part of this suite's run
    ],
)
def test_a_pass_needs_no_review_when_the_run_is_mains(path):
    passed = run((SUITE.result_job, "success"))
    labelled = pr(changed=changed("atom/x.py", path), labels=frozenset({"ci:atom"}))
    assert state(d.decide(labelled, (passed,), SUITE, NOW)) == d.SUCCESS


def test_failed_result_is_failure():
    decision = d.decide(
        pr(**APPROVED), (run((SUITE.result_job, "failure")),), SUITE, NOW
    )
    assert state(decision) == d.FAILURE


def test_authorized_running_waits():
    decision = d.decide(pr(**APPROVED), (run(status="in_progress"),), SUITE, NOW)
    assert decision.status[1].startswith("Running") and decision.plan == d.PLAN_HEAVY


@pytest.mark.parametrize("old", [True, False], ids=["old-definition", "new"])
def test_an_earlier_run_still_going_waits_when_a_later_one_proved_nothing(old):
    # A label or a comment review starts a later run that ends proving nothing
    # while the heavy run goes on: wait for it, neither re-run nor ask a push.
    heavy = run(status="in_progress", run_id=1, created=100, old=old)
    nothing = (SIGNAL, "skipped") if old else NOT_RUN
    later = run(nothing, run_id=2, created=200, old=old)
    decision = d.decide(pr(**APPROVED), (heavy, later), SUITE, NOW)
    assert decision.status == (d.PENDING, f"{d.RUNNING} (run 1)")
    assert decision.action == d.ACT_NONE


def test_fork_waiting_for_approval_is_never_rerun():
    fork_run = run(status="action_required")
    decision = d.decide(pr(head_repo="x/ATOM", **APPROVED), (fork_run,), SUITE, NOW)
    assert "maintainer" in decision.status[1] and decision.action == d.ACT_NONE


def test_authorized_not_run_is_rerun_once_by_the_bot():
    decision = d.decide(pr(**APPROVED), (run(NOT_RUN),), SUITE, NOW)
    assert decision.action == d.ACT_RERUN and decision.run_id == 5


def test_no_second_bot_rerun():
    jobs = (d.Job(SUITE.not_run_job, "success"),)
    attempts = (
        d.Attempt(1, "author", "completed", jobs),
        d.Attempt(2, d.BOT_LOGIN, "completed", jobs),
    )
    decision = d.decide(pr(**APPROVED), (run(attempts=attempts),), SUITE, NOW)
    assert decision.action == d.ACT_NONE and "re-run" in decision.status[1]


def test_fork_not_run_is_left_to_a_maintainer():
    decision = d.decide(pr(head_repo="x/ATOM", **APPROVED), (run(NOT_RUN),), SUITE, NOW)
    assert decision.action == d.ACT_NONE


def test_a_fork_prs_approval_run_speaks_for_it():
    # GitHub reports the base repository for a fork PR's review runs; the
    # commit, not the repository, says what was tested.
    passed = run((SUITE.result_job, "success"), event="pull_request_review")
    assert state(d.decide(pr(head_repo="x/ATOM"), (passed,), SUITE, NOW)) == d.SUCCESS


def test_stale_merge_commit_asks_for_a_push_and_is_not_rerun():
    decision = d.decide(
        pr(**APPROVED), (run((SUITE.stale_merge_job, "success")),), SUITE, NOW
    )
    assert (
        decision.status == (d.PENDING, d.STALE_MERGE) and decision.action == d.ACT_NONE
    )


@pytest.mark.parametrize("age, attempt", [(31 * 24 * 3600, 1), (0, 50)])
def test_rerun_limits(age, attempt):
    jobs = (d.Job(SUITE.not_run_job, "success"),)
    attempts = (d.Attempt(attempt, "author", "completed", jobs),)
    decision = d.decide(
        pr(**APPROVED), (run(attempts=attempts, created=NOW - age),), SUITE, NOW
    )
    assert decision.action == d.ACT_NONE


def test_unauthorized_waits_and_plans_smoke():
    decision = d.decide(pr(), (run(NOT_RUN),), SUITE, NOW)
    assert decision.status == (d.PENDING, d.WAITING_APPROVAL)
    assert decision.plan == d.PLAN_SMOKE


def test_unauthorized_smoke_failure_is_failure():
    decision = d.decide(pr(), (run(NOT_RUN, (SUITE.smoke_job, "failure")),), SUITE, NOW)
    assert decision.status == (d.FAILURE, "Smoke test failed")


@pytest.mark.parametrize(
    "event_base, draft, merge_ok, expected",
    [
        ("main", False, True, d.PLAN_HEAVY),
        ("feat", False, True, d.PLAN_NONE),
        ("main", True, True, d.PLAN_NONE),
        ("main", False, False, d.PLAN_NONE),
    ],
)
def test_executor_refuses_a_run_not_against_main(event_base, draft, merge_ok, expected):
    heavy = d.Decision((d.PENDING, ""), d.PLAN_HEAVY)
    assert d.plan_for_executor(heavy, event_base, draft, merge_ok) == expected


def test_result_job_name_records_what_the_attempt_set_out_to_prove():
    assert d.result_job_name(SUITE, d.PLAN_HEAVY, True) == SUITE.result_job
    assert d.result_job_name(SUITE, d.PLAN_SMOKE, True) == SUITE.not_run_job
    assert d.result_job_name(SUITE, d.PLAN_NONE, False) == SUITE.stale_merge_job
