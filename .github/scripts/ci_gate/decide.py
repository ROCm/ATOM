# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The heavy-CI decision for one PR and suite: pure functions over the PR's
current state and its runs. The evaluator writes the Status from it; the
executor's plan job (after cutover) takes only the plan.

Two steps: whether the gate applies to the PR at all (`decide` returns no
Status for a closed PR or a draft), then the gate's verdict.

Invariant I1: a Status is success only when a run that speaks for the PR's
head (`speaks_for`) proved heavy CI passed on it merged into main, or the
change needs no suite run (`needs_suite`).
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from .suites import PRODUCT_DIRS, Suite

BOT_LOGIN = "github-actions[bot]"
MAX_ATTEMPTS = 50
RERUN_WINDOW_S = 30 * 24 * 3600

# Runs of a PR's own code: a manual dispatch may use other inputs, and
# schedule / push runs test a branch, so none of them speaks for a PR.
PR_EVENTS = frozenset({"pull_request", "pull_request_review"})

PENDING = "pending"
SUCCESS = "success"
FAILURE = "failure"

PLAN_HEAVY = "heavy"
PLAN_SMOKE = "smoke"
PLAN_NONE = "none"

ACT_NONE = "none"
ACT_RERUN = "rerun"
ACT_CANCEL_ALL = "cancel_all_runs"
ACT_HOLD = "dependency_hold"

WAITING_BASE = "Waiting: base is"
WAITING_APPROVAL = "Waiting for approval or a ci label"
BLOCKED = "⛔ BLOCKED by unmerged"
RUNNING = "Running"
PUSH_NEEDED = "No valid CI run on this commit: push to start one"
RERUN_OR_PUSH = "Authorized; re-run the workflow from the Actions tab, or push"
CI_REVIEW_NEEDED = (
    "Passed, but this PR changes its own CI definition: needs an approved review"
)
STALE_MERGE = (
    "No valid CI run on this commit (merge commit not against main): push to start one"
)

_ACTIVE = ("queued", "in_progress")


@dataclass(frozen=True)
class ChangedPaths:
    """Every path a PR adds, removes or renames from / to. Incomplete when
    the file list was cut short: then nothing can be ruled out."""

    paths: tuple[str, ...]
    complete: bool


@dataclass(frozen=True)
class Job:
    name: str
    conclusion: str  # success | failure | skipped | cancelled | ...


@dataclass(frozen=True)
class Attempt:
    number: int
    triggering_actor: str
    status: str  # queued | in_progress | completed | action_required | ...
    jobs: tuple[Job, ...] = ()
    conclusion: str = ""  # of a completed attempt: success | failure | cancelled | ...

    def job(self, name: str) -> Job | None:
        return next((j for j in self.jobs if j.name == name), None)


@dataclass(frozen=True)
class Run:
    run_id: int
    event: str
    head_sha: str
    created_at: float
    attempts: tuple[Attempt, ...]
    old_definition: bool  # no aggregate job of this design: the pre-cutover workflow

    @property
    def latest(self) -> Attempt:
        return self.attempts[-1]


@dataclass(frozen=True)
class PullRequest:
    number: int
    is_open: bool
    draft: bool
    base_ref: str
    head_sha: str
    head_repo: str
    base_repo: str
    labels: frozenset[str]
    review_decision: str
    changed: ChangedPaths
    last_base_change_at: float | None  # None: the base never changed
    open_dependencies: tuple[int, ...]


@dataclass(frozen=True)
class Decision:
    status: tuple[str, str] | None  # (state, description); None: write nothing
    plan: str
    action: str = ACT_NONE
    run_id: int | None = None


def speaks_for(pr: PullRequest, event: str, head_sha: str, created_at: float) -> bool:
    """Whether a run tested the PR's current head as the PR: a PR event, of
    the head commit (a commit is its content, whichever repository a run
    reports: a fork PR's review runs report the base repository), and, when
    the base changed, created after the last change (an earlier run's merge
    commit may be against the old base; a run in the same second may be
    too)."""
    return (
        event in PR_EVENTS
        and head_sha == pr.head_sha
        and (pr.last_base_change_at is None or created_at > pr.last_base_change_at)
    )


def decide(
    pr: PullRequest, runs: tuple[Run, ...], suite: Suite, now: float
) -> Decision:
    if not pr.is_open:
        return Decision(None, PLAN_NONE, ACT_CANCEL_ALL)
    if pr.draft:
        return Decision(None, PLAN_NONE)
    return _verdict(pr, runs, suite, now)


def _verdict(
    pr: PullRequest, runs: tuple[Run, ...], suite: Suite, now: float
) -> Decision:
    if pr.base_ref != "main":
        return Decision((PENDING, f"{WAITING_BASE} {pr.base_ref}"), PLAN_NONE)
    if pr.open_dependencies:
        deps = " ".join(f"#{n}" for n in pr.open_dependencies)
        return Decision((PENDING, f"{BLOCKED} {deps}"), PLAN_NONE, ACT_HOLD)
    if not needs_suite(pr.changed, suite):
        return Decision((SUCCESS, "Not required for this change"), PLAN_NONE)

    # The verdict follows the evidence: a pass anywhere proves the head; a run
    # still going may add evidence; else the latest failure. A run that proves
    # nothing either way (cancelled, not run, ...) leaves the verdict open.
    valid = sorted(
        (r for r in runs if speaks_for(pr, r.event, r.head_sha, r.created_at)),
        key=lambda r: r.created_at,
    )
    judged = [(run, evidence(run, suite)) for run in valid]
    passed = next((run for run, ev in judged if ev.kind == PASS), None)
    if passed is not None:
        if (
            changes_ci_definition(pr.changed, suite)
            and pr.review_decision != "APPROVED"
        ):
            # The run that passed ran this PR's own CI definition.
            return Decision((PENDING, CI_REVIEW_NEEDED), PLAN_NONE)
        return Decision((SUCCESS, f"Passed (run {passed.run_id})"), PLAN_NONE)

    authorized = is_authorized(pr, suite)
    plan = PLAN_HEAVY if authorized else PLAN_SMOKE
    latest, latest_evidence = judged[-1] if judged else (None, NO_RUN)
    # Any run still going may yet prove the head, not only the latest: a later
    # one (a label, a comment review) may have ended proving nothing.
    running = next(
        (run for run, ev in reversed(judged) if ev.reason == RUNNING_REASON), None
    )
    if authorized and running is not None:
        return Decision((PENDING, f"{RUNNING} (run {running.run_id})"), PLAN_HEAVY)
    failed = [run for run, ev in judged if ev.kind == FAIL]
    if failed:
        return Decision((FAILURE, f"Failed (run {failed[-1].run_id})"), plan)
    if authorized:
        return _authorized(pr, latest, latest_evidence, now)
    if latest is not None and _smoke_failed(latest, suite):
        return Decision((FAILURE, "Smoke test failed"), PLAN_SMOKE)
    return Decision((PENDING, WAITING_APPROVAL), PLAN_SMOKE)


def is_authorized(pr: PullRequest, suite: Suite) -> bool:
    return bool(pr.labels & suite.labels) or pr.review_decision == "APPROVED"


def _authorized(
    pr: PullRequest, latest: Run | None, ev: Evidence, now: float
) -> Decision:
    """An authorized PR without a verdict: why, and what moves it on."""
    if latest is None:
        return Decision((PENDING, PUSH_NEEDED), PLAN_HEAVY)
    if ev.reason == APPROVAL_REASON:
        return Decision(
            (PENDING, "Waiting for a maintainer to approve the fork run"), PLAN_HEAVY
        )
    if ev.reason == STALE_MERGE_REASON:
        # A re-run keeps the run's merge commit: only a push helps.
        return Decision((PENDING, STALE_MERGE), PLAN_HEAVY)
    if latest.old_definition:
        # A pre-cutover run re-runs its own gate, which can still run heavy
        # CI; never re-run it automatically, whatever it then decides.
        return Decision((PENDING, RERUN_OR_PUSH), PLAN_HEAVY)
    if _can_rerun(pr, latest, now):
        return Decision((PENDING, "Re-running"), PLAN_HEAVY, ACT_RERUN, latest.run_id)
    return Decision((PENDING, RERUN_OR_PUSH), PLAN_HEAVY)


@dataclass(frozen=True)
class Evidence:
    """What a run's latest attempt proves about heavy CI on its commit:
    PASS, FAIL, or NONE with the reason it proves nothing."""

    kind: str
    reason: str = ""


PASS = "pass"
FAIL = "fail"
NONE = "none"
RUNNING_REASON = "running"
APPROVAL_REASON = "waiting for fork approval"
CANCELLED_REASON = "cancelled"
NOT_RUN_REASON = "heavy CI not run"
STALE_MERGE_REASON = "merge commit not against main"
NO_RUN = Evidence(NONE, "no run")


def evidence(run: Run, suite: Suite) -> Evidence:
    attempt = run.latest
    if attempt.status in _ACTIVE:
        return Evidence(NONE, RUNNING_REASON)
    if attempt.status == "action_required":
        return Evidence(NONE, APPROVAL_REASON)
    if attempt.conclusion == "cancelled":
        # A cancelled attempt's jobs fail on their cancelled needs.
        return Evidence(NONE, CANCELLED_REASON)
    if run.old_definition:
        return _old_definition_evidence(attempt, suite)
    result = attempt.job(suite.result_job)
    if result is not None:
        return Evidence(PASS if result.conclusion == "success" else FAIL)
    if attempt.job(suite.stale_merge_job) is not None:
        return Evidence(NONE, STALE_MERGE_REASON)
    return Evidence(NONE, NOT_RUN_REASON)


def _old_definition_evidence(attempt: Attempt, suite: Suite) -> Evidence:
    """A pre-cutover run, read as its own gate read it: heavy CI was required
    when the gate opened (its signal job ran), and passed when the gate did."""
    signal = attempt.job(suite.old_signal_job)
    if signal is None or signal.conclusion == "skipped":
        return Evidence(NONE, NOT_RUN_REASON)
    gate = attempt.job(suite.old_gate_job)
    passed = (
        signal.conclusion == "success"
        and gate is not None
        and gate.conclusion == "success"
    )
    return Evidence(PASS if passed else FAIL)


def changes_ci_definition(changed: ChangedPaths, suite: Suite) -> bool:
    """Whether the PR may change how the suite's run tests it, not only what
    it tests: a path the run executes (one the suite is relevant to) that is
    not product code (PRODUCT_DIRS): its workflow, the actions, scripts,
    Dockerfiles and data it uses, how the product is installed. An incomplete
    file list may."""
    return not changed.complete or any(
        _relevant_to(path, suite) and not path.startswith(PRODUCT_DIRS)
        for path in changed.paths
    )


def _smoke_failed(run: Run, suite: Suite) -> bool:
    if suite.smoke_job is None or run.latest.status != "completed":
        return False
    smoke = run.latest.job(suite.smoke_job)
    return smoke is not None and smoke.conclusion == "failure"


def _can_rerun(pr: PullRequest, run: Run, now: float) -> bool:
    return (
        pr.head_repo == pr.base_repo
        and run.latest.number < MAX_ATTEMPTS
        and now - run.created_at < RERUN_WINDOW_S
        and all(a.triggering_actor != BOT_LOGIN for a in run.attempts)
    )


def _path_matches(pattern: str, path: str) -> bool:
    # The old ci-gate's matching (check_heavy_ci_gate.sh), kept as is.
    if pattern == "**/*.md":
        return path.endswith(".md")
    if pattern.endswith("/**"):
        prefix = pattern[:-3].rstrip("/")
        return path == prefix or path.startswith(prefix + "/")
    return fnmatch.fnmatchcase(path, pattern)


def needs_suite(changed: ChangedPaths, suite: Suite) -> bool:
    """Whether the change may affect what the suite tests: true unless every
    path is known and none is one the suite tests."""
    if not changed.complete:
        return True
    return any(_relevant_to(path, suite) for path in changed.paths)


def _relevant_to(path: str, suite: Suite) -> bool:
    return (
        not suite.include_globs
        or any(_path_matches(g, path) for g in suite.include_globs)
    ) and not any(_path_matches(g, path) for g in suite.ignore_globs)


def plan_for_executor(
    decision: Decision, event_base_ref: str, event_draft: bool, merge_base_in_main: bool
) -> str:
    """The executor's plan: the decision's plan, refused when the run's own
    event or merge commit is not a commit against main."""
    if event_draft or event_base_ref != "main" or not merge_base_in_main:
        return PLAN_NONE
    return decision.plan


def result_job_name(suite: Suite, plan: str, merge_base_in_main: bool) -> str:
    """The executor's aggregate job name: the only record, per attempt, of
    whether it set out to prove heavy CI."""
    if plan == PLAN_HEAVY:
        return suite.result_job
    return suite.not_run_job if merge_base_in_main else suite.stale_merge_job
