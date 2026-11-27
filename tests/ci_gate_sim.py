# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Protocol simulator for the heavy-CI gate (.github/scripts/ci_gate): a small
model of the GitHub behaviour the design relies on, with each behaviour it does
not rely on modelled both ways (HYPOTHESES). Random event interleavings; I1 is
checked after every step, and once the world goes quiet nothing may be left
waiting on something that will not happen.

tests/test_ci_gate_sim.py runs a short pass. A long one:
python tests/ci_gate_sim.py [SEQUENCES_PER_HYPOTHESIS] [--buggy-aggregator]
[--no-merge-base-check] [--no-time-rule] [--trust-manual-runs]
[--trust-incomplete-file-lists] [--trust-own-ci-definition]  (each flag
removes one defence and
must produce violations)
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".github" / "scripts"))

from ci_gate import decide as d
from ci_gate.suites import SUITES

SUITE = SUITES[0]
HEAVY_JOB = "Accuracy (model)"
REPO = "ROCm/ATOM"
FORK = "someone/ATOM"
RECONCILE_EVERY = 600.0
# Paths that change how SUITE's run tests a PR, in and out of .github/.
OWN_CI_DEFINITION = (
    ".github/workflows/atom-test.yaml",
    ".github/scripts/atom_test.sh",
    "scripts/wait_infer_drain.sh",
    "setup.py",
)


def _github_time(t: float) -> float:
    """A time as GitHub reports it: to the second."""
    return float(int(t))


HYPOTHESES = (
    "rerun_payload_refresh",  # T1: a re-run's event payload shows the current PR
    "merge_recompute_on_retarget",  # T2-T4 / F11: retarget recomputes the merge commit
    "workflow_run_after_bot_rerun",  # S3
    "edited_on_auto_retarget",  # F15
    "relay_reaches_pr",  # F11 for the relay: the PR's merge commit has the relay
)


@dataclass
class SimAttempt:
    number: int
    actor: str
    status: str  # queued | in_progress | completed | action_required
    plan: str | None = None
    jobs: dict[str, str] = field(default_factory=dict)
    rerun_failed: bool = False
    merge_ok: bool = True
    conclusion: str = ""  # of the run when this attempt completed


@dataclass
class SimRun:
    run_id: int
    head_sha: str
    created_at: float
    event_base: str
    event_draft: bool
    merge_base: str
    old_definition: bool
    attempts: list[SimAttempt]
    event: str = "pull_request"

    @property
    def latest(self) -> SimAttempt:
        return self.attempts[-1]


class Violation(Exception):
    pass


class World:
    def __init__(self, rng: random.Random, hyp: dict[str, bool], opts: dict[str, bool]):
        self.rng = rng
        self.hyp = hyp
        self.opts = opts
        self.now = 1000.0
        self.next_reconcile = self.now + RECONCILE_EVERY
        self.fork = rng.random() < 0.25
        self.is_open = True
        self.draft = rng.random() < 0.2
        self.base = "main" if rng.random() < 0.7 else "feat"
        self.labels: set[str] = set()
        self.approved = False
        self.deps_open = rng.random() < 0.2
        self.sha_counter = 0
        self.files: dict[str, d.ChangedPaths] = {}
        # Ground truth: whether a head truly touches only paths no suite tests
        # (an incomplete file list may hide one that does).
        self.truly_docs_only: dict[str, bool] = {}
        self.touches_ci: dict[str, bool] = {}
        self.fakes_result: dict[str, bool] = {}
        self.approved_heads: set[str] = set()  # heads an approval covered
        self.head = ""
        self.merge_base = self.base
        self.last_base_change_at: float | None = None
        self.runs: list[SimRun] = []
        self.statuses: dict[str, tuple[str, str]] = {}
        self.tested_main: set[str] = set()
        self.events: list[str] = []
        self.stats: Counter[str] = Counter()
        self._new_head()
        if rng.random() < 0.3:
            self._old_definition_run()
        self._create_run()
        self._emit("pull_request_target")

    # ---- helpers -------------------------------------------------------
    def _new_head(self) -> None:
        self.sha_counter += 1
        self.head = f"sha{self.sha_counter}"
        docs_only = self.rng.random() < 0.15
        # A cut-short file list shows only docs; the paths it hides are code.
        cut_short = self.rng.random() < 0.1
        self.truly_docs_only[self.head] = docs_only and not cut_short
        shown = ("docs/a.md",) if docs_only or cut_short else ("atom/x.py",)
        # Some heads change the suite's own CI definition; some of those make
        # it report a pass without running heavy CI.
        touches_ci = not docs_only and self.rng.random() < 0.15
        if touches_ci:
            shown += (self.rng.choice(OWN_CI_DEFINITION),)
        self.touches_ci[self.head] = touches_ci or cut_short
        self.fakes_result[self.head] = touches_ci and self.rng.random() < 0.5
        self.files[self.head] = d.ChangedPaths(shown, complete=not cut_short)

    def _old_definition_run(self) -> None:
        # The old gate opens (its signal job runs) when it requires heavy CI;
        # closed, it passes without heavy CI.
        opened = self.rng.random() < 0.6
        signal = (
            self.rng.choice(("success", "success", "failure")) if opened else "skipped"
        )
        heavy = (
            self.rng.choice(("success", "failure"))
            if signal == "success"
            else "skipped"
        )
        smoke = "failure" if self.rng.random() < 0.15 else "success"
        required_ok = not opened or (signal == "success" and heavy == "success")
        gate = "success" if required_ok and smoke == "success" else "failure"
        jobs = {
            SUITE.old_signal_job: signal,
            SUITE.old_gate_job: gate,
            "Accuracy (m)": heavy,
            SUITE.smoke_job: smoke,
        }
        self.runs.append(
            SimRun(
                run_id=len(self.runs) + 1,
                head_sha=self.head,
                created_at=self.now,
                event_base=self.base,
                event_draft=self.draft,
                merge_base=self.merge_base,
                old_definition=True,
                attempts=[SimAttempt(1, "author", "completed", jobs=jobs)],
                # The old gate runs heavy CI on a push or on an approval.
                event=self.rng.choice(("pull_request", "pull_request_review")),
            )
        )
        if heavy == "success" and self.merge_base == "main":
            self.tested_main.add(self.head)
        self.now += 1

    def _emit(self, kind: str, drop: float = 0.15) -> None:
        if self.rng.random() >= drop:
            self.events.append(kind)

    def _group_active(self, sha: str) -> list[SimRun]:
        return [
            r
            for r in self.runs
            if r.head_sha == sha
            and r.event in d.PR_EVENTS  # a manual run has a group of its own
            and not r.old_definition
            and r.latest.status in ("queued", "in_progress")
        ]

    def _cancel(self, run: SimRun) -> None:
        attempt = run.latest
        if attempt.status not in ("queued", "in_progress"):
            return
        plan = attempt.plan or d.PLAN_NONE
        attempt.jobs["plan"] = "success" if attempt.plan else "cancelled"
        attempt.jobs[HEAVY_JOB] = "cancelled" if plan == d.PLAN_HEAVY else "skipped"
        attempt.jobs[SUITE.smoke_job] = (
            "cancelled" if plan != d.PLAN_NONE else "skipped"
        )
        self._finish_result(attempt, plan)
        attempt.status = "completed"
        attempt.conclusion = "cancelled"
        self.stats["cancelled"] += 1

    def _create_run(self) -> None:
        if self.draft and self.rng.random() < 0.5:
            pass  # draft PRs still get pull_request runs; their plan is none
        for run in self._group_active(self.head):
            self._cancel(run)
        status = "action_required" if self.fork else "queued"
        self.runs.append(
            SimRun(
                run_id=len(self.runs) + 1,
                head_sha=self.head,
                created_at=self.now,
                event_base=self.base,
                event_draft=self.draft,
                merge_base=self.merge_base,
                old_definition=False,
                attempts=[SimAttempt(1, "author", status)],
            )
        )
        self.now += 1

    def _pr_view(self) -> d.PullRequest:
        return d.PullRequest(
            number=1,
            is_open=self.is_open,
            draft=self.draft,
            base_ref=self.base,
            head_sha=self.head,
            head_repo=FORK if self.fork else REPO,
            base_repo=REPO,
            labels=frozenset(self.labels),
            review_decision="APPROVED" if self.approved else "REVIEW_REQUIRED",
            changed=self._changed_view(),
            last_base_change_at=(
                None
                if self.opts["no_time_rule"] or self.last_base_change_at is None
                else _github_time(self.last_base_change_at)
            ),
            open_dependencies=(7,) if self.deps_open else (),
        )

    def _changed_view(self) -> d.ChangedPaths:
        changed = self.files[self.head]
        paths = changed.paths
        if self.opts["trust_own_ci_definition"]:
            paths = tuple(
                p
                for p in paths
                if not d.changes_ci_definition(d.ChangedPaths((p,), True), SUITE)
            )
        complete = changed.complete or self.opts["trust_incomplete_file_lists"]
        return d.ChangedPaths(paths, complete)

    def _runs_view(self) -> tuple[d.Run, ...]:
        return tuple(
            d.Run(
                run_id=r.run_id,
                event=("pull_request" if self.opts["trust_manual_runs"] else r.event),
                head_sha=r.head_sha,
                created_at=_github_time(r.created_at),
                old_definition=r.old_definition,
                attempts=tuple(
                    d.Attempt(
                        number=a.number,
                        triggering_actor=a.actor,
                        status=a.status,
                        jobs=tuple(d.Job(n, c) for n, c in a.jobs.items()),
                        conclusion=a.conclusion,
                    )
                    for a in r.attempts
                ),
            )
            for r in self.runs
        )

    def _decision(self) -> d.Decision:
        return d.decide(self._pr_view(), self._runs_view(), SUITE, self.now)

    def _authorized(self) -> bool:
        return bool(self.labels & SUITE.labels) or self.approved

    # ---- executor --------------------------------------------------------
    def _start(self, run: SimRun) -> None:
        attempt = run.latest
        if attempt.rerun_failed:
            attempt.plan = run.attempts[-2].plan  # F7: plan job output carried over
            attempt.merge_ok = run.attempts[-2].merge_ok
        else:
            refresh = attempt.number > 1 and self.hyp["rerun_payload_refresh"]
            event_base = self.base if refresh else run.event_base
            event_draft = self.draft if refresh else run.event_draft
            merge_ok = run.merge_base == "main" or self.opts["no_merge_base_check"]
            attempt.merge_ok = merge_ok
            attempt.plan = d.plan_for_executor(
                self._decision(), event_base, event_draft, merge_ok
            )
        attempt.status = "in_progress"
        if attempt.plan == d.PLAN_HEAVY:
            if not self._authorized():
                self.stats["heavy_unauthorized_start"] += 1
            if run.head_sha in self.tested_main:
                self.stats["heavy_on_tested_sha"] += 1

    def _finish(self, run: SimRun) -> None:
        attempt = run.latest
        plan = attempt.plan or d.PLAN_NONE
        previous = run.attempts[-2].jobs if attempt.rerun_failed else {}
        attempt.jobs["plan"] = "success"

        def outcome(job: str, wanted: bool, p_fail: float) -> str:
            if not wanted:
                return "skipped"
            if previous.get(job) == "success":
                return "success"  # rerun-failed keeps the jobs that passed
            if attempt.rerun_failed and previous.get(job) == "skipped":
                return "skipped"  # rerun-failed does not run skipped jobs
            return "failure" if self.rng.random() < p_fail else "success"

        attempt.jobs[SUITE.smoke_job] = outcome(
            SUITE.smoke_job, plan != d.PLAN_NONE, 0.1
        )
        attempt.jobs[HEAVY_JOB] = outcome(HEAVY_JOB, plan == d.PLAN_HEAVY, 0.2)
        self._finish_result(attempt, plan)
        if self.fakes_result[run.head_sha] and run.event in d.PR_EVENTS:
            # This head's own CI definition reports a pass with no heavy CI.
            for name in (SUITE.result_job, SUITE.not_run_job, SUITE.stale_merge_job):
                attempt.jobs.pop(name, None)
            attempt.jobs[HEAVY_JOB] = "skipped"
            attempt.jobs[SUITE.result_job] = "success"
        attempt.status = "completed"
        failed = any(c == "failure" for c in attempt.jobs.values())
        attempt.conclusion = "failure" if failed else "success"
        if (
            plan == d.PLAN_HEAVY
            and attempt.jobs[HEAVY_JOB] == "success"
            and run.merge_base == "main"
            and run.event in d.PR_EVENTS  # a manual run's inputs are its own
        ):
            self.tested_main.add(run.head_sha)
        bot = attempt.actor == d.BOT_LOGIN
        if not bot or self.hyp["workflow_run_after_bot_rerun"]:
            self._emit("workflow_run")

    def _finish_result(self, attempt: SimAttempt, plan: str) -> None:
        name = d.result_job_name(SUITE, plan, attempt.merge_ok)
        heavy_ok = plan != d.PLAN_HEAVY or attempt.jobs.get(HEAVY_JOB) == "success"
        smoke_ok = attempt.jobs.get(SUITE.smoke_job) in ("success", "skipped")
        ok = heavy_ok and smoke_ok
        if self.opts["buggy_aggregator"]:
            ok = True  # F9: if: always() with no check of its needs
        for stale in (SUITE.result_job, SUITE.not_run_job, SUITE.stale_merge_job):
            attempt.jobs.pop(stale, None)
        attempt.jobs[name] = "success" if ok else "failure"

    def _rerun(self, run: SimRun, actor: str, failed_only: bool) -> None:
        if run.latest.status != "completed" or run.old_definition:
            return
        if failed_only and all(c != "failure" for c in run.latest.jobs.values()):
            return
        for other in self._group_active(run.head_sha):
            if other is not run:
                self._cancel(other)  # F8: a re-run still takes its group
        run.attempts.append(
            SimAttempt(len(run.attempts) + 1, actor, "queued", rerun_failed=failed_only)
        )
        self.stats[f"rerun_{actor}"] += 1

    # ---- evaluator -------------------------------------------------------
    def _evaluate(self, kind: str) -> None:
        if kind == "pull_request_target_sync":
            for run in self.runs:
                if run.head_sha != self.head and run.latest.status in (
                    "queued",
                    "in_progress",
                ):
                    self._cancel(run)  # section 3.4
        decision = self._decision()
        if decision.status is not None:
            self.statuses[self.head] = decision.status
        if decision.action == d.ACT_RERUN:
            run = next(r for r in self.runs if r.run_id == decision.run_id)
            self._rerun(run, d.BOT_LOGIN, failed_only=False)
        elif decision.action == d.ACT_CANCEL_ALL:
            for run in self.runs:
                self._cancel(run)

    def _reconcile(self) -> None:
        # The pass reaches every open non-draft PR, whatever its base.
        if self.is_open and not self.draft:
            self._evaluate("schedule")

    # ---- driver --------------------------------------------------------------
    def human_step(self) -> None:
        choice = self.rng.choice(
            [
                "push",
                "label",
                "approve",
                "retarget",
                "draft",
                "close",
                "deps",
                "ui_rerun",
                "ui_rerun_failed",
                "maintainer_approve",
                "manual_dispatch",
            ]
        )
        if not self.is_open and choice != "close":
            return
        if choice == "push":
            self._new_head()
            self.approved = False  # dismiss_stale_reviews_on_push
            self.merge_base = self.base
            self._create_run()
            self._emit("pull_request_target_sync")
        elif choice == "label":
            self.labels ^= {"ci:atom"}
            self._emit("pull_request_target")
        elif choice == "approve":
            self.approved = True
            self.approved_heads.add(self.head)
            if self.hyp["relay_reaches_pr"]:
                self._emit("relay")
        elif choice == "retarget":
            self.base = "main" if self.base == "feat" else "feat"
            self.last_base_change_at = self.now
            if self.hyp["merge_recompute_on_retarget"]:
                self.merge_base = self.base
            automatic = self.rng.random() < 0.5
            if not automatic or self.hyp["edited_on_auto_retarget"]:
                self._emit("pull_request_target")
        elif choice == "draft":
            self.draft = not self.draft
            if not self.draft:
                self._create_run()
            self._emit("pull_request_target")
        elif choice == "close":
            self.is_open = not self.is_open
            if self.is_open:
                self._create_run()
            self._emit("pull_request_target")
        elif choice == "deps":
            self.deps_open = False  # the dependency merges; no event on this PR
        elif choice in ("ui_rerun", "ui_rerun_failed") and self.runs:
            self._rerun(
                self.rng.choice(self.runs), "maintainer", choice == "ui_rerun_failed"
            )
        elif choice == "maintainer_approve":
            for run in self.runs:
                if run.latest.status == "action_required":
                    run.attempts.append(SimAttempt(2, "maintainer", "queued"))  # F13
                    break
        elif choice == "manual_dispatch":
            # A maintainer's manual run on the PR's head, maybe with another
            # aiter branch: it passes, but proves nothing about the PR.
            self.runs.append(
                SimRun(
                    run_id=len(self.runs) + 1,
                    head_sha=self.head,
                    created_at=self.now,
                    event_base="main",
                    event_draft=False,
                    merge_base="main",
                    old_definition=False,
                    attempts=[
                        SimAttempt(
                            1,
                            "maintainer",
                            "completed",
                            plan=d.PLAN_HEAVY,
                            jobs={SUITE.result_job: "success"},
                        )
                    ],
                    event="workflow_dispatch",
                )
            )
        self.now += self.rng.uniform(1, 900)

    def system_step(self) -> bool:
        options = []
        for run in self.runs:
            if run.latest.status == "queued":
                options.append(("start", run))
            elif run.latest.status == "in_progress":
                options.append(("finish", run))
        if self.events:
            options.append(("event", None))
        if not options:
            return False
        kind, run = self.rng.choice(options)
        if kind == "start":
            self._start(run)
        elif kind == "finish":
            self._finish(run)
        else:
            self._evaluate(self.events.pop(self.rng.randrange(len(self.events))))
        self.now += self.rng.uniform(1, 120)
        if self.now >= self.next_reconcile:
            self.next_reconcile = self.now + RECONCILE_EVERY
            self._reconcile()
        return True

    def check_i1(self) -> None:
        for sha, (state, _) in self.statuses.items():
            if state != d.SUCCESS:
                continue
            if sha in self.tested_main or self.truly_docs_only[sha]:
                continue
            # A pass by the head's own CI definition stands once a reviewer
            # approved that head (and with it the definition).
            if self.touches_ci[sha] and sha in self.approved_heads:
                continue
            raise Violation(
                f"I1: {sha} is success but never passed heavy CI against main"
            )

    def quiesce(self) -> None:
        for _ in range(400):
            progressed = self.system_step()
            if not progressed:
                self.now += RECONCILE_EVERY + 1
                self.next_reconcile = self.now
                before = (
                    dict(self.statuses),
                    len(self.events),
                    [r.latest.status for r in self.runs],
                )
                self._reconcile()
                self.now += 7 * 3600  # past the 6 h "Running" re-check
                self._reconcile()
                after = (
                    dict(self.statuses),
                    len(self.events),
                    [r.latest.status for r in self.runs],
                )
                if before == after and not any(
                    r.latest.status in ("queued", "in_progress") for r in self.runs
                ):
                    return
            self.check_i1()

    def outcome(self) -> str:
        if not self.is_open:
            return "closed"
        if self.base != "main":
            return "base-not-main"
        if self.draft:
            return "draft"
        status = self.statuses.get(self.head)
        if self.deps_open:
            return (
                "blocked"
                if status and status[1].startswith("⛔")
                else "BUG:blocked-not-shown"
            )
        if status is None:
            return "BUG:missing-status"
        state, desc = status
        if state in (d.SUCCESS, d.FAILURE):
            return state
        for stuck in ("Running", "Re-running"):
            if desc.startswith(stuck):
                return f"BUG:stuck-{stuck}"
        if self._authorized():
            return "needs-human:" + desc.split(":")[0].split(";")[0]
        return "waiting-approval"


def simulate(
    seed: int, hyp: dict[str, bool], opts: dict[str, bool], depth: int
) -> tuple[str, Counter[str]]:
    rng = random.Random(seed)
    world = World(rng, hyp, opts)
    for _ in range(depth):
        if rng.random() < 0.45:
            world.human_step()
        else:
            world.system_step()
        world.check_i1()
    world.quiesce()
    return world.outcome(), world.stats


DEFENCES_ON = {
    "buggy_aggregator": False,
    "no_merge_base_check": False,
    "no_time_rule": False,
    "trust_manual_runs": False,
    "trust_incomplete_file_lists": False,
    "trust_own_ci_definition": False,
}


@dataclass
class Tally:
    violations: Counter[str] = field(default_factory=Counter)
    outcomes: Counter[str] = field(default_factory=Counter)
    stats: Counter[str] = field(default_factory=Counter)
    examples: dict[str, tuple[int, dict[str, bool]]] = field(default_factory=dict)

    @property
    def bugs(self) -> int:
        return sum(c for n, c in self.outcomes.items() if n.startswith("BUG"))


def run_all(per: int, opts: dict[str, bool]) -> Tally:
    """`per` seeded sequences under every combination of HYPOTHESES."""
    tally = Tally()
    for values in itertools.product((False, True), repeat=len(HYPOTHESES)):
        hyp = dict(zip(HYPOTHESES, values))
        for i in range(per):
            seed = (
                i * 1_000_003 + sum(v << k for k, v in enumerate(values))
            ) & 0xFFFFFFFF
            try:
                outcome, run_stats = simulate(seed, hyp, opts, depth=24)
            except Violation as exc:
                tally.violations[str(exc).split(":")[0]] += 1
                tally.examples.setdefault(str(exc), (seed, hyp))
                continue
            tally.outcomes[outcome] += 1
            tally.stats.update(run_stats)
            if outcome.startswith("BUG"):
                tally.examples.setdefault(outcome, (seed, hyp))
    return tally


def main() -> int:
    per = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 2000
    opts = {flag: f"--{flag.replace('_', '-')}" in sys.argv for flag in DEFENCES_ON}
    tally = run_all(per, opts)
    violations, outcomes, stats, examples = (
        tally.violations,
        tally.outcomes,
        tally.stats,
        tally.examples,
    )
    total = per * 2 ** len(HYPOTHESES)
    print(f"sequences: {total}  options: {opts}")
    print(f"I1 violations: {sum(violations.values())} {dict(violations)}")
    for msg, (seed, hyp) in list(examples.items())[:3]:
        print(f"  e.g. seed={seed} {msg} hyp={ {k: v for k, v in hyp.items() if v} }")
    print("final outcomes:")
    for name, count in outcomes.most_common():
        print(f"  {name:70s} {count}")
    print(f"stats: {dict(stats)}")
    return 1 if violations or tally.bugs else 0


if __name__ == "__main__":
    sys.exit(main())
