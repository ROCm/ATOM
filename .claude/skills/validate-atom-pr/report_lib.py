"""Pure report-assembly logic for validate-atom-pr.

validate_pr.sh does the "doing" (git apply, claiming a GPU, invoking pytest at base and at
head); this module does the "reporting" (turning the raw facts that doing produced into a
report_schema.json-valid document, and deciding the verdict). Splitting it out this way
means the verdict table -- the whole point of this validator -- can be unit tested directly,
with synthetic facts, with no GPU, no pytest subprocess and no network.

The verdict table, from the user's own scope decision (no perf scraping, no shape grid):

    base    head    verdict
    pass    pass    PASS
    pass    fail    BLOCK        -- this PR introduced it
    fail    fail    NEEDS_WORK   -- pre-existing, not this PR's fault
    fail    pass    PASS         -- the PR fixes it (said so in findings, severity "note")

An earlier stage that could not complete (merge conflict, no runtime_compat, no collectable
target, no GPU claimed) means base and head were never both actually run -- that is
INCONCLUSIVE, a fifth state, not a sixth cell of the table above.
"""

from __future__ import annotations

import copy

NOT_DERIVED_ROUTE = "not-derived: route detection is out of scope for validate-atom-pr"
NOT_DERIVED_SHAPE_VARS = (
    "not-derived: shape-grid instrumentation is out of scope for validate-atom-pr"
)

SKIP_REASONS = {
    "correctness_s1_grid": (
        "shape grid explicitly out of scope for validate-atom-pr (base/head single-target "
        "comparison only) -- see .claude/skills/validate-atom-pr/SKILL.md"
    ),
    "execution_receipt": (
        "route/kernel-symbol execution receipt not implemented by this minimal validator; "
        "no route or shape-grid instrumentation is performed"
    ),
    "index_width_scan": (
        "performed separately by review-pr/fetch.sh via scan_index_width.py; not duplicated "
        "here"
    ),
}

# report_schema.json's `stages` requires exactly these 9 keys on every report, including an
# INCONCLUSIVE one that stopped after only the first one or two stages actually ran -- a
# report cannot overclaim by omission, so a stage the run never reached is still present,
# 'skip', with a reason that says so, not a silently missing key.
ALL_STAGE_NAMES = (
    "merge_sim",
    "gpu_claim",
    "runtime_compat",
    "test_policy",
    "baseline_control",
    "correctness_repo_tests",
    "correctness_s1_grid",
    "execution_receipt",
    "index_width_scan",
)
NOT_REACHED_REASON = "run stopped at an earlier stage before this one was reached"


def _stat_ok(stats: dict) -> bool:
    return (
        stats.get("executed", 0) >= 1
        and stats.get("failures", 0) == 0
        and stats.get("errors", 0) == 0
    )


def decide_verdict(
    base_ok: bool, head_ok: bool, base_missing: bool = False
) -> tuple[str, list[dict]]:
    """Implements the base/head table. Returns (verdict, extra_findings).

    `base_missing` is true when --target's file did not exist before the patch at all (the
    PR adds it) -- there was no baseline to run, as opposed to a baseline that ran and failed.
    It only changes which of BLOCK/NEEDS_WORK applies when head still fails: a newly-added
    failing test is this PR's own regression, never a "pre-existing" one, since nothing
    existed to pre-exist.
    """
    if base_ok and head_ok:
        return "PASS", []
    if base_ok and not head_ok:
        return "BLOCK", [
            {
                "severity": "blocker",
                "stage": "correctness_repo_tests",
                "detail": "target passed at base and fails at head: this PR introduced the "
                "failure.",
            }
        ]
    if not base_ok and not head_ok:
        if base_missing:
            return "BLOCK", [
                {
                    "severity": "blocker",
                    "stage": "correctness_repo_tests",
                    "detail": "target does not exist at base (added by this PR) and fails at "
                    "head: there is no 'pre-existing' baseline here, so a failing new test "
                    "is this PR's own regression.",
                }
            ]
        return "NEEDS_WORK", [
            {
                "severity": "should-fix",
                "stage": "baseline_control",
                "detail": "target already fails at base; pre-existing, not this PR's fault. "
                "Still needs fixing, but not by this PR.",
            }
        ]
    # not base_ok and head_ok
    if base_missing:
        return "PASS", [
            {
                "severity": "note",
                "stage": "baseline_control",
                "detail": "target does not exist at base (added by this PR); no baseline "
                "comparison was possible. Head run passes.",
            }
        ]
    return "PASS", [
        {
            "severity": "note",
            "stage": "baseline_control",
            "detail": "target failed at base and passes at head: this PR fixes a "
            "pre-existing failure.",
        }
    ]


def _exit_code_for(verdict: str) -> int:
    return {"PASS": 0, "INCONCLUSIVE": 2}.get(verdict, 1)


def build_inconclusive_report(
    *,
    label: str,
    started_utc: str,
    finished_utc: str,
    repo: dict,
    runtime_identity: dict,
    test_selection: dict,
    stages: dict,
    reason: str,
    blocking_stage: str,
    degraded_mode: str | None = None,
) -> dict:
    """A stage before baseline_control/correctness_repo_tests could not complete.

    Neither run happened (or happened only partially), so base and head are not both known;
    this is INCONCLUSIVE rather than a guess at BLOCK/NEEDS_WORK/PASS.
    """
    full_stages = _fill_stage_skeleton(stages)
    report = _skeleton(
        label=label,
        started_utc=started_utc,
        finished_utc=finished_utc,
        repo=repo,
        runtime_identity=runtime_identity,
        test_selection=test_selection,
        stages=full_stages,
        verdict="INCONCLUSIVE",
        findings=[
            {
                "severity": "note",
                "stage": blocking_stage,
                "detail": reason,
            }
        ],
        arch_coverage={},
        arch_coverage_basis={},
        degraded_mode=degraded_mode,
    )
    report["process_exit_code"] = _exit_code_for("INCONCLUSIVE")
    return report


def build_report(
    *,
    label: str,
    started_utc: str,
    finished_utc: str,
    repo: dict,
    runtime_identity: dict,
    test_selection: dict,
    stages: dict,
    base_stats: dict,
    head_stats: dict,
    gpu_arch: str,
    base_missing: bool = False,
) -> dict:
    """The common path: merge_sim/gpu_claim/runtime_compat/test_policy all completed, and
    the head run actually executed (whether it passed or not). The base run executed too,
    unless base_missing (the target did not exist before the patch)."""
    base_ok = _stat_ok(base_stats)
    head_ok = _stat_ok(head_stats)
    verdict, extra_findings = decide_verdict(base_ok, head_ok, base_missing)

    full_stages = _fill_stage_skeleton(stages)
    complete = verdict == "PASS"
    arch_coverage = {gpu_arch: "runtime"} if complete and gpu_arch else {}
    arch_coverage_basis = (
        {gpu_arch: f"pytest-junit-executed:{head_stats.get('executed', 0)}"}
        if arch_coverage
        else {}
    )

    report = _skeleton(
        label=label,
        started_utc=started_utc,
        finished_utc=finished_utc,
        repo=repo,
        runtime_identity=runtime_identity,
        test_selection=test_selection,
        stages=full_stages,
        verdict=verdict,
        findings=extra_findings,
        arch_coverage=arch_coverage,
        arch_coverage_basis=arch_coverage_basis,
    )
    report["process_exit_code"] = _exit_code_for(verdict)
    return report


def _fill_stage_skeleton(stages: dict) -> dict:
    """Every one of the 9 required stage keys is present on exit, no matter how early the
    run stopped. A stage this validator permanently skips by scope decision gets its
    stated reason (SKIP_REASONS); any other stage the caller did not reach yet (the common
    case for an INCONCLUSIVE report that bailed after gpu_claim or runtime_compat) gets a
    generic 'not reached' skip instead of a silently-missing key."""
    full = copy.deepcopy(stages)
    for name in ALL_STAGE_NAMES:
        if name in full:
            continue
        reason = SKIP_REASONS.get(name, NOT_REACHED_REASON)
        full[name] = {"status": "skip", "note": reason}
    return full


def _skeleton(
    *,
    label: str,
    started_utc: str,
    finished_utc: str,
    repo: dict,
    runtime_identity: dict,
    test_selection: dict,
    stages: dict,
    verdict: str,
    findings: list[dict],
    arch_coverage: dict,
    arch_coverage_basis: dict,
    degraded_mode: str | None = None,
) -> dict:
    full_selection = {
        "shape_env": "",
        "shape_arg": "",
        "grid": "",
        "expected_route": NOT_DERIVED_ROUTE,
        "shape_vars": NOT_DERIVED_SHAPE_VARS,
        **test_selection,
    }
    return {
        "label": label,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "isolation": {
            "level": "git-worktree + private caches",
            "container": False,
            "reason": "runs directly in the supplied --repo checkout, no container layer",
        },
        "arch_coverage": arch_coverage,
        "arch_coverage_basis": arch_coverage_basis,
        "degraded_mode": degraded_mode,
        "repo": repo,
        "runtime_identity": runtime_identity,
        "test_selection": full_selection,
        "stages": stages,
        "findings": findings,
        "verdict": verdict,
        "process_exit_code": None,  # filled in by the caller
    }
