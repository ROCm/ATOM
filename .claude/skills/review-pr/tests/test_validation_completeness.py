"""A validator may declare a stage out of scope; it may not declare away its evidence.

fetch.sh decides a validation report is `complete` -- which is what entitles it to a
PASS rather than INCONCLUSIVE -- by checking all nine stages. The original check demanded
every stage be "pass", which no validator honestly limited to a base/head comparison can
satisfy, since three stages name work it never claimed to do. Relaxing that to "pass, or
skipped with a reason" fixed the false rejection and opened a worse hole: a report with
all nine stages skipped was complete, and so was one that skipped baseline_control.

Skipping baseline_control is not a scope decision. Without the base run, "head fails" and
"was already failing" produce the same report -- the one distinction the validator exists
to make. So the tiers are pinned here: out-of-scope stages may be skipped with a stated
reason, evidence stages may not, and a skip with an empty reason never counts.

The logic lives in a heredoc inside fetch.sh, so it is extracted and exercised rather
than imported. If that block moves or is renamed, these tests fail loudly instead of
silently testing nothing -- the extraction asserts it found something.
"""

import pathlib
import re

import pytest

FETCH = pathlib.Path(__file__).resolve().parent.parent / "fetch.sh"
STAGES = [
    "merge_sim",
    "gpu_claim",
    "runtime_compat",
    "test_policy",
    "baseline_control",
    "correctness_repo_tests",
    "correctness_s1_grid",
    "execution_receipt",
    "index_width_scan",
]


def _complete(stages):
    lines = FETCH.read_text(encoding="utf-8").split("\n")
    try:
        d0 = next(i for i, x in enumerate(lines) if x.startswith("SCOPE_DECLARABLE"))
        d1 = next(
            i for i in range(d0, len(lines)) if lines[i].strip() == "return False"
        )
        c0 = next(i for i, x in enumerate(lines) if re.match(r"\s*complete\s*=", x))
    except StopIteration:  # pragma: no cover - the guard is the point
        pytest.fail(
            "could not find the completeness block in fetch.sh; it moved or was renamed"
        )
    ns = {}
    decl = "\n".join(lines[d0 : d1 + 1])
    exec(decl, ns)  # noqa: S102 - running fetch.sh's real block is the point
    body = "\n".join(x.removeprefix("    ") for x in lines[c0 : c0 + 14])
    ns.update(
        report={"stages": stages},
        selection={"runner": "pytest"},
        coverage={"gfx950": True},
        coverage_basis={"basis": "declared"},
    )
    exec(body, ns)  # noqa: S102
    return ns["complete"]


def _all_pass():
    s = {n: {"status": "pass"} for n in STAGES}
    s["index_width_scan"] = {"status": "info"}
    return s


def test_a_fully_performed_run_is_complete():
    assert _complete(_all_pass()) is True


def test_every_stage_skipped_is_not_complete():
    skipped = {n: {"status": "skip", "note": "a reason"} for n in STAGES}
    assert (
        _complete(skipped) is False
    ), "a report that ran nothing must not entitle PASS"


@pytest.mark.parametrize("stage", ["baseline_control", "correctness_repo_tests"])
def test_evidence_stages_cannot_be_skipped(stage):
    s = _all_pass()
    s[stage] = {"status": "skip", "note": "a plausible-sounding reason"}
    assert _complete(s) is False, f"{stage} is the evidence, not a scope decision"


def test_out_of_scope_stages_may_be_skipped_with_a_reason():
    s = _all_pass()
    for n in ("correctness_s1_grid", "execution_receipt", "index_width_scan"):
        s[n] = {"status": "skip", "note": "outside this validator's declared scope"}
    assert _complete(s) is True


def test_a_skip_with_no_reason_never_counts():
    s = _all_pass()
    s["correctness_s1_grid"] = {"status": "skip", "note": ""}
    assert _complete(s) is False


def test_a_stage_that_ran_and_failed_still_blocks():
    s = _all_pass()
    s["correctness_s1_grid"] = {"status": "fail", "note": "it ran and it failed"}
    assert _complete(s) is False
