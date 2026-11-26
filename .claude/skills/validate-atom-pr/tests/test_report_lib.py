"""Hermetic tests for validate-atom-pr's report_lib.py -- the verdict table (the whole
point of this validator) and schema-valid report assembly.

Hermetic on purpose: no network, no real GPU, no pytest subprocess, no dependence on any
untracked file (an earlier branch had a bug where an untracked `reports/` directory passed
locally but failed on a fresh CI checkout -- these tests only ever read files that ship in
this skill directory, loaded by absolute path computed from this test file's own location).
"""

import importlib.util
import json
import pathlib

import jsonschema
import pytest

SKILL = pathlib.Path(__file__).resolve().parent.parent


def _load_report_lib():
    spec = importlib.util.spec_from_file_location("report_lib", SKILL / "report_lib.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report_lib = _load_report_lib()


def _stats(executed=1, failures=0, errors=0, tests=None, skipped=0):
    return {
        "tests": tests if tests is not None else executed + skipped,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "executed": executed,
    }


def _schema():
    return json.loads((SKILL / "report_schema.json").read_text())


def _identity():
    return {
        "repo": {
            "worktree": "/tmp/repo",
            "base": "a" * 40,
            "head": "b" * 40,
            "kind": "atom",
            "patch_sha256": "c" * 64,
        },
        "runtime_identity": {
            "module_path": "/tmp/repo/atom/__init__.py",
            "python_executable": "/usr/bin/python3",
            "native_artifacts": [],
        },
        "test_selection": {
            "target": "tests/test_example.py",
            "runner": "pytest",
            "runner_reason": "validate-atom-pr always invokes --target as a pytest node/file",
        },
    }


# ---------------------------------------------------------------------------------------
# decide_verdict -- the four base/head cells plus the base_missing variants.
# ---------------------------------------------------------------------------------------


def test_pass_pass_is_pass():
    verdict, findings = report_lib.decide_verdict(True, True)
    assert verdict == "PASS"
    assert findings == []


def test_pass_fail_is_block():
    verdict, findings = report_lib.decide_verdict(True, False)
    assert verdict == "BLOCK"
    assert findings[0]["severity"] == "blocker"


def test_fail_fail_is_needs_work():
    verdict, findings = report_lib.decide_verdict(False, False)
    assert verdict == "NEEDS_WORK"
    assert findings[0]["severity"] == "should-fix"


def test_fail_pass_is_pass_with_note():
    verdict, findings = report_lib.decide_verdict(False, True)
    assert verdict == "PASS"
    assert findings[0]["severity"] == "note"
    assert "fixes a" in findings[0]["detail"]


def test_base_missing_and_head_fails_is_block_not_needs_work():
    """A newly-added failing test is this PR's own regression -- there is no 'pre-existing'
    baseline for a file that did not exist before the patch."""
    verdict, findings = report_lib.decide_verdict(False, False, base_missing=True)
    assert verdict == "BLOCK"
    assert findings[0]["severity"] == "blocker"


def test_base_missing_and_head_passes_is_pass_with_note():
    verdict, findings = report_lib.decide_verdict(False, True, base_missing=True)
    assert verdict == "PASS"
    assert "does not exist at base" in findings[0]["detail"]


# ---------------------------------------------------------------------------------------
# build_report -- full report assembly, each of the four verdict combos, schema-valid.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_stats,head_stats,expected_verdict,expected_exit",
    [
        (_stats(executed=5), _stats(executed=5), "PASS", 0),
        (_stats(executed=5), _stats(executed=5, failures=1), "BLOCK", 1),
        (
            _stats(executed=5, failures=1),
            _stats(executed=5, failures=1),
            "NEEDS_WORK",
            1,
        ),
        (_stats(executed=5, failures=1), _stats(executed=5), "PASS", 0),
    ],
)
def test_build_report_verdict_table_and_schema(
    base_stats, head_stats, expected_verdict, expected_exit
):
    identity = _identity()
    report = report_lib.build_report(
        label="unit-test",
        started_utc="2026-01-01T00:00:00Z",
        finished_utc="2026-01-01T00:05:00Z",
        repo=identity["repo"],
        runtime_identity=identity["runtime_identity"],
        test_selection=identity["test_selection"],
        stages={
            "merge_sim": {"status": "pass"},
            "gpu_claim": {"status": "pass", "hip_index": 0, "arch": "gfx942"},
            "runtime_compat": {"status": "pass"},
            "test_policy": {"status": "pass"},
            "baseline_control": {"status": "pass", "stats": base_stats},
            "correctness_repo_tests": {
                "status": "pass" if report_lib._stat_ok(head_stats) else "fail",
                "stats": head_stats,
            },
        },
        base_stats=base_stats,
        head_stats=head_stats,
        gpu_arch="gfx942",
    )
    assert report["verdict"] == expected_verdict
    assert report["process_exit_code"] == expected_exit
    jsonschema.validate(report, _schema())


def test_build_report_always_fills_the_nine_required_stages():
    """Stages this minimal validator permanently skips (shape grid, execution receipt,
    index-width scan) must still appear in the report, 'skip' with a reason -- never
    silently omitted. report_schema.json requires all 9 stage keys for exactly this
    reason."""
    identity = _identity()
    report = report_lib.build_report(
        label="unit-test",
        started_utc="2026-01-01T00:00:00Z",
        finished_utc="2026-01-01T00:05:00Z",
        repo=identity["repo"],
        runtime_identity=identity["runtime_identity"],
        test_selection=identity["test_selection"],
        stages={
            "merge_sim": {"status": "pass"},
            "gpu_claim": {"status": "pass"},
            "runtime_compat": {"status": "pass"},
            "test_policy": {"status": "pass"},
            "baseline_control": {"status": "pass", "stats": _stats()},
            "correctness_repo_tests": {"status": "pass", "stats": _stats()},
        },
        base_stats=_stats(),
        head_stats=_stats(),
        gpu_arch="gfx942",
    )
    required = {
        "merge_sim",
        "gpu_claim",
        "runtime_compat",
        "test_policy",
        "baseline_control",
        "correctness_repo_tests",
        "correctness_s1_grid",
        "execution_receipt",
        "index_width_scan",
    }
    assert required <= report["stages"].keys()
    for name in ("correctness_s1_grid", "execution_receipt", "index_width_scan"):
        assert report["stages"][name]["status"] == "skip"
        assert report["stages"][name]["note"]
    jsonschema.validate(report, _schema())


def test_report_identity_fields_come_from_real_inputs():
    """repo.base/repo.head/repo.patch_sha256 are exactly the fields fetch.sh's identity
    gate cross-checks against the PR it is reviewing -- this test pins that build_report
    passes them through verbatim rather than re-deriving or dropping them."""
    identity = _identity()
    identity["repo"]["base"] = "1" * 40
    identity["repo"]["head"] = "2" * 40
    identity["repo"]["patch_sha256"] = "3" * 64
    report = report_lib.build_report(
        label="unit-test",
        started_utc="2026-01-01T00:00:00Z",
        finished_utc="2026-01-01T00:05:00Z",
        repo=identity["repo"],
        runtime_identity=identity["runtime_identity"],
        test_selection=identity["test_selection"],
        stages={
            "merge_sim": {"status": "pass"},
            "gpu_claim": {"status": "pass"},
            "runtime_compat": {"status": "pass"},
            "test_policy": {"status": "pass"},
            "baseline_control": {"status": "pass", "stats": _stats()},
            "correctness_repo_tests": {"status": "pass", "stats": _stats()},
        },
        base_stats=_stats(),
        head_stats=_stats(),
        gpu_arch="gfx942",
    )
    assert report["repo"]["base"] == "1" * 40
    assert report["repo"]["head"] == "2" * 40
    assert report["repo"]["patch_sha256"] == "3" * 64
    jsonschema.validate(report, _schema())


def test_inconclusive_report_is_still_schema_valid_and_still_written():
    """BLOCK/NEEDS_WORK/INCONCLUSIVE all still write a report worth consuming -- only a
    missing file at --out means nothing to read. This pins that an INCONCLUSIVE report
    (an earlier stage failed to complete) is itself schema-valid, not a bare early exit.
    """
    identity = _identity()
    report = report_lib.build_inconclusive_report(
        label="unit-test",
        started_utc="2026-01-01T00:00:00Z",
        finished_utc="2026-01-01T00:01:00Z",
        repo=identity["repo"],
        runtime_identity=identity["runtime_identity"],
        test_selection=identity["test_selection"],
        stages={"gpu_claim": {"status": "fail", "note": "no idle GPU"}},
        reason="no idle GPU could be claimed on this box",
        blocking_stage="gpu_claim",
        degraded_mode="NO_GPU",
    )
    assert report["verdict"] == "INCONCLUSIVE"
    assert report["process_exit_code"] == 2
    assert report["degraded_mode"] == "NO_GPU"
    jsonschema.validate(report, _schema())


def test_block_report_is_schema_valid_and_has_nonzero_exit():
    """validate_pr.sh exits non-zero on BLOCK but must always still write the report --
    this pins the report side of that contract (report_lib never special-cases "don't
    write" for any verdict)."""
    identity = _identity()
    report = report_lib.build_report(
        label="unit-test",
        started_utc="2026-01-01T00:00:00Z",
        finished_utc="2026-01-01T00:05:00Z",
        repo=identity["repo"],
        runtime_identity=identity["runtime_identity"],
        test_selection=identity["test_selection"],
        stages={
            "merge_sim": {"status": "pass"},
            "gpu_claim": {"status": "pass"},
            "runtime_compat": {"status": "pass"},
            "test_policy": {"status": "pass"},
            "baseline_control": {"status": "pass", "stats": _stats()},
            "correctness_repo_tests": {"status": "fail", "stats": _stats(failures=1)},
        },
        base_stats=_stats(),
        head_stats=_stats(failures=1),
        gpu_arch="gfx942",
    )
    assert report["verdict"] == "BLOCK"
    assert report["process_exit_code"] == 1
    jsonschema.validate(report, _schema())


# ---------------------------------------------------------------------------------------
# report_schema.json itself -- the 14 top-level required fields and 9 stage entries, as a
# direct assertion against the schema file, independent of report_lib's own behavior.
# ---------------------------------------------------------------------------------------


def test_schema_top_level_required_fields():
    schema = _schema()
    assert set(schema["required"]) == {
        "label",
        "started_utc",
        "finished_utc",
        "isolation",
        "arch_coverage",
        "arch_coverage_basis",
        "degraded_mode",
        "repo",
        "runtime_identity",
        "test_selection",
        "stages",
        "findings",
        "verdict",
        "process_exit_code",
    }


def test_schema_stages_required_entries():
    schema = _schema()
    assert set(schema["properties"]["stages"]["required"]) == {
        "merge_sim",
        "gpu_claim",
        "runtime_compat",
        "test_policy",
        "baseline_control",
        "correctness_repo_tests",
        "correctness_s1_grid",
        "execution_receipt",
        "index_width_scan",
    }
