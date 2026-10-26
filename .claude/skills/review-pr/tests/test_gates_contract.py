"""Pin the gate roster and the required-artifact list.

A gate that is silently renamed or dropped does not fail anything -- the
review simply stops being checked on that axis. This test is the only thing
that notices.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import _lib

EXPECTED_GATES = [
    "answers", "diagnostic", "corefiles", "ledger",
    "card", "refutations", "independent",
]
EXPECTED_REQUIRED = [
    "card.md", "verdicts.txt", "answers.txt", "ai_diagnostic.txt",
    "core_files.txt", "refutations.txt", "independent.txt",
    "rules.txt", "pr_meta.json",
]
PROJ = str(HERE.parent.parent.parent)


def test_required_artifacts_are_pinned():
    assert _lib.REQUIRED == EXPECTED_REQUIRED
    assert _lib.OPTIONAL == ["late_findings.txt"]


def test_gate_roster_is_pinned(tmp_path):
    (tmp_path / "pr.diff").write_text("", encoding="utf-8")
    (tmp_path / "rules.txt").write_text("", encoding="utf-8")
    res = _lib.run_gates(str(tmp_path), PROJ)
    assert [name for name, _, _, _ in res] == EXPECTED_GATES


def test_empty_work_dir_fails_closed(tmp_path):
    """An empty WORK dir means no step ran. Every gate must go red, not green."""
    (tmp_path / "pr.diff").write_text("", encoding="utf-8")
    (tmp_path / "rules.txt").write_text("", encoding="utf-8")
    res = _lib.run_gates(str(tmp_path), PROJ)
    assert all(rc != 0 for _, rc, _, _ in res), \
        "a gate went green on an empty WORK dir -- it fails open"
