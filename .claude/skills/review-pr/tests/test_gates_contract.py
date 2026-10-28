"""Pin the gate roster and the required-artifact list.

A gate that is silently renamed or dropped does not fail anything -- the
review simply stops being checked on that axis. This test is the only thing
that notices.
"""

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import _lib

EXPECTED_GATES = [
    "answers",
    "diagnostic",
    "corefiles",
    "ledger",
    "card",
    "refutations",
    "independent",
]
EXPECTED_REQUIRED = [
    "card.md",
    "verdicts.txt",
    "answers.txt",
    "ai_diagnostic.txt",
    "core_files.txt",
    "refutations.txt",
    "independent.txt",
    "rules.txt",
    "pr_meta.json",
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
    """An empty WORK dir means no step ran.

    Four gates own "did the work happen at all": answers, diagnostic, corefiles and
    ledger each read one artifact that Step 1/1b should have produced, and each must
    go red when that artifact is empty. `card` is deliberately not asserted here --
    its scope is card-vs-verdicts consistency (every FIRE traced to a card line and
    vice versa), and an empty card with an empty verdicts.txt is genuinely nothing
    for it to catch; that is not the gate being broken, it is a different gate's job
    (the four above) to notice that nothing ran. What the pipeline actually depends
    on is that the gate *run* as a whole cannot go green on an empty WORK dir, which
    is asserted directly below via `_gates.py`'s own exit code.
    """
    (tmp_path / "pr.diff").write_text("", encoding="utf-8")
    (tmp_path / "rules.txt").write_text("", encoding="utf-8")
    res = _lib.run_gates(str(tmp_path), PROJ)
    by_name = {name: rc for name, rc, _, _ in res}
    for gate in ("answers", "diagnostic", "corefiles", "ledger"):
        assert (
            by_name[gate] != 0
        ), f"{gate} went green on an empty WORK dir -- it fails open"

    gates_py = HERE / "_gates.py"
    p = subprocess.run(
        [sys.executable, str(gates_py), str(tmp_path), PROJ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode != 0, (
        "_gates.py exited 0 on an empty WORK dir -- the pipeline would treat "
        f"an unrun review as a passing one\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    )
