"""check_citations.py splits its verdict in two, and the split is the whole design.

Deterministic failures (the cited file does not resolve, the line is past EOF) exit
non-zero. The proximity check -- does an identifier the finding quotes appear near the
line it cites -- only prints a note, because matching prose against source is fuzzy and
it ran 1-for-2 on seven real ATOM reviews. These tests pin that split: a tightening that
makes proximity gate, or a loosening that stops the deterministic half failing, breaks
one of them.
"""

import pathlib
import subprocess
import sys

SKILL = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = SKILL / "check_citations.py"


def run(work, tree):
    p = subprocess.run(
        [sys.executable, str(SCRIPT), str(work), str(tree)],
        capture_output=True,
        text=True,
        check=False,  # a non-zero exit is the thing under test, not an error
    )
    return p.returncode, p.stdout


def setup(tmp_path, card, src="x = 1\ny = 2\nTARGET = 3\nz = 4\n", path="a/b.py"):
    work, tree = tmp_path / "work", tmp_path / "tree"
    (work).mkdir()
    f = tree / path
    f.parent.mkdir(parents=True)
    f.write_text(src, encoding="utf-8")
    (work / "card.md").write_text(card, encoding="utf-8")
    return work, tree


def test_resolving_citation_passes(tmp_path):
    w, t = setup(tmp_path, "📝 a/b.py:3 sets `TARGET` wrongly\n")
    rc, out = run(w, t)
    assert rc == 0
    assert "every cited file:line resolves" in out


def test_missing_file_is_a_hard_failure(tmp_path):
    w, t = setup(tmp_path, "📝 nope/gone.py:3 does a thing\n")
    rc, out = run(w, t)
    assert rc == 1
    assert "no such file" in out


def test_line_past_eof_is_a_hard_failure(tmp_path):
    w, t = setup(tmp_path, "📝 a/b.py:9000 does a thing\n")
    rc, out = run(w, t)
    assert rc == 1
    assert "file has" in out


def test_proximity_miss_is_only_a_note(tmp_path):
    # TARGET is at :3; citing :1 while quoting it is a 2-line drift -> within tolerance.
    # Put it far away instead: a 40-line file with the symbol at the end.
    src = "\n".join(["pad"] * 40 + ["TARGET = 3"]) + "\n"
    w, t = setup(tmp_path, "📝 a/b.py:2 sets `TARGET` wrongly\n", src=src)
    rc, out = run(w, t)
    assert rc == 0, "a proximity miss must not withhold a review"
    assert "CITATIONS NOTE" in out
    assert "TARGET" in out


def test_small_line_drift_is_tolerated(tmp_path):
    w, t = setup(tmp_path, "📝 a/b.py:1 sets `TARGET` wrongly\n")
    rc, out = run(w, t)
    assert rc == 0
    assert "NOTE" not in out, "2 lines of drift is normal on a moving branch"


def test_bare_basename_resolves_through_the_diff(tmp_path):
    w, t = setup(tmp_path, "📝 b.py:3 sets `TARGET` wrongly\n")
    (w / "pr.diff").write_text("+++ b/a/b.py\n", encoding="utf-8")
    rc, out = run(w, t)
    assert rc == 0, out


def test_no_card_fails_closed(tmp_path):
    work, tree = tmp_path / "work", tmp_path / "tree"
    work.mkdir()
    tree.mkdir()
    rc, out = run(work, tree)
    assert rc == 1
    assert "no card.md" in out


def test_non_finding_lines_are_not_checked(tmp_path):
    # Prose outside a finding may cite anything; only findings carry the contract.
    w, t = setup(tmp_path, "Review: LGTM\nsee also nope/gone.py:9000 for context\n")
    rc, out = run(w, t)
    assert rc == 0, out
