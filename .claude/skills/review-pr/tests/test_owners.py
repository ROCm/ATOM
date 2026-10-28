"""Every failure class run_one.sh can emit must have an owner.

A class with no owner routes a real failure to nobody, silently: the review
died, someone needs to look, and no one is told.
"""

import pathlib
import re
import sys

SKILL = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL))
import _notify


def test_every_emitted_class_has_an_owner():
    run_one = (SKILL / "run_one.sh").read_text(encoding="utf-8")
    emitted = set(re.findall(r"\bfail\s+(\w+)\s+\d+", run_one))
    assert emitted, "found no `fail <class> <code>` calls in run_one.sh"
    missing = sorted(c for c in emitted if c not in _notify.OWNERS)
    assert not missing, f"run_one.sh emits classes with no owner: {missing}"


def test_unknown_class_routes_to_flow():
    assert _notify.owner_for("no-such-class") == _notify.owner_for("flow")
