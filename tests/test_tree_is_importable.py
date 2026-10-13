"""Every module in the tree parses, and none carries a merge marker.

Both failures this covers were committed and pushed. A merge left raw
`<<<<<<< HEAD` in two files -- I had read the merge's output through
`tail -8` and taken the last line for the whole conflict list, then
`git add -A` staged the markers. The unit suite stayed green because
nothing in it imports those modules, and the engine found them by failing
to start four workers with `IndentationError`.

A syntax check is not a substitute for tests. It is a substitute for an
engine boot, which is what was paying for this.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
MARKERS = ("<<<<<<< ", "=======\n>>>>>>> ", ">>>>>>> ")


def _sources():
    for base in ("atom", "tests", "tools"):
        yield from (ROOT / base).rglob("*.py")


def test_every_module_parses():
    broken = []
    for path in _sources():
        try:
            ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            broken.append(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
    assert not broken, "modules that do not parse:\n  " + "\n  ".join(broken)


def test_no_file_carries_a_merge_marker():
    """Separate from parsing: a marker inside a docstring or a comment
    parses fine and is still an unfinished merge."""
    marked = []
    for path in _sources():
        if path.name == pathlib.Path(__file__).name:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.startswith("<<<<<<< ") or line.startswith(">>>>>>> "):
                marked.append(f"{path.relative_to(ROOT)}:{n}")
                break
    assert not marked, "unfinished merges:\n  " + "\n  ".join(marked)
