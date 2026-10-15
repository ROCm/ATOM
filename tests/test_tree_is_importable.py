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


def test_calls_to_v41_bridge_helpers_match_their_signatures():
    """Catch a merge that took one side's call and the other side's def.

    `_check_row_alignment` went out with a three-argument call against a
    four-argument definition: the call was inside the conflict region and
    the definition was not, so resolving the conflict changed only one of
    them. Every unit test passed -- none of them reaches that path -- and
    four workers failed to start.

    Checked for the handful of module-level helpers this branch and the
    sibling one both edit, by walking the AST rather than by importing:
    importing drags in vLLM and torch, and the point is to be cheap enough
    that nobody skips it.
    """
    import inspect

    WATCHED = {
        "_check_row_alignment",
        "_v41_state_checkpoints_around_forward",
        "v41_store_images_after_forward",
        "snapshot_v41_batch",
    }
    src = (ROOT / "atom/plugin/vllm/deepseek_v41_bridge.py").read_text()
    tree = ast.parse(src)
    arity = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WATCHED:
            required = len(node.args.args) - len(node.args.defaults)
            arity[node.name] = (required, len(node.args.args))

    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in arity or node.keywords:
            continue
        lo, hi = arity[name]
        n = len(node.args)
        if not (lo <= n <= hi):
            bad.append(f"line {node.lineno}: {name}({n} args), def takes {lo}-{hi}")
    assert not bad, "call/def mismatch:\n  " + "\n  ".join(bad)
