"""The vLLM plugin calls `h2d_group_members`; it does not read it.

`AttentionMetadataBuilder.h2d_group_members` became a method in #3136. The
plugin composes the builder's groups itself (`_publish_v41_metadata`), and a
read of the old attribute fails at the first forward with
`'function' object has no attribute 'items'` -- in a module the non-GPU unit
run cannot import, so this checks the source instead.
"""

import ast
import pathlib

PLUGIN = pathlib.Path(__file__).resolve().parents[1] / "atom" / "plugin"


def _uncalled_reads() -> list[str]:
    found = []
    for path in sorted(PLUGIN.rglob("*.py")):
        tree = ast.parse(path.read_text())
        called = {
            id(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "h2d_group_members"
                and isinstance(node.ctx, ast.Load)
                and id(node) not in called
            ):
                found.append(f"{path.relative_to(PLUGIN.parents[1])}:{node.lineno}")
    return found


def test_the_plugin_calls_h2d_group_members():
    assert _uncalled_reads() == []


def test_the_check_sees_the_v41_publication():
    # Guard against the scan going vacuous: the V4.1 publication wrapper is
    # the reader this test exists for.
    sources = [p.read_text() for p in PLUGIN.rglob("*.py")]
    assert any("h2d_group_members()" in s for s in sources)
