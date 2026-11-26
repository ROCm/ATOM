# SPDX-License-Identifier: MIT
"""The SGLang mono-refusal stand-in must match the function it replaces.

`install_minimax_mono_plugin_refusal` rebinds
`atom.models.minimax_m3.mono.dispatch._config_refusal` at runtime. A rebind
is invisible to every static check in this repo -- ruff and black see two
unrelated functions, and the only code path that exercises the pairing is an
SGLang MiniMax-M3 launch, which lives in the `ci:sglang` heavy job rather
than pre-checkin. That is how `#2479` dropped `text_config` from the native
signature and its one call site while the stand-in kept demanding it: every
SGLang MiniMax-M3 start raised

    TypeError: install_minimax_mono_plugin_refusal.<locals>._config_refusal()
               missing 1 required positional argument: 'text_config'

at model construction for ~20 days before anyone read the traceback.

Read with `ast` rather than imported: `dispatch` pulls in the model stack and
the bridge pulls in SGLang, neither of which exists on the CPU-only runner.
An import-based test would `importorskip` here and guard nothing where it
actually needs to run.
"""

from __future__ import annotations

import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
NATIVE = REPO / "atom" / "models" / "minimax_m3" / "mono" / "dispatch.py"
BRIDGE = REPO / "atom" / "plugin" / "sglang" / "minimax_m3_bridge.py"
INSTALLER = "install_minimax_mono_plugin_refusal"


def _args(node: ast.FunctionDef) -> list[str]:
    a = node.args
    assert not a.vararg and not a.kwarg, f"{node.name} grew *args/**kwargs"
    return [arg.arg for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs)]


def _module_level(path: pathlib.Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path.name} defines no module-level {name}()")


def _nested(path: pathlib.Path, outer: str, inner: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name == outer:
            for child in ast.walk(node):
                if isinstance(child, ast.FunctionDef) and child.name == inner:
                    return child
            raise AssertionError(f"{outer}() defines no nested {inner}()")
    raise AssertionError(f"{path.name} defines no {outer}()")


def test_stand_in_takes_what_the_native_refusal_takes():
    """The whole point: one signature, two definitions, kept equal."""
    native = _args(_module_level(NATIVE, "_config_refusal"))
    stand_in = _args(_nested(BRIDGE, INSTALLER, "_config_refusal"))
    assert stand_in == native, (
        f"dispatch._config_refusal{tuple(native)} is rebound by {INSTALLER} to "
        f"a stand-in taking {tuple(stand_in)}. Every SGLang MiniMax-M3 launch "
        "would raise TypeError at model construction."
    )


def test_call_site_passes_what_the_signature_declares():
    """Guards the other half: the native signature and its caller.

    `MonoDecode.__init__` is the only caller. Pinning it here means a future
    change to `_config_refusal` has to move all three together -- signature,
    call site, stand-in -- rather than any two of them.
    """
    tree = ast.parse(NATIVE.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_config_refusal"
    ]
    assert calls, "dispatch.py no longer calls _config_refusal"
    declared = _args(_module_level(NATIVE, "_config_refusal"))
    for call in calls:
        assert not call.keywords, "call site started using keywords; update this test"
        assert len(call.args) == len(declared), (
            f"_config_refusal declares {len(declared)} parameter(s) {tuple(declared)} "
            f"but line {call.lineno} passes {len(call.args)}"
        )
