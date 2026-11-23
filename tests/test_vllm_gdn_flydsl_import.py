# SPDX-License-Identifier: MIT
"""The vLLM GDN opt-in must neither probe when off nor hide broken imports."""

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "atom/plugin/vllm/attention/layer_gdn.py"
DEPENDENCY = "aiter.ops.flydsl.linear_attention_kernels"


def _selection(import_dependency, enabled):
    # Execute the actual module-level selection without importing vLLM or the
    # GPU stack. Keep the whole block through the first class, including any
    # exception handler, so the previous silent fallback fails this test.
    tree = ast.parse(SOURCE.read_text())
    start = next(
        i
        for i, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "USE_FLYDSL_GDR" for t in node.targets
        )
    )
    end = next(
        i
        for i in range(start, len(tree.body))
        if isinstance(tree.body[i], ast.ClassDef)
    )
    namespace = {
        "envs": SimpleNamespace(ATOM_USE_FLYDSL_GDR=enabled),
        "__builtins__": {**vars(builtins), "__import__": import_dependency},
    }
    code = compile(ast.Module(tree.body[start:end], []), str(SOURCE), "exec")
    exec(code, namespace)  # noqa: S102 - trusted repository source
    return namespace


@pytest.mark.parametrize("available", [False, True])
def test_disabled_does_not_import_flydsl(available):
    imports = []

    def dependency(name, *args, **kwargs):
        imports.append(name)
        if not available:
            raise ImportError("FlyDSL is not installed")
        return SimpleNamespace(flydsl_gdr_decode=object())

    namespace = _selection(dependency, False)
    assert namespace["USE_FLYDSL_GDR"] is False
    assert not imports
    assert "flydsl_gdr_decode" not in namespace


def test_enabled_propagates_dependency_import_error():
    failure = ImportError("broken FlyDSL dependency")

    def dependency(name, *args, **kwargs):
        assert name == DEPENDENCY
        raise failure

    with pytest.raises(ImportError) as caught:
        _selection(dependency, True)
    assert caught.value is failure


def test_enabled_binds_the_requested_kernel():
    kernel = object()
    imports = []

    def dependency(name, *args, **kwargs):
        imports.append(name)
        return SimpleNamespace(flydsl_gdr_decode=kernel)

    namespace = _selection(dependency, True)
    assert imports == [DEPENDENCY]
    assert namespace["USE_FLYDSL_GDR"] is True
    assert namespace["flydsl_gdr_decode"] is kernel
