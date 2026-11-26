# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Compatibility tests for the package-level quantization API."""

import ast
from pathlib import Path

ESTABLISHED_PUBLIC_API = {
    "assert_mxfp4_arch_supported",
    "disable",
    "enable",
    "get_attention_backend",
    "get_quant_backend",
    "is_under_bf16_prefix",
    "register_fp8_weight_optimizer_hooks",
    "register_mxfp4_weight_optimizer_hooks",
    "store_weights_fp8",
    "store_weights_fp8_blockwise2d",
}


def _declared_exports() -> set[str]:
    source = (
        Path(__file__).resolve().parents[3]
        / "atomrl"
        / "lumen"
        / "quantize"
        / "__init__.py"
    )
    module = ast.parse(source.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError("atomrl.lumen.quantize must declare __all__")


def test_declared_exports_preserve_established_quantize_api() -> None:
    assert ESTABLISHED_PUBLIC_API <= _declared_exports()


def test_star_import_preserves_established_quantize_api() -> None:
    from atomrl.lumen import quantize

    namespace: dict[str, object] = {}
    # Test executes Python's star-import statement to verify exports.
    exec(  # noqa: S102 -- test executes generated source
        "from atomrl.lumen.quantize import *", namespace
    )

    assert ESTABLISHED_PUBLIC_API <= namespace.keys()
    assert ESTABLISHED_PUBLIC_API <= set(quantize.__all__)
