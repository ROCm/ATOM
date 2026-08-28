# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""`prepare_mtp_decode` overrides must match what the drafter will call them with.

Importing any backend pulls in aiter and a GPU, so this reads the source AST
instead and stays in the non-GPU pre-checkin suite -- the same trick
`test_model_runner_staging_fence.py` uses.

The pairing this pins has been wrong twice: `GDNAttentionMetadataBuilder`
inherited `fuse_mtp_decode_position_update = True` while declaring an override
that takes none of the fused arguments, and `TritonMLAMetadataBuilder` grew an
override that stopped at `num_reject_tokens`. Both are a `TypeError` on the
first mid-step of any speculative decode, on a backend no CI job runs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

BACKENDS = Path(__file__).resolve().parents[1] / "atom/model_ops/attentions"

# What `EagleProposer.propose` adds when it takes the fused branch.
FUSED_KWARGS = {"update_context_lens", "positions_out"}


def _classes(path: Path) -> dict[str, ast.ClassDef]:
    module = ast.parse(path.read_text())
    return {n.name: n for n in ast.walk(module) if isinstance(n, ast.ClassDef)}


def _all_classes() -> dict[str, tuple[Path, ast.ClassDef]]:
    found: dict[str, tuple[Path, ast.ClassDef]] = {}
    for path in sorted(BACKENDS.glob("*.py")):
        for name, node in _classes(path).items():
            found[name] = (path, node)
    return found


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _fuses(cls: ast.ClassDef, classes: dict) -> bool:
    """The flag as this class sees it, following `Builder(...)` bases by name."""
    for node in cls.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "fuse_mtp_decode_position_update"
            for t in node.targets
        ):
            return bool(getattr(node.value, "value", False))
    for base in cls.bases:
        name = base.id if isinstance(base, ast.Name) else None
        if name in classes and _fuses(classes[name][1], classes):
            return True
    return False


def _accepts_fused(fn: ast.FunctionDef) -> bool:
    if fn.args.kwarg is not None:  # **kwargs swallows whatever the base takes
        return True
    named = {a.arg for a in fn.args.kwonlyargs} | {a.arg for a in fn.args.args}
    return FUSED_KWARGS <= named


_OVERRIDES = sorted(
    name
    for name, (_, cls) in _all_classes().items()
    if _method(cls, "prepare_mtp_decode") is not None
)


@pytest.mark.parametrize("class_name", _OVERRIDES)
def test_a_fusing_backend_accepts_the_arguments_fusing_means(class_name):
    """Declaring the flag is a promise to take `positions_out` and update it.

    A backend that cannot absorb the position bump must say so on the class, not
    accept the keywords and drop them -- that trades a `TypeError` for a draft
    reading last step's positions, which nothing would report.
    """
    classes = _all_classes()
    path, cls = classes[class_name]
    fn = _method(cls, "prepare_mtp_decode")
    assert _fuses(cls, classes) == _accepts_fused(fn), (
        f"{path.name}:{cls.name}.prepare_mtp_decode "
        f"{'takes' if _accepts_fused(fn) else 'does not take'} the fused "
        f"arguments while the class resolves "
        f"fuse_mtp_decode_position_update={_fuses(cls, classes)}"
    )


def test_the_scan_found_the_backends_it_is_meant_to_cover():
    """A rename that emptied the sweep would leave every case above vacuous."""
    assert len(_OVERRIDES) >= 4, _OVERRIDES
