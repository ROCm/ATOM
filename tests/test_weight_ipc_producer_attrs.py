# SPDX-License-Identifier: MIT
"""Post-load state that only the producer computes, and the consumer needs.

The rapidserve decode process builds its model on the meta device and imports
every parameter from prefill, so it never runs `process_weights_after_loading`.
Anything that hook decides therefore has to travel, and two kinds of decision
travel by different means:

* tensors it stashes as plain module attributes -- carried by the generic
  `__attr__` sweep in `export_model_weight_handles`, which looks at every CUDA
  tensor in `mod.__dict__` regardless of name;
* flags it sets -- carried by `_module_meta_attrs`, which filters to public
  `str`/`bool` names and so drops anything private or otherwise typed.

The second filter is the trap. A missed flag leaves the consumer on a
construction-time default while holding the producer's post-load tensors, and
the two disagree about what those tensors ARE. V4's `_wo_a_mxscale` is the
worked example: miss it and decode keeps wo_a in FP8, takes the BF16 einsum
branch, and dies with `expected scalar type BFloat16 but found Float8_e4m3fn`
-- a dtype error about a weight that is exactly right.

`_PRODUCER_ONLY_ATTRS` is the escape hatch the module documents for both cases.
This file checks nothing has been added to a post-load hook without going
through it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
IPC = ROOT / "atom/model_engine/ipc_utils.py"
# The native engine's models. The vLLM plugin path does not use this IPC
# handoff -- its weights are vLLM's -- so its hooks are not this file's
# business.
MODELS = ROOT / "atom/models"


def _producer_only_attrs() -> set[str]:
    """`_PRODUCER_ONLY_ATTRS`, read statically (the module imports torch)."""
    tree = ast.parse(IPC.read_text(), filename=str(IPC))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "_PRODUCER_ONLY_ATTRS"
            for t in node.targets
        ):
            continue
        return {
            e.value
            for e in node.value.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        }
    raise AssertionError("_PRODUCER_ONLY_ATTRS not found in ipc_utils.py")


def _post_load_self_attrs() -> dict[str, str]:
    """`self.<attr> = ...` assignments inside post-load hooks, attr -> where."""
    found: dict[str, str] = {}
    for src in sorted(MODELS.rglob("*.py")):
        tree = ast.parse(src.read_text(), filename=str(src))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            if fn.name != "process_weights_after_loading":
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        found.setdefault(
                            target.attr, f"{src.name}:{node.lineno}"
                        )
    return found


def _carried_by_the_public_sweep(attr: str) -> bool:
    """Whether `_module_meta_attrs`' filter would pick the name up."""
    return not attr.startswith("_") and attr != "training"


# ── Every private post-load flag is declared ─────────────────────────────


def test_no_post_load_attribute_is_silently_dropped():
    """The sweep the fix came from. A private name set by a post-load hook is
    invisible to `_module_meta_attrs`, so it must either be a tensor (carried
    by the `__attr__` sweep, which ignores names) or be declared here.

    A new one is not necessarily a bug -- it is a decision. Add it to
    `_PRODUCER_ONLY_ATTRS` if the consumer reads it, and to `_TENSOR_VALUED`
    below if it holds a tensor.
    """
    declared = _producer_only_attrs()
    undeclared = {
        attr: where
        for attr, where in _post_load_self_attrs().items()
        if not _carried_by_the_public_sweep(attr)
        and attr not in declared
        and attr not in _TENSOR_VALUED
    }
    assert not undeclared, (
        "post-load hooks set these private attributes, which "
        "_module_meta_attrs drops and nothing else carries: "
        f"{undeclared}. Declare each in _PRODUCER_ONLY_ATTRS (a flag the "
        "consumer reads) or in this test's _TENSOR_VALUED (a tensor the "
        "__attr__ sweep already carries)."
    )


#: Private post-load attributes that hold CUDA tensors. The `__attr__` sweep in
#: `export_model_weight_handles` carries these without help -- it walks
#: `mod.__dict__` for tensors and does not look at names -- so they do not
#: belong in `_PRODUCER_ONLY_ATTRS`, whose entries ride the non-tensor sidecar.
_TENSOR_VALUED = {
    "_wo_a_w_fp8",  # deepseek_v4.py:2512
    "_wo_a_w_scale",  # deepseek_v4.py:2513
}


# ── The V4 case specifically ─────────────────────────────────────────────


@pytest.mark.parametrize("attr", ["_wo_a_mxscale", "_wo_a_fp8_dtype"])
def test_the_v4_wo_a_selector_travels(attr):
    """Without these the consumer's `_wo_a_mxscale` stays at its
    construction-time False (deepseek_v4.py:2343) while wo_a aliases the
    producer's FP8 weight, and the grouped LoRA takes the BF16 branch."""
    assert attr in _producer_only_attrs()


def test_the_selector_is_not_reachable_through_the_public_sweep():
    """i.e. the declaration above is load-bearing, not belt-and-braces."""
    assert not _carried_by_the_public_sweep("_wo_a_mxscale")


def test_the_dtype_beside_it_is_not_a_str_or_bool():
    """`_wo_a_fp8_dtype` is a `torch.dtype`, which `_module_meta_attrs`' type
    filter would drop even if the name were public -- the second, independent
    reason it has to be declared."""
    import torch

    assert not isinstance(torch.float8_e4m3fn, (str, bool))
