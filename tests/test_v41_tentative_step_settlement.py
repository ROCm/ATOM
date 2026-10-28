# SPDX-License-Identifier: MIT
"""Who owns a tentative step's accepted prefix, on either staging path.

A tentative step runs the draft's whole block and leaves its cursor to one
side; only the accepted prefix advances it, and nobody knows that prefix until
the sampler has rejected what it rejects. So every tentative step has to be
settled: a real one is handed to the sampler, a synthetic one is dropped,
because nothing will ever sample a capture.

Leaving one unsettled is not a silent loss. The cache refuses the *next* step
while a prefix is outstanding ("Commit the accepted prefix before reusing or
checkpointing state"), so the failure lands a step later and in another
function -- which is why this is worth pinning rather than reading.

Both staging paths reach `_prepare` through the same `_v41_prepare_kwargs`, so
either can produce a tentative step. The settlement used to be inline in the
builder path only.

Executed by AST extraction, so the real function is the one under test with no
vLLM, aiter or GPU present.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

BRIDGE = Path(__file__).resolve().parents[1] / "atom/plugin/vllm/deepseek_v41_bridge.py"


def _settle():
    """The real `_v41_settle_tentative_step`, with its module global."""
    tree = ast.parse(BRIDGE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_v41_settle_tentative_step"
    )
    scope = {"_V41_PENDING_SPECULATIVE_STEP": None}
    exec(  # noqa: S102 - execute only AST nodes from this repository's source.
        compile(ast.Module(body=[function], type_ignores=[]), str(BRIDGE), "exec"),
        scope,
    )
    return scope


def _metadata(tentative):
    return SimpleNamespace(
        step=SimpleNamespace(tentative=tentative),
        cache=SimpleNamespace(pending="an outstanding accepted prefix"),
    )


def test_a_real_tentative_step_is_handed_to_the_sampler():
    scope = _settle()
    builder, metadata = object(), _metadata(True)

    scope["_v41_settle_tentative_step"](builder, metadata, False)

    assert scope["_V41_PENDING_SPECULATIVE_STEP"] == (builder, metadata)
    # Still outstanding on purpose: the sampler commits it.
    assert metadata.cache.pending == "an outstanding accepted prefix"


def test_a_synthetic_tentative_step_is_dropped():
    """Nothing samples a capture, so its prefix is not a quantity that exists."""
    scope = _settle()
    metadata = _metadata(True)

    scope["_v41_settle_tentative_step"](object(), metadata, True)

    assert metadata.cache.pending is None
    assert scope["_V41_PENDING_SPECULATIVE_STEP"] is None


@pytest.mark.parametrize("synthetic", [False, True], ids=["real", "synthetic"])
def test_a_step_that_is_not_tentative_is_left_alone(synthetic):
    scope = _settle()
    metadata = _metadata(False)

    scope["_v41_settle_tentative_step"](object(), metadata, synthetic)

    assert metadata.cache.pending == "an outstanding accepted prefix"
    assert scope["_V41_PENDING_SPECULATIVE_STEP"] is None


def test_a_step_without_the_attribute_is_not_tentative():
    """`_prepare` omits it entirely when speculation is off."""
    scope = _settle()
    metadata = SimpleNamespace(
        step=SimpleNamespace(), cache=SimpleNamespace(pending=None)
    )

    scope["_v41_settle_tentative_step"](object(), metadata, False)

    assert scope["_V41_PENDING_SPECULATIVE_STEP"] is None


def test_both_staging_paths_settle_their_step():
    """The divergence this exists to prevent, read off the source.

    The in-forward fallback produced tentative steps and settled none of them,
    so a capture through it left a prefix outstanding and the next step raised.
    """
    tree = ast.parse(BRIDGE.read_text())
    callers = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(inner, ast.Call)
            and getattr(inner.func, "id", "") == "_v41_settle_tentative_step"
            for inner in ast.walk(node)
        )
    }
    assert {"_v41_stage_outside_forward", "v41_stage_step"} <= callers, (
        "both staging paths must settle a tentative step; "
        f"found callers: {sorted(callers)}"
    )
