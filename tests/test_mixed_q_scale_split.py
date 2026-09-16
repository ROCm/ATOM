# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The mixed branch must split `q_scale` wherever it splits `q`.

`q_scale` is the fused qkv_a projection's per-token output scale, positionally
aligned with `q`. The mixed prefill+decode branch slices `q` into two halves and
projects each separately; passing the unsliced scale to either half makes it
dequantize against the other half's rows.

That failure is silent. The shapes still broadcast, the outputs stay finite,
nothing is logged, and neither a prefill-only nor a decode-only accuracy run
splits a batch at all -- so the only thing standing between this and a shipped
accuracy regression is a test that looks straight at it.

The static check below needs no aiter and therefore runs in CI, which is the
point: it guards the defect's exact shape (a bare `q_scale=` inside the mixed
branch) on the machines that never get to execute the behavioural tests.
"""

import importlib.util
import re
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "atom/model_ops/attention_mla.py"

requires_aiter = pytest.mark.skipif(
    importlib.util.find_spec("aiter") is None,
    reason="attention_mla imports aiter at module scope; CI has none",
)


def _mixed_branch_source() -> str:
    """The mixed prefill+decode implementation, wherever it currently lives.

    Located by AST rather than by slicing the `if context.is_mixed:` arm out of
    `forward_impl`: the implementation moved into its own method and a
    text-anchored test silently stops covering anything when that happens. It
    did not, here -- it went red, which is how this got updated -- but only
    because the marker it keyed off disappeared entirely.
    """
    import ast

    src = _SRC.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "_forward_mixed":
            return "\n".join(src.splitlines()[node.lineno - 1 : node.end_lineno])
    pytest.fail("attention_mla.py has no _forward_mixed")


def test_mixed_branch_never_passes_the_unsliced_scale():
    """No `x_scale=q_scale` in the mixed arm -- only the split halves."""
    body = _mixed_branch_source()
    bare = re.findall(r"x_scale=q_scale\b(?!_[pd])", body)
    assert not bare, (
        f"mixed branch passes the unsliced q_scale {len(bare)} time(s); each "
        "half must get its own slice (q_scale_p / q_scale_d)"
    )


def test_mixed_branch_slices_q_and_scale_the_same_number_of_times():
    """Every `q[...]` slice should have a scale slice to go with it."""
    body = _mixed_branch_source()
    assert "q_scale_p" in body and "q_scale_d" in body, (
        "mixed branch no longer computes the per-half scales; if the split "
        "moved, move this test with it"
    )
    assert "_split_per_token_q_scale(" in body


@requires_aiter
class TestSplitBehaviour:
    def test_plain_per_token_layout_splits_by_row(self):
        import torch

        from atom.model_ops.attention_mla import _split_per_token_q_scale

        n_tokens, n_prefill = 10, 6
        scale = torch.arange(n_tokens * 4, dtype=torch.float32).view(n_tokens, 4)
        p, d = _split_per_token_q_scale(scale, n_prefill, n_tokens)

        assert p.shape == (n_prefill, 4)
        assert d.shape == (n_tokens - n_prefill, 4)
        assert torch.equal(p, scale[:n_prefill])
        assert torch.equal(d, scale[n_prefill:])
        # The decode half must NOT be reading the prefill rows -- the exact
        # aliasing this whole module exists to prevent.
        assert not torch.equal(d, scale[: n_tokens - n_prefill])

    def test_batch_wide_scalar_passes_through_to_both(self):
        import torch

        from atom.model_ops.attention_mla import _split_per_token_q_scale

        one = torch.tensor(1.0)
        p, d = _split_per_token_q_scale(one, 6, 10)
        assert p is one and d is one

    def test_none_passes_through(self):
        from atom.model_ops.attention_mla import _split_per_token_q_scale

        assert _split_per_token_q_scale(None, 6, 10) == (None, None)

    def test_shuffled_mxfp4_layout_is_refused_not_sliced(self):
        """uint8 is the fp4x2 producer's swizzled layout: unsliceable."""
        import torch

        from atom.model_ops.attention_mla import _split_per_token_q_scale

        # 256-row padding makes this shape indistinguishable from a plain
        # per-token scale, which is why the check is on dtype.
        shuffled = torch.zeros((256, 8), dtype=torch.uint8)
        with pytest.raises(NotImplementedError, match="MXFP4"):
            _split_per_token_q_scale(shuffled, 100, 256)

    def test_row_count_mismatch_is_refused(self):
        import torch

        from atom.model_ops.attention_mla import _split_per_token_q_scale

        scale = torch.zeros((7, 4), dtype=torch.float32)
        with pytest.raises(ValueError, match="one row per token"):
            _split_per_token_q_scale(scale, 3, 10)
