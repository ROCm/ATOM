# SPDX-License-Identifier: MIT
"""Layer/native width limits and DP unification use the same admission rule."""

from types import SimpleNamespace

import pytest
import torch

from atom.models.deepseek_v41.mono import decode_step


@pytest.mark.parametrize(
    "rows,max_rows,dp_ranks,unified,accepted",
    [
        (48, 48, 0, True, True),
        (49, 48, 0, True, False),
        (49, 64, 0, True, True),
        (64, 64, 0, True, True),
        (65, 64, 0, True, False),
        (24, 64, 4, True, True),
        (25, 64, 4, True, False),
        (8, 64, 2, False, False),
    ],
)
def test_shared_decode_admission(
    monkeypatch, rows, max_rows, dp_ranks, unified, accepted
):
    context = SimpleNamespace(is_prefill=False, running_tokens_are_unified=unified)
    step = SimpleNamespace(decode=True, running_tokens=rows)
    forward = SimpleNamespace(
        context=context, ubatch_slices=None, attn_metadata=SimpleNamespace(step=step)
    )
    monkeypatch.setattr(decode_step, "get_forward_context", lambda: forward)
    ids = torch.arange(rows)
    assert (
        decode_step.step_supported(ids, None, dp_ranks, max_rows=max_rows) is accepted
    )
