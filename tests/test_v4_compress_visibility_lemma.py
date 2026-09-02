# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The arithmetic behind dropping `min(per-token, ctx // ratio)`, exhausted.

The V4 index builders used to bound a token's compress-group count twice: by
the token's own position and again by its sequence's context length. The second
bound is dropped on the grounds that it can never be the binding one.

Only those grounds are settled here. Whether every site that applied it was
actually visited is a property of the change, not of anything this file checks
-- it belongs to review and to the commit message, and a test that went looking
for the leftovers would be reading the source it is supposed to be testing.

The grounds are pure arithmetic over two integers, so this file settles them by
enumeration rather than by sampling, and it does so on the CPU with no aiter,
no Triton and no GPU -- which is what lets it run in CI, where every
kernel-vs-reference test in this repo skips itself.

What enumeration CANNOT settle is the premise `pos < ctx`. That is a property
of the code that BUILDS positions, not of the builders that consume them, and
those live in the scheduler, in three metadata paths, and in two out-of-tree
bridges. `test_decode_indptr_qlen_invariance.py` pins the consumer side; the
premise itself is enforced by `require_step_within_full_q`, whose own tests are
at the bottom of this file -- next to the lemma it exists to keep true.
"""

import numpy as np
import pytest

from atom.model_ops.attentions.v4_pool_geometry import (
    CSA_RATIO,
    HCA_RATIO,
    require_step_within_full_q,
)

RATIOS = [CSA_RATIO, HCA_RATIO]
# Past two HCA groups and 1024 CSA groups: every residue of both ratios appears
# many times over, and so does every carry across a group boundary.
LIMIT = 4096


def _grid():
    """Every `(ctx, pos)` pair with `1 <= ctx <= LIMIT` and `0 <= pos < LIMIT`."""
    return np.meshgrid(
        np.arange(1, LIMIT + 1, dtype=np.int64),
        np.arange(LIMIT, dtype=np.int64),
        indexing="ij",
    )


@pytest.mark.parametrize("ratio", RATIOS)
def test_the_dropped_bound_never_binds_for_a_legal_position(ratio):
    """Exhaustive over every `(ctx, pos)` with `0 <= pos < ctx <= LIMIT`."""
    ctx, pos = _grid()
    legal = pos < ctx
    per_token = (pos + 1) // ratio
    per_sequence = ctx // ratio
    # Where a position is legal, the sequence's bound is never the smaller, so
    # taking `min` of the two returns the per-token one unchanged.
    assert np.array_equal(
        np.minimum(per_token, per_sequence)[legal], per_token[legal]
    ), f"ratio {ratio}: some legal (ctx, pos) has ctx//r < (pos+1)//r"


@pytest.mark.parametrize("ratio", RATIOS)
def test_the_bound_does_bind_once_a_position_runs_past_its_sequence(ratio):
    """Teeth check. If the two expressions agreed everywhere, the test above
    would hold for reasons having nothing to do with the premise, and deleting
    the premise's guard would look safe when it is not."""
    ctx, pos = _grid()
    illegal = pos >= ctx
    differs = (pos + 1) // ratio > ctx // ratio
    assert np.any(differs & illegal), (
        f"ratio {ratio}: no illegal position separates the two rules, so the "
        f"exhaustive test above proves nothing about the premise"
    )
    # And every disagreement is on the illegal side -- none on the legal one.
    assert not np.any(differs & ~illegal)


@pytest.mark.parametrize("topk", [16, 64, 512])
def test_the_index_topk_cap_is_not_redundant_and_must_stay(topk):
    """CSA carries a third bound, `index_topk`. Unlike the sequence's count it
    is a property of the OUTPUT buffer, so it does bind and had to be kept.
    Asserted separately so a later cleanup does not sweep it up with the other.

    CSA only: `index_topk` never reaches the HCA section, whose slice length is
    the committed group count itself.
    """
    visible = (np.arange(LIMIT, dtype=np.int64) + 1) // CSA_RATIO
    assert np.any(visible > topk), (
        f"index_topk={topk} never binds within {LIMIT} positions, so this file "
        f"says nothing about it"
    )
    assert np.any(visible <= topk), "and it must not bind everywhere either"


# --- the premise's guard ----------------------------------------------------
# Everything above is conditional on `pos < ctx`. `require_step_within_full_q`
# is where that becomes true rather than assumed, so what these check is that
# its accept/reject line is the SAME line as the premise's -- not that some
# inequality raises.

FULL_Q = 8


def _last_position(ctx, length):
    """Where a step's last token lands. `< ctx` is the whole premise."""
    return ctx - FULL_Q + length - 1


@pytest.mark.parametrize("length", range(1, 2 * FULL_Q + 1))
def test_the_guard_accepts_exactly_the_steps_that_keep_a_position_legal(length):
    """The guard's line and the premise's line are one line, checked at every
    context long enough to hold a full step."""
    legal = all(_last_position(ctx, length) < ctx for ctx in range(FULL_Q, 4 * FULL_Q))
    assert legal == (length <= FULL_Q), "the position arithmetic moved"
    if legal:
        require_step_within_full_q(length, FULL_Q, "a step")
    else:
        with pytest.raises(ValueError, match="past its own context"):
            require_step_within_full_q(length, FULL_Q, "a step")


def test_the_guard_survives_python_dash_O():
    """`assert` would not: the rectangle failure is a device-side out-of-bounds
    read, so the check has to outlive the optimizer that strips asserts."""
    with pytest.raises(ValueError) as excinfo:
        require_step_within_full_q(FULL_Q + 1, FULL_Q, "a named source")
    assert not isinstance(excinfo.value, AssertionError)
    # The source is in the message: three producers call this, and "some
    # sequence forwarded too much" is not actionable without knowing which.
    assert "a named source" in str(excinfo.value)


def test_an_empty_batch_is_not_a_violation():
    """Callers pass 0 for an empty batch rather than branching around the call;
    `max()` of nothing is what they would otherwise have to special-case."""
    require_step_within_full_q(0, FULL_Q, "an empty step")
