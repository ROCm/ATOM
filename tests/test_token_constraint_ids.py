# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from atom.model_ops.token_constraints import TokenConstraint, apply_bitmask_reference


@pytest.mark.parametrize(
    "constraint,allowed",
    [
        (TokenConstraint.allow([0, 31, 32, 64]), {0, 31, 32, 64}),
        (TokenConstraint.deny([0, 31, 32, 64]), set(range(65)) - {0, 31, 32, 64}),
        (TokenConstraint.force(31), {31}),
        (TokenConstraint.deny([]), set(range(65))),
        (TokenConstraint.allow([31, 31]), {31}),
    ],
)
def test_id_rules_preserve_allowed_logits(constraint, allowed):
    logits = torch.arange(130, dtype=torch.float32).view(2, 65)
    before = logits.clone()
    apply_bitmask_reference(logits, constraint.to_bitmask(65), [1])
    torch.testing.assert_close(logits[0], before[0])
    assert set(torch.isfinite(logits[1]).nonzero().flatten().tolist()) == allowed
    for token in allowed:
        assert logits[1, token] == before[1, token]


@pytest.mark.parametrize(
    "constraint",
    [
        TokenConstraint.allow([]),
        TokenConstraint.force(-1),
        TokenConstraint.force(4),
        TokenConstraint.force(True),
        TokenConstraint.deny(range(4)),
    ],
)
def test_invalid_or_empty_token_sets_fail_before_sampling(constraint):
    with pytest.raises(ValueError):
        constraint.to_bitmask(4)
