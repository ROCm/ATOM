# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`any_greedy=False` skips the greedy fallback without changing a sampled id."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("rows", [1, 5, 12])
@pytest.mark.parametrize("independent", [False, True])
def test_no_greedy_row_matches_fallback(rows, independent):
    from atom.model_ops.sampler import Sampler

    torch.manual_seed(0)
    logits = torch.randn(rows, 129280, device="cuda").to(torch.bfloat16)
    temps = torch.full((rows,), 1.0, device="cuda")
    s = Sampler()
    torch.manual_seed(1)
    ref = s(logits, temps, needs_independent_noise=independent)
    torch.manual_seed(1)
    out = s(logits, temps, needs_independent_noise=independent, any_greedy=False)
    assert torch.equal(ref, out)


def test_greedy_row_still_takes_argmax():
    from atom.model_ops.sampler import SAMPLER_EPS, Sampler

    torch.manual_seed(0)
    logits = torch.randn(3, 129280, device="cuda").to(torch.bfloat16)
    temps = torch.tensor([1.0, SAMPLER_EPS, 1.0], device="cuda")
    out = Sampler()(logits, temps, any_greedy=True)
    assert int(out[1]) == int(logits[1].float().argmax())
