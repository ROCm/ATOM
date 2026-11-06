# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_MTP_DRAFT_LM_HEAD_FP8: `DraftLMHeadFP8.mm` against the bf16 LM head.

The FP8 copy's logits stay within the 128 x 128 block FP8 error of the bf16
ones, and a row whose top logit leads by a clear margin keeps its argmax."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops.draft_lm_head_fp8 import DraftLMHeadFP8


@pytest.mark.parametrize("rows", [1, 3, 8, 16])
def test_fp8_draft_logits_track_the_bf16_head(rows):
    torch.manual_seed(rows)
    n, k = 8192, 2048
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    x = torch.randn(rows, k, device="cuda").to(torch.bfloat16)
    # Every row has one clearly leading token: x points along that weight row.
    lead = torch.randint(0, n, (rows,), device="cuda")
    x = (x + 40.0 * w[lead].float()).to(torch.bfloat16)
    head = DraftLMHeadFP8(w)
    ref = x.float() @ w.float().t()
    got = head.mm(x).float()
    assert got.shape == (rows, n) and torch.isfinite(got).all()
    err = (got - ref).abs().max().item()
    assert err <= 0.08 * ref.abs().max().item(), err
    assert torch.equal(got.argmax(-1), ref.argmax(-1))
    assert torch.equal(ref.argmax(-1), lead)
