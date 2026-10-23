# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Triton routing kernel must reproduce AITER's `topk_softmax` exactly.

`rocm_aiter_topk_softmax_impl` swaps one for the other inside captured graphs
and keeps AITER for eager calls, so the same request routes through both
depending on the batch it lands in. Any difference -- an expert id, the order
of tied experts, or the last bit of a weight -- would make outputs depend on
batching, hence bit-exactness rather than a tolerance.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("compares two GPU kernels; needs a real GPU", allow_module_level=True)

aiter = pytest.importorskip("aiter")

from atom.model_ops.triton_topk_softmax import (
    triton_topk_softmax,
    triton_topk_softmax_supported,
)

EXPERTS, TOPK = 512, 10


def _logits(rows, width, scale, ties):
    x = torch.randn(rows, width, device="cuda") * scale
    if ties:
        # Coarse values: many equal logits, including +0.0 and -0.0.
        x = (x * 4).round() / 4
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("shared", [0, 1])
@pytest.mark.parametrize("rows", [1, 4, 33, 1024])
@pytest.mark.parametrize("scale,ties", [(2.0, False), (0.05, True), (8.0, False)])
def test_matches_aiter_bit_for_bit(shared, rows, scale, ties):
    torch.manual_seed(rows * 7 + shared)
    logits = _logits(rows, EXPERTS + shared, scale, ties)
    assert triton_topk_softmax_supported(logits, TOPK, True, shared)

    # The fused-shared layout: one [rows, k + shared] buffer per output, the
    # kernels write the routed columns through a [:, :k] view.
    w_ref = torch.full((rows, TOPK + shared), -1.0, device="cuda")
    ids_ref = torch.full(
        (rows, TOPK + shared), EXPERTS, dtype=torch.int32, device="cuda"
    )
    w_new, ids_new = w_ref.clone(), ids_ref.clone()
    scratch = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")

    aiter.topk_softmax(
        w_ref, ids_ref[:, :TOPK], scratch, logits, True, shared,
        "sigmoid" if shared else "",
    )  # fmt: skip
    triton_topk_softmax(w_new, ids_new[:, :TOPK], logits, shared)

    assert torch.equal(ids_new, ids_ref)
    assert torch.equal(w_new, w_ref)


def test_unsupported_shapes_fall_back():
    logits = torch.zeros(4, 513, dtype=torch.bfloat16, device="cuda")
    assert not triton_topk_softmax_supported(logits, TOPK, False, 1)
    assert not triton_topk_softmax_supported(logits, TOPK, True, 0)  # 513 experts
    assert not triton_topk_softmax_supported(logits[:, ::2], TOPK, True, 0)
