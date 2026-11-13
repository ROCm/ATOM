# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The in-place LSE mask in `_forward_prefill_mla`.

`_forward_prefill_mla` zeroes the attention output rows whose per-head LSE is
not finite before handing them to the cross-rank combine. It used to spell that

    o = torch.where(torch.isfinite(lse).unsqueeze(-1), o, torch.zeros_like(o))

which torch expands into six launches: four to build the mask, one to fill an
o-sized buffer with zeros, and one to select between them. It is now a single
Triton kernel, `zero_nonfinite_rows`, that writes into `o` in place and skips
the rows whose LSE is finite. These tests pin the two properties that swap
depends on: the result is bit-identical to the old expression, and the write
really does land in the caller's buffer rather than a copy.
"""

import pytest
import torch

# Order matters: importing triton without an active GPU driver raises rather
# than failing the import, and importorskip only catches the latter. Same guard
# order as tests/test_dcp_a2a_fused_quant.py.
if not torch.cuda.is_available():
    pytest.skip(
        "compares a Triton kernel against its reference; needs a real GPU",
        allow_module_level=True,
    )

pytest.importorskip("triton")


def _old(o, lse):
    """The expression this replaced."""
    return torch.where(torch.isfinite(lse).unsqueeze(-1), o, torch.zeros_like(o))


def _new(o, lse):
    """What `_forward_prefill_mla` does now: the fused Triton kernel."""
    from atom.model_ops.dcp_ops import zero_nonfinite_rows

    return zero_nonfinite_rows(o, lse)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
# 513 and 129 are the point of this list, not padding: BLOCK rounds the head dim
# up to a power of two, so only a non-power-of-two size exercises the masked tail
# of the last block. Every shape here used to be 512, which never touched it.
@pytest.mark.parametrize(
    "shape",
    [
        (1, 16, 512),
        (7, 64, 512),
        (128, 1, 512),
        (5, 3, 513),
        (2, 7, 129),
        (3, 2, 1025),
        (1, 33, 129),
        # Around a multiple of the kernel's 32-row group.
        (8191, 1, 7),
        (8192, 1, 7),
        (8193, 1, 7),
    ],
)
def test_matches_the_where_it_replaced(dtype, shape):
    tokens, heads, _dim = shape
    torch.manual_seed(0)
    o = torch.randn(shape, dtype=dtype, device="cuda")
    lse = torch.randn(tokens, heads, dtype=torch.float32, device="cuda")
    # Scatter the three non-finite kinds through the LSE, plus an all-finite and
    # an all-masked row so neither extreme is only covered by luck.
    flat = lse.view(-1)
    if flat.numel() >= 4:
        flat[0] = float("inf")
        flat[1] = float("-inf")
        flat[2] = float("nan")
    lse[0, :] = float("nan")

    expected = _old(o.clone(), lse.clone())
    got = _new(o.clone(), lse.clone())
    assert torch.equal(got, expected)


def test_keeps_the_dtype_and_writes_in_place():
    """The kernel writes o itself; a widened or reallocated result would
    silently double what reaches the cross-rank combine."""
    for dtype in (torch.bfloat16, torch.float16):
        o = torch.randn(4, 8, 512, dtype=dtype, device="cuda")
        lse = torch.zeros(4, 8, device="cuda")
        lse[1] = float("inf")
        ptr = o.data_ptr()
        out = _new(o, lse)
        assert out.dtype == dtype
        assert out.data_ptr() == ptr and out is o


def test_masks_the_rows_it_should():
    o = torch.randn(4, 8, 512, device="cuda")
    lse = torch.zeros(4, 8, device="cuda")
    lse[1] = float("inf")
    out = _new(o, lse)
    assert torch.count_nonzero(out[1]) == 0
    assert torch.count_nonzero(out[0]) > 0


def test_does_not_disturb_the_lse_the_caller_still_returns():
    """`logical_not_` is in place on isfinite's output, not on the LSE."""
    lse = torch.tensor([[1.0, float("inf")], [float("nan"), 2.0]], device="cuda")
    before = lse.clone()
    _new(torch.randn(2, 2, 512, device="cuda"), lse)
    assert torch.equal(torch.isnan(lse), torch.isnan(before))
    assert torch.equal(lse[~torch.isnan(lse)], before[~torch.isnan(before)])


def test_all_finite_leaves_every_row_untouched():
    o = torch.randn(3, 4, 512, device="cuda")
    expected = o.clone()
    assert torch.equal(_new(o, torch.zeros(3, 4, device="cuda")), expected)


def test_all_non_finite_zeroes_everything():
    o = torch.randn(3, 4, 512, device="cuda")
    assert (
        torch.count_nonzero(_new(o, torch.full((3, 4), float("nan"), device="cuda")))
        == 0
    )


def test_mask_is_per_head_not_per_token():
    """A non-finite head must not take its token's other heads down with it."""
    o = torch.ones(2, 3, 512, device="cuda")
    lse = torch.zeros(2, 3, device="cuda")
    lse[0, 1] = float("inf")
    out = _new(o, lse)
    assert torch.count_nonzero(out[0, 0]) == 512
    assert torch.count_nonzero(out[0, 1]) == 0
    assert torch.count_nonzero(out[0, 2]) == 512
    assert torch.count_nonzero(out[1]) == 3 * 512


def test_grouped_mask_preserves_strided_storage_and_finite_nan_payloads():
    storage = torch.full((5, 68, 137), -13, device="cuda", dtype=torch.bfloat16)
    output = storage[1:4, 1:67:2, 3:132]
    output.fill_(float("nan"))
    output[0, 0].fill_(-0.0)
    lse_storage = torch.zeros(68, 8, device="cuda")
    lse = lse_storage[1:67:2, 1:7:2].transpose(0, 1)
    # Both sides of a grouped row boundary and the final masked tile.
    lse[0, 31] = float("-inf")
    lse[0, 32] = float("nan")
    lse[-1, -1] = float("inf")
    expected = storage.clone()
    expected_output = expected[1:4, 1:67:2, 3:132]
    expected_output[~torch.isfinite(lse)] = 0
    original_lse_bits = lse_storage.view(torch.int32).clone()

    assert _new(output, lse) is output
    # Compare the entire backing allocation: guards, NaN payloads and signed
    # zeros must survive whenever their corresponding LSE is finite.
    assert torch.equal(storage.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(lse_storage.view(torch.int32), original_lse_bits)
