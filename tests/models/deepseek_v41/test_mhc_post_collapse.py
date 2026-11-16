# SPDX-License-Identifier: MIT
"""An owed mHC post fused into the collapse / the DSpark tap mean.

`post_collapse_streams` must give the bits `expand_residual` (aiter's
`mhc_post`) followed by `collapse_streams` or the BF16 stream mean give:
equality is the contract, not a tolerance.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)


def _mhc():
    pytest.importorskip("aiter", reason="Single-Pass mHC calls AITER kernels")
    from atom.model_ops.deepseek_v41 import mhc

    return mhc


def _data(tokens, hidden, seed, skew):
    g = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(tokens, 4, hidden, device="cuda", generator=g) * 1.5
    if skew:
        magnitude = torch.exp(
            torch.rand(tokens, 1, 1, device="cuda", generator=g) * 9.2 - 4.6
        )
        residual = residual * magnitude
        residual[:, :, ::97] *= 20
    residual = residual.to(torch.bfloat16)
    pending = (torch.randn(tokens, hidden, device="cuda", generator=g) * 3).to(
        torch.bfloat16
    )
    post = torch.rand(tokens, 4, 1, device="cuda", generator=g) * 2
    comb = torch.softmax(torch.randn(tokens, 4, 4, device="cuda", generator=g) * 2, -1)
    pre = torch.sigmoid(torch.randn(tokens, 4, device="cuda", generator=g)) + 1e-6
    return residual, pending, post, comb, pre


@pytest.mark.parametrize("tokens", [1, 7, 192, 1344, 1536])
@pytest.mark.parametrize("hidden", [2048, 5120, 7168])
@pytest.mark.parametrize("skew", [False, True])
def test_post_collapse_is_bit_exact(tokens, hidden, skew):
    mhc = _mhc()
    residual, pending, post, comb, pre = _data(tokens, hidden, tokens + hidden, skew)
    settled = mhc.expand_residual(pending, residual, post, comb)
    assert mhc.post_collapse_supported(residual, pending)

    mean = mhc.post_collapse_streams(pending, residual, post, comb)
    assert torch.equal(mean, settled.mean(dim=-2))

    collapsed = mhc.post_collapse_streams(pending, residual, post, comb, pre)
    assert torch.equal(collapsed, mhc.collapse_streams(settled, pre))

    state = mhc.SinglePassHCState(residual, pre, pending, post, comb)
    assert torch.equal(state.collapse(), mhc.collapse_streams(settled, pre))


def test_post_collapse_keeps_the_leading_shape():
    mhc = _mhc()
    residual, pending, post, comb, pre = _data(10, 5120, 3, False)
    state = mhc.SinglePassHCState(
        residual.view(2, 5, 4, 5120),
        pre.view(2, 5, 4),
        pending.view(2, 5, 5120),
        post.view(2, 5, 4, 1),
        comb.view(2, 5, 4, 4),
    )
    expected = mhc.collapse_streams(
        mhc.expand_residual(pending, residual, post, comb), pre
    ).view(2, 5, 5120)
    assert torch.equal(state.collapse(), expected)


def test_post_collapse_empty_batch():
    mhc = _mhc()
    residual, pending, post, comb, _ = _data(0, 256, 0, False)
    assert mhc.post_collapse_streams(pending, residual, post, comb).shape == (0, 256)


def test_unsupported_shapes_keep_the_settle_path():
    mhc = _mhc()
    residual, pending, _, _, _ = _data(4, 256, 1, False)
    assert not mhc.post_collapse_supported(residual[:, :2], pending)
    assert not mhc.post_collapse_supported(residual[..., :128], pending[:, :128])
    assert not mhc.post_collapse_supported(
        residual.transpose(0, 1).contiguous().transpose(0, 1), pending
    )
    assert not mhc.post_collapse_supported(residual.cpu(), pending.cpu())
