# SPDX-License-Identifier: MIT
"""An owed mHC post fused into the collapse / the DSpark tap mean.

`post_reduce_streams` (aiter's `mhc_post_reduce`) must give the bits
`expand_residual` (aiter's `mhc_post`) followed by `collapse_streams` or the
BF16 stream mean give: equality is the contract, not a tolerance.
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


def _data(tokens, hidden, seed, skew, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(tokens, 4, hidden, device="cuda", generator=g) * 1.5
    if skew:
        magnitude = torch.exp(
            torch.rand(tokens, 1, 1, device="cuda", generator=g) * 9.2 - 4.6
        )
        residual = residual * magnitude
        residual[:, :, ::97] *= 20
    residual = residual.to(dtype)
    pending = (torch.randn(tokens, hidden, device="cuda", generator=g) * 3).to(dtype)
    post = torch.rand(tokens, 4, 1, device="cuda", generator=g) * 2
    comb = torch.softmax(torch.randn(tokens, 4, 4, device="cuda", generator=g) * 2, -1)
    pre = torch.sigmoid(torch.randn(tokens, 4, device="cuda", generator=g)) + 1e-6
    return residual, pending, post, comb, pre


@pytest.mark.parametrize("tokens", [1, 7, 192, 1344, 1536])
@pytest.mark.parametrize("hidden", [2048, 5120, 7168])
@pytest.mark.parametrize("skew", [False, True])
def test_post_reduce_is_bit_exact(tokens, hidden, skew):
    mhc = _mhc()
    residual, pending, post, comb, pre = _data(tokens, hidden, tokens + hidden, skew)
    settled = mhc.expand_residual(pending, residual, post, comb)
    assert mhc.post_reduce_supported(pending, residual)

    mean = mhc.post_reduce_streams(pending, residual, post, comb)
    assert torch.equal(mean, settled.mean(dim=-2))

    collapsed = mhc.post_reduce_streams(pending, residual, post, comb, pre)
    assert torch.equal(collapsed, mhc.collapse_streams(settled, pre))

    state = mhc.SinglePassHCState(residual, pre, pending, post, comb)
    assert torch.equal(state.collapse(), mhc.collapse_streams(settled, pre))
    assert torch.equal(state.stream_mean(), settled.mean(dim=-2))


def test_post_reduce_keeps_the_leading_shape():
    mhc = _mhc()
    residual, pending, post, comb, pre = _data(10, 5120, 3, False)
    state = mhc.SinglePassHCState(
        residual.view(2, 5, 4, 5120),
        pre.view(2, 5, 4),
        pending.view(2, 5, 5120),
        post.view(2, 5, 4, 1),
        comb.view(2, 5, 4, 4),
    )
    settled = mhc.expand_residual(pending, residual, post, comb)
    expected = mhc.collapse_streams(settled, pre).view(2, 5, 5120)
    assert torch.equal(state.collapse(), expected)
    assert torch.equal(state.stream_mean(), settled.mean(dim=-2).view(2, 5, 5120))


def test_post_reduce_empty_batch():
    mhc = _mhc()
    residual, pending, post, comb, pre = _data(0, 2048, 0, False)
    assert mhc.post_reduce_streams(pending, residual, post, comb).shape == (0, 2048)
    assert mhc.post_reduce_streams(pending, residual, post, comb, pre).shape == (
        0,
        2048,
    )


def test_stream_mean_without_an_owed_post_is_the_plain_mean():
    mhc = _mhc()
    residual, _, _, _, pre = _data(5, 2048, 4, False)
    state = mhc.SinglePassHCState(residual, pre)
    assert torch.equal(state.stream_mean(), residual.mean(dim=-2))


def test_unsupported_shapes_keep_the_settle_path():
    mhc = _mhc()
    # Hidden 768: mhc_post serves it (256-column tiles), the reduce does not.
    residual, pending, post, comb, pre = _data(4, 768, 1, False)
    assert mhc.mhc_post_covers(residual)
    assert not mhc.post_reduce_supported(pending, residual)
    state = mhc.SinglePassHCState(residual, pre, pending, post, comb)
    settled = mhc.expand_residual(pending, residual, post, comb)
    assert torch.equal(state.stream_mean(), settled.mean(dim=-2))
    assert torch.equal(state.collapse(), mhc.collapse_streams(settled, pre))

    residual, pending, _, _, _ = _data(4, 2048, 1, False)
    assert not mhc.post_reduce_supported(pending, residual[:, :2])
    assert not mhc.post_reduce_supported(
        pending, residual.transpose(0, 1).contiguous().transpose(0, 1)
    )
    assert not mhc.post_reduce_supported(pending.cpu(), residual.cpu())


def test_coverage_is_what_mhc_post_takes_without_trapping():
    mhc = _mhc()
    residual, pending, post, comb, _ = _data(3, 2048, 2, False)
    assert mhc.mhc_post_covers(residual)
    assert mhc.mhc_post_covers(residual.half())
    # Fewer than two tiles of the hidden size: mhc_post traps on these, so
    # expand_residual keeps the torch body.
    for hidden in (256, 512, 1024):
        small, small_pending, small_post, small_comb, _ = _data(3, hidden, 3, False)
        assert not mhc.mhc_post_covers(small)
        # The torch body takes the post mix as [tokens, 4].
        settled = mhc.expand_residual(
            small_pending, small, small_post.squeeze(-1), small_comb
        )
        assert settled.shape == small.shape
    # FP32 streams are not mhc_post's either.
    assert not mhc.mhc_post_covers(residual.float())
    assert not mhc.post_reduce_supported(pending.float(), residual.float())
    torch_body = mhc.expand_residual(
        pending.float(), residual.float(), post.squeeze(-1), comb
    )
    assert torch_body.dtype == torch.float32


def test_post_reduce_matches_in_fp16():
    mhc = _mhc()
    residual, pending, post, comb, pre = _data(33, 5120, 5, True, torch.float16)
    settled = mhc.expand_residual(pending, residual, post, comb)
    assert torch.equal(
        mhc.post_reduce_streams(pending, residual, post, comb, pre),
        mhc.collapse_streams(settled, pre),
    )
    assert torch.equal(
        mhc.post_reduce_streams(pending, residual, post, comb),
        settled.mean(dim=-2),
    )
