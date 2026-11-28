# SPDX-License-Identifier: MIT
"""`SinglePassHCState.from_embeddings` in one launch: the same bits, shape,
strides and dtypes as the torch body it replaces."""

import pytest
import torch


def _mhc():
    pytest.importorskip("aiter", reason="Single-Pass mHC calls AITER kernels")
    from atom.model_ops.deepseek_v41 import mhc

    return mhc


def _torch_body(hidden, hc_mult):
    residual = (
        hidden.unsqueeze(-2)
        .expand(*hidden.shape[:-1], hc_mult, hidden.shape[-1])
        .contiguous()
    )
    pre_mix = torch.zeros(
        (*hidden.shape[:-1], hc_mult), dtype=torch.float32, device=hidden.device
    )
    pre_mix[..., 0] = 1
    return residual, pre_mix


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("shape", [(1, 5120), (17, 5120), (1344, 5120), (3, 5, 256)])
@pytest.mark.parametrize("hc_mult", [1, 4])
def test_embed_state_is_bit_exact(shape, hc_mult):
    mhc = _mhc()
    torch.manual_seed(0)
    hidden = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    hidden.view(-1)[::7] = float("nan")
    hidden.view(-1)[1::11] = float("inf")
    # The fused path is the one under test: a gate that stopped taking it
    # would leave the torch body to pass the comparison below on its own.
    assert mhc.embed_state(hidden, hc_mult) is not None
    state = mhc.SinglePassHCState.from_embeddings(hidden, hc_mult)
    residual, pre_mix = _torch_body(hidden, hc_mult)
    assert state.residual.shape == residual.shape
    assert state.residual.stride() == residual.stride()
    assert state.residual.dtype == residual.dtype
    assert torch.equal(state.residual.view(torch.int16), residual.view(torch.int16))
    assert state.pre_mix.dtype == torch.float32
    assert torch.equal(state.pre_mix, pre_mix)
    assert state.pending is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_embed_state_under_torch_compile():
    compiled = torch.compile(_mhc().embed_state, fullgraph=True, dynamic=True)
    for rows in (17, 33):
        hidden = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        fused = compiled(hidden, 4)
        assert fused is not None
        residual, pre_mix = _torch_body(hidden, 4)
        assert torch.equal(fused[0], residual)
        assert torch.equal(fused[1], pre_mix)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_embed_state_empty_and_strided_rows():
    mhc = _mhc()
    empty = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)
    state = mhc.SinglePassHCState.from_embeddings(empty, 4)
    assert state.residual.shape == (0, 4, 256) and state.pre_mix.shape == (0, 4)
    wide = torch.randn(6, 512, device="cuda", dtype=torch.bfloat16)
    strided = wide[:, :256]
    assert mhc.embed_state(strided, 4) is None  # the torch body serves it
    state = mhc.SinglePassHCState.from_embeddings(strided, 4)
    assert torch.equal(state.residual, _torch_body(strided, 4)[0])


def test_embed_state_leaves_cpu_and_non_power_of_two_hc_mult_to_torch():
    mhc = _mhc()
    assert mhc.embed_state(torch.zeros(2, 8), 4) is None
    if torch.cuda.is_available():
        assert mhc.embed_state(torch.zeros(2, 8, device="cuda"), 3) is None
