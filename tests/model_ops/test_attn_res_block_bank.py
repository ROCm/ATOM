"""Attention-residual block bank: slot bookkeeping and kernels reading a bank view."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.model_ops.attention_residual import AttnRes, new_block_bank


def _closing_site(layer_idx, block_size):
    return AttnRes(
        proj=object(),
        norm=SimpleNamespace(eps=1e-6),
        block_size=block_size,
        layer_idx=layer_idx,
    )


@pytest.mark.parametrize("layers,block_size", [(25, 12), (93, 12), (7, 3)])
def test_bank_views_match_concatenated_blocks(layers, block_size):
    """Every site sees exactly the blocks the old torch.cat bank held, as views
    of one bank, and a closing layer hands back no prefix."""
    tokens, hidden = 3, 4
    num_blocks = -(-layers // block_size)
    bank = new_block_bank(torch.zeros(tokens, hidden), num_blocks)
    assert bank.shape == (tokens, num_blocks, hidden)
    assert bank[:, 0].is_contiguous()  # slot-major: each slot is a dense [T, H]
    closed = []  # what torch.cat would have accumulated
    for layer_idx in range(layers):
        site = _closing_site(layer_idx, block_size)
        before = site.blocks_before(bank)
        # a view of the bank, not a copy (data_ptr() is 0 for an empty view)
        assert before.untyped_storage().data_ptr() == bank.untyped_storage().data_ptr()
        assert before.shape[1] == len(closed)
        if closed:
            assert torch.equal(before, torch.stack(closed, dim=1))
        prefix = torch.randn(tokens, hidden)
        out = site.maybe_close_block(prefix, bank)
        if layer_idx % block_size == 0:
            assert out is None
            closed.append(prefix)
        else:
            assert out is prefix
        after = site.blocks_after(bank)
        assert after.shape[1] == len(closed)
        assert torch.equal(after, torch.stack(closed, dim=1))
    # the output site mixes every slot, and every slot was written
    assert len(closed) == num_blocks
    assert torch.equal(bank, torch.stack(closed, dim=1))


def test_disabled_sites_pass_through():
    site = AttnRes(enabled=False)
    assert site.blocks_before(None) is None
    assert site.blocks_after(None) is None
    prefix = torch.randn(2, 4)
    assert site.maybe_close_block(prefix, None) is prefix


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("gluon_tdm", [True, False])
@pytest.mark.parametrize("quant", [False, True])
@torch.inference_mode()
def test_kernels_read_bank_view_in_place(monkeypatch, gluon_tdm, quant):
    """A [:, :n] view of the slot-major bank goes to the kernel as is (no
    .contiguous() copy) and gives bit-identical results to the same candidates
    passed contiguous."""
    from atom.model_ops.kimi_k3 import attention_residual as attn_res

    tdm_module = getattr(attn_res, "_attn_res_tdm_module", None)
    if tdm_module is None:  # no gfx1250 Gluon path in this tree: Triton only
        if gluon_tdm:
            pytest.skip("no Gluon TDM attn-res path")
    else:
        monkeypatch.setitem(
            attn_res.envs.environment_variables,
            "ATOM_KIMI_K3_ATTN_RES_GLUON_TDM",
            lambda: gluon_tdm,
        )
        tdm_module.cache_clear()
    generator = torch.Generator(device="cuda").manual_seed(20261101)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cuda")

    rows, hidden, used = 301, 7168, 5
    bank = new_block_bank(randn(rows, hidden).bfloat16(), 8)
    for slot in range(used):
        bank[:, slot].copy_(randn(rows, hidden).bfloat16())
    view = bank[:, :used]
    assert not view.is_contiguous()
    prefix, add = randn(rows, hidden).bfloat16(), randn(rows, hidden).bfloat16()
    score_weight, norm_weight = randn(hidden) * 0.01, randn(hidden).bfloat16()
    kwargs = {
        "add_hidden": add,
        "out_norm_weight": norm_weight,
        "out_eps": 1e-5,
        "quant_dtype": torch.float8_e4m3fn if quant else None,
    }
    try:
        strided = attn_res._apply_attn_res_impl(
            prefix, view, score_weight, 1e-5, **kwargs
        )
        dense = attn_res._apply_attn_res_impl(
            prefix, view.contiguous(), score_weight, 1e-5, **kwargs
        )
    finally:
        if tdm_module is not None:
            tdm_module.cache_clear()
    for a, b in zip(strided, dense):
        if a.dtype == torch.float8_e4m3fn:
            a, b = a.view(torch.uint8), b.view(torch.uint8)
        assert torch.equal(a, b)
