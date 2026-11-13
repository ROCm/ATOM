# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""apply_attn_res must read a strided slice of a block bank exactly as its copy."""

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")

from aiter import dtypes

from atom.model_ops.kimi_k3 import attention_residual
from atom.model_ops.kimi_k3.attention_residual import _apply_attn_res_impl

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)

HIDDEN = 7168
BANK = 8  # Kimi-K3's 93 layers close a block every 12


def _inputs(tokens, seed):
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=generator) * scale).bfloat16().cuda()

    return {
        "bank": randn(tokens, BANK, HIDDEN),
        "prefix": randn(tokens, HIDDEN),
        "adds": [randn(tokens, HIDDEN) for _ in range(2)],
        "score_weight": randn(HIDDEN, scale=0.01).float(),
        "out_norm": randn(HIDDEN),
    }


def _call(x, block_residual, adds, out_norm, quant):
    return _apply_attn_res_impl(
        x["prefix"],
        block_residual,
        x["score_weight"],
        1e-6,
        add_hidden=x["adds"][0] if adds >= 1 else None,
        out_norm_weight=x["out_norm"] if out_norm else None,
        add_hidden2=x["adds"][1] if adds == 2 else None,
        quant_dtype=dtypes.fp8 if quant else None,
    )


def _assert_bitwise(got, want, what=""):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.dtype == w.dtype and g.shape == w.shape, what
        assert torch.equal(g.view(torch.uint8), w.view(torch.uint8)), what


@pytest.mark.parametrize("tokens", [1, 6, 16, 48, 130])
@pytest.mark.parametrize(
    "adds,out_norm,quant",
    [
        (0, False, False),
        (1, False, False),
        (0, True, False),
        (0, True, True),
        (2, True, False),
        (2, True, True),
    ],
)
@torch.inference_mode()
def test_bank_slice_matches_its_copy(tokens, adds, out_norm, quant):
    x = _inputs(tokens, 20261029 + tokens)
    for blocks in range(1, BANK + 1):
        view = x["bank"][:, :blocks]
        # a single token's slice is contiguous at any width
        assert view.is_contiguous() == (blocks == BANK or tokens == 1)
        want = _call(x, view.contiguous(), adds, out_norm, quant)
        _assert_bitwise(_call(x, view, adds, out_norm, quant), want, f"{blocks} blocks")


@pytest.mark.parametrize("tokens", [1, 16])
@torch.inference_mode()
def test_offset_slices_match_their_copies(tokens):
    # a slice past block 0, and one past token 0, are read in place too
    x = _inputs(tokens + 1, 20261031 + tokens)
    past_t0 = {**x, "prefix": x["prefix"][1:], "adds": [a[1:] for a in x["adds"]]}
    for inputs, view in ((x, x["bank"][:, 2:5]), (past_t0, x["bank"][1:, :3])):
        assert view.storage_offset() > 0
        want = _call(inputs, view.contiguous(), 2, True, True)
        _assert_bitwise(_call(inputs, view, 2, True, True), want)


@torch.inference_mode()
def test_the_kernel_reads_the_bank_in_place(monkeypatch):
    # what the guard is for: the launch gets the slice itself, not a copy of it
    kernel = attention_residual._attn_res_fused_kernel
    launched = []

    class Recorder:
        def __getitem__(self, grid):
            def launch(br, *args, **kwargs):
                launched.append((br.data_ptr(), br.stride()))
                return kernel[grid](br, *args, **kwargs)

            return launch

    monkeypatch.setattr(attention_residual, "_attn_res_fused_kernel", Recorder())
    x = _inputs(16, 20261032)
    view = x["bank"][:, :3]
    _call(x, view, 2, True, False)
    assert launched == [(view.data_ptr(), view.stride())]


@torch.inference_mode()
def test_strided_hidden_dim_is_still_copied():
    x = _inputs(8, 20261030)
    # [T, H, B] storage viewed as [T, B, H]: the hidden dim is not unit-stride
    bank = x["bank"].transpose(1, 2).contiguous().transpose(1, 2)
    assert bank.stride(-1) != 1
    _assert_bitwise(
        _call(x, bank, 2, True, False), _call(x, bank.contiguous(), 2, True, False)
    )
