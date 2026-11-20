# SPDX-License-Identifier: MIT
"""An active count reaches each existing attention-residual op and launch."""

import inspect

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def ops():
    from torch._dynamo.backends.registry import lookup_backend

    lookup_backend("eager")
    with atom_without_aiter() as load:
        yield load("atom.model_ops.kimi_k3.attention_residual")


@pytest.mark.parametrize(
    "add,quant", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("num_blocks", [None, 2])
def test_active_count_reaches_every_op(monkeypatch, ops, add, quant, num_blocks):
    seen = []
    signature = inspect.signature(ops._apply_attn_res_impl)

    def implementation(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        seen.append(bound.arguments)
        x = args[0]
        if quant:
            return x.to(torch.float8_e4m3fn), x.new_ones(x.shape[0], 1).float(), x
        return x, x

    monkeypatch.setattr(ops, "_apply_attn_res_impl", implementation)
    for suffix in ("", "_add", "_quant", "_add_quant"):
        monkeypatch.setattr(
            torch.ops.aiter,
            "kimi_k3_apply_attn_res" + suffix,
            getattr(ops, "_apply_attn_res" + suffix + "_op"),
        )
    x, bank, weight = torch.ones(2, 8), torch.ones(2, 3, 8), torch.ones(8)
    output, prefix = ops.apply_attn_res(
        x,
        bank,
        weight,
        1e-6,
        x if add else None,
        weight,
        1e-6,
        x if add else None,
        torch.float8_e4m3fn if quant else None,
        num_blocks,
    )
    assert prefix is x and seen[0]["block_residual"] is bank
    assert seen[0]["num_blocks"] == num_blocks
    if quant:
        assert output[0].dtype == torch.float8_e4m3fn and output[1].shape == (2, 1)
    else:
        assert output is x


@pytest.mark.parametrize(
    "add,quant", [(False, False), (True, False), (False, True), (True, True)]
)
def test_active_count_preserves_fake_op_metadata(ops, add, quant):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x, bank, weight = torch.empty(2, 8), torch.empty(2, 3, 8), torch.empty(8)
        out, prefix = ops.apply_attn_res(
            x,
            bank,
            weight,
            1e-6,
            x if add else None,
            weight,
            1e-6,
            x if add else None,
            torch.float8_e4m3fn if quant else None,
            2,
        )
        assert prefix.shape == x.shape and prefix.dtype == x.dtype
        if quant:
            assert out[0].shape == x.shape and out[0].dtype == torch.float8_e4m3fn
            assert out[1].shape == (2, 1) and out[1].dtype == torch.float32
        else:
            assert out.shape == x.shape and out.dtype == x.dtype


@pytest.mark.parametrize("count", [0, 1, 3])
def test_launch_uses_active_count_and_full_bank_stride(monkeypatch, ops, count):
    seen = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: seen.append(args)

    monkeypatch.setattr(ops, "_attn_res_fused_kernel", Kernel(), raising=False)
    x, bank = torch.ones(2, 8).bfloat16(), torch.ones(2, 3, 8).bfloat16()
    ops._apply_attn_res_impl(x, bank, torch.ones(8), 1e-6, num_blocks=count)
    assert seen[0][0] is bank
    assert seen[0][9:12] == (count, count + 1, 8)
    assert seen[0][16:18] == bank.stride()[:2]


@pytest.mark.parametrize("count", [-1, 4])
def test_active_count_must_fit_bank(ops, count):
    with pytest.raises(ValueError, match="num_blocks"):
        ops._apply_attn_res_impl(
            torch.ones(2, 8), torch.ones(2, 3, 8), torch.ones(8), 1e-6, num_blocks=count
        )
