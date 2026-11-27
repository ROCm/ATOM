# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""An active count reaches each existing attention-residual op and launch."""

import inspect
import sys
from types import SimpleNamespace

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


@pytest.mark.parametrize("quant", [False, True])
@pytest.mark.parametrize("count", [1, 2])
def test_columns_and_active_count_reach_the_same_kernel(monkeypatch, ops, quant, count):
    owner = object.__new__(ops.PeerColumnGather)
    owner.rank, owner.world, owner.rows, owner.cols = 0, 2, 2, 2
    owner.closed = False
    owner.scratch = torch.zeros(16, dtype=torch.int64)
    owner.epochs = torch.zeros(2, dtype=torch.int32)
    owner.buffer = SimpleNamespace(addresses=torch.zeros(2, dtype=torch.int64))
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={"gather": owner})
    )
    monkeypatch.setitem(
        sys.modules,
        "atom.config",
        SimpleNamespace(get_current_atom_config=lambda: config),
    )
    prefix = torch.ones(2, 4, dtype=torch.bfloat16)
    columns = torch.full((2, 2), 2, dtype=torch.bfloat16)
    bank = torch.full((2, 4, 4), float("nan"), dtype=torch.bfloat16)
    bank[:, :count].fill_(3)
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((args, kwargs))
                assert args[0] is bank and args[9] == count
                assert kwargs["GATHER"] and not kwargs["DO_ADD"]
                assert kwargs["share_ptr"] is columns
                settled = prefix + torch.cat([columns, columns], dim=-1)
                mixed = settled + args[0][:, : args[9]].sum(dim=1)
                args[3].copy_(mixed)
                args[4].fill_(1)
                args[7].copy_(settled)

            return launch

    monkeypatch.setattr(ops, "_attn_res_fused_kernel", Kernel(), raising=False)
    suffix = "_add_quant" if quant else "_add"
    monkeypatch.setattr(
        torch.ops.aiter,
        "kimi_k3_apply_attn_res" + suffix,
        getattr(ops, "_apply_attn_res" + suffix + "_op"),
    )
    mixed, settled = ops.apply_attn_res(
        prefix,
        bank,
        torch.ones(4),
        1e-6,
        ops.ColumnTail(columns, "gather", 2),
        torch.ones(4),
        1e-6,
        None,
        torch.float8_e4m3fn if quant else None,
        count,
    )
    expected = torch.full_like(prefix, 3 + 3 * count)
    output = mixed[0] if quant else mixed
    torch.testing.assert_close(output.float(), expected.to(output.dtype).float())
    assert len(launches) == 1 and torch.equal(settled, torch.full_like(prefix, 3))
