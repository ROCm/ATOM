# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
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


@pytest.mark.parametrize("add,quant", [(False, False), (True, False), (True, True)])
def test_close_dispatch_and_fake_metadata(monkeypatch, ops, add, quant):
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
            quant_dtype=torch.float8_e4m3fn if quant else None,
            num_blocks=1,
            close_block=True,
        )
        assert prefix is None
        if quant:
            assert out[0].shape == x.shape and out[0].dtype == torch.float8_e4m3fn
            assert out[1].shape == (2, 1) and out[1].dtype == torch.float32
        else:
            assert out.shape == x.shape and out.dtype == x.dtype

    seen = []

    def implementation(*args, **kwargs):
        seen.append((args, kwargs))
        x = args[0]
        return x, x.new_ones(2, 1), None

    monkeypatch.setattr(ops, "_apply_attn_res_impl", implementation)
    monkeypatch.setattr(
        torch.ops.aiter, "kimi_k3_apply_attn_res_close", ops._apply_attn_res_close_op
    )
    x, bank, weight = torch.ones(2, 8), torch.ones(2, 3, 8), torch.ones(8)
    _, prefix = ops.apply_attn_res(
        x,
        bank,
        weight,
        1e-6,
        x if add else None,
        weight,
        quant_dtype=torch.float8_e4m3fn if quant else None,
        num_blocks=1,
        close_block=True,
    )
    assert prefix is None and seen[0][0][1] is bank
    assert seen[0][0][9] == 1 and seen[0][1] == {"close_block": True}


def test_close_launch_writes_original_bank_without_prefix_allocation(monkeypatch, ops):
    seen = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: seen.append((args, kwargs))

    monkeypatch.setattr(ops, "_attn_res_fused_kernel", Kernel(), raising=False)
    bank = torch.ones(4, 5, 16)[1:3, 1:4]
    x = torch.ones(2, bank.shape[-1])
    _, _, prefix = ops._apply_attn_res_impl(
        x, bank, torch.ones(x.shape[-1]), 1e-6, x, num_blocks=1, close_block=True
    )
    args, kwargs = seen[0]
    assert prefix is None and args[7] is x
    assert args[23] is bank and args[24:27] == bank.stride()
    assert kwargs["CLOSE_BLOCK"] and not kwargs["WRITE_PREF"]


@pytest.mark.parametrize("count", [None, 3])
def test_close_requires_an_unused_slot(ops, count):
    with pytest.raises(ValueError, match="inactive bank slot"):
        ops._apply_attn_res_impl(
            torch.ones(2, 8),
            torch.ones(2, 3, 8),
            torch.ones(8),
            1e-6,
            num_blocks=count,
            close_block=True,
        )


def test_only_close_schema_mutates_bank(ops):
    for suffix in ("", "_add", "_quant", "_add_quant", "_close"):
        schema = getattr(
            torch.ops.aiter, "kimi_k3_apply_attn_res" + suffix
        ).default._schema
        mutated = [
            arg.name
            for arg in schema.arguments
            if arg.alias_info and arg.alias_info.is_write
        ]
        assert mutated == (["block_residual"] if suffix == "_close" else [])
        assert all(result.alias_info is None for result in schema.returns)


def test_close_mutation_survives_compiled_calls_and_bank_views(ops):
    # A CPU implementation exercises the real schema's functionalization and
    # mutation ordering; kernel arithmetic is covered by the GPU parity test.
    def cpu_close(prefix, bank, score, eps, blocks, add=None, *unused):
        value = prefix if add is None else prefix + add
        bank[:, blocks].copy_(value)
        return value.clone(), value.new_empty(1)

    library = torch.library.Library("aiter", "IMPL")
    library.impl("kimi_k3_apply_attn_res_close", cpu_close, "CPU")

    def run(prefix, bank, add):
        mixed, remainder = ops.apply_attn_res(
            prefix,
            bank,
            torch.ones(prefix.shape[1]),
            1e-6,
            add,
            num_blocks=1,
            close_block=True,
        )
        assert remainder is None
        return mixed + bank[:, 1]

    compiled = torch.compile(run, backend="aot_eager", fullgraph=True, dynamic=True)
    for tokens in (2, 5, 2):
        storage = torch.full((tokens + 2, 5, 16), -7.0)
        bank = storage[1:-1, 1:4, :8]
        prefix = torch.arange(tokens * 8).view(tokens, 8).float()
        add = torch.full_like(prefix, 0.5)
        expected = storage.clone()
        expected[1:-1, 2, :8] = prefix + add
        result = compiled(prefix, bank, add)
        assert torch.equal(result, 2 * (prefix + add))
        assert torch.equal(storage, expected)
