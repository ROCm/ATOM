# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""CPU coverage of ATOM's architecture dispatch and AITER output contract."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from aiter_stub import atom_under_stubs


@contextmanager
def _runtime(architecture):
    with atom_under_stubs() as load:
        arch = load("aiter.ops.triton.utils._triton.arch_info")
        arch.get_arch = Mock(return_value=architecture)
        gate_module = load("aiter.ops.triton.fusions.attn_res")
        gate_module.attn_res_gate = Mock()
        triton = load("triton")
        triton.next_power_of_2 = lambda value: 1 << (value - 1).bit_length()
        # Exercise import-time selection even when another test imported ATOM.
        # atom_under_stubs restores both sys.modules and package attributes.
        name = "atom.model_ops.kimi_k3.attention_residual"
        sys.modules.pop(name, None)
        module = load(name)
        kernel = MagicMock()
        module._attn_res_fused_kernel = kernel
        yield SimpleNamespace(
            module=module,
            get_arch=arch.get_arch,
            gate=gate_module.attn_res_gate,
            kernel=kernel,
            launch=kernel.__getitem__.return_value,
        )


@pytest.fixture
def runtime():
    with _runtime("gfx1250") as value:
        yield value


def _inputs(rows=3):
    hidden = 32
    bank = torch.empty(rows, 8, hidden, dtype=torch.bfloat16)
    # Preserve both a noncontiguous prefix and an offset bank view at the API.
    prefix = torch.empty(rows, hidden * 2, dtype=torch.bfloat16)[:, ::2]
    return prefix, bank[:, 2:5], torch.empty(hidden, dtype=torch.float32)


@pytest.mark.parametrize("architecture", ["gfx1250", "gfx950"])
def test_architecture_is_resolved_once(architecture):
    with _runtime(architecture) as runtime:
        assert runtime.module._USE_AITER_ATTN_RES == (architecture == "gfx1250")
        # A subsequent call must use the binding established at import time.
        runtime.get_arch.return_value = (
            "gfx950" if architecture == "gfx1250" else "gfx1250"
        )
        for rows in (1, 257):
            prefix, bank, weight = _inputs(rows)
            runtime.gate.return_value = (torch.empty_like(prefix), prefix)
            runtime.module._apply_attn_res_impl(prefix, bank, weight, 1e-5)
        runtime.get_arch.assert_called_once_with()
        if architecture == "gfx1250":
            assert runtime.gate.call_count == 2
            assert all(
                call.kwargs["backend"] == "arch" for call in runtime.gate.call_args_list
            )
            runtime.launch.assert_not_called()
        else:
            runtime.gate.assert_not_called()
            assert runtime.launch.call_count == 2


@pytest.mark.parametrize("add_count", [0, 1, 2])
@pytest.mark.parametrize("quant", [False, True])
def test_aiter_preserves_inputs_and_output_contract(runtime, add_count, quant):
    prefix, bank, weight = _inputs()
    norm = torch.empty(32, dtype=torch.bfloat16)
    adds = [torch.empty_like(prefix) if i < add_count else None for i in range(2)]
    prefix_out = torch.empty_like(prefix) if add_count else prefix
    dtype = torch.float8_e4m3fn if quant else None
    y = torch.empty(prefix.shape, dtype=dtype or prefix.dtype)
    scale = torch.empty((len(prefix), 1), dtype=torch.float32)
    runtime.gate.return_value = ((y, scale) if quant else y, prefix_out)

    output = runtime.module._apply_attn_res_impl(
        prefix,
        bank,
        weight,
        1e-5,
        add_hidden=adds[0],
        out_norm_weight=norm,
        out_eps=1e-4,
        add_hidden2=adds[1],
        quant_dtype=dtype,
    )

    runtime.gate.assert_called_once()
    args, kwargs = runtime.gate.call_args
    for actual, expected in zip(args[:3], (prefix, bank, weight)):
        assert actual is expected
    assert args[1].data_ptr() == bank.data_ptr()
    assert args[1].stride() == bank.stride()
    assert args[3] == 1e-5
    assert args[4] is adds[0] and args[5] is adds[1]
    assert kwargs == {
        "output_rms_weight": norm,
        "output_rms_eps": 1e-4,
        "out_quant_dtype": dtype,
        "backend": "arch",
    }
    expected = (y, scale, prefix_out) if quant else (y, prefix_out)
    assert len(output) == len(expected)
    assert all(actual is wanted for actual, wanted in zip(output, expected))
    runtime.launch.assert_not_called()


def test_aiter_launch_failure_is_not_retried_with_inline_triton(runtime):
    runtime.gate.side_effect = RuntimeError("kernel launch failed")
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        runtime.module._apply_attn_res_impl(*_inputs(), 1e-5)
    runtime.gate.assert_called_once()
    runtime.launch.assert_not_called()


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"add_hidden2": torch.empty(3, 32)}, "add_hidden2 requires add_hidden"),
        ({"quant_dtype": torch.float8_e4m3fn}, "quant_dtype requires out_norm_weight"),
    ],
)
def test_invalid_fusions_fail_before_dispatch(runtime, kwargs, error):
    with pytest.raises(ValueError, match=error):
        runtime.module._apply_attn_res_impl(*_inputs(), 1e-5, **kwargs)
    runtime.gate.assert_not_called()
    runtime.launch.assert_not_called()


def test_inline_triton_preserves_bank_view_and_bf16_layout(runtime, monkeypatch):
    monkeypatch.setattr(runtime.module, "_USE_AITER_ATTN_RES", False)
    layouts = set()
    for rows in (1, 256, 257, 512, 513):
        prefix, bank, weight = _inputs(rows)
        add = torch.empty_like(prefix)
        output, stored_prefix = runtime.module._apply_attn_res_impl(
            prefix,
            bank,
            weight,
            1e-5,
            add_hidden=add,
            add_hidden2=add,
            out_norm_weight=torch.empty(32, dtype=torch.bfloat16),
        )
        args, kwargs = runtime.launch.call_args
        assert args[0].data_ptr() == bank.data_ptr()
        assert args[0].stride() == bank.stride()
        assert output.shape == prefix.shape and output.dtype == torch.bfloat16
        assert stored_prefix is not prefix and stored_prefix.shape == prefix.shape
        assert kwargs["DO_ADD"] and kwargs["DO_ADD2"] and kwargs["WRITE_PREF"]
        assert kwargs["OUT_NORM"] and not kwargs["QUANT"]
        layouts.add(
            tuple(kwargs[key] for key in ("BL", "BD", "num_warps", "num_stages"))
        )
    assert len(layouts) == 1
    runtime.gate.assert_not_called()
