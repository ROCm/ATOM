# SPDX-License-Identifier: MIT
"""SP FP8 attention transport and head exchange fall back off their contract."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter

from atom import config


@pytest.fixture(scope="module")
def modules():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            chip_info=load("aiter.jit.utils.chip_info"),
            sp_attention=load("atom.model_ops.sp_attention"),
        )


@pytest.fixture
def supported(monkeypatch, modules):
    settings = SimpleNamespace(torch_dtype=torch.bfloat16, sequence_parallel_size=4)
    monkeypatch.setattr(modules.chip_info, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(modules.sp_attention.dtypes, "fp8", torch.float8_e4m3fn)
    monkeypatch.setenv("ATOM_USE_CUSTOM_ALL_GATHER", "1")
    with config.use_custom_atom_config(settings):
        yield settings


def test_validated_layout_selects_fp8_without_opt_in(modules, supported):
    assert modules.sp_attention.sp_fp8_attention_output_supported()


@pytest.mark.parametrize("case", ["sp_off", "dtype", "gfx"])
def test_unsupported_runtime_keeps_existing_kernels(
    monkeypatch, modules, supported, case
):
    if case == "sp_off":
        supported.sequence_parallel_size = 1
    elif case == "dtype":
        supported.torch_dtype = torch.float16
    else:
        monkeypatch.setattr(modules.chip_info, "get_gfx_runtime", lambda: "gfx942")
    assert not modules.sp_attention.sp_fp8_attention_output_supported()


def test_head_exchange_requires_usable_collective(monkeypatch, modules, supported):
    transport = modules.sp_attention
    ca = SimpleNamespace(
        disabled=False,
        should_head_exchange=lambda x: True,
        registered_input_view=lambda shape, dtype: torch.empty(shape, dtype=dtype),
    )
    group = SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=ca))
    x = torch.ones(8, 16, dtype=torch.bfloat16)

    def view(*args):
        return transport.head_exchange_view(ca, x, *args)

    assert transport.head_exchange_comm(group) is ca
    assert view(8) is x
    # Padded rows, or the FP8 payload, are staged in the registered pool.
    assert view(12).shape == (12, 16)
    assert view(8, torch.float8_e4m3fn).dtype == torch.float8_e4m3fn
    assert transport.head_exchange_view(None, x, 8) is None
    ca.should_head_exchange = lambda x: False
    assert view(8) is None
    ca.should_head_exchange = lambda x: True
    ca.registered_input_view = lambda shape, dtype: None
    assert view(12) is None and view(8) is x
    monkeypatch.setenv("ATOM_USE_CUSTOM_ALL_GATHER", "0")
    assert transport.head_exchange_comm(group) is None
