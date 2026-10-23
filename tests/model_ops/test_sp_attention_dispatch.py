# SPDX-License-Identifier: MIT
"""M3 SP FP8 attention transport and head exchange fall back off their contract."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from aiter.dist import parallel_state
from aiter.jit.utils import chip_info

from atom import config
from atom.distributed import sp_head_exchange, ulysses_sp
from atom.model_ops.minimax_m3 import attention_fp8, input_norm_fp8
from atom.plugin import prepare


@pytest.fixture
def supported(monkeypatch):
    settings = SimpleNamespace(
        torch_dtype=torch.bfloat16,
        hf_config=SimpleNamespace(
            architectures=["MiniMaxM3SparseForConditionalGeneration"]
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ulysses_sp, "_SP_WORLD_SIZE", 4)
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", "atom")
    monkeypatch.setattr(attention_fp8.dtypes, "fp8", torch.float8_e4m3fn)
    monkeypatch.setattr(input_norm_fp8, "_fused_per_token_quant", lambda *a, **k: None)
    monkeypatch.setattr(sp_head_exchange, "_exchange", lambda *args: None)
    monkeypatch.setenv("ATOM_USE_CUSTOM_ALL_GATHER", "1")
    with config.use_custom_atom_config(settings):
        yield settings


def _head_input(**overrides):
    fields = {
        "is_cuda": True,
        "ndim": 2,
        "shape": (8192, 2048),
        "dtype": torch.bfloat16,
        "is_contiguous": lambda: True,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_validated_layout_selects_fp8_without_opt_in(supported):
    assert attention_fp8.supports_m3_attention_fp8(8192)
    assert input_norm_fp8.supports_m3_fused_gemma_fp8(6144)


@pytest.mark.parametrize("case", ["cpu", "sp1", "sp2", "tp4", "dtype", "gfx", "plugin"])
def test_unsupported_runtime_keeps_existing_kernels(monkeypatch, supported, case):
    if case == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(config, "_current_atom_config", None)
        assert (
            sp_head_exchange.head_exchange_communicator(_head_input(is_cuda=False))
            is None
        )
    elif case.startswith("sp"):
        monkeypatch.setattr(ulysses_sp, "_SP_WORLD_SIZE", int(case[-1]))
    elif case == "tp4":
        monkeypatch.setattr(
            parallel_state, "get_tensor_model_parallel_world_size", lambda: 4
        )
    elif case == "dtype":
        supported.torch_dtype = torch.float16
    elif case == "gfx":
        monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx942")
    else:
        monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", "vllm")
    assert not attention_fp8.supports_m3_attention_fp8(8192)
    assert not input_norm_fp8.supports_m3_fused_gemma_fp8(6144)


def test_head_exchange_requires_m3_and_usable_collective(monkeypatch, supported):
    ca = SimpleNamespace(disabled=False, _pool={}, should_custom_ag=lambda x: True)
    group = SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=ca))
    monkeypatch.setattr(ulysses_sp, "get_sp_group", lambda: group)
    assert sp_head_exchange.head_exchange_communicator(_head_input()) is ca
    assert (
        sp_head_exchange.head_exchange_communicator(_head_input(shape=(8192, 1024)))
        is ca
    )
    for tensor in (
        _head_input(is_cuda=False),
        _head_input(shape=(4, 2048)),
        _head_input(shape=(8193, 2048)),
        _head_input(shape=(8192, 4096)),
        _head_input(dtype=torch.float16),
        _head_input(is_contiguous=lambda: False),
    ):
        assert sp_head_exchange.head_exchange_communicator(tensor) is None
    ca.disabled = True
    assert sp_head_exchange.head_exchange_communicator(_head_input()) is None
    ca.disabled = False
    ca.should_custom_ag = lambda x: False
    assert sp_head_exchange.head_exchange_communicator(_head_input()) is None
    ca.should_custom_ag = lambda x: True
    supported.hf_config.architectures = ["LlamaForCausalLM"]
    assert sp_head_exchange.head_exchange_communicator(_head_input()) is None
