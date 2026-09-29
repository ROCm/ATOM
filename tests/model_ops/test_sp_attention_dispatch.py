# SPDX-License-Identifier: MIT
"""CPU coverage for M3 SP dispatch, optional kernels, and transport ordering."""

import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from aiter.dist import parallel_state
from aiter.jit.utils import chip_info
from aiter.ops import fused_qk_rmsnorm_group_quant, quant

from atom import config
from atom.distributed import sp_head_exchange, sp_registered_buffer, ulysses_sp
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


def test_tp_replicated_o_proj_reuses_fp8_transport_on_tp_group(monkeypatch, supported):
    supported.m3_tp_replicated_o_proj = True
    monkeypatch.setattr(ulysses_sp, "_SP_WORLD_SIZE", 1)
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: 4
    )
    ca = SimpleNamespace(disabled=False, _pool={}, should_custom_ag=lambda x: True)
    group = SimpleNamespace(
        world_size=4, device_communicator=SimpleNamespace(ca_comm=ca)
    )
    assert attention_fp8.supports_m3_attention_fp8(2048, tp_replicated_o_proj=True)
    assert input_norm_fp8.supports_m3_fused_gemma_fp8(6144, tp_replicated_o_proj=True)
    assert not attention_fp8.supports_m3_attention_fp8(8192)
    assert sp_head_exchange.head_exchange_communicator(_head_input(), group=group) is ca
    supported.m3_tp_replicated_o_proj = False
    assert (
        sp_head_exchange.head_exchange_communicator(_head_input(), group=group) is None
    )


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


def test_other_widths_and_fp8_formats_keep_existing_kernels(monkeypatch, supported):
    assert not attention_fp8.supports_m3_attention_fp8(4096)
    assert not input_norm_fp8.supports_m3_fused_gemma_fp8(4096)
    monkeypatch.setattr(attention_fp8.dtypes, "fp8", torch.float8_e4m3fnuz)
    assert not attention_fp8.supports_m3_attention_fp8(8192)
    assert not input_norm_fp8.supports_m3_fused_gemma_fp8(6144)


def _load_copy(module):
    spec = importlib.util.spec_from_file_location(
        module.__name__ + "_test", module.__file__
    )
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    return copy


def test_missing_fused_norm_entry_is_an_import_safe_fallback(monkeypatch, supported):
    monkeypatch.delattr(
        fused_qk_rmsnorm_group_quant, "fused_qk_rmsnorm_per_token_quant", raising=False
    )
    copy = _load_copy(input_norm_fp8)
    assert not copy.supports_m3_fused_gemma_fp8(6144)


def test_missing_head_kernel_is_an_import_safe_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiter.ops.sp_head_exchange", None)
    copy = _load_copy(sp_head_exchange)
    assert copy.head_exchange_communicator(_head_input()) is None


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


def test_decode_gathers_before_using_existing_per_token_quantizer(monkeypatch):
    monkeypatch.setattr(ulysses_sp, "_SP_WORLD_SIZE", 4)
    x = torch.empty((8, 4), dtype=torch.bfloat16)
    gathered = torch.empty((2, 16), dtype=torch.bfloat16)
    expected = object(), object()

    def gather(received):
        assert received.data_ptr() == x.data_ptr()
        return gathered

    def quantize(received, *, quant_dtype):
        assert received is gathered
        assert quant_dtype == attention_fp8.dtypes.fp8
        return expected

    def forbidden(*args):
        raise AssertionError("decode must not launch the prefill amax exchange")

    monkeypatch.setattr(attention_fp8, "ulysses_gather_heads", gather)
    monkeypatch.setattr(quant, "per_token_quant_hip", quantize)
    monkeypatch.setattr(attention_fp8, "head_amax", forbidden)
    assert attention_fp8.gather_heads_fp8(x) == expected


@pytest.mark.parametrize("registered", [False, True])
def test_prefill_exchanges_global_amax_before_fp8_payload(monkeypatch, registered):
    monkeypatch.setattr(ulysses_sp, "_SP_WORLD_SIZE", 4)
    group = SimpleNamespace(rank_in_group=2)
    monkeypatch.setattr(attention_fp8, "get_sp_group", lambda: group)
    x = torch.empty((8192, 4), dtype=torch.bfloat16)
    local_amax = torch.empty((8192, 1), dtype=torch.bfloat16)
    global_amax = torch.empty((32768, 1), dtype=torch.bfloat16)
    payload = torch.empty(x.shape, dtype=attention_fp8.dtypes.fp8)
    scale = torch.empty((2048, 1), dtype=torch.float32)
    output = torch.empty((2048, 8), dtype=torch.bfloat16)
    calls = []
    ca = object() if registered else None

    def amax(received):
        assert received.data_ptr() == x.data_ptr()
        calls.append("amax")
        return local_amax

    def gather_amax(received):
        assert received is local_amax
        calls.append("gather_amax")
        return global_amax

    def quantize(received, maxima, world, rank, out=None):
        assert received.data_ptr() == x.data_ptr()
        assert maxima is global_amax and (world, rank) == (4, 2)
        assert out is (payload if registered else None)
        calls.append("quantize")
        return payload, scale

    def exchange(received, *args, **kwargs):
        assert received.dtype == torch.bfloat16
        assert received.data_ptr() == payload.data_ptr()
        if registered:
            assert args == (ca,) and kwargs == {"registered": True}
        calls.append("exchange")
        return output

    monkeypatch.setattr(attention_fp8, "head_amax", amax)
    monkeypatch.setattr(attention_fp8, "_all_gather_tokens", gather_amax)
    monkeypatch.setattr(attention_fp8, "quantize_with_gathered_amax", quantize)
    monkeypatch.setattr(attention_fp8, "ulysses_gather_heads", exchange)
    monkeypatch.setattr(sp_head_exchange, "head_exchange_communicator", lambda x: ca)
    monkeypatch.setattr(sp_head_exchange, "exchange_heads", exchange)
    monkeypatch.setattr(
        sp_registered_buffer, "registered_input_view", lambda *args: (payload, ca)
    )
    result, result_scale = attention_fp8.gather_heads_fp8(x)
    assert calls == ["amax", "gather_amax", "quantize", "exchange"]
    assert result.dtype == attention_fp8.dtypes.fp8
    assert result.shape == (2048, 16)
    assert result.data_ptr() == output.data_ptr()
    assert result_scale is scale


def test_fused_norm_execution_errors_remain_visible(monkeypatch):
    error = RuntimeError("configured fused kernel failed")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(input_norm_fp8, "_fused_per_token_quant", fail)
    with pytest.raises(RuntimeError) as raised:
        input_norm_fp8.fused_m3_gemma_norm_fp8(
            torch.empty((1, 6144), dtype=torch.bfloat16),
            torch.empty(6144, dtype=torch.bfloat16),
            1e-6,
        )
    assert raised.value is error


def test_head_exchange_execution_errors_remain_visible(monkeypatch):
    error = RuntimeError("configured head exchange failed")

    def fail(*args, **kwargs):
        raise error

    ca = SimpleNamespace(
        _ptr=1,
        _IS_CAPTURING=False,
        _pool={"input": SimpleNamespace(data_ptr=2, max_size=128)},
    )
    monkeypatch.setattr(sp_head_exchange, "_exchange", fail)
    with pytest.raises(RuntimeError) as raised:
        sp_head_exchange.exchange_heads(torch.empty((4, 4), dtype=torch.bfloat16), ca)
    assert raised.value is error
