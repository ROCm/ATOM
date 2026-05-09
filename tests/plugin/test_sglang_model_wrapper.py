"""Tests for the real SGLang OOT wrapper under mocked SGLang deps."""

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


class _Obj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeLogitsProcessor:
    def __init__(self, config):
        self.config = config

    def __call__(self, input_ids, hidden_states, lm_head, forward_batch):
        return _Obj(
            input_ids=input_ids,
            hidden_states=hidden_states,
            lm_head=lm_head,
            forward_batch=forward_batch,
        )


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _make_fake_modules(*, is_last_rank: bool, setup_hook=None) -> dict[str, ModuleType]:
    sglang_pkg = _package("sglang")
    srt_pkg = _package("sglang.srt")
    layers_pkg = _package("sglang.srt.layers")
    quant_pkg = _package("sglang.srt.layers.quantization")
    model_executor_pkg = _package("sglang.srt.model_executor")

    distributed_mod = ModuleType("sglang.srt.distributed")
    distributed_mod.get_pp_group = lambda: _Obj(is_last_rank=is_last_rank)

    logits_mod = ModuleType("sglang.srt.layers.logits_processor")
    logits_mod.LogitsProcessor = _FakeLogitsProcessor
    logits_mod.LogitsProcessorOutput = object

    quant_base_mod = ModuleType("sglang.srt.layers.quantization.base_config")
    quant_base_mod.QuantizationConfig = object

    forward_batch_mod = ModuleType("sglang.srt.model_executor.forward_batch_info")
    forward_batch_mod.ForwardBatch = object
    forward_batch_mod.PPProxyTensors = object

    attn_backend_pkg = _package("atom.plugin.sglang.attention_backend")
    mla_mod = ModuleType("atom.plugin.sglang.attention_backend.sgl_attention_mla")
    mla_mod.setup_deepseek_for_sglang = setup_hook or (lambda model: None)

    return {
        "sglang": sglang_pkg,
        "sglang.srt": srt_pkg,
        "sglang.srt.distributed": distributed_mod,
        "sglang.srt.layers": layers_pkg,
        "sglang.srt.layers.logits_processor": logits_mod,
        "sglang.srt.layers.quantization": quant_pkg,
        "sglang.srt.layers.quantization.base_config": quant_base_mod,
        "sglang.srt.model_executor": model_executor_pkg,
        "sglang.srt.model_executor.forward_batch_info": forward_batch_mod,
        "atom.plugin.sglang.attention_backend": attn_backend_pkg,
        "atom.plugin.sglang.attention_backend.sgl_attention_mla": mla_mod,
    }


def _import_wrapper_module(
    monkeypatch, fake_model, *, is_last_rank=False, setup_hook=None
):
    import atom

    monkeypatch.setattr(
        atom,
        "prepare_model",
        MagicMock(return_value=fake_model),
        raising=False,
    )

    fake_modules = _make_fake_modules(
        is_last_rank=is_last_rank,
        setup_hook=setup_hook,
    )
    patcher = patch.dict(sys.modules, fake_modules)
    patcher.start()
    sys.modules.pop("atom.plugin.sglang.models.base_model_wrapper", None)
    module = importlib.import_module("atom.plugin.sglang.models.base_model_wrapper")
    module = importlib.reload(module)
    return module, patcher


def test_qwen_wrapper_forwards_forward_batch_and_pp_proxy_tensors(monkeypatch):
    """Non-DeepSeek models should receive wrapper kwargs directly."""
    fake_model = MagicMock(return_value="hidden_states")
    fake_model.lm_head = object()

    module, patcher = _import_wrapper_module(
        monkeypatch,
        fake_model,
        is_last_rank=False,
    )
    try:
        wrapper = module.Qwen3MoeForCausalLM(
            _Obj(vocab_size=32000, architectures=["Qwen3MoeForCausalLM"])
        )
        forward_batch = _Obj(tag="fb")
        pp_proxy_tensors = _Obj(hidden_states="hs", residual="res")

        result = wrapper.forward(
            input_ids="input_ids",
            positions="positions",
            forward_batch=forward_batch,
            input_embeds="input_embeds",
            pp_proxy_tensors=pp_proxy_tensors,
            attn_metadata="meta",
        )
    finally:
        patcher.stop()

    assert result == "hidden_states"
    fake_model.assert_called_once_with(
        input_ids="input_ids",
        positions="positions",
        intermediate_tensors=pp_proxy_tensors,
        inputs_embeds="input_embeds",
        forward_batch=forward_batch,
        get_embedding=False,
        pp_proxy_tensors=pp_proxy_tensors,
        attn_metadata="meta",
    )


def test_qwen_wrapper_last_rank_returns_logits_output(monkeypatch):
    """Last PP rank should run the logits processor on wrapper output."""
    fake_model = MagicMock(return_value="hidden_states")
    fake_model.lm_head = object()

    module, patcher = _import_wrapper_module(
        monkeypatch,
        fake_model,
        is_last_rank=True,
    )
    try:
        wrapper = module.Qwen3MoeForCausalLM(
            _Obj(vocab_size=32000, architectures=["Qwen3MoeForCausalLM"])
        )
        forward_batch = _Obj(tag="fb")

        result = wrapper.forward(
            input_ids="input_ids",
            positions="positions",
            forward_batch=forward_batch,
        )
    finally:
        patcher.stop()

    assert result.input_ids == "input_ids"
    assert result.hidden_states == "hidden_states"
    assert result.lm_head is fake_model.lm_head
    assert result.forward_batch is forward_batch


def test_deepseek_wrapper_sets_and_clears_forward_batch_context(monkeypatch):
    """DeepSeek path should expose forward_batch via ContextVar only during forward."""
    captured = {}

    def _forward_impl(**kwargs):
        captured["current_forward_batch"] = module.get_current_forward_batch()
        captured["kwargs"] = kwargs
        return "hidden_states"

    fake_model = MagicMock(side_effect=_forward_impl)
    fake_model.lm_head = object()
    setup_hook = MagicMock()

    module, patcher = _import_wrapper_module(
        monkeypatch,
        fake_model,
        is_last_rank=False,
        setup_hook=setup_hook,
    )
    try:
        wrapper = module.DeepseekV3ForCausalLM(
            _Obj(vocab_size=32000, architectures=["DeepseekV3ForCausalLM"])
        )
        forward_batch = _Obj(tag="fb")
        pp_proxy_tensors = _Obj(hidden_states="hs", residual="res")

        result = wrapper.forward(
            input_ids="input_ids",
            positions="positions",
            forward_batch=forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
        )
    finally:
        patcher.stop()

    setup_hook.assert_called_once_with(fake_model)
    assert result == "hidden_states"
    assert captured["current_forward_batch"] is forward_batch
    assert captured["kwargs"] == {
        "input_ids": "input_ids",
        "positions": "positions",
        "intermediate_tensors": pp_proxy_tensors,
        "inputs_embeds": None,
    }
    assert module.get_current_forward_batch() is None


def test_deepseek_wrapper_resets_forward_batch_context_on_exception(monkeypatch):
    """DeepSeek ContextVar should not leak across failed forwards."""
    fake_model = MagicMock(side_effect=RuntimeError("boom"))
    fake_model.lm_head = object()

    module, patcher = _import_wrapper_module(
        monkeypatch,
        fake_model,
        is_last_rank=False,
    )
    try:
        wrapper = module.DeepseekV3ForCausalLM(
            _Obj(vocab_size=32000, architectures=["DeepseekV3ForCausalLM"])
        )

        with pytest.raises(RuntimeError, match="boom"):
            wrapper.forward(
                input_ids="input_ids",
                positions="positions",
                forward_batch=_Obj(tag="fb"),
            )
    finally:
        patcher.stop()

    assert module.get_current_forward_batch() is None
