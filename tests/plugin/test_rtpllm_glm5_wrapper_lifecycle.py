"""Lifecycle tests for the GLM5 rtp-llm wrapper."""

import ast
from contextlib import nullcontext
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

_ATOM_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_IMPORT_TIME_SPARSE_KERNELS = {
    "flashmla_sparse",
    "flash_mla",
    "sparse_mla",
    "attention_mla_sparse",
}


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _install_fake_rtp_modules() -> dict[str, ModuleType]:
    fake_config_mod = ModuleType("rtp_llm.config.model_config")

    class _FakeModelConfig:
        pass

    fake_config_mod.ModelConfig = _FakeModelConfig

    fake_factory_register_mod = ModuleType("rtp_llm.model_factory_register")
    fake_factory_register_mod.register_model = MagicMock()
    fake_factory_register_mod._model_factory = {}
    fake_factory_register_mod._hf_architecture_2_ft = {}

    fake_deepseek_mod = ModuleType("rtp_llm.models.deepseek_v2")

    class _FakeDeepSeekV2:
        def _get_device_str(self):
            return "cpu"

        def _create_python_model(self):
            self.native_create_python_model_called = True

        def load(self, skip_python_model=False):
            self.native_load_called = skip_python_model

    fake_deepseek_mod.DeepSeekV2 = _FakeDeepSeekV2

    fake_weight_info_mod = ModuleType("rtp_llm.model_loader.model_weight_info")

    class _FakeModelWeights:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.global_weights = {}

        def set_global_weight(self, name, tensor):
            self.global_weights[name] = tensor

    class _FakeModelDeployWeightInfo:
        pass

    fake_weight_info_mod.ModelDeployWeightInfo = _FakeModelDeployWeightInfo
    fake_weight_info_mod.ModelWeights = _FakeModelWeights

    fake_module_base_mod = ModuleType("rtp_llm.models_py.model_desc.module_base")

    class _FakeGptModelBase:
        def __init__(self, *args, **kwargs):
            self.init_args = args
            self.init_kwargs = kwargs

    fake_module_base_mod.GptModelBase = _FakeGptModelBase

    fake_ops_mod = ModuleType("rtp_llm.ops")

    class _FakeParallelismConfig:
        pass

    fake_ops_mod.ParallelismConfig = _FakeParallelismConfig

    fake_compute_ops_mod = ModuleType("rtp_llm.ops.compute_ops")

    class _FakePyModelInputs:
        pass

    class _FakePyModelOutputs:
        def __init__(self, hidden_states):
            self.hidden_states = hidden_states

    fake_compute_ops_mod.PyModelInputs = _FakePyModelInputs
    fake_compute_ops_mod.PyModelOutputs = _FakePyModelOutputs

    fake_weight_mod = ModuleType("rtp_llm.utils.model_weight")
    fake_weight_mod.W = SimpleNamespace(
        lm_head="lm_head",
        embedding="embedding",
        final_ln_gamma="final_ln_gamma",
    )

    fake_loader_mod = ModuleType("atom.model_loader.loader")

    class _FakeWeightsMapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_loader_mod.WeightsMapper = _FakeWeightsMapper
    fake_loader_mod.load_model_in_plugin_mode = MagicMock()

    return {
        "atom.model_loader": _package("atom.model_loader"),
        "atom.model_loader.loader": fake_loader_mod,
        "rtp_llm": _package("rtp_llm"),
        "rtp_llm.config": _package("rtp_llm.config"),
        "rtp_llm.config.model_config": fake_config_mod,
        "rtp_llm.model_factory_register": fake_factory_register_mod,
        "rtp_llm.models": _package("rtp_llm.models"),
        "rtp_llm.models.deepseek_v2": fake_deepseek_mod,
        "rtp_llm.model_loader": _package("rtp_llm.model_loader"),
        "rtp_llm.model_loader.model_weight_info": fake_weight_info_mod,
        "rtp_llm.models_py": _package("rtp_llm.models_py"),
        "rtp_llm.models_py.model_desc": _package("rtp_llm.models_py.model_desc"),
        "rtp_llm.models_py.model_desc.module_base": fake_module_base_mod,
        "rtp_llm.ops": fake_ops_mod,
        "rtp_llm.ops.compute_ops": fake_compute_ops_mod,
        "rtp_llm.utils": _package("rtp_llm.utils"),
        "rtp_llm.utils.model_weight": fake_weight_mod,
    }


def _make_wrapper_instance(cls):
    instance = cls.__new__(cls)
    instance.model_config = SimpleNamespace(
        num_layers=1,
        compute_dtype=torch.bfloat16,
    )
    instance.parallelism_config = SimpleNamespace()
    instance.max_generate_batch_size = 1
    instance.fmha_config = None
    instance.hw_kernel_config = None
    instance.device_resource_config = None
    return instance


def test_glm5_load_skip_python_model_does_not_create_atom_model():
    fake_modules = _install_fake_rtp_modules()

    with (
        patch.dict(sys.modules, fake_modules),
        patch.dict(
            os.environ,
            {"RTP_LLM_EXTERNAL_MODEL_PACKAGES": "atom.plugin.rtpllm.models"},
        ),
    ):
        sys.modules.pop("atom.plugin.rtpllm.models.glm5", None)
        module = importlib.import_module("atom.plugin.rtpllm.models.glm5")
        module = importlib.reload(module)
        instance = _make_wrapper_instance(module.ATOMGlm5Moe)
        instance._create_python_model = MagicMock()

        instance.load(skip_python_model=True)

        instance._create_python_model.assert_not_called()
        assert instance.device == "cpu"
        assert isinstance(instance.model_weights_loader, module._NoopModelWeightsLoader)
        assert isinstance(instance.weight_manager, module._NoopWeightManager)


def _patch_optional_attr(module, attr):
    if hasattr(module, attr):
        return patch.object(module, attr)
    return nullcontext(MagicMock(name=attr))


def _read_plugin_file(relative_path: str) -> str:
    return (_ATOM_ROOT / relative_path).read_text()


def test_glm5_create_python_model_lets_prepare_model_own_mla_patching():
    fake_modules = _install_fake_rtp_modules()
    fake_atom_model = MagicMock(name="atom_model")
    fake_atom_model.to.return_value = fake_atom_model
    fake_utils_mod = ModuleType("atom.plugin.rtpllm.utils")

    class _FakeRTPForwardMLAContext:
        @staticmethod
        def collect_layer_maps(model):
            return ({}, {}, {})

    fake_utils_mod.RTPForwardMLAContext = _FakeRTPForwardMLAContext

    with (
        patch.dict(
            sys.modules,
            fake_modules,
        ),
        patch.dict(
            os.environ,
            {"RTP_LLM_EXTERNAL_MODEL_PACKAGES": "atom.plugin.rtpllm.models"},
        ),
        patch.dict(sys.modules, {"atom.plugin.rtpllm.utils": fake_utils_mod}),
        patch(
            "atom.prepare_model", return_value=fake_atom_model, create=True
        ) as prepare_model,
    ):
        sys.modules.pop("atom.plugin.rtpllm.models.glm5", None)
        module = importlib.import_module("atom.plugin.rtpllm.models.glm5")
        module = importlib.reload(module)
        instance = _make_wrapper_instance(module.ATOMGlm5Moe)
        instance.device = "cpu"
        instance.weight = MagicMock()

        with (
            _patch_optional_attr(
                module, "apply_attention_mla_rtpllm_patch"
            ) as mla_patch,
            _patch_optional_attr(
                module, "apply_deepseek_mla_rtpllm_patch"
            ) as deepseek_patch,
        ):
            result = instance._create_python_model()

        prepare_model.assert_called_once_with(config=instance, engine="rtpllm")
        mla_patch.assert_not_called()
        deepseek_patch.assert_not_called()
        load_model_in_plugin_mode = fake_modules[
            "atom.model_loader.loader"
        ].load_model_in_plugin_mode
        load_model_in_plugin_mode.assert_called_once()
        assert result is instance.py_model


def test_glm5_support_cuda_graph_honors_eager_env():
    fake_modules = _install_fake_rtp_modules()

    with (
        patch.dict(sys.modules, fake_modules),
        patch.dict(
            os.environ,
            {
                "RTP_LLM_EXTERNAL_MODEL_PACKAGES": "atom.plugin.rtpllm.models",
                "ENABLE_CUDA_GRAPH": "0",
            },
        ),
    ):
        sys.modules.pop("atom.plugin.rtpllm.models.glm5", None)
        module = importlib.import_module("atom.plugin.rtpllm.models.glm5")
        module = importlib.reload(module)
        instance = _make_wrapper_instance(module.ATOMGlm5Moe)

        assert instance.support_cuda_graph() is False


def test_glm5_runtime_uses_mla_forward_context_class():
    fake_modules = _install_fake_rtp_modules()
    fake_utils_mod = ModuleType("atom.plugin.rtpllm.utils")
    marker_context_cls = object()
    fake_utils_mod.RTPForwardMLAContext = marker_context_cls

    with (
        patch.dict(sys.modules, fake_modules),
        patch.dict(sys.modules, {"atom.plugin.rtpllm.utils": fake_utils_mod}),
        patch.dict(
            os.environ,
            {"RTP_LLM_EXTERNAL_MODEL_PACKAGES": "atom.plugin.rtpllm.models"},
        ),
    ):
        sys.modules.pop("atom.plugin.rtpllm.models.glm5", None)
        module = importlib.import_module("atom.plugin.rtpllm.models.glm5")
        module = importlib.reload(module)
        module.RTPForwardContext = None

        context_cls = module._ATOMGlm5MoeRuntime._get_forward_context_cls()

    assert context_cls is marker_context_cls


def test_glm5_runtime_forward_wraps_model_call_in_rtp_context(monkeypatch):
    fake_modules = _install_fake_rtp_modules()
    expected_input_ids = torch.tensor([10, 11], dtype=torch.int64)
    position_ids = torch.tensor([5, 6], dtype=torch.int32)
    hidden_states = torch.randn(2, 4)
    events = []

    class _FakeAtomModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, *, input_ids, positions, intermediate_tensors, inputs_embeds):
            events.append(("model", bool(_FakeRTPForwardContext.in_context)))
            assert torch.equal(input_ids, expected_input_ids)
            assert torch.equal(positions, position_ids.to(torch.long))
            assert positions.dtype == torch.long
            assert intermediate_tensors is None
            assert inputs_embeds is None
            return hidden_states

    class _FakeBind:
        def __enter__(self):
            _FakeRTPForwardContext.in_context = True
            events.append(("enter", None))

        def __exit__(self, exc_type, exc, tb):
            events.append(("exit", None))
            _FakeRTPForwardContext.in_context = False

    class _FakeRTPForwardContext:
        in_context = False

        @staticmethod
        def collect_layer_maps(model):
            return ({}, {}, {})

        @staticmethod
        def bind(**kwargs):
            assert torch.equal(kwargs["positions"], position_ids.to(torch.long))
            assert kwargs["positions"].dtype == torch.long
            return _FakeBind()

    with (
        patch.dict(sys.modules, fake_modules),
        patch.dict(
            os.environ,
            {"RTP_LLM_EXTERNAL_MODEL_PACKAGES": "atom.plugin.rtpllm.models"},
        ),
    ):
        sys.modules.pop("atom.plugin.rtpllm.models.glm5", None)
        module = importlib.import_module("atom.plugin.rtpllm.models.glm5")
        module = importlib.reload(module)
        monkeypatch.setattr(module, "RTPForwardContext", _FakeRTPForwardContext)
        runtime = module._ATOMGlm5MoeRuntime(
            model_config=SimpleNamespace(max_seq_len=16),
            parallelism_config=SimpleNamespace(),
            weights=MagicMock(),
            max_generate_batch_size=2,
            atom_model=_FakeAtomModel(),
        )
        runtime.kv_cache = SimpleNamespace()
        inputs = SimpleNamespace(
            input_ids=expected_input_ids,
            input_hiddens=None,
            attention_inputs=SimpleNamespace(position_ids=position_ids),
        )

        output = runtime.forward(inputs)

    assert output.hidden_states is hidden_states
    assert events == [("enter", None), ("model", True), ("exit", None)]


def test_glm5_runtime_prepare_fmha_impl_bypasses_native_mla_factory(monkeypatch):
    fake_modules = _install_fake_rtp_modules()

    class _FakeRTPForwardContext:
        @staticmethod
        def collect_layer_maps(model):
            return ({}, {}, {})

    with (
        patch.dict(sys.modules, fake_modules),
        patch.dict(
            os.environ,
            {"RTP_LLM_EXTERNAL_MODEL_PACKAGES": "atom.plugin.rtpllm.models"},
        ),
    ):
        sys.modules.pop("atom.plugin.rtpllm.models.glm5", None)
        module = importlib.import_module("atom.plugin.rtpllm.models.glm5")
        module = importlib.reload(module)
        monkeypatch.setattr(module, "RTPForwardContext", _FakeRTPForwardContext)
        atom_model = torch.nn.Linear(1, 1)
        runtime = module._ATOMGlm5MoeRuntime(
            model_config=SimpleNamespace(max_seq_len=16),
            parallelism_config=SimpleNamespace(),
            weights=MagicMock(),
            max_generate_batch_size=2,
            atom_model=atom_model,
        )
        inputs = SimpleNamespace(attention_inputs=SimpleNamespace())

        attn_pyobj = runtime.prepare_fmha_impl(inputs, is_cuda_graph=False)

    assert attn_pyobj.fmha_params is None
    assert attn_pyobj.is_cuda_graph is False
    assert hasattr(attn_pyobj, "prepare_cuda_graph")


def test_glm5_runtime_decode_positions_prefer_sequence_lengths_plus_one():
    fake_modules = _install_fake_rtp_modules()

    with (
        patch.dict(sys.modules, fake_modules),
        patch.dict(
            os.environ,
            {"RTP_LLM_EXTERNAL_MODEL_PACKAGES": "atom.plugin.rtpllm.models"},
        ),
    ):
        sys.modules.pop("atom.plugin.rtpllm.models.glm5", None)
        module = importlib.import_module("atom.plugin.rtpllm.models.glm5")
        module = importlib.reload(module)
        runtime = object.__new__(module._ATOMGlm5MoeRuntime)
        attn_inputs = SimpleNamespace(
            input_lengths=torch.tensor([1, 2], dtype=torch.int32),
            is_prefill=False,
            sequence_lengths=torch.tensor([999, 999], dtype=torch.int32),
            sequence_lengths_plus_1_d=torch.tensor([35, 50], dtype=torch.int32),
        )

        positions = runtime._build_positions_from_attention_inputs(
            attn_inputs=attn_inputs,
            model_device=torch.device("cpu"),
        )

    assert positions.cpu().tolist() == [34, 48, 49]


def test_glm5_runtime_graph_decode_ignores_stale_position_ids():
    fake_modules = _install_fake_rtp_modules()

    with (
        patch.dict(sys.modules, fake_modules),
        patch.dict(
            os.environ,
            {"RTP_LLM_EXTERNAL_MODEL_PACKAGES": "atom.plugin.rtpllm.models"},
        ),
    ):
        sys.modules.pop("atom.plugin.rtpllm.models.glm5", None)
        module = importlib.import_module("atom.plugin.rtpllm.models.glm5")
        module = importlib.reload(module)
        runtime = object.__new__(module._ATOMGlm5MoeRuntime)
        inputs = SimpleNamespace(
            bert_embedding_inputs=None,
            attention_inputs=SimpleNamespace(
                input_lengths=torch.tensor([1, 2], dtype=torch.int32),
                is_prefill=False,
                is_cuda_graph=True,
                position_ids=torch.tensor([0, 0, 0], dtype=torch.int32),
                sequence_lengths_plus_1_d=torch.tensor([35, 50], dtype=torch.int32),
            ),
        )

        positions = runtime._extract_positions(
            inputs=inputs,
            model_device=torch.device("cpu"),
            token_num=3,
        )

    assert positions.cpu().tolist() == [34, 48, 49]


def test_rtpllm_wrapper_registers_glm5_override_and_alias():
    register_model_mock = MagicMock()

    fake_rtp_register_mod = ModuleType("rtp_llm.model_factory_register")
    fake_rtp_register_mod.register_model = register_model_mock
    fake_rtp_register_mod._model_factory = {}
    fake_rtp_register_mod._hf_architecture_2_ft = {}

    fake_atom_register_mod = ModuleType("atom.plugin.register")
    fake_atom_register_mod._ATOM_SUPPORTED_MODELS = {}

    fake_atom_deepseek_mod = ModuleType("atom.models.deepseek_v2")

    class _FakeGlmMoeDsaForCausalLM:
        pass

    fake_atom_deepseek_mod.GlmMoeDsaForCausalLM = _FakeGlmMoeDsaForCausalLM

    fake_atom_qwen_mod = ModuleType("atom.plugin.rtpllm.models.qwen3_5")

    class _FakeATOMQwen35Moe:
        pass

    fake_atom_qwen_mod.ATOMQwen35Moe = _FakeATOMQwen35Moe

    fake_atom_glm_mod = ModuleType("atom.plugin.rtpllm.models.glm5")

    class _FakeATOMGlm5Moe:
        pass

    fake_atom_glm_mod.ATOMGlm5Moe = _FakeATOMGlm5Moe

    fake_modules = {
        "rtp_llm": _package("rtp_llm"),
        "rtp_llm.models": _package("rtp_llm.models"),
        "rtp_llm.model_factory_register": fake_rtp_register_mod,
        "atom.models.deepseek_v2": fake_atom_deepseek_mod,
        "atom.plugin.register": fake_atom_register_mod,
        "atom.plugin.rtpllm.models.qwen3_5": fake_atom_qwen_mod,
        "atom.plugin.rtpllm.models.glm5": fake_atom_glm_mod,
    }

    with patch.dict(sys.modules, fake_modules):
        sys.modules.pop("atom.plugin.rtpllm.models", None)
        sys.modules.pop("atom.plugin.rtpllm.models.base_model_wrapper", None)
        importlib.import_module("atom.plugin.rtpllm.models")

        assert fake_rtp_register_mod._model_factory["glm_5"] is _FakeATOMGlm5Moe
        assert (
            fake_rtp_register_mod._hf_architecture_2_ft["GlmMoeDsaForCausalLM"]
            == "glm_5"
        )
        assert (
            fake_atom_register_mod._ATOM_SUPPORTED_MODELS["GlmMoeDsaForCausalLM"]
            is _FakeGlmMoeDsaForCausalLM
        )
        register_model_mock.assert_has_calls(
            [
                call("atom_qwen35_moe", _FakeATOMQwen35Moe, []),
                call("atom_glm5_moe", _FakeATOMGlm5Moe, []),
            ],
            any_order=False,
        )


def test_mla_attention_legacy_boundary_shape_stays_executable_during_migration():
    from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention

    q = torch.empty(2, 4, 256)
    compressed_kv = torch.empty(2, 512)
    k_pe = torch.empty(2, 64)
    positions = torch.arange(2, dtype=torch.int32)
    attention = RTPMLAAttention(mla_modules=SimpleNamespace(v_head_dim=128))

    output = attention(q, compressed_kv, k_pe, positions=positions)

    assert output.shape == (2, 4, 128)


def test_mla_attention_is_marked_as_mla_adapter():
    from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention

    assert RTPMLAAttention.use_mla is True


def test_glm5_wrapper_does_not_use_mha_or_qwen_patches():
    source = _read_plugin_file("atom/plugin/rtpllm/models/glm5.py")

    assert "RTPFullAttention" not in source
    assert "apply_attention_mha_rtpllm_patch" not in source
    assert "apply_attention_gdn_rtpllm_patch" not in source
    assert "apply_qwen3_next_rtpllm_patch" not in source


def test_glm5_wrapper_does_not_import_or_call_deepseek_mla_patch():
    source = _read_plugin_file("atom/plugin/rtpllm/models/glm5.py")

    assert "apply_deepseek_mla_rtpllm_patch" not in source


def test_rtp_mla_prepare_no_longer_contains_deepseek_forward_monkey_patch():
    assert not (
        _ATOM_ROOT / "atom/plugin/rtpllm/attention_backend/rtp_mla_prepare.py"
    ).exists()


def test_glm5_mla_backend_is_not_full_attention_adapter():
    source = _read_plugin_file(
        "atom/plugin/rtpllm/attention_backend/rtp_mla_attention.py"
    )

    assert "class RTPMLAAttention" in source
    assert "use_mla" in source
    assert "RTPFullAttention" not in source


def test_sparse_mla_backend_has_no_import_time_cuda_sparse_kernel_dependencies():
    backend_path = (
        _ATOM_ROOT / "atom/plugin/rtpllm/attention_backend/rtp_sparse_mla_backend.py"
    )
    assert backend_path.exists()

    tree = ast.parse(backend_path.read_text())
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert not any(
        forbidden in module_name.split(".")
        for module_name in imported_modules
        for forbidden in _FORBIDDEN_IMPORT_TIME_SPARSE_KERNELS
    )


def test_rtp_mla_patch_updates_deepseek_attention_symbol(monkeypatch):
    import types

    from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import (
        RTPMLAAttention,
        apply_attention_mla_rtpllm_patch,
    )

    sentinel = object()
    fake_ops = types.ModuleType("atom.model_ops")
    fake_ops.Attention = sentinel
    fake_base_attention = types.ModuleType("atom.model_ops.base_attention")
    fake_base_attention.Attention = sentinel
    fake_deepseek = types.ModuleType("atom.models.deepseek_v2")
    fake_deepseek.Attention = sentinel
    monkeypatch.setitem(sys.modules, "atom.model_ops", fake_ops)
    monkeypatch.setitem(
        sys.modules, "atom.model_ops.base_attention", fake_base_attention
    )
    monkeypatch.setitem(sys.modules, "atom.models.deepseek_v2", fake_deepseek)

    apply_attention_mla_rtpllm_patch()

    assert fake_ops.Attention is RTPMLAAttention
    assert fake_base_attention.Attention is RTPMLAAttention
    assert fake_deepseek.Attention is RTPMLAAttention
